# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:19:41 2025

@author: DEVLINA
"""

#!/usr/bin/env python3
import os, time, random, warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve, auc,
                             confusion_matrix, mean_squared_error)
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import json
# ---------------- CONFIG ----------------
CSV_PATH = "contact_primary_school_temporal_edges.csv"   # <-- change to your dataset CSV (u,v,t)
# NOTE: GCN_IN_DIM must match the number of node features produced below
GCN_IN_DIM = 8            # deg, pagerank, clustering, betweenness, eigenvector, closeness, core, eccentricity
GCN_OUT_DIM = 64          # gcn output dim
CONTROLLER_DIM = 128      # controller GRU hidden dim
NODEGRU_HID = 64          # node GRU hidden dim
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42
BETWEENNESS_MAX_NODES = 2000   # approximate betweenness when N large
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "ct_ev_gcn_output"
PLOT_DPI = 200
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------

# ---------------- Load CSV and detect columns ----------------
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
df = pd.read_csv(CSV_PATH)
print("CSV columns:", df.columns.tolist())

cols = df.columns.tolist()
src_col = None; tgt_col = None; time_col = None
for c in cols:
    lc = c.lower()
    if lc in ("source","src","node1","u","from"): src_col = c
    if lc in ("target","tgt","node2","v","to"): tgt_col = c
    if lc in ("time","timestamp","ts","t"): time_col = c
if src_col is None or tgt_col is None:
    src_col, tgt_col = cols[0], cols[1]
if time_col is None and len(cols) >= 3:
    time_col = cols[2]
if time_col is None:
    df["_time_dummy"] = np.arange(len(df)); time_col = "_time_dummy"

df = df[[src_col, tgt_col, time_col]].dropna().copy()
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
print(f"Loaded {len(df)} events; time range {df[time_col].min()} -> {df[time_col].max()}")

# map nodes to contiguous indices
nodes = pd.unique(df[[src_col, tgt_col]].values.ravel())
nodes = list(nodes)
nodes_map = {n:i for i,n in enumerate(sorted(nodes, key=lambda x: str(x)))}
inv_map = {i:n for n,i in nodes_map.items()}
N = len(nodes_map)
print("Num nodes:", N)
src_arr = df[src_col].map(nodes_map).values.astype(np.int32)
tgt_arr = df[tgt_col].map(nodes_map).values.astype(np.int32)
time_arr = df[time_col].astype(float).values

# ---------------- Build continuous snapshots (natural timestamps) ----------------
# We'll create a snapshot every time the timestamp changes (i.e. at each unique timestamp),
# updating edge counts incrementally so edge frequency is preserved.
unique_times = np.unique(time_arr)
K = len(unique_times)
print("Unique timestamps (snapshots):", K)

A_counts = np.zeros((N,N), dtype=np.int32)
H_features = []    # list length K of N x feat
A_norm_list = []   # list length K of N x N
role_labels = []   # list length K of N

# helper: safe (approx) betweenness for large graphs
def safe_betweenness(G):
    if G.number_of_nodes() <= BETWEENNESS_MAX_NODES:
        return nx.betweenness_centrality(G, normalized=True)
    else:
        k = min(200, max(10, int(math.sqrt(G.number_of_nodes()))))
        return nx.betweenness_centrality(G, k=k, normalized=True, seed=RANDOM_SEED)

def build_features_and_adj(A_counts):
    A = A_counts.astype(float)
    # self-loop to avoid zero rows
    np.fill_diagonal(A, A.diagonal()+1.0)
    degs = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.power(degs, -0.5)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    D = np.diag(deg_inv_sqrt)
    A_norm = D.dot(A).dot(D)

    Gtmp = nx.Graph()
    Gtmp.add_nodes_from(range(N))
    rows, cols = np.nonzero(A_counts)
    edges = list(zip(rows.tolist(), cols.tolist()))
    Gtmp.add_edges_from(edges)

    # compute node attributes (robust to small/empty graphs)
    deg = np.array([Gtmp.degree(i) for i in range(N)], dtype=float).reshape(-1,1)
    if Gtmp.number_of_edges() > 0:
        try:
            pr_dict = nx.pagerank(Gtmp, max_iter=200)
            pr = np.array([pr_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
        except Exception:
            pr = np.zeros((N,1))
    else:
        pr = np.zeros((N,1))
    cl = np.array([nx.clustering(Gtmp,i) for i in range(N)], dtype=float).reshape(-1,1)
    try:
        bet_dict = safe_betweenness(Gtmp)
        bet = np.array([bet_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
    except Exception:
        bet = np.zeros((N,1))
    try:
        eig_dict = nx.eigenvector_centrality_numpy(Gtmp) if Gtmp.number_of_edges()>0 else {}
        eig = np.array([eig_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
    except Exception:
        eig = np.zeros((N,1))
    try:
        clo_dict = nx.closeness_centrality(Gtmp)
        clo = np.array([clo_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
    except Exception:
        clo = np.zeros((N,1))
    try:
        core_dict = nx.core_number(Gtmp)
        core = np.array([core_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
    except Exception:
        core = np.zeros((N,1))
    try:
        ecc_dict = nx.eccentricity(Gtmp) if nx.is_connected(Gtmp) else {i:0.0 for i in Gtmp.nodes()}
        ecc = np.array([ecc_dict.get(i,0.0) for i in range(N)], dtype=float).reshape(-1,1)
    except Exception:
        ecc = np.zeros((N,1))

    feats = np.hstack([deg, pr, cl, bet, eig, clo, core, ecc]).astype(np.float32)
    feats = np.nan_to_num(feats)
    nonzero = np.std(feats, axis=0) > 1e-8
    if np.any(nonzero):
        sc = StandardScaler()
        feats[:, nonzero] = sc.fit_transform(feats[:, nonzero])
    return feats, A_norm, Gtmp

# iterate events and create snapshots at natural timestamps
idx = 0
prev_centers = None  
for step, t in enumerate(unique_times, 1):
    # ---------------- Update adjacency counts ---------------- #
    while idx < len(time_arr) and time_arr[idx] == t:
        u = int(src_arr[idx]); v = int(tgt_arr[idx])
        A_counts[u, v] += 1
        A_counts[v, u] += 1
        idx += 1

    feats, A_norm, G_t = build_features_and_adj(A_counts)
    H_features.append(feats)
    A_norm_list.append(A_norm)

    # ---------------- Role discovery ---------------- #
    labels = None
    try:
        km = MiniBatchKMeans(
            n_clusters=3,
            random_state=RANDOM_SEED,
            batch_size=256,
            max_iter=50,
            n_init=1,
            init=prev_centers if prev_centers is not None else 'k-means++'
        )
        cluster_ids = km.fit_predict(feats)
        prev_centers = km.cluster_centers_

        # ---- Compute cluster stats ---- #
        cluster_stats = {}
        for c in np.unique(cluster_ids):
            idxs = np.where(cluster_ids == c)[0]
            cluster_stats[c] = {
                'avg_deg':  float(np.mean(feats[idxs,0])),
                'avg_bet':  float(np.mean(feats[idxs,3])),
                'avg_eig':  float(np.mean(feats[idxs,4])),
                'avg_core': float(np.mean(feats[idxs,6])),
                'size':     len(idxs)
            }

        # ---- Map clusters to roles ---- #
        central_c = max(cluster_stats, key=lambda c: cluster_stats[c]['avg_deg'] + cluster_stats[c]['avg_eig'])
        bridge_c  = max(cluster_stats, key=lambda c: cluster_stats[c]['avg_bet'])
        if central_c == bridge_c:
            sorted_by_bet = sorted(cluster_stats, key=lambda c: cluster_stats[c]['avg_bet'], reverse=True)
            bridge_c = sorted_by_bet[1] if len(sorted_by_bet) > 1 else sorted_by_bet[0]

        all_cs = set(cluster_stats.keys())
        periph_cands = list(all_cs - {central_c, bridge_c})
        periph_c = (min(cluster_stats, key=lambda c: cluster_stats[c]['size'])
                    if len(periph_cands) == 0 else periph_cands[0])

        labels = np.zeros(N, dtype=np.int64)
        for node_idx in range(N):
            cid = int(cluster_ids[node_idx])
            if cid == periph_c:
                labels[node_idx] = 0
            elif cid == bridge_c:
                labels[node_idx] = 1
            elif cid == central_c:
                labels[node_idx] = 2
            else:
                labels[node_idx] = 0

    except Exception:
        # ---- Fallback heuristic ---- #
        deg_vals = np.array([G_t.degree(i) for i in range(N)], dtype=float)
        try:
            bet_dict = nx.betweenness_centrality(G_t, k=min(100, N))
            bet_vals = np.array([bet_dict.get(i, 0.0) for i in range(N)], dtype=float)
        except Exception:
            bet_vals = np.zeros(N)

        deg_rank = deg_vals.argsort().argsort()
        bet_rank = bet_vals.argsort().argsort()

        labels = np.zeros(N, dtype=np.int64)
        for i_node in range(N):
            if bet_rank[i_node] > 0.8*(N-1):
                labels[i_node] = 1  # bridge
            elif deg_rank[i_node] > 0.8*(N-1):
                labels[i_node] = 2  # central
            else:
                labels[i_node] = 0  # periphery

    role_labels.append(labels)
    if step % 50 == 0 or step == len(unique_times):
        print(f"[{step}/{len(unique_times)}] Processed timestamp {t}, assigned roles.")

print("Prepared snapshots (first 5) label counts:")
for i in range(min(5, len(role_labels))):
    u,c = np.unique(role_labels[i], return_counts=True)
    mapping = {0:'periphery',1:'bridge',2:'central'}
    d = {mapping.get(int(k),str(k)):int(v) for k,v in zip(u,c)}
    print(f" snapshot {i}: {d}")

# ---------------- sequence length and indices ----------------
K = len(H_features)   # number of snapshots
if K < 2:
    raise ValueError("Need at least 2 snapshots in time to train and test.")
# As explained: we use H_features[0..K-2] to build H_seq (length S = K-1), train on transitions j->j+1 for j in 0..S-2,
# and final evaluation uses j = S-1 to predict role_labels[S] (final snapshot K-1).
S = K - 1
if S < 1:
    raise ValueError("Need at least 2 snapshots (S>=1).")

# ---------------- Model components ----------------
class Controller(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_w_flat):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, out_w_flat)
        self.hidden_dim = hidden_dim
    def forward(self, x, h):
        h_new = self.gru(x, h)
        wflat = self.decoder(h_new)
        return wflat, h_new

def gcn_forward(A_norm_np, X_np, W):
    X = torch.as_tensor(X_np, dtype=torch.float32, device=DEVICE)
    A = torch.as_tensor(A_norm_np, dtype=torch.float32, device=DEVICE)
    if isinstance(W, torch.Tensor):
        W_t = W.to(device=DEVICE, dtype=torch.float32)
    else:
        W_t = torch.as_tensor(W, dtype=torch.float32, device=DEVICE)
    out = A.matmul(X.matmul(W_t))
    return F.relu(out)

class NodeSeqClassifier(nn.Module):
    def __init__(self, gcn_out_dim, node_gru_hid, num_roles=3):
        super().__init__()
        self.node_gru = nn.GRU(input_size=gcn_out_dim, hidden_size=node_gru_hid, batch_first=False)
        self.classifier = nn.Sequential(
            nn.Linear(node_gru_hid, max(8, node_gru_hid//2)),
            nn.ReLU(),
            nn.Linear(max(8, node_gru_hid//2), num_roles)
        )
    def forward(self, H_seq):  # H_seq: seq_len x N x gcn_out
        out_seq, _ = self.node_gru(H_seq)
        last_hidden = out_seq[-1]  # N x hid
        logits = self.classifier(last_hidden)  # N x num_roles
        return logits, out_seq

# instantiate modules
in_dim = GCN_IN_DIM
out_dim = GCN_OUT_DIM
controller_hidden = CONTROLLER_DIM
controller = Controller(input_dim=out_dim, hidden_dim=controller_hidden, out_w_flat=in_dim*out_dim).to(DEVICE)
node_seq_clf = NodeSeqClassifier(gcn_out_dim=out_dim, node_gru_hid=NODEGRU_HID, num_roles=3).to(DEVICE)
optimizer = torch.optim.AdamW(list(controller.parameters()) + list(node_seq_clf.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

# ---------------- sequence split indices ----------------
# Train transitions j->j+1 for j in [0 .. S-2], final evaluation uses j = S-1 to predict final snapshot (index S)
if S >= 2:
    train_idx = list(range(0, S-1))   # 0..S-2
else:
    train_idx = []
val_idx = []
test_idx = [S-1]
print("Sequence steps S:", S, "train/val/test sizes:", len(train_idx), len(val_idx), len(test_idx))

# ---------------- helper metrics ----------------
def mrr_from_probs(probs, true_labels):
    rr = []
    for i in range(len(true_labels)):
        p = probs[i]
        order = np.argsort(-p)
        rank = int(np.where(order == true_labels[i])[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))

def compute_metrics_snapshot(true_roles, probs, preds):
    res = {}
    res['accuracy'] = accuracy_score(true_roles, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(true_roles, preds, average='macro', zero_division=0)
    res.update({'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1})
    try:
        y_true_oh = label_binarize(true_roles, classes=[0,1,2])
        if y_true_oh.ndim == 1:
            y_true_oh = y_true_oh.reshape(-1, 1)
        res['roc_auc_macro'] = float(roc_auc_score(y_true_oh, probs, average='macro', multi_class='ovr'))
        res['pr_auc_macro'] = float(average_precision_score(y_true_oh, probs, average='macro'))
    except Exception:
        res['roc_auc_macro'] = None
        res['pr_auc_macro'] = None
    res['mrr'] = mrr_from_probs(probs, true_roles)
    try:
        y_true_oh = label_binarize(true_roles, classes=[0,1,2])
        mse = mean_squared_error(y_true_oh, probs)
        res['mse'] = float(mse)
    except Exception:
        res['mse'] = None
    return res

# ---------------- Training loop ----------------
print("Begin training for", EPOCHS, "epochs")
best_state = None
best_val = -1.0
# ================================================================
#   TRAINING LOOP (memory-optimized)
# ================================================================
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    controller.train(); node_seq_clf.train()
    optimizer.zero_grad()

    controller_h_t = torch.zeros(controller_hidden, device=DEVICE)

    # we keep a rolling list of embeddings, never the full (S,N,out_dim)
    H_history = []
    total_loss = 0.0
    count = 0

    for j in range(S):
        if j == 0:
            summary = torch.zeros(out_dim, device=DEVICE)
        else:
            summary = H_history[-1].mean(dim=0)

        # controller + gcn step
        wflat, controller_h_t = controller(summary, controller_h_t)
        W_j = wflat.view(in_dim, out_dim)
        H_j = gcn_forward(A_norm_list[j], H_features[j], W_j)

        H_history.append(H_j.detach())   # keep small references

        # compute loss only when j in train_idx
        if j in train_idx:
            # build only the subsequence up to j
            H_sub = torch.stack(H_history[:(j+1)], dim=0).to(DEVICE)
            logits, _ = node_seq_clf(H_sub)

            labels_next = torch.tensor(role_labels[j], dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, labels_next)

            total_loss += loss
            count += 1

    if count > 0:
        total_loss = total_loss / count
        total_loss.backward()
        optimizer.step()
    else:
        total_loss = torch.tensor(0.0)

    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} loss={total_loss.item():.6f} time={elapsed:.1f}s")
# save model
torch.save(controller.state_dict(), os.path.join(OUTPUT_DIR, "controller_final.pt"))
torch.save(node_seq_clf.state_dict(), os.path.join(OUTPUT_DIR, "node_seq_clf_final.pt"))

# ---------------- Final evaluation on final snapshot ----------------
controller.eval(); node_seq_clf.eval()
with torch.no_grad():
    controller_h_t = torch.zeros(controller_hidden, device=DEVICE)
    H_seq_final = torch.zeros((S, N, out_dim), dtype=torch.float32, device=DEVICE)
    for j in range(S):
        if j == 0:
            summary = torch.zeros(out_dim, device=DEVICE)
        else:
            summary = H_seq_final[j-1].mean(dim=0)
        wflat, controller_h_t = controller(summary, controller_h_t)
        W_j = wflat.view(in_dim, out_dim)
        H_j = gcn_forward(A_norm_list[j], H_features[j], W_j)
        H_seq_final[j] = H_j

    agg_true = []
    agg_probs = []
    agg_preds = []
    metrics_rows = []

    for j in test_idx:
        # j is S-1 -> predicting labels at index S (final snapshot)
        H_sub = H_seq_final[:(j+1)]
        logits, _ = node_seq_clf(H_sub)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        true_roles = role_labels[j+1]

        # save per-node probabilities too
        df_out = pd.DataFrame({
            "node_index": np.arange(N),
            "node_id": [inv_map[i] for i in range(N)],
            "true_role": true_roles,
            "pred_role": preds
        })
        proba_df = pd.DataFrame(probs, columns=["prob_periphery","prob_bridge","prob_central"])
        df_save = pd.concat([df_out, proba_df], axis=1)
        csv_path = os.path.join(OUTPUT_DIR, f"roles_snapshot_{j+1}.csv")
        df_save.to_csv(csv_path, index=False)
        print(f"Saved predictions+probs CSV: {csv_path}")

        m = compute_metrics_snapshot(true_roles, probs, preds)
        m_row = {
            "snapshot": j+1,
            "accuracy": m['accuracy'],
            "precision_macro": m['precision_macro'],
            "recall_macro": m['recall_macro'],
            "f1_macro": m['f1_macro'],
            "mrr": m['mrr'],
            "mse": m['mse'],
            "roc_auc_macro": m['roc_auc_macro'],
            "pr_auc_macro": m['pr_auc_macro']
        }
        metrics_rows.append(m_row)
        print(f"Final snapshot {j+1} metrics:", m_row)

        agg_true.append(true_roles)
        agg_probs.append(probs)
        agg_preds.append(preds)

    # write metrics summary CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("Saved metrics summary:", metrics_csv)

    # plots & heatmap (micro-average across final snapshot)
    if len(agg_true) > 0:
        Y_true = np.vstack(agg_true).ravel()
        Y_prob = np.vstack(agg_probs)
        Y_pred = np.vstack(agg_preds).ravel()

        try:
            y_true_oh = label_binarize(Y_true, classes=[0,1,2])
            precision, recall, _ = precision_recall_curve(y_true_oh.ravel(), Y_prob.ravel())
            pr_auc_val = auc(recall, precision)
            fig = plt.figure(figsize=(6,5), dpi=PLOT_DPI)
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"PR Curve (micro)  PR-AUC={pr_auc_val:.4f}")
            plt.grid(True, linestyle='--', alpha=0.3)
            pr_pdf = os.path.join(OUTPUT_DIR, "PR_curve_micro_blue.pdf")
            fig.tight_layout(); fig.savefig(pr_pdf); plt.close(fig)
            print("Saved PR curve (blue):", pr_pdf)
        except Exception as e:
            print("Could not plot PR curve:", e)

        try:
            fpr, tpr, _ = roc_curve(y_true_oh.ravel(), Y_prob.ravel())
            roc_auc_val = auc(fpr, tpr)
            fig = plt.figure(figsize=(6,5), dpi=PLOT_DPI)
            plt.plot(fpr, tpr, color='red', lw=2); plt.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve (micro)  ROC-AUC={roc_auc_val:.4f}")
            plt.grid(True, linestyle='--', alpha=0.3)
            roc_pdf = os.path.join(OUTPUT_DIR, "ROC_curve_micro_red.pdf")
            fig.tight_layout(); fig.savefig(roc_pdf); plt.close(fig)
            print("Saved ROC curve (red):", roc_pdf)
        except Exception as e:
            print("Could not plot ROC curve:", e)

        try:
            cm = confusion_matrix(Y_true, Y_pred, labels=[0,1,2])
            fig, ax = plt.subplots(figsize=(5,4), dpi=PLOT_DPI)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['periphery','bridge','central'],
                        yticklabels=['periphery','bridge','central'])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (final snapshot)")
            hm_pdf = os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.pdf")
            fig.tight_layout(); fig.savefig(hm_pdf); plt.close(fig)
            print("Saved confusion matrix heatmap:", hm_pdf)
        except Exception as e:
            print("Could not plot heatmap:", e)
    else:
        print("No test snapshots to aggregate and plot.")

# ---------------- Save numpy artifacts ----------------
np.save(os.path.join(OUTPUT_DIR, "role_labels.npy"), np.stack(role_labels, axis=0))
np.save(os.path.join(OUTPUT_DIR, "A_norm_list.npy"), np.stack(A_norm_list, axis=0))
np.save(os.path.join(OUTPUT_DIR, "H_features.npy"), np.stack(H_features, axis=0))
print("All done. Outputs are in:", OUTPUT_DIR)
