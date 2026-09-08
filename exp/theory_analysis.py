#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
theory_analysis.py - theory experiment hooks (P1 / P2 / P3) for NSPGNN -> TAG.

These hooks quantify the three theoretical claims in docs/theory_extension_roadmap.md:

  P1 (structure-preference transfer): neighbor-similarity density / KL divergence
      between benign vs malicious edges on Omega(X), Omega(AX), Omega(A^2 X)
      (replicates NSPGNN Fig.2 / Table I).
  P2 (text attack = structural attack in embedding space): decompose the SGC
      surrogate attack loss into structural / text / cross terms.
  P3 (text-attack class direction): cosine consistency between the text-induced
      embedding shift DeltaX = E(S')-E(S) and the geometric direction
      (away from own-class centroid / toward neighbor-dominant-class centroid).

The core geometry is pure numpy/scipy (no torch) so it can be smoke-tested on CPU
(see smoke_theory.py). The CLI wraps torch-based data loading/encoding from
taglas_bench.py for real TAG data on H100.

Authoring note: the theoretical coefficients M (text term) and C (cross term) of
P2 are NOT yet derived; the P2 hook returns the *empirical* loss decomposition
(struct / text / cross) plus the Theorem-1 structural magnitude, so the
derivation can be validated against real numbers later.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:  # sparse adjacency is optional; dense fallback below
    import scipy.sparse as sp
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


# --------------------------------------------------------------------------- math
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def adjacency_normalized(edge_index, n: int, add_self_loops: bool = True):
    """Symmetrically normalized adjacency A^ = D^{-1/2} (A+I) D^{-1/2}.

    Returns a scipy sparse matrix (or dense ndarray if scipy is missing).
    Works with `A_norm @ Z` for both types.
    """
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    if add_self_loops:
        src = np.concatenate([src, np.arange(n)])
        dst = np.concatenate([dst, np.arange(n)])
    if HAVE_SCIPY:
        A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        A = (A + A.T).astype(bool).astype(np.float64)  # undirected 0/1
        deg = np.asarray(A.sum(1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D = sp.diags(dinv)
        return (D @ A @ D).tocsr()
    A = np.zeros((n, n))
    A[src, dst] = 1.0
    A = (A + A.T > 0).astype(np.float64)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return dinv[:, None] * A * dinv[None, :]


def aggregated_features(X: np.ndarray, A_norm, tau: int) -> np.ndarray:
    """A^tau X : apply the normalized adjacency tau times to X (tau=0 -> X)."""
    Z = np.asarray(X, dtype=np.float64)
    for _ in range(int(tau)):
        Z = A_norm @ Z
    return Z


def edge_cosine_sim(Z: np.ndarray, edge_index) -> np.ndarray:
    """Omega(Z)[u,v] = cosine similarity of the aggregated features of edge endpoints."""
    ei = np.asarray(edge_index)
    u = Z[ei[0]]
    v = Z[ei[1]]
    num = (u * v).sum(1)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12
    return num / den


def edge_diff(clean_ei, attacked_ei):
    """Return (added, removed) edge sets between two (2,E) edge lists.

    Robust across any attack implementation (works for TGRB attack products too).
    """
    clean = set(map(tuple, np.asarray(clean_ei).T.tolist()))
    atk = set(map(tuple, np.asarray(attacked_ei).T.tolist()))
    added = sorted(atk - clean)
    removed = sorted(clean - atk)
    return (np.asarray(added, dtype=np.int64).T if added else np.empty((2, 0), dtype=np.int64),
            np.asarray(removed, dtype=np.int64).T if removed else np.empty((2, 0), dtype=np.int64))


# --------------------------------------------------------------------------- P1 diagnostics
def _auc(scores_neg, scores_pos) -> float:
    """AUC = P(score_neg > score_pos) via Mann-Whitney U (no sklearn needed)."""
    a = np.asarray(scores_neg, dtype=np.float64)
    b = np.asarray(scores_pos, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    all_ = np.concatenate([a, b])
    order = np.argsort(all_)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(all_.size)
    R = ranks[: a.size].sum()
    U = R - a.size * (a.size + 1) / 2.0
    return U / (a.size * b.size)


def density_stats(sim_benign, sim_mal, bins: int = 50, rng: tuple = (-1.0, 1.0)):
    """KL / JS / Cohen's d / AUC / overlap between benign and malicious sim distributions.

    `sim_benign` should be the *higher* similarities (benign links), `sim_mal` the lower
    (malicious injected links). Returns a dict of scalars for one tau.
    """
    sb = np.asarray(sim_benign, dtype=np.float64)
    sm = np.asarray(sim_mal, dtype=np.float64)
    hb, _ = np.histogram(sb, bins=bins, range=rng)
    hm, _ = np.histogram(sm, bins=bins, range=rng)
    eps = 1e-9
    pb = (hb + eps) / (hb.sum() + eps * bins)
    pm = (hm + eps) / (hm.sum() + eps * bins)
    kl_bm = float(np.sum(pb * np.log(pb / pm)))  # KL(benign || malicious)
    kl_mb = float(np.sum(pm * np.log(pm / pb)))  # KL(malicious || benign)
    m = 0.5 * (pb + pm)
    js = 0.5 * (np.sum(pb * np.log(pb / m)) + np.sum(pm * np.log(pm / m)))
    mean_b, mean_m = sb.mean(), sm.mean()
    var_b, var_m = sb.var(ddof=1), sm.var(ddof=1)
    pooled = np.sqrt(0.5 * (var_b + var_m) + 1e-12)
    cohens_d = (mean_b - mean_m) / pooled
    overlap = float(np.sum(np.minimum(pb, pm)))
    return {
        "n_benign": int(sb.size),
        "n_malicious": int(sm.size),
        "mean_benign": float(mean_b),
        "mean_malicious": float(mean_m),
        "std_benign": float(np.sqrt(var_b)),
        "std_malicious": float(np.sqrt(var_m)),
        "separation": float(mean_b - mean_m),
        "cohens_d": float(cohens_d),
        "auc": float(_auc(sb, sm)),
        "kl_benign_malicious": kl_bm,
        "kl_malicious_benign": kl_mb,
        "js_divergence": float(js),
        "overlap": overlap,
        "1_overlap": float(1.0 - overlap),
        "hist_benign": hb.tolist(),
        "hist_malicious": hm.tolist(),
    }


def neighbor_similarity_analysis(X, edge_index, malicious_edges, n, taus=(0, 1, 2),
                                 bins=50, rng=(-1.0, 1.0)):
    """P1: density/KL of benign vs malicious edges on Omega(A^tau X) for each tau."""
    A_norm = adjacency_normalized(edge_index, n)
    out = {}
    for tau in taus:
        Z = aggregated_features(X, A_norm, tau)
        out[f"Omega_A{tau}X"] = density_stats(
            edge_cosine_sim(Z, edge_index),      # benign = original edges
            edge_cosine_sim(Z, malicious_edges),  # malicious = injected edges
            bins=bins, rng=rng,
        )
    return out


# --------------------------------------------------------------------------- P2 loss decomposition
def train_sgc_numpy(H, y, train_idx, val_idx, lr=0.1, epochs=500, weight_decay=1e-4,
                    seed=0, patience=50):
    """Train the SGC head W on fixed aggregated features H = A^tau X (softmax regression)."""
    rng = np.random.default_rng(seed)
    C = int(y.max()) + 1
    d = H.shape[1]
    W = rng.normal(0.0, 0.01, size=(d, C))
    best_acc, best_W, bad = -1.0, W.copy(), 0
    for ep in range(epochs):
        p = softmax(H @ W)
        loss = -np.log(p[train_idx, y[train_idx]] + 1e-12).mean()
        g = p.copy()
        g[train_idx, y[train_idx]] -= 1.0
        g /= train_idx.size
        grad = H.T @ g + weight_decay * W
        W -= lr * grad
        if ep % 10 == 0:
            acc = (p[val_idx].argmax(1) == y[val_idx]).mean()
            if acc > best_acc:
                best_acc, best_W, bad = acc, W.copy(), 0
            else:
                bad += 1
                if bad >= patience // 10:
                    break
    return best_W


def sgc_nll(H, W, y, idx):
    p = softmax(H @ W)
    return float(-np.log(p[idx, y[idx]] + 1e-12).mean())


def theorem1_magnitude(H, W, y, idx):
    """Structural attack-loss proxy from NSPGNN Thm 1:
    -||H^T (S* - Y)||_F^2  (computed without materializing the N^2 kernel K=H H^T)."""
    p = softmax(H @ W)
    delta = p.copy()
    delta[idx, y[idx]] -= 1.0
    delta /= idx.size
    return float(-np.linalg.norm(H.T @ delta) ** 2)


def loss_decomposition(X, A_norm_clean, A_norm_attacked, X_attacked, y,
                       train_idx, val_idx, test_idx, tau=1, seed=0):
    """P2: decompose SGC attack-loss increase into structural / text / cross terms.

    Returns empirical NLL deltas (struct / text / hybrid / cross) plus the
    Theorem-1 structural magnitude for each configuration.
    """
    H = aggregated_features(X, A_norm_clean, tau)
    H_struct = aggregated_features(X, A_norm_attacked, tau)
    H_text = aggregated_features(X_attacked, A_norm_clean, tau)
    H_hybrid = aggregated_features(X_attacked, A_norm_attacked, tau)

    W = train_sgc_numpy(H, y, train_idx, val_idx, seed=seed)

    nll_clean = sgc_nll(H, W, y, test_idx)
    nll_struct = sgc_nll(H_struct, W, y, test_idx)
    nll_text = sgc_nll(H_text, W, y, test_idx)
    nll_hybrid = sgc_nll(H_hybrid, W, y, test_idx)

    d_struct = nll_struct - nll_clean
    d_text = nll_text - nll_clean
    d_hybrid = nll_hybrid - nll_clean
    cross = d_hybrid - d_struct - d_text  # >0 => superadditive (hybrid harder)

    return {
        "tau": int(tau),
        "nll_clean": nll_clean,
        "nll_struct": nll_struct,
        "nll_text": nll_text,
        "nll_hybrid": nll_hybrid,
        "delta_struct": d_struct,
        "delta_text": d_text,
        "delta_hybrid": d_hybrid,
        "cross_term": cross,
        "thm1_magnitude_clean": theorem1_magnitude(H, W, y, test_idx),
        "thm1_magnitude_struct": theorem1_magnitude(H_struct, W, y, test_idx),
        "thm1_magnitude_text": theorem1_magnitude(H_text, W, y, test_idx),
        "thm1_magnitude_hybrid": theorem1_magnitude(H_hybrid, W, y, test_idx),
    }


# --------------------------------------------------------------------------- P3 text direction
def _class_centroids(X, y):
    centroids = {}
    for c in np.unique(y):
        centroids[int(c)] = X[y == c].mean(0)
    return centroids


def _neighbor_dominant_class(v, y, edge_index, n):
    """Most frequent neighbor class (excluding own class); None if no such neighbor."""
    ei = np.asarray(edge_index)
    neigh = ei[1][ei[0] == v]
    if neigh.size == 0:
        return None
    ny = y[neigh]
    own = y[v]
    others = ny[ny != own]
    if others.size == 0:
        return None
    counts = np.bincount(others, minlength=int(y.max()) + 1)
    return int(counts.argmax())


def text_direction_analysis(X, X_attacked, y, edge_index, attacked_ids):
    """P3: cosine between text-induced shift dX and the geometric class direction.

    - cos_away_own  : cosine(dX, X[v] - centroid[own])   (moving away from own centroid)
    - cos_toward_dom: cosine(dX, centroid[dom] - X[v])   (moving toward neighbor-dominant centroid)
    """
    centroids = _class_centroids(X, y)
    cos_away, cos_toward = [], []
    for v in attacked_ids:
        dX = np.asarray(X_attacked[v], dtype=np.float64) - np.asarray(X[v], dtype=np.float64)
        nrm = np.linalg.norm(dX)
        if nrm < 1e-9:
            continue
        own = int(y[v])
        away = np.asarray(X[v]) - centroids[own]
        if np.linalg.norm(away) > 1e-9:
            cos_away.append(float(dX @ away / (nrm * np.linalg.norm(away))))
        dom = _neighbor_dominant_class(v, y, edge_index, int(y.max()) + 1)
        if dom is not None:
            toward = centroids[dom] - np.asarray(X[v])
            if np.linalg.norm(toward) > 1e-9:
                cos_toward.append(float(dX @ toward / (nrm * np.linalg.norm(toward))))
    return {
        "n_attacked": int(len(attacked_ids)),
        "cos_away_own_centroid_mean": float(np.mean(cos_away)) if cos_away else float("nan"),
        "cos_away_own_centroid_std": float(np.std(cos_away)) if cos_away else float("nan"),
        "cos_toward_dom_centroid_mean": float(np.mean(cos_toward)) if cos_toward else float("nan"),
        "cos_toward_dom_centroid_std": float(np.std(cos_toward)) if cos_toward else float("nan"),
        "cos_away_own_centroid": [float(x) for x in cos_away],
        "cos_toward_dom_centroid": [float(x) for x in cos_toward],
    }


# --------------------------------------------------------------------------- synthetic utilities (smoke)
def make_synthetic_graph(n=300, d=32, n_class=4, intra_deg=4, inter_deg=1, seed=0):
    """Homophilic Gaussian-blob graph for smoke testing the hooks."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_class, size=n)
    centers = rng.normal(0, 2.0, size=(n_class, d))
    X = centers[y] + rng.normal(0, 0.6, size=(n, d))
    edges = []
    for v in range(n):
        same = np.where(y == y[v])[0]
        same = same[same != v]
        for u in rng.choice(same, size=min(intra_deg, same.size), replace=False):
            edges.append((v, int(u)))
        diff = np.where(y != y[v])[0]
        if diff.size:
            for u in rng.choice(diff, size=min(inter_deg, diff.size), replace=False):
                edges.append((v, int(u)))
    edge_index = np.asarray(edges, dtype=np.int64).T
    return X, edge_index, y


def add_low_similarity_edges(edge_index, X, y, n_add, seed=0):
    """Structural attack proxy: add edges between the lowest Omega(AX)-similarity pairs
    (NSPGNN Thm 1 attack preference), returning (new_ei, added_ei)."""
    rng = np.random.default_rng(seed)
    A_norm = adjacency_normalized(edge_index, X.shape[0])
    Z = aggregated_features(X, A_norm, tau=1)
    existing = set(map(tuple, np.asarray(edge_index).T.tolist()))
    # candidate low-similarity pairs: sample cross-class pairs, keep lowest sim
    cand = set()
    while len(cand) < n_add * 20:
        u, v = int(rng.integers(0, X.shape[0])), int(rng.integers(0, X.shape[0]))
        if u == v or y[u] == y[v]:
            continue
        if (u, v) in existing or (v, u) in existing:
            continue
        cand.add((u, v))
    cand = sorted(cand, key=lambda e: _cos(Z[e[0]], Z[e[1]]))[:n_add]
    added = np.asarray(cand, dtype=np.int64).T if cand else np.empty((2, 0), dtype=np.int64)
    new_ei = np.concatenate([np.asarray(edge_index), added], axis=1)
    return new_ei, added


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def synthetic_text_perturbation(X, y, edge_index, node_ids, scale=1.5, seed=0):
    """Shift attacked nodes' embeddings toward their neighbor-dominant class centroid
    (mimics a neighborhood-aware text attack). Returns X_attacked."""
    rng = np.random.default_rng(seed)
    centroids = _class_centroids(X, y)
    Xa = X.copy()
    for v in node_ids:
        dom = _neighbor_dominant_class(v, y, edge_index, int(y.max()) + 1)
        if dom is None:
            continue
        dX = centroids[dom] - X[v]
        dX = dX / (np.linalg.norm(dX) + 1e-12) * scale
        Xa[v] = X[v] + dX + rng.normal(0, 0.2, size=X.shape[1])
    return Xa


# --------------------------------------------------------------------------- CLI (real TAG data, H100)
def run_real_p1(args, cfg):
    """Load TAG data -> encode -> structural attack -> edge diff -> P1 density/KL table."""
    import torch  # noqa: F401  (torch-only path)
    from taglas_bench import load_data, get_features, add_low_sim_edges as _als, heuristic_perturb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data(os.path.join(cfg["data_dir"], f"{cfg['dataset']}.pt"))
    n = d["y"].shape[0]
    X = get_features(d, args.encoder, os.path.join(cfg["cache_dir"], args.dataset, args.encoder + ".pt"), device)
    X_np = X.cpu().numpy()
    ei = d["edge_index"]
    if args.attack == "pgd_evasion":
        ei_atk = _als(ei, X, n, int(ei.shape[1] * args.ptb), args.seed, device)
    else:
        ei_atk = heuristic_perturb(ei, d["y"], n, args.ptb, args.seed)
    clean_np = ei.cpu().numpy()
    atk_np = ei_atk.cpu().numpy()
    added, removed = edge_diff(clean_np, atk_np)
    print(f"[p1] dataset={args.dataset} encoder={args.encoder} attack={args.attack} ptb={args.ptb}")
    print(f"[p1] edges: clean={clean_np.shape[1]} attacked={atk_np.shape[1]} added={added.shape[1]} removed={removed.shape[1]}")
    taus = [int(t) for t in args.taus.split(",")]
    res = neighbor_similarity_analysis(X_np, clean_np, added, n, taus=taus)
    for k, v in res.items():
        print(f"[p1] {k}: KL(benign||mal)={v['kl_benign_malicious']:.3f} "
              f"JS={v['js_divergence']:.3f} d={v['cohens_d']:.3f} AUC={v['auc']:.3f} "
              f"mean_b={v['mean_benign']:.3f} mean_m={v['mean_malicious']:.3f}")
    json.dump(res, open(os.path.join(cfg["out_dir"], args.dataset, args.encoder, f"p1_{args.attack}_{args.ptb}.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description="theory hooks (P1/P2/P3) for NSPGNN -> TAG")
    ap.add_argument("--task", default="p1", choices=["p1", "p2", "p3"])
    ap.add_argument("--dataset", default="cora_node")
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--attack", default="pgd_evasion", choices=["pgd_evasion", "heur_poison"])
    ap.add_argument("--ptb", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--taus", default="0,1,2")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.task == "p1":
        run_real_p1(args, cfg)
    else:
        raise SystemExit(
            f"[theory_analysis] real-data P2/P3 need the text-attack pipeline + re-encoding "
            f"(vLLM). Use the numpy functions loss_decomposition() / text_direction_analysis() "
            f"directly with X_attacked from re-encoded attacked texts (see smoke_theory.py)."
        )


if __name__ == "__main__":
    main()
