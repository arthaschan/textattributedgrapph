#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_theory.py - smoke test for theory_analysis.py hooks (P1 / P2 / P3) on a
synthetic homophilic graph. No torch / GPU needed: validates the numpy core and
the numbers the hooks report, so the real-data CLI can be trusted on H100.

Run:
    python smoke_theory.py
"""
import numpy as np

from theory_analysis import (
    add_low_similarity_edges,
    adjacency_normalized,
    loss_decomposition,
    make_synthetic_graph,
    neighbor_similarity_analysis,
    synthetic_text_perturbation,
    text_direction_analysis,
)


def split(y, frac_train=0.1, frac_val=0.1, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    a, b = int(len(y) * frac_train), int(len(y) * (frac_train + frac_val))
    return perm[:a], perm[a:b], perm[b:]


def main():
    n, d, C = 400, 32, 4
    X, edge_index, y = make_synthetic_graph(n=n, d=d, n_class=C, intra_deg=4, inter_deg=1, seed=0)
    train, val, test = split(y)
    A_clean = adjacency_normalized(edge_index, n)

    print("=" * 70)
    print("P1  neighbor-similarity density / KL  (benign vs injected malicious edges)")
    print("=" * 70)
    n_add = max(20, int(edge_index.shape[1] * 0.05))
    ei_atk, added = add_low_similarity_edges(edge_index, X, y, n_add, seed=0)
    res = neighbor_similarity_analysis(X, edge_index, added, n, taus=(0, 1, 2))
    for k, v in res.items():
        print(f"  {k:10s}  KL(b||m)={v['kl_benign_malicious']:6.3f}  "
              f"KL(m||b)={v['kl_malicious_benign']:6.3f}  d={v['cohens_d']:6.3f}  "
              f"AUC={v['auc']:5.3f}  mean_b={v['mean_benign']:5.3f}  mean_m={v['mean_malicious']:5.3f}")

    print()
    print("=" * 70)
    print("P2  SGC attack-loss decomposition  (struct / text / cross)")
    print("=" * 70)
    # structural attack on the adjacency
    ei_atk, _ = add_low_similarity_edges(edge_index, X, y, n_add, seed=1)
    A_attacked = adjacency_normalized(ei_atk, n)
    # text attack on a subset of test nodes (synthetic: shift toward neighbor-dominant class)
    atk_ids = test[: max(1, int(len(test) * 0.4))]
    X_attacked = synthetic_text_perturbation(X, y, edge_index, atk_ids, scale=2.0, seed=0)
    dec = loss_decomposition(X, A_clean, A_attacked, X_attacked, y, train, val, test, tau=1, seed=0)
    for k in ("nll_clean", "nll_struct", "nll_text", "nll_hybrid"):
        print(f"  {k:12s} = {dec[k]:.4f}")
    print(f"  delta_struct = {dec['delta_struct']:+.4f}")
    print(f"  delta_text   = {dec['delta_text']:+.4f}")
    print(f"  delta_hybrid = {dec['delta_hybrid']:+.4f}")
    print(f"  cross_term   = {dec['cross_term']:+.4f}   (0 => additive; >0 => hybrid is harder)")
    print(f"  thm1 magnitude: clean={dec['thm1_magnitude_clean']:.4f} "
          f"struct={dec['thm1_magnitude_struct']:.4f} "
          f"text={dec['thm1_magnitude_text']:.4f} hybrid={dec['thm1_magnitude_hybrid']:.4f}")

    print()
    print("=" * 70)
    print("P3  text-attack direction vs geometric class direction")
    print("=" * 70)
    p3 = text_direction_analysis(X, X_attacked, y, edge_index, atk_ids)
    print(f"  n_attacked = {p3['n_attacked']}")
    print(f"  cos(away from own centroid)   mean={p3['cos_away_own_centroid_mean']:+.3f}  "
          f"std={p3['cos_away_own_centroid_std']:.3f}")
    print(f"  cos(toward neighbor-dom centroid) mean={p3['cos_toward_dom_centroid_mean']:+.3f}  "
          f"std={p3['cos_toward_dom_centroid_std']:.3f}")
    print()
    print("smoke done.")


if __name__ == "__main__":
    main()
