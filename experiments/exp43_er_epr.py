"""
Experiment 43: Discrete ER=EPR — Deep Investigation

Initial finding: r=0.879 between |W[i,j]| and causal connectivity for
spacelike pairs. Partial r=0.817 controlling for spatial distance.
Random DAGs give r=0.71 (partly generic but causet is stronger).

This experiment:
1. Scale with N — does the correlation persist/strengthen?
2. Phase comparison — continuum vs crystalline
3. Null model: shuffle W while preserving spectrum, check if r drops
4. Quantitative ER=EPR: is |W[i,j]| proportional to (common connections)?
5. Does the relationship hold in 4D?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.two_orders import TwoOrder
from causal_sets.two_orders_v2 import mcmc_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.d_orders import DOrder

rng = np.random.default_rng(42)


def sj_wightman(cs):
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N))
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W


def er_epr_correlation(cs, W):
    """Compute correlation between |W[i,j]| and causal connectivity
    for spacelike-separated pairs."""
    N = cs.n
    order = cs.order
    order_int = order.astype(np.int32)

    w_vals = []
    conn_vals = []

    for i in range(N):
        for j in range(i + 1, N):
            if not order[i, j] and not order[j, i]:
                w_ij = abs(W[i, j])
                cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                w_vals.append(w_ij)
                conn_vals.append(cp + cf)

    if len(w_vals) < 10:
        return float('nan'), float('nan'), len(w_vals)

    w_arr = np.array(w_vals)
    c_arr = np.array(conn_vals)
    r = np.corrcoef(w_arr, c_arr)[0, 1]

    return r, np.mean(w_arr), len(w_vals)


def er_epr_partial(cs, W, two_order=None):
    """Partial correlation controlling for spatial distance."""
    N = cs.n
    order = cs.order
    order_int = order.astype(np.int32)

    if two_order is not None:
        v_rank = np.zeros(N, dtype=int)
        for rank, idx in enumerate(np.argsort(two_order.v)):
            v_rank[idx] = rank

    w_vals, conn_vals, dist_vals = [], [], []

    for i in range(N):
        for j in range(i + 1, N):
            if not order[i, j] and not order[j, i]:
                w_vals.append(abs(W[i, j]))
                cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                conn_vals.append(cp + cf)
                if two_order is not None:
                    dist_vals.append(abs(int(v_rank[i]) - int(v_rank[j])))
                else:
                    dist_vals.append(abs(i - j))

    if len(w_vals) < 10:
        return float('nan')

    w_arr = np.array(w_vals)
    c_arr = np.array(conn_vals)
    d_arr = np.array(dist_vals)

    r_wc = np.corrcoef(w_arr, c_arr)[0, 1]
    r_wd = np.corrcoef(w_arr, d_arr)[0, 1]
    r_cd = np.corrcoef(c_arr, d_arr)[0, 1]

    denom = np.sqrt((1 - r_wd ** 2) * (1 - r_cd ** 2))
    if denom < 1e-10:
        return float('nan')

    return (r_wc - r_wd * r_cd) / denom


def main():
    print("=" * 75)
    print("EXPERIMENT 43: Discrete ER=EPR — Deep Investigation")
    print("=" * 75)

    # Phase 1: Scaling with N
    print("\n--- Phase 1: Correlation vs N (2-orders) ---")
    print(f"  {'N':>5} {'r':>8} {'r_partial':>10} {'<|W|>':>8} {'n_pairs':>8}")
    print("-" * 45)

    for N in [20, 30, 40, 50, 60, 70]:
        r_vals, rp_vals = [], []
        for _ in range(10):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            W = sj_wightman(cs)
            r, w_mean, n_pairs = er_epr_correlation(cs, W)
            rp = er_epr_partial(cs, W, two_order=to)
            if not np.isnan(r):
                r_vals.append(r)
            if not np.isnan(rp):
                rp_vals.append(rp)

        print(f"  {N:>5} {np.mean(r_vals):>8.3f} {np.mean(rp_vals):>10.3f} "
              f"{'':>8} {len(r_vals):>8}", flush=True)

    # Phase 2: Continuum vs crystalline
    print("\n--- Phase 2: Continuum vs Crystalline ---")
    N = 40
    eps = 0.12
    beta_c = 6.64 / (N * eps ** 2)

    for beta, label in [(0, "Continuum"), (beta_c * 3, "Crystalline")]:
        res = mcmc_corrected(N, beta=beta, eps=eps, n_steps=30000,
                              n_therm=15000, record_every=100, rng=rng)
        r_vals = []
        for cs in res['samples'][-15:]:
            W = sj_wightman(cs)
            r, _, _ = er_epr_correlation(cs, W)
            if not np.isnan(r):
                r_vals.append(r)

        print(f"  {label:>15}: r = {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}", flush=True)

    # Phase 3: Null model — shuffle W off-diagonal while preserving eigenvalues
    print("\n--- Phase 3: Null model (shuffled W) ---")
    to = TwoOrder(40, rng=rng)
    cs = to.to_causet()
    W = sj_wightman(cs)

    # True correlation
    r_true, _, _ = er_epr_correlation(cs, W)

    # Shuffled: randomly permute off-diagonal elements of W
    r_shuffled = []
    for _ in range(50):
        perm = rng.permutation(40)
        W_shuf = W[np.ix_(perm, perm)]  # permute rows and columns
        r_s, _, _ = er_epr_correlation(cs, W_shuf)
        if not np.isnan(r_s):
            r_shuffled.append(r_s)

    print(f"  True r: {r_true:.3f}")
    print(f"  Shuffled r: {np.mean(r_shuffled):.3f} ± {np.std(r_shuffled):.3f}")
    print(f"  z-score: {(r_true - np.mean(r_shuffled)) / (np.std(r_shuffled) + 1e-10):.1f}")

    # Phase 4: Quantitative relationship
    print("\n--- Phase 4: Functional form ---")
    print("  Is |W[i,j]| ∝ (connections)? Or |W[i,j]| ∝ (connections)^alpha?")

    to = TwoOrder(50, rng=rng)
    cs = to.to_causet()
    W = sj_wightman(cs)

    w_vals, conn_vals = [], []
    order_int = cs.order.astype(np.int32)
    for i in range(50):
        for j in range(i + 1, 50):
            if not cs.order[i, j] and not cs.order[j, i]:
                w_vals.append(abs(W[i, j]))
                cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                conn_vals.append(cp + cf)

    # Bin by connectivity and compute mean |W|
    conn_arr = np.array(conn_vals)
    w_arr = np.array(w_vals)
    unique_conn = sorted(set(conn_arr))

    print(f"  {'connections':>12} {'<|W|>':>8} {'n_pairs':>8}")
    print("-" * 32)
    conn_means, w_means = [], []
    for c in unique_conn:
        if c > 0:
            mask = conn_arr == c
            if np.sum(mask) > 5:
                conn_means.append(c)
                w_means.append(np.mean(w_arr[mask]))
                print(f"  {c:>12} {w_means[-1]:>8.4f} {np.sum(mask):>8}")

    if len(conn_means) > 3:
        log_c = np.log(np.array(conn_means))
        log_w = np.log(np.array(w_means))
        alpha = np.polyfit(log_c, log_w, 1)[0]
        print(f"\n  Power law fit: |W| ~ conn^{alpha:.2f}")
        print(f"  Linear (alpha=1): |W| proportional to connections")
        print(f"  Sub-linear (alpha<1): weaker than proportional")

    # Phase 5: 4D test
    print("\n--- Phase 5: ER=EPR in 4D ---")

    for d in [2, 4]:
        r_vals = []
        for _ in range(10):
            if d == 2:
                to = TwoOrder(40, rng=rng)
                cs = to.to_causet()
            else:
                do = DOrder(d=4, N=30, rng=rng)
                cs = do.to_causet_fast()

            W = sj_wightman(cs)
            r, _, _ = er_epr_correlation(cs, W)
            if not np.isnan(r):
                r_vals.append(r)

        print(f"  d={d}: r = {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}", flush=True)

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
