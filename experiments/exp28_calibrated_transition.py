"""
Experiment 28: BD Phase Transition with CORRECTED Action

The factor-of-4 error is fixed. Now:
  S = eps*(N - 2*eps*sum N_n*f(n,eps))

At eps=0.12, N=50: <S> ≈ 3.5 (matches Surya's 3.846)
beta_c = 1.66/(N*eps^2) should now work directly.

This is the definitive experiment for the publishable result.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.two_orders import TwoOrder
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet


def interval_entropy(cs, max_k=15):
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0:
        return 0.0
    p = dist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def chain_dim(cs):
    N = cs.n
    c = cs.longest_chain()
    return np.log(N) / np.log(c) if c > 1 else float('nan')


def n_layers(cs):
    N = cs.n
    heights = np.zeros(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:, j])[0]
        if len(preds) > 0:
            heights[j] = np.max(heights[preds]) + 1
    return len(np.unique(heights))


def main():
    rng = np.random.default_rng(42)
    eps = 0.12  # Surya's primary value

    print("=" * 80)
    print("EXPERIMENT 28: BD Phase Transition — CORRECTED Action (factor-of-4 fixed)")
    print(f"eps={eps}")
    print("=" * 80)

    # Phase 1: Validate action at beta=0
    print(f"\n--- Validation: <S> at beta=0 ---")
    for N in [30, 50, 70]:
        S_vals = []
        for _ in range(200):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            S_vals.append(bd_action_corrected(cs, eps))
        bc = 1.66 / (N * eps ** 2)
        print(f"  N={N}: <S>={np.mean(S_vals):.3f} ± {np.std(S_vals):.3f}, "
              f"beta_c={bc:.2f}")
    print(f"  Surya (N=50, eps=0.12, beta≈0): S = 3.846")

    # Phase 2: Scan beta for N=50
    N = 50
    beta_c = 1.66 / (N * eps ** 2)
    print(f"\n--- N={N}, beta_c = {beta_c:.2f} ---")

    betas = sorted(set(
        list(np.linspace(0, beta_c * 0.7, 5)) +
        list(np.linspace(beta_c * 0.7, beta_c * 1.5, 8)) +
        list(np.linspace(beta_c * 1.5, beta_c * 3.0, 5))
    ))

    print(f"\n  {'beta':>7} {'beta/bc':>8} {'<S>':>7} {'H':>6} {'d_c':>5} "
          f"{'lay':>4} {'f':>6} {'acc':>6} {'chi_S':>7}")
    print("-" * 70)

    results = []
    for beta in betas:
        t0 = time.time()
        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=50000, n_therm=25000,
                              record_every=25, rng=rng)

        H_vals, dc_vals, lay_vals, of_vals = [], [], [], []
        for cs in res['samples']:
            H_vals.append(interval_entropy(cs))
            dc_vals.append(chain_dim(cs))
            lay_vals.append(n_layers(cs))
            of_vals.append(cs.ordering_fraction())

        S_arr = res['actions']
        chi_S = np.var(S_arr) * N
        elapsed = time.time() - t0

        r = {
            'beta': beta, 'ratio': beta / beta_c,
            'S': np.mean(S_arr), 'H': np.mean(H_vals),
            'dc': np.nanmean(dc_vals), 'lay': np.mean(lay_vals),
            'of': np.mean(of_vals), 'acc': res['accept_rate'],
            'chi_S': chi_S, 'chi_H': np.var(H_vals) * N,
            'H_vals': H_vals, 'S_arr': S_arr,
        }
        results.append(r)

        marker = ""
        if r['ratio'] > 0.8 and r['ratio'] < 1.2:
            marker = " <-- NEAR beta_c"

        print(f"  {beta:>7.2f} {r['ratio']:>8.2f} {r['S']:>7.3f} {r['H']:>6.3f} "
              f"{r['dc']:>5.2f} {r['lay']:>4.1f} {r['of']:>6.3f} "
              f"{r['acc']:>6.3f} {r['chi_S']:>7.2f}{marker}  ({elapsed:.0f}s)")

    # Find transition
    chi_vals = [r['chi_S'] for r in results]
    peak = np.argmax(chi_vals)
    beta_c_meas = results[peak]['beta']

    print(f"\n  Susceptibility peak: beta = {beta_c_meas:.2f}")
    print(f"  Glaser prediction:  beta = {beta_c:.2f}")
    print(f"  Ratio: {beta_c_meas/beta_c:.2f}")

    # Phase 3: Finite-size scaling with corrected action
    print(f"\n--- Finite-size scaling ---")
    print(f"  {'N':>4} {'bc_pred':>8} {'bc_meas':>8} {'ratio':>6} {'chi_max':>8} "
          f"{'H_lo':>6} {'H_hi':>6}")
    print("-" * 55)

    for N in [30, 50, 70, 90]:
        bc = 1.66 / (N * eps ** 2)
        betas_fss = sorted(set(
            list(np.linspace(0, bc * 0.5, 3)) +
            list(np.linspace(bc * 0.5, bc * 2.0, 8)) +
            list(np.linspace(bc * 2.0, bc * 4.0, 3))
        ))

        n_mcmc = max(20000, int(40000 * (50 / N) ** 1.5))

        fss_results = []
        for beta in betas_fss:
            res = mcmc_corrected(N, beta=beta, eps=eps,
                                  n_steps=n_mcmc, n_therm=n_mcmc // 2,
                                  record_every=max(1, n_mcmc // 400), rng=rng)
            chi_S = np.var(res['actions']) * N

            H_list = [interval_entropy(cs) for cs in res['samples']]

            fss_results.append({
                'beta': beta, 'chi_S': chi_S,
                'H': np.mean(H_list), 'acc': res['accept_rate'],
            })

        chis = [r['chi_S'] for r in fss_results]
        pk = np.argmax(chis)
        bc_m = fss_results[pk]['beta']

        lo = [r for r in fss_results if r['beta'] < bc_m * 0.5]
        hi = [r for r in fss_results if r['beta'] > bc_m * 2.0]
        H_lo = np.mean([r['H'] for r in lo]) if lo else float('nan')
        H_hi = np.mean([r['H'] for r in hi]) if hi else float('nan')

        print(f"  {N:>4} {bc:>8.2f} {bc_m:>8.2f} {bc_m/bc:>6.2f} "
              f"{chis[pk]:>8.2f} {H_lo:>6.3f} {H_hi:>6.3f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
