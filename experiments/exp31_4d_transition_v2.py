"""
Experiment 31v2: Refined 4D phase transition scan with longer chains.

Based on v1 findings:
  - Transition around beta ~ 2.0-2.5
  - High autocorrelation times at large beta — need longer chains
  - Ordering fraction jumps from ~0.12 to ~0.50
  - Entropy shows non-monotonic behavior (rises then drops then rises)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.d_orders import (
    DOrder, swap_move, bd_action_4d_fast,
    interval_entropy, mcmc_d_order
)
from causal_sets.bd_action import count_intervals_by_size


def autocorr_time(x, max_lag=500):
    x = np.array(x)
    n = len(x)
    if n < 10:
        return 1.0
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 1.0
    tau = 0.5
    for lag in range(1, min(max_lag, n // 4)):
        c = np.mean(x[:n - lag] * x[lag:]) / var
        if c < 0:
            break
        tau += c
    return max(1.0, tau)


def scan_beta(N, d, betas, n_steps, n_therm, record_every, seed=42):
    results = []
    for beta in betas:
        t0 = time.time()
        rng = np.random.default_rng(seed)

        res = mcmc_d_order(
            d=d, N=N, beta=beta,
            n_steps=n_steps, n_thermalize=n_therm,
            record_every=record_every,
            rng=rng, verbose=False
        )
        dt = time.time() - t0

        tau_S = autocorr_time(res['actions'])
        tau_H = autocorr_time(res['entropies'])
        n_eff_S = len(res['actions']) / (2 * tau_S)
        n_eff_H = len(res['entropies']) / (2 * tau_H)

        mean_S = np.mean(res['actions'])
        mean_H = np.mean(res['entropies'])
        mean_ord = np.mean(res['ordering_fracs'])
        mean_ht = np.mean(res['heights'])

        err_S = np.std(res['actions']) / np.sqrt(max(1, n_eff_S))
        err_H = np.std(res['entropies']) / np.sqrt(max(1, n_eff_H))
        err_ord = np.std(res['ordering_fracs']) / np.sqrt(max(1, n_eff_S))
        err_ht = np.std(res['heights']) / np.sqrt(max(1, n_eff_S))

        # Also compute action susceptibility (variance * beta^2) as transition indicator
        chi_S = np.var(res['actions']) * (beta ** 2 if beta > 0 else 1.0)

        row = {
            'beta': beta,
            'mean_S': mean_S, 'err_S': err_S,
            'mean_H': mean_H, 'err_H': err_H,
            'mean_ord': mean_ord, 'err_ord': err_ord,
            'mean_ht': mean_ht, 'err_ht': err_ht,
            'chi_S': chi_S,
            'accept_rate': res['accept_rate'],
            'tau_S': tau_S, 'tau_H': tau_H,
            'n_samples': len(res['actions']),
            'time_s': dt,
        }
        results.append(row)

        print(f"beta={beta:5.2f}  S={mean_S:7.2f}+-{err_S:.2f}  "
              f"H={mean_H:.3f}+-{err_H:.3f}  "
              f"ord={mean_ord:.3f}  ht={mean_ht:.1f}  "
              f"chi_S={chi_S:.1f}  acc={res['accept_rate']:.3f}  "
              f"tau={tau_S:.0f}  n_eff={n_eff_S:.0f}  ({dt:.1f}s)")

    return results


def main():
    N = 30
    d = 4

    print("=" * 90)
    print(f"Experiment 31v2: Refined 4D phase transition (N={N}, d={d})")
    print("=" * 90)

    # Fine scan around the transition with longer chains
    print(f"\n--- Fine scan: beta = 0 to 6 with longer chains ---")
    betas = np.array([0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
                      3.0, 3.5, 4.0, 5.0, 6.0])

    results = scan_beta(
        N=N, d=d, betas=betas,
        n_steps=40000, n_therm=15000, record_every=10,
        seed=42
    )

    # Also run with different seed to check reproducibility
    print(f"\n--- Reproducibility check (seed=999) at key betas ---")
    key_betas = np.array([1.5, 2.0, 2.5, 3.0, 4.0])
    results_check = scan_beta(
        N=N, d=d, betas=key_betas,
        n_steps=40000, n_therm=15000, record_every=10,
        seed=999
    )

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"FULL RESULTS (seed=42)")
    print(f"{'=' * 90}")
    print(f"{'beta':>6s}  {'S':>8s}  {'err_S':>6s}  {'H':>6s}  {'err_H':>6s}  "
          f"{'ord':>6s}  {'ht':>5s}  {'chi_S':>7s}  {'acc':>5s}  {'tau':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['beta']:6.2f}  {r['mean_S']:8.2f}  {r['err_S']:6.2f}  "
              f"{r['mean_H']:6.3f}  {r['err_H']:6.3f}  "
              f"{r['mean_ord']:6.3f}  {r['mean_ht']:5.1f}  "
              f"{r['chi_S']:7.1f}  {r['accept_rate']:5.3f}  {r['tau_S']:5.0f}")

    # Transition analysis
    H_vals = np.array([r['mean_H'] for r in results])
    ord_vals = np.array([r['mean_ord'] for r in results])
    betas_arr = np.array([r['beta'] for r in results])
    chi_vals = np.array([r['chi_S'] for r in results])

    print(f"\n--- Transition analysis ---")

    # dH/dbeta — steepest descent in entropy
    dH = np.diff(H_vals) / np.diff(betas_arr)
    for i, b in enumerate(betas_arr[:-1]):
        b_mid = (betas_arr[i] + betas_arr[i+1]) / 2
        print(f"  dH/dbeta at beta={b_mid:.2f}: {dH[i]:.4f}")

    # dord/dbeta — steepest rise in ordering
    dord = np.diff(ord_vals) / np.diff(betas_arr)
    max_dord_idx = np.argmax(dord)
    beta_c_ord = (betas_arr[max_dord_idx] + betas_arr[max_dord_idx + 1]) / 2
    print(f"\n  Max dord/dbeta at beta ~ {beta_c_ord:.2f}")

    # Susceptibility peak
    chi_peak_idx = np.argmax(chi_vals[1:]) + 1  # skip beta=0
    print(f"  Chi_S peak at beta = {betas_arr[chi_peak_idx]:.2f} (chi = {chi_vals[chi_peak_idx]:.1f})")

    print(f"\n  ESTIMATED beta_c (ordering jump) ~ {beta_c_ord:.2f}")

    # Compare with 2D
    print(f"\n--- Comparison with 2D ---")
    print(f"  2D (Glaser et al.): beta_c ~ 1.66 / (N * eps^2)")
    print(f"  4D (this work):     beta_c ~ {beta_c_ord:.2f} at N={N}")
    print(f"  4D ordering fraction: {ord_vals[0]:.3f} (beta=0) -> {ord_vals[-1]:.3f} (beta=6)")
    print(f"  4D entropy:          {H_vals[0]:.3f} (beta=0) -> {H_vals[-1]:.3f} (beta=6)")


if __name__ == '__main__':
    main()
