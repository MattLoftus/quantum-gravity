"""
Experiment 31: Phase transition in 4-orders (4D causal sets).

EXPLORATORY — first scan of MCMC on 4-orders with the 4D BD action.

A 4-order is the intersection of 4 total orders, which embeds faithfully
in 4D Minkowski spacetime. We scan coupling beta to look for a phase
transition analogous to the 2D case (Glaser et al. 2018).

Key question: does interval entropy H show a sharp transition in 4D?

Setup:
  - N = 30 (small, since 4-orders are more expensive)
  - d = 4 (4D Minkowski embedding)
  - 4D BD action: S = N/24 - 4L/24 + 6*I_2/24 - 4*I_3/24
  - MCMC move: swap two elements in one coordinate
  - Observables: interval entropy H, ordering fraction, height, action
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.d_orders import (
    DOrder, swap_move, bd_action_4d_fast,
    interval_entropy, mcmc_d_order
)
from causal_sets.fast_core import FastCausalSet


def autocorr_time(x, max_lag=200):
    """Estimate integrated autocorrelation time."""
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
    """Scan over beta values and collect observables."""
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

        # Compute means and errors
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

        row = {
            'beta': beta,
            'mean_S': mean_S,
            'err_S': err_S,
            'mean_H': mean_H,
            'err_H': err_H,
            'mean_ord': mean_ord,
            'err_ord': err_ord,
            'mean_ht': mean_ht,
            'err_ht': err_ht,
            'accept_rate': res['accept_rate'],
            'tau_S': tau_S,
            'tau_H': tau_H,
            'n_samples': len(res['actions']),
            'time_s': dt,
        }
        results.append(row)

        print(f"beta={beta:6.2f}  S={mean_S:7.2f}+-{err_S:.2f}  "
              f"H={mean_H:.3f}+-{err_H:.3f}  "
              f"ord={mean_ord:.3f}+-{err_ord:.3f}  "
              f"ht={mean_ht:.1f}+-{err_ht:.1f}  "
              f"acc={res['accept_rate']:.3f}  "
              f"tau_S={tau_S:.1f}  ({dt:.1f}s)")

    return results


def main():
    N = 30
    d = 4

    print(f"=" * 80)
    print(f"Experiment 31: 4D phase transition scan")
    print(f"  N = {N}, d = {d} (4-orders)")
    print(f"  4D BD action: S = N/24 - 4L/24 + 6*I_2/24 - 4*I_3/24")
    print(f"=" * 80)

    # First: characterize beta=0 (uniform random 4-orders)
    print(f"\n--- Phase 1: Random 4-order baseline (beta=0) ---")
    rng = np.random.default_rng(42)
    n_random = 200
    S_vals, H_vals, ord_vals, ht_vals = [], [], [], []
    for _ in range(n_random):
        do = DOrder(d, N, rng=rng)
        cs = do.to_causet_fast()
        S_vals.append(bd_action_4d_fast(cs))
        H_vals.append(interval_entropy(cs))
        ord_vals.append(cs.ordering_fraction())
        ht_vals.append(cs.longest_chain())

    print(f"  Random 4-order (N={N}): ")
    print(f"    S = {np.mean(S_vals):.3f} +- {np.std(S_vals)/np.sqrt(n_random):.3f}")
    print(f"    H = {np.mean(H_vals):.3f} +- {np.std(H_vals)/np.sqrt(n_random):.3f}")
    print(f"    ordering = {np.mean(ord_vals):.4f} +- {np.std(ord_vals)/np.sqrt(n_random):.4f}")
    print(f"    height = {np.mean(ht_vals):.2f} +- {np.std(ht_vals)/np.sqrt(n_random):.2f}")

    # Phase 2: Coarse scan to find transition region
    print(f"\n--- Phase 2: Coarse beta scan ---")
    # 4D action scale is ~1/24 * N, so action values are smaller than 2D
    # Start with a wide range
    betas_coarse = np.array([0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])

    results_coarse = scan_beta(
        N=N, d=d, betas=betas_coarse,
        n_steps=15000, n_therm=5000, record_every=10,
        seed=42
    )

    # Find where entropy changes most (rough transition location)
    H_vals_scan = [r['mean_H'] for r in results_coarse]
    dH = [abs(H_vals_scan[i+1] - H_vals_scan[i]) for i in range(len(H_vals_scan)-1)]
    max_dH_idx = np.argmax(dH)
    beta_lo = betas_coarse[max_dH_idx]
    beta_hi = betas_coarse[max_dH_idx + 1]

    print(f"\n  Largest entropy change between beta={beta_lo:.1f} and beta={beta_hi:.1f}")
    print(f"  dH = {dH[max_dH_idx]:.4f}")

    # Phase 3: Fine scan around transition
    print(f"\n--- Phase 3: Fine scan around transition ---")
    betas_fine = np.linspace(beta_lo, beta_hi, 10)

    results_fine = scan_beta(
        N=N, d=d, betas=betas_fine,
        n_steps=20000, n_therm=8000, record_every=10,
        seed=123
    )

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")

    all_results = results_coarse + results_fine
    all_results.sort(key=lambda r: r['beta'])

    print(f"\n{'beta':>8s}  {'S':>8s}  {'H':>7s}  {'ord':>7s}  {'ht':>6s}  {'acc':>5s}")
    print(f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}")
    for r in all_results:
        print(f"{r['beta']:8.2f}  {r['mean_S']:8.2f}  {r['mean_H']:7.3f}  "
              f"{r['mean_ord']:7.4f}  {r['mean_ht']:6.1f}  {r['accept_rate']:5.3f}")

    # Check for transition signatures
    H_all = np.array([r['mean_H'] for r in all_results])
    ord_all = np.array([r['mean_ord'] for r in all_results])
    beta_all = np.array([r['beta'] for r in all_results])

    H_range = np.max(H_all) - np.min(H_all)
    ord_range = np.max(ord_all) - np.min(ord_all)

    print(f"\nEntropy range: {np.min(H_all):.3f} to {np.max(H_all):.3f} (delta={H_range:.3f})")
    print(f"Ordering range: {np.min(ord_all):.4f} to {np.max(ord_all):.4f} (delta={ord_range:.4f})")

    if H_range > 0.3:
        print(f"\n** TRANSITION DETECTED ** — entropy changes by {H_range:.3f}")
        # Find steepest point
        dH_db = np.abs(np.diff(H_all) / np.diff(beta_all))
        idx = np.argmax(dH_db)
        beta_c_est = (beta_all[idx] + beta_all[idx+1]) / 2
        print(f"   Estimated beta_c ~ {beta_c_est:.2f}")
    else:
        print(f"\n   No sharp transition detected (entropy range < 0.3)")
        print(f"   May need wider beta range or larger N")


if __name__ == '__main__':
    main()
