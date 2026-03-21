"""
Experiment 32: Large-N BD Phase Transition via Parallel Tempering

Push finite-size scaling to N=150, 200 using parallel tempering to overcome
poor thermalization at high beta (crystalline phase).

Key question: does chi_S_max continue to grow with N? If so, confirms
first-order phase transition.

Uses corrected BD action with eps=0.12.
beta_c = 6.64 / (N * eps^2)  [calibrated from exp28 finite-size scaling]
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders_v2 import (
    bd_action_corrected, mcmc_corrected, parallel_tempering
)
from causal_sets.two_orders import TwoOrder
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet


def interval_entropy(cs, max_k=15):
    """Shannon entropy of the interval-size distribution."""
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0:
        return 0.0
    p = dist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def make_beta_ladder(beta_c, n_betas=10, max_ratio=6.0):
    """
    Create a beta ladder spanning the transition.

    Low betas explore freely (high acceptance), high betas probe the
    crystalline phase. Denser spacing near the transition region.
    """
    # The actual transition seems to be at ~1.2-4x beta_c depending on N.
    # Extend to max_ratio * beta_c to capture it.
    betas = sorted(set(
        list(np.linspace(0.1 * beta_c, 0.6 * beta_c, 2)) +
        list(np.linspace(0.6 * beta_c, 2.0 * beta_c, n_betas - 5)) +
        list(np.linspace(2.0 * beta_c, max_ratio * beta_c, 3))
    ))
    return np.array(betas)


def analyze_chain(chain_data, N):
    """Compute observables from a single chain's samples."""
    S_arr = chain_data['actions']
    if len(S_arr) == 0:
        return None

    H_vals = [interval_entropy(cs) for cs in chain_data['samples']]

    return {
        'beta': chain_data['beta'],
        'S_mean': np.mean(S_arr),
        'S_std': np.std(S_arr),
        'chi_S': np.var(S_arr) * N,
        'H_mean': np.mean(H_vals),
        'H_std': np.std(H_vals),
        'chi_H': np.var(H_vals) * N,
        'accept_rate': chain_data['accept_rate'],
        'n_samples': len(S_arr),
    }


def run_parallel_tempering_scan(N, eps, n_betas=10, n_steps=50000,
                                  swap_interval=20, rng=None):
    """Run parallel tempering and extract phase transition observables."""
    beta_c = 6.64 / (N * eps ** 2)
    betas = make_beta_ladder(beta_c, n_betas)

    print(f"\n  Beta ladder ({len(betas)} chains):")
    for i, b in enumerate(betas):
        print(f"    [{i}] beta={b:.3f}  (beta/bc={b/beta_c:.2f})")

    n_therm = n_steps // 2
    record_every = max(1, n_steps // 500)

    print(f"\n  Running {n_steps} steps, {n_therm} thermalization, "
          f"record every {record_every}, swap every {swap_interval}...")

    t0 = time.time()
    result = parallel_tempering(
        N, betas, eps,
        n_steps=n_steps,
        swap_interval=swap_interval,
        n_therm=n_therm,
        record_every=record_every,
        rng=rng,
    )
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"\n  Swap acceptance rates between adjacent chains:")
    for i, rate in enumerate(result['swap_matrix']):
        print(f"    chains {i}<->{i+1}: {rate:.3f}")

    # Analyze each chain
    print(f"\n  {'beta':>7} {'b/bc':>6} {'<S>':>8} {'H':>7} {'chi_S':>8} "
          f"{'chi_H':>8} {'acc':>6} {'n':>5}")
    print("  " + "-" * 65)

    results = []
    for chain in result['chains']:
        r = analyze_chain(chain, N)
        if r is None:
            continue
        results.append(r)
        marker = ""
        if 0.8 < r['beta'] / beta_c < 1.2:
            marker = " <-- NEAR bc"
        print(f"  {r['beta']:>7.3f} {r['beta']/beta_c:>6.2f} {r['S_mean']:>8.3f} "
              f"{r['H_mean']:>7.3f} {r['chi_S']:>8.2f} {r['chi_H']:>8.4f} "
              f"{r['accept_rate']:>6.3f} {r['n_samples']:>5d}{marker}")

    return results, beta_c, elapsed


def main():
    rng = np.random.default_rng(42)
    eps = 0.12

    print("=" * 80)
    print("EXPERIMENT 32: Large-N Phase Transition via Parallel Tempering")
    print(f"eps={eps}, beta_c = 6.64/(N*eps^2)")
    print("=" * 80)

    # --- Quick validation at N=50 (should match exp28) ---
    print("\n--- Validation: N=50 parallel tempering ---")
    N = 50
    results_50, bc_50, t_50 = run_parallel_tempering_scan(
        N, eps, n_betas=10, n_steps=30000, rng=rng
    )

    chi_vals = [r['chi_S'] for r in results_50]
    peak = np.argmax(chi_vals)
    print(f"\n  chi_S peak at beta={results_50[peak]['beta']:.3f} "
          f"(bc_pred={bc_50:.3f}, ratio={results_50[peak]['beta']/bc_50:.2f})")
    print(f"  chi_S_max = {chi_vals[peak]:.2f}")

    # --- N=100 ---
    print("\n\n--- N=100 ---")
    N = 100
    results_100, bc_100, t_100 = run_parallel_tempering_scan(
        N, eps, n_betas=10, n_steps=40000, rng=rng
    )

    chi_vals = [r['chi_S'] for r in results_100]
    peak = np.argmax(chi_vals)
    print(f"\n  chi_S peak at beta={results_100[peak]['beta']:.3f} "
          f"(bc_pred={bc_100:.3f}, ratio={results_100[peak]['beta']/bc_100:.2f})")
    print(f"  chi_S_max = {chi_vals[peak]:.2f}")

    # --- N=150 ---
    print("\n\n--- N=150 ---")
    N = 150
    results_150, bc_150, t_150 = run_parallel_tempering_scan(
        N, eps, n_betas=10, n_steps=40000, swap_interval=15, rng=rng
    )

    chi_vals = [r['chi_S'] for r in results_150]
    peak = np.argmax(chi_vals)
    print(f"\n  chi_S peak at beta={results_150[peak]['beta']:.3f} "
          f"(bc_pred={bc_150:.3f}, ratio={results_150[peak]['beta']/bc_150:.2f})")
    print(f"  chi_S_max = {chi_vals[peak]:.2f}")

    # --- N=200 (push as high as feasible) ---
    print("\n\n--- N=200 ---")
    N = 200
    t_start = time.time()
    results_200, bc_200, t_200 = run_parallel_tempering_scan(
        N, eps, n_betas=10, n_steps=30000, swap_interval=15, rng=rng
    )

    chi_vals = [r['chi_S'] for r in results_200]
    peak = np.argmax(chi_vals)
    print(f"\n  chi_S peak at beta={results_200[peak]['beta']:.3f} "
          f"(bc_pred={bc_200:.3f}, ratio={results_200[peak]['beta']/bc_200:.2f})")
    print(f"  chi_S_max = {chi_vals[peak]:.2f}")

    # --- Summary: Finite-size scaling ---
    print("\n\n" + "=" * 80)
    print("FINITE-SIZE SCALING SUMMARY")
    print("=" * 80)
    print(f"\n  {'N':>4} {'bc_pred':>8} {'bc_meas':>8} {'ratio':>6} {'chi_max':>9} "
          f"{'H_lo':>7} {'H_hi':>7} {'DeltaH':>7} {'time':>6}")
    print("  " + "-" * 70)

    all_results = [
        (50, results_50, bc_50, t_50),
        (100, results_100, bc_100, t_100),
        (150, results_150, bc_150, t_150),
        (200, results_200, bc_200, t_200),
    ]

    chi_maxes = []
    Ns = []

    for N_val, results, bc, elapsed in all_results:
        chi_vals = [r['chi_S'] for r in results]
        peak = np.argmax(chi_vals)
        bc_meas = results[peak]['beta']

        # H below and above transition
        lo = [r for r in results if r['beta'] < bc_meas * 0.5]
        hi = [r for r in results if r['beta'] > bc_meas * 2.0]
        H_lo = np.mean([r['H_mean'] for r in lo]) if lo else float('nan')
        H_hi = np.mean([r['H_mean'] for r in hi]) if hi else float('nan')
        delta_H = H_lo - H_hi

        chi_maxes.append(chi_vals[peak])
        Ns.append(N_val)

        print(f"  {N_val:>4} {bc:>8.3f} {bc_meas:>8.3f} {bc_meas/bc:>6.2f} "
              f"{chi_vals[peak]:>9.2f} {H_lo:>7.3f} {H_hi:>7.3f} "
              f"{delta_H:>7.3f} {elapsed:>5.0f}s")

    # Log-log fit for chi_max vs N
    Ns = np.array(Ns, dtype=float)
    chi_maxes = np.array(chi_maxes)
    if len(Ns) >= 2 and all(c > 0 for c in chi_maxes):
        log_N = np.log(Ns)
        log_chi = np.log(chi_maxes)
        slope, intercept = np.polyfit(log_N, log_chi, 1)
        print(f"\n  chi_S_max ~ N^alpha: alpha = {slope:.2f}")
        print(f"  (alpha > 0 confirms diverging susceptibility => first-order transition)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
