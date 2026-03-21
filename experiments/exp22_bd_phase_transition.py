"""
Experiment 22: Benincasa-Dowker Action Phase Transition Study

Reproduce and extend the Surya (2012) / Glaser et al. (2018) result:
a phase transition in 2D causal set quantum gravity at critical coupling beta_c.

On one side: manifold-like causets (low action).
On the other: non-manifold "layered" causets (different structure).

We measure across the transition:
  1. Action per element <S/N> (order parameter)
  2. Myrheim-Meyer dimension
  3. Spectral dimension (peak and plateau width)
  4. Link density (links per element)
  5. Ordering fraction
  6. Susceptibility (variance of action — peaks at transition)

The partition function is: Z = sum_C exp(-beta * S_BD(C))
We use MCMC with fixed N (volume-fixing penalty).

NEW contribution: spectral dimension and composite manifold-likeness
across the transition. Nobody has published this measurement.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.fast_core import sprinkle_fast, spectral_dimension_fast
from causal_sets.bd_action import bd_action_2d
from causal_sets.mcmc import mcmc_bd_action
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer


def measure_observables(causet, compute_spectral=True):
    """Compute all observables for a single causet sample."""
    n = causet.n
    if n < 5:
        return None

    S = bd_action_2d(causet)
    links = int(np.sum(causet.link_matrix()))
    of = causet.ordering_fraction()
    chain = causet.longest_chain()

    cs_old = CausalSet(n)
    cs_old.order = causet.order.astype(np.int8)
    mm = myrheim_meyer(cs_old)

    ds_peak = float('nan')
    ds_plateau = 0
    if compute_spectral and n >= 20:
        sigmas, d_s = spectral_dimension_fast(causet, sigma_range=(0.1, 100.0), n_sigma=40)
        if len(d_s) > 0:
            ds_peak = np.max(d_s)
            ds_plateau = int(np.sum(np.abs(d_s - 2.0) < 0.3))

    return {
        'S': S, 'S_per_N': S / n, 'N': n,
        'links_per_N': links / n, 'ordering_frac': of,
        'mm_dim': mm, 'chain': chain, 'chain_ratio': chain / n,
        'ds_peak': ds_peak, 'ds_plateau': ds_plateau
    }


def run_phase_scan(N: int, betas: np.ndarray, n_mcmc: int = 50000,
                   n_burn: int = 20000, n_measure: int = 100,
                   rng: np.random.Generator = None):
    """
    For each beta, run MCMC and measure observables after burn-in.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []

    for i, beta in enumerate(betas):
        t_start = time.time()

        # Start from a fresh sprinkled causet (warm start)
        cs_init, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)

        # Run MCMC
        mcmc_result = mcmc_bd_action(
            cs_init, beta=beta, n_steps=n_mcmc,
            target_n=N, n_size_penalty=0.5,  # stronger volume fixing
            rng=rng, record_every=max(1, (n_mcmc - n_burn) // n_measure)
        )

        # Measure observables on post-burn-in samples
        samples = mcmc_result['samples']
        n_total = len(samples)
        burn_idx = max(0, n_total - n_measure)
        post_burn = samples[burn_idx:]

        # Measure all observables
        obs_list = []
        for j, cs in enumerate(post_burn):
            # Only compute spectral dim for every 10th sample (expensive)
            obs = measure_observables(cs, compute_spectral=(j % 10 == 0))
            if obs:
                obs_list.append(obs)

        if not obs_list:
            continue

        # Aggregate
        S_vals = [o['S_per_N'] for o in obs_list]
        mm_vals = [o['mm_dim'] for o in obs_list if not np.isnan(o['mm_dim'])]
        ds_vals = [o['ds_peak'] for o in obs_list if not np.isnan(o['ds_peak'])]
        ln_vals = [o['links_per_N'] for o in obs_list]
        of_vals = [o['ordering_frac'] for o in obs_list]
        n_vals = [o['N'] for o in obs_list]
        plat_vals = [o['ds_plateau'] for o in obs_list if o['ds_plateau'] > 0]

        elapsed = time.time() - t_start

        result = {
            'beta': beta,
            'S_mean': np.mean(S_vals), 'S_std': np.std(S_vals),
            'S_suscept': np.var(S_vals) * N,  # susceptibility = N * var(S/N)
            'mm_mean': np.mean(mm_vals) if mm_vals else float('nan'),
            'mm_std': np.std(mm_vals) if mm_vals else float('nan'),
            'ds_mean': np.mean(ds_vals) if ds_vals else float('nan'),
            'ds_std': np.std(ds_vals) if ds_vals else float('nan'),
            'plateau_mean': np.mean(plat_vals) if plat_vals else 0,
            'links_mean': np.mean(ln_vals),
            'of_mean': np.mean(of_vals),
            'N_mean': np.mean(n_vals),
            'accept': mcmc_result['accept_rate'],
            'time': elapsed,
        }
        results.append(result)

        print(f"  beta={beta:>6.3f}: S/N={result['S_mean']:>7.3f}±{result['S_std']:>5.3f}  "
              f"MM={result['mm_mean']:>5.2f}  d_s={result['ds_mean']:>5.2f}  "
              f"L/N={result['links_mean']:>5.2f}  f={result['of_mean']:>5.3f}  "
              f"<N>={result['N_mean']:>5.0f}  χ={result['S_suscept']:>7.2f}  "
              f"acc={result['accept']:>5.3f}  ({elapsed:.1f}s)")

    return results


def main():
    rng = np.random.default_rng(42)

    print("=" * 90)
    print("EXPERIMENT 22: BD Action Phase Transition Study")
    print("=" * 90)

    # Phase 1: Ground truth — what do sprinkled 2D causets look like?
    print("\n--- Ground truth: Sprinkled 2D causets ---")
    print(f"  {'N':>5} {'S/N':>8} {'MM':>6} {'d_s':>6} {'L/N':>6} {'f':>6}")
    print("-" * 45)

    for N in [50, 80, 100, 150]:
        vals = {'S': [], 'mm': [], 'ds': [], 'ln': [], 'of': []}
        for _ in range(10):
            cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
            obs = measure_observables(cs)
            if obs:
                vals['S'].append(obs['S_per_N'])
                if not np.isnan(obs['mm_dim']): vals['mm'].append(obs['mm_dim'])
                if not np.isnan(obs['ds_peak']): vals['ds'].append(obs['ds_peak'])
                vals['ln'].append(obs['links_per_N'])
                vals['of'].append(obs['ordering_frac'])

        print(f"  {N:>5} {np.mean(vals['S']):>8.3f} {np.mean(vals['mm']):>6.2f} "
              f"{np.mean(vals['ds']):>6.2f} {np.mean(vals['ln']):>6.2f} "
              f"{np.mean(vals['of']):>6.3f}")

    # Phase 2: Coarse scan to find the transition
    print("\n--- Phase 2: Coarse beta scan (N=80, 20000 MCMC steps) ---")
    betas_coarse = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
    coarse_results = run_phase_scan(N=80, betas=betas_coarse,
                                     n_mcmc=20000, n_burn=10000, n_measure=50, rng=rng)

    # Find the transition: where does the susceptibility peak?
    if coarse_results:
        suscept = [r['S_suscept'] for r in coarse_results]
        peak_idx = np.argmax(suscept)
        beta_peak = coarse_results[peak_idx]['beta']
        print(f"\n  Susceptibility peak at beta ≈ {beta_peak:.1f}")

        # Phase 3: Fine scan around the transition
        print(f"\n--- Phase 3: Fine scan around beta_c ≈ {beta_peak:.1f} (N=100) ---")
        beta_lo = max(0, beta_peak - 3)
        beta_hi = beta_peak + 3
        betas_fine = np.linspace(beta_lo, beta_hi, 15)
        fine_results = run_phase_scan(N=100, betas=betas_fine,
                                       n_mcmc=30000, n_burn=15000, n_measure=80, rng=rng)

        # Find refined transition point
        if fine_results:
            suscept_fine = [r['S_suscept'] for r in fine_results]
            peak_fine = np.argmax(suscept_fine)
            beta_c = fine_results[peak_fine]['beta']
            print(f"\n  Refined beta_c ≈ {beta_c:.3f}")

    # Phase 4: Finite size scaling
    print(f"\n--- Phase 4: Finite size scaling (does transition sharpen with N?) ---")
    print(f"  {'N':>5} {'beta_c':>8} {'chi_max':>10} {'S_jump':>8}")
    print("-" * 40)

    for N in [50, 80, 100, 150]:
        # Scan a few betas around the expected transition
        betas_fss = np.linspace(max(0, beta_peak - 4), beta_peak + 4, 10)
        fss_results = run_phase_scan(N=N, betas=betas_fss,
                                      n_mcmc=15000, n_burn=8000, n_measure=40, rng=rng)

        if fss_results:
            suscept_fss = [r['S_suscept'] for r in fss_results]
            peak_fss = np.argmax(suscept_fss)
            beta_c_fss = fss_results[peak_fss]['beta']
            chi_max = fss_results[peak_fss]['S_suscept']

            # Action jump across transition
            S_low = fss_results[0]['S_mean']
            S_high = fss_results[-1]['S_mean']
            S_jump = abs(S_high - S_low)

            print(f"  {N:>5} {beta_c_fss:>8.3f} {chi_max:>10.2f} {S_jump:>8.3f}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Critical coupling: beta_c ≈ {beta_peak:.1f} (coarse)")
    print("  If susceptibility chi_max scales with N: first-order transition")
    print("  If chi_max scales as N^gamma with gamma < 1: continuous transition")
    print("  Novel observable: spectral dimension d_s across transition")
    print("=" * 90)


if __name__ == '__main__':
    main()
