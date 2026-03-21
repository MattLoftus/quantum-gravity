"""
Experiment 23: BD Phase Transition — Correct Methodology

Using the Surya/Glaser 2-order representation with coordinate-swap moves
and the non-local BD action (epsilon=1).

This matches the published methodology exactly, then adds the NOVEL
measurement: spectral dimension across the phase transition.

According to Glaser et al.:
  beta_c = 1.66 / (N * epsilon^2)
  For epsilon=1, N=50: beta_c ≈ 0.033
  For epsilon=1, N=80: beta_c ≈ 0.021

The transition is FIRST ORDER: bimodal action histogram at beta_c.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import (
    TwoOrder, mcmc_two_order, bd_action_2d_fast
)
from causal_sets.fast_core import spectral_dimension_fast
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer


def measure(cs, compute_spectral=True):
    """Measure all observables on a causet sample."""
    n = cs.n
    S = bd_action_2d_fast(cs)
    of = cs.ordering_fraction()
    chain = cs.longest_chain()
    links = int(np.sum(cs.link_matrix()))

    cs_old = CausalSet(n)
    cs_old.order = cs.order.astype(np.int8)
    mm = myrheim_meyer(cs_old)

    ds_peak = float('nan')
    ds_plateau = 0
    if compute_spectral and n >= 15:
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.1, 50.0), n_sigma=30)
        if len(d_s) > 0:
            ds_peak = np.max(d_s)
            ds_plateau = int(np.sum(np.abs(d_s - 2.0) < 0.3))

    return {
        'S': S, 'S_N': S / n, 'N': n, 'of': of, 'chain': chain,
        'links_N': links / n, 'mm': mm,
        'ds_peak': ds_peak, 'ds_plateau': ds_plateau,
    }


def phase_scan(N, betas, n_mcmc=50000, n_therm=20000, n_samples=200,
               epsilon=1.0, rng=None):
    """Run MCMC at each beta, measure observables."""
    if rng is None:
        rng = np.random.default_rng()

    record_every = max(1, (n_mcmc - n_therm) // n_samples)
    results = []

    for beta in betas:
        t0 = time.time()

        res = mcmc_two_order(N, beta=beta, epsilon=epsilon,
                              n_steps=n_mcmc, n_thermalize=n_therm,
                              record_every=record_every, rng=rng)

        samples = res['samples']
        actions = res['actions']

        # Measure observables on samples
        obs_list = []
        for j, cs in enumerate(samples):
            obs = measure(cs, compute_spectral=(j % 20 == 0))
            obs_list.append(obs)

        if not obs_list:
            continue

        S_vals = [o['S_N'] for o in obs_list]
        mm_vals = [o['mm'] for o in obs_list if not np.isnan(o['mm'])]
        ds_vals = [o['ds_peak'] for o in obs_list if not np.isnan(o['ds_peak'])]
        of_vals = [o['of'] for o in obs_list]
        ln_vals = [o['links_N'] for o in obs_list]
        ch_vals = [o['chain'] for o in obs_list]
        plat_vals = [o['ds_plateau'] for o in obs_list if o['ds_plateau'] > 0]

        elapsed = time.time() - t0

        r = {
            'beta': beta,
            'S_mean': np.mean(S_vals), 'S_std': np.std(S_vals),
            'suscept': np.var(S_vals) * N,
            'mm_mean': np.mean(mm_vals) if mm_vals else float('nan'),
            'ds_mean': np.mean(ds_vals) if ds_vals else float('nan'),
            'of_mean': np.mean(of_vals),
            'links_mean': np.mean(ln_vals),
            'chain_mean': np.mean(ch_vals),
            'plateau_mean': np.mean(plat_vals) if plat_vals else 0,
            'accept': res['accept_rate'],
            'time': elapsed,
            'S_histogram': np.array(S_vals),  # keep for bimodality check
        }
        results.append(r)

        print(f"  β={beta:>7.4f}: S/N={r['S_mean']:>7.3f}±{r['S_std']:>5.3f}  "
              f"MM={r['mm_mean']:>5.2f}  d_s={r['ds_mean']:>5.2f}  "
              f"L/N={r['links_mean']:>5.2f}  f={r['of_mean']:>5.3f}  "
              f"h={r['chain_mean']:>4.1f}  χ={r['suscept']:>7.1f}  "
              f"acc={r['accept']:>5.3f}  ({elapsed:.0f}s)")

    return results


def check_bimodality(S_vals, n_bins=30):
    """Check if the action histogram is bimodal (sign of first-order transition)."""
    hist, edges = np.histogram(S_vals, bins=n_bins)
    # Find peaks: local maxima
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 2:
            peaks.append((edges[i], hist[i]))
    return len(peaks) >= 2, peaks


def main():
    rng = np.random.default_rng(42)

    print("=" * 90)
    print("EXPERIMENT 23: BD Phase Transition — Proper 2-Order Methodology")
    print("Following Surya (2012) / Glaser et al. (2018)")
    print("NOVEL: spectral dimension measurement across the transition")
    print("=" * 90)

    # Glaser et al. formula: beta_c = 1.66 / (N * epsilon^2)
    # For epsilon=1: beta_c(N=50) ≈ 0.033, beta_c(N=80) ≈ 0.021

    # Phase 1: Reproduce at N=50 (matches Surya's primary size)
    N = 50
    beta_c_expected = 1.66 / (N * 1.0 ** 2)
    print(f"\n--- Phase 1: N={N}, expected beta_c ≈ {beta_c_expected:.4f} ---")

    betas = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
                       0.035, 0.040, 0.050, 0.060, 0.080, 0.100, 0.150])

    results_50 = phase_scan(N=50, betas=betas, n_mcmc=60000, n_therm=30000,
                             n_samples=200, rng=rng)

    # Find susceptibility peak
    if results_50:
        chi_vals = [r['suscept'] for r in results_50]
        peak_idx = np.argmax(chi_vals)
        beta_c_measured = results_50[peak_idx]['beta']
        print(f"\n  Susceptibility peak at beta = {beta_c_measured:.4f}")
        print(f"  Expected: {beta_c_expected:.4f}")
        print(f"  Agreement: {abs(beta_c_measured - beta_c_expected) / beta_c_expected * 100:.1f}%")

        # Check bimodality at the peak
        bimodal, peaks = check_bimodality(results_50[peak_idx]['S_histogram'])
        print(f"  Bimodal histogram: {'YES (first order)' if bimodal else 'NO'}")
        if peaks:
            print(f"  Action peaks at: {[f'{p[0]:.2f}' for p in peaks]}")

    # Phase 2: Larger N for better statistics
    N = 80
    beta_c_expected = 1.66 / (N * 1.0 ** 2)
    print(f"\n--- Phase 2: N={N}, expected beta_c ≈ {beta_c_expected:.4f} ---")

    betas_80 = np.linspace(0.005, 0.060, 12)
    results_80 = phase_scan(N=80, betas=betas_80, n_mcmc=40000, n_therm=20000,
                             n_samples=150, rng=rng)

    if results_80:
        chi_vals_80 = [r['suscept'] for r in results_80]
        peak_80 = np.argmax(chi_vals_80)
        beta_c_80 = results_80[peak_80]['beta']
        print(f"\n  Susceptibility peak at beta = {beta_c_80:.4f}")
        print(f"  Expected: {beta_c_expected:.4f}")

    # Phase 3: Spectral dimension in each phase
    print(f"\n--- Phase 3: Spectral dimension in each phase ---")
    if results_50:
        # Low beta (continuum phase)
        low_beta_results = [r for r in results_50 if r['beta'] < beta_c_measured * 0.5]
        high_beta_results = [r for r in results_50 if r['beta'] > beta_c_measured * 2.0]

        if low_beta_results:
            ds_low = [r['ds_mean'] for r in low_beta_results if not np.isnan(r['ds_mean'])]
            mm_low = [r['mm_mean'] for r in low_beta_results if not np.isnan(r['mm_mean'])]
            print(f"  Continuum phase (beta < {beta_c_measured*0.5:.3f}):")
            print(f"    d_s = {np.mean(ds_low):.2f} ± {np.std(ds_low):.2f}")
            print(f"    MM dim = {np.mean(mm_low):.2f} ± {np.std(mm_low):.2f}")

        if high_beta_results:
            ds_high = [r['ds_mean'] for r in high_beta_results if not np.isnan(r['ds_mean'])]
            mm_high = [r['mm_mean'] for r in high_beta_results if not np.isnan(r['mm_mean'])]
            print(f"  Crystalline phase (beta > {beta_c_measured*2:.3f}):")
            print(f"    d_s = {np.mean(ds_high):.2f} ± {np.std(ds_high):.2f}")
            print(f"    MM dim = {np.mean(mm_high):.2f} ± {np.std(mm_high):.2f}")

        if low_beta_results and high_beta_results and ds_low and ds_high:
            delta_ds = np.mean(ds_high) - np.mean(ds_low)
            print(f"\n  SPECTRAL DIMENSION JUMP: Δd_s = {delta_ds:.2f}")
            print(f"  (positive = crystalline has higher d_s; negative = lower)")

    # Phase 4: Finite size scaling
    print(f"\n--- Phase 4: Finite size scaling ---")
    print(f"  {'N':>5} {'beta_c':>8} {'chi_max':>10} {'S_jump':>8} {'bimodal':>8}")
    print("-" * 50)

    for N in [30, 50, 70, 90]:
        beta_c_exp = 1.66 / N
        betas_fss = np.linspace(max(0.001, beta_c_exp * 0.3), beta_c_exp * 3.0, 10)
        fss = phase_scan(N=N, betas=betas_fss, n_mcmc=30000, n_therm=15000,
                          n_samples=100, rng=rng)

        if fss:
            chi_fss = [r['suscept'] for r in fss]
            peak = np.argmax(chi_fss)
            S_lo = fss[0]['S_mean']
            S_hi = fss[-1]['S_mean']
            bimod, _ = check_bimodality(fss[peak]['S_histogram'])

            print(f"  {N:>5} {fss[peak]['beta']:>8.4f} {chi_fss[peak]:>10.1f} "
                  f"{abs(S_hi-S_lo):>8.3f} {'YES' if bimod else 'no':>8}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    if results_50:
        print(f"  Measured beta_c(N=50) = {beta_c_measured:.4f}")
        print(f"  Glaser prediction:      {1.66/50:.4f}")
        print(f"  Phase transition is {'first-order (bimodal)' if bimodal else 'continuous or unclear'}")
        if low_beta_results and high_beta_results and ds_low and ds_high:
            print(f"  NOVEL: spectral dimension changes from {np.mean(ds_low):.2f} "
                  f"to {np.mean(ds_high):.2f} across transition")
    print("=" * 90)


if __name__ == '__main__':
    main()
