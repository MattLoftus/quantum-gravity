"""
Experiment 24: BD Action Calibration + Deep Crystalline Phase

Two goals:
1. Resolve the beta_c discrepancy (we got 0.09-0.15 vs Glaser's 0.033)
2. Push to MUCH higher beta to definitively sample the crystalline phase
   and check if spectral dimension changes there

Glaser's formula: beta_c = 1.66 / (N * epsilon^2)
They use a DIFFERENT action normalization. Let's test at multiple epsilon
values to calibrate, and also check what happens at beta >> beta_c.

Key diagnostic: in the crystalline phase (Surya 2012):
  - ordering fraction rises to ~0.6 (from ~0.5 in continuum)
  - height drops to ~3 (from ~10 in continuum)
  - action drops to S ~ -45 (from S ~ 4 in continuum)
  at N=50. If we don't see these signatures, we haven't reached the phase.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder, mcmc_two_order, bd_action_2d_fast
from causal_sets.fast_core import spectral_dimension_fast
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer
from causal_sets.bd_action import count_intervals_by_size


def measure(cs, compute_spectral=True):
    n = cs.n
    if n < 5:
        return None

    counts = count_intervals_by_size(cs, max_size=3)
    N0 = counts.get(0, 0)  # links
    N1 = counts.get(1, 0)  # 1-intervals
    N2 = counts.get(2, 0)  # 2-intervals
    N3 = counts.get(3, 0)  # 3-intervals

    S_our = n - 2 * N0 + 4 * N1 - 2 * N2  # our formula
    S_simple = n - 2 * N0  # simplest BD (no non-locality)
    S_original = n - 2 * N0 + N1  # original exp16 formula

    of = cs.ordering_fraction()
    chain = cs.longest_chain()
    links_n = N0 / n if n > 0 else 0

    cs_old = CausalSet(n)
    cs_old.order = cs.order.astype(np.int8)
    mm = myrheim_meyer(cs_old)

    ds_peak = float('nan')
    if compute_spectral and n >= 15:
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.1, 50.0), n_sigma=30)
        if len(d_s) > 0:
            ds_peak = np.max(d_s)

    return {
        'S_our': S_our, 'S_simple': S_simple, 'S_original': S_original,
        'N0': N0, 'N1': N1, 'N2': N2, 'N3': N3,
        'of': of, 'chain': chain, 'links_n': links_n,
        'mm': mm, 'ds_peak': ds_peak, 'n': n,
    }


def main():
    rng = np.random.default_rng(42)
    N = 50

    print("=" * 90)
    print("EXPERIMENT 24: BD Action Calibration + Deep Crystalline Phase")
    print("=" * 90)

    # Part 1: Check what a random 2-order looks like (beta=0)
    print("\n--- Part 1: Random 2-orders (beta=0, N=50) — Surya's 'continuum phase' baseline ---")
    print("  Surya reports: f≈0.50, height≈10, S≈4 for beta=0")

    res = mcmc_two_order(N, beta=0.0, n_steps=30000, n_thermalize=10000,
                          record_every=20, rng=rng)
    obs_list = [measure(cs, compute_spectral=(i % 20 == 0))
                for i, cs in enumerate(res['samples'])]
    obs_list = [o for o in obs_list if o]

    print(f"\n  Our beta=0 results (N={N}):")
    print(f"    S (our formula):     {np.mean([o['S_our'] for o in obs_list]):>8.2f} ± {np.std([o['S_our'] for o in obs_list]):>5.2f}")
    print(f"    S (simple N-2L):     {np.mean([o['S_simple'] for o in obs_list]):>8.2f} ± {np.std([o['S_simple'] for o in obs_list]):>5.2f}")
    print(f"    S (N-2L+I2):         {np.mean([o['S_original'] for o in obs_list]):>8.2f} ± {np.std([o['S_original'] for o in obs_list]):>5.2f}")
    print(f"    Ordering fraction:   {np.mean([o['of'] for o in obs_list]):>8.4f}")
    print(f"    Height (chain):      {np.mean([o['chain'] for o in obs_list]):>8.1f}")
    print(f"    Links/N:             {np.mean([o['links_n'] for o in obs_list]):>8.2f}")
    print(f"    N0 (links):          {np.mean([o['N0'] for o in obs_list]):>8.1f}")
    print(f"    N1 (1-intervals):    {np.mean([o['N1'] for o in obs_list]):>8.1f}")
    print(f"    N2 (2-intervals):    {np.mean([o['N2'] for o in obs_list]):>8.1f}")
    print(f"    MM dimension:        {np.mean([o['mm'] for o in obs_list if not np.isnan(o['mm'])]):>8.2f}")
    ds_vals = [o['ds_peak'] for o in obs_list if not np.isnan(o['ds_peak'])]
    if ds_vals:
        print(f"    Spectral dim peak:   {np.mean(ds_vals):>8.2f}")

    # Part 2: Push to VERY high beta to find the crystalline phase
    print("\n--- Part 2: Scanning to very high beta ---")
    print("  Surya's crystalline phase: f≈0.60, height≈3, S≈-45")
    print(f"\n  {'beta':>8} {'S_our':>8} {'S_simple':>10} {'f':>6} {'height':>7} {'L/N':>6} {'d_s':>6} {'accept':>7}")
    print("-" * 70)

    for beta in [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        t0 = time.time()
        res = mcmc_two_order(N, beta=beta, n_steps=40000, n_thermalize=20000,
                              record_every=20, rng=rng)

        obs = [measure(cs, compute_spectral=(i % 30 == 0))
               for i, cs in enumerate(res['samples'])]
        obs = [o for o in obs if o]

        S_our = np.mean([o['S_our'] for o in obs])
        S_simple = np.mean([o['S_simple'] for o in obs])
        f = np.mean([o['of'] for o in obs])
        h = np.mean([o['chain'] for o in obs])
        ln = np.mean([o['links_n'] for o in obs])
        ds = np.mean([o['ds_peak'] for o in obs if not np.isnan(o['ds_peak'])])
        elapsed = time.time() - t0

        marker = ""
        if f > 0.55:
            marker = " <-- HIGH f (crystalline?)"
        if h < 4:
            marker = " <-- LOW height (crystalline!)"

        print(f"  {beta:>8.2f} {S_our:>8.1f} {S_simple:>10.1f} {f:>6.3f} {h:>7.1f} "
              f"{ln:>6.2f} {ds:>6.2f} {res['accept_rate']:>7.3f}{marker}")

    # Part 3: Try with the SIMPLE action (S = N - 2L) which is what Surya
    # may actually be using as the "local" BD action
    print("\n--- Part 3: MCMC with SIMPLE action S = N - 2*Links ---")
    print("  (Surya may use this rather than the non-local version)")

    # Redefine the action to use S_simple = N - 2*L
    # We need to modify the MCMC to use this action
    # Quick approach: run with our existing code but interpret the
    # transition through the simple action lens

    # Actually, let's check: at beta=0, S_simple ≈ ?
    # If S_simple ≈ 4 at beta=0 (matching Surya), then she uses the simple action
    # If S_our ≈ 4 at beta=0, then she uses the non-local action

    print(f"\n  At beta=0: S_our={np.mean([o['S_our'] for o in obs_list]):.1f}, "
          f"S_simple={np.mean([o['S_simple'] for o in obs_list]):.1f}, "
          f"S_original={np.mean([o['S_original'] for o in obs_list]):.1f}")
    print(f"  Surya reports S≈4 at beta=0 for N=50")
    print(f"  --> Closest match tells us which action Surya uses")

    # Part 4: Detailed spectral dimension profile comparison
    # between low-beta and the highest-beta sample we can get
    print("\n--- Part 4: Spectral dimension profiles: low vs high beta ---")

    for beta_val, label in [(0.0, "Continuum (β=0)"), (50.0, "Deep crystalline (β=50)")]:
        res = mcmc_two_order(N, beta=beta_val, n_steps=30000, n_thermalize=20000,
                              record_every=100, rng=rng)

        # Take last sample
        cs = res['samples'][-1]
        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.05, 100.0), n_sigma=40)

        print(f"\n  {label}:")
        print(f"    f={cs.ordering_fraction():.3f}, chain={cs.longest_chain()}, "
              f"links={int(np.sum(cs.link_matrix()))}")

        if len(d_s) > 0:
            print(f"    Peak d_s = {np.max(d_s):.2f}")
            print(f"    {'sigma':>10} {'d_s':>8}")
            for idx in range(0, len(sigmas), max(1, len(sigmas) // 8)):
                print(f"    {sigmas[idx]:>10.3f} {d_s[idx]:>8.2f}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == '__main__':
    main()
