"""
Experiment 18: Extended BD Action MCMC — longer chains, larger N.

Exp16 showed BD-weighted MCMC moves in the right direction but
needs more steps for convergence.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.bd_action import bd_action_2d
from causal_sets.mcmc import mcmc_bd_action
from causal_sets.fast_core import sprinkle_fast, spectral_dimension_fast
from causal_sets.core import CausalSet
from causal_sets.dimension import myrheim_meyer


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 18: Extended BD Action MCMC")
    print("=" * 70)

    # Ground truth
    print("\n--- Ground truth: 2D sprinkled causets (N=60) ---")
    gt_mm = []
    gt_sbd = []
    for _ in range(5):
        cs, _ = sprinkle_fast(60, dim=2, extent_t=1.0, region='diamond', rng=rng)
        S = bd_action_2d(cs)
        cs_old = CausalSet(60)
        cs_old.order = cs.order.astype(np.int8)
        d = myrheim_meyer(cs_old)
        gt_mm.append(d)
        gt_sbd.append(S / 60)
        links = int(np.sum(cs.link_matrix()))
        print(f"  S_BD={S:>8.0f}, S/N={S/60:>7.3f}, MM_dim={d:.2f}, L/N={links/60:.2f}")
    print(f"  Mean: MM={np.mean(gt_mm):.2f}, S/N={np.mean(gt_sbd):.3f}")

    # MCMC with longer chains
    print("\n--- MCMC: 5000 steps, target N=60 ---")
    print(f"  {'beta':>6} {'accept':>8} {'<S/N>':>8} {'<N>':>6} {'<MM>':>6} {'<d_s>':>6} {'<L/N>':>6}")
    print("-" * 60)

    for beta in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        cs_init, _ = sprinkle_fast(60, dim=2, extent_t=1.0, region='diamond', rng=rng)

        result = mcmc_bd_action(cs_init, beta=beta, n_steps=5000,
                                target_n=60, n_size_penalty=0.1,
                                rng=rng, record_every=5)

        samples = result['samples']
        accept_rate = result.get('accept_rate', len(samples) / 5000)

        # Measure on last 200 samples (burn-in)
        burn = max(0, len(samples) - 200)
        recent = samples[burn:]

        s_vals, mm_vals, ds_vals, ln_vals, n_vals = [], [], [], [], []

        for cs in recent[::5]:  # subsample
            S = bd_action_2d(cs)
            s_vals.append(S / cs.n if cs.n > 0 else 0)
            n_vals.append(cs.n)

            cs_old = CausalSet(cs.n)
            cs_old.order = cs.order.astype(np.int8)
            d = myrheim_meyer(cs_old)
            if not np.isnan(d):
                mm_vals.append(d)

            sigmas, d_s = spectral_dimension_fast(cs)
            if len(d_s) > 0:
                ds_vals.append(np.max(d_s))

            links = int(np.sum(cs.link_matrix()))
            ln_vals.append(links / cs.n if cs.n > 0 else 0)

        mm_mean = np.mean(mm_vals) if mm_vals else float('nan')
        ds_mean = np.mean(ds_vals) if ds_vals else float('nan')

        print(f"  {beta:>6.1f} {accept_rate:>8.3f} {np.mean(s_vals):>8.3f} "
              f"{np.mean(n_vals):>6.0f} {mm_mean:>6.2f} {ds_mean:>6.2f} "
              f"{np.mean(ln_vals):>6.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
