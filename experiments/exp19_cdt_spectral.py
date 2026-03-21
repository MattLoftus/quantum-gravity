"""
Experiment 19: 2D CDT Spectral Dimension — Direct Comparison.

The first apples-to-apples comparison of spectral dimension between
CDT and causal sets using identical measurement methodology.

Key question: Does 2D CDT show d_s ≈ 2 as a genuine plateau, or
is it a transient crossing like we found for random graphs (Exp04)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from cdt.triangulation import mcmc_cdt, spectral_dimension_cdt


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 19: 2D CDT Spectral Dimension")
    print("=" * 70)

    # Phase 1: CDT at different couplings
    print("\n--- Phase 1: CDT volume profile and spectral dimension ---")

    T = 30  # time slices
    s_init = 15  # initial spatial volume per slice
    target_vol = T * s_init

    for lambda2, mu in [(0.0, 0.01), (0.1, 0.01), (0.5, 0.01), (-0.1, 0.01)]:
        samples = mcmc_cdt(T=T, s_init=s_init, lambda2=lambda2, mu=mu,
                           n_steps=20000, target_volume=target_vol, rng=rng)

        # Take last 500 samples
        recent = samples[-500:]

        # Average volume profile
        mean_profile = np.mean(recent, axis=0)
        total_v = np.mean([np.sum(s) for s in recent])

        # Spectral dimension of a representative sample
        sigmas_list = []
        ds_list = []
        peaks = []

        for s in recent[::50]:  # every 50th sample
            sigmas, d_s = spectral_dimension_cdt(s.astype(int))
            if len(sigmas) > 0:
                sigmas_list.append(sigmas)
                ds_list.append(d_s)
                peaks.append(np.max(d_s))

        if peaks:
            mean_peak = np.mean(peaks)
            std_peak = np.std(peaks)

            # Find d_s ≈ 2 region
            mean_ds = np.mean(ds_list, axis=0)
            mean_sigmas = sigmas_list[0]

            # Check for plateau at d_s = 2
            near_2 = np.abs(mean_ds - 2.0) < 0.3
            plateau_width = np.sum(near_2)

            print(f"\n  lambda2={lambda2:>5.1f}, mu={mu}: <N>={total_v:.0f}")
            print(f"    Peak d_s: {mean_peak:.2f} ± {std_peak:.2f}")
            print(f"    d_s ≈ 2 plateau width: {plateau_width} points out of {len(mean_ds)}")
            print(f"    Profile (mean across samples):")
            print(f"    {'sigma':>10} {'d_s':>8}")
            for idx in np.linspace(2, len(mean_sigmas) - 2, 10, dtype=int):
                marker = " <--" if abs(mean_ds[idx] - 2.0) < 0.3 else ""
                print(f"    {mean_sigmas[idx]:>10.3f} {mean_ds[idx]:>8.2f}{marker}")

    # Phase 2: Comparison with causal sets and random graphs
    print("\n--- Phase 2: Direct comparison using SAME spectral dimension code ---")

    from causal_sets.fast_core import sprinkle_fast, spectral_dimension_fast

    # Sprinkled 2D causet
    cs2d, _ = sprinkle_fast(450, dim=2, region='diamond', rng=rng)
    sigmas_cs, ds_cs = spectral_dimension_fast(cs2d, sigma_range=(0.1, 100.0), n_sigma=50)

    # CDT with ~450 vertices
    samples_cdt = mcmc_cdt(T=30, s_init=15, lambda2=0.0, mu=0.01,
                            n_steps=20000, target_volume=450, rng=rng)
    last_cdt = samples_cdt[-1].astype(int)
    sigmas_cdt, ds_cdt = spectral_dimension_cdt(last_cdt, sigma_range=(0.1, 100.0), n_sigma=50)

    # Random graph with ~450 nodes
    from causal_sets.fast_core import FastCausalSet
    n_rand = 450
    rand_adj = np.zeros((n_rand, n_rand), dtype=bool)
    for i in range(n_rand):
        for j in range(i + 1, n_rand):
            if rng.random() < 0.01:
                rand_adj[i, j] = True
                rand_adj[j, i] = True
    degree = np.sum(rand_adj, axis=1).astype(float)
    mask = degree > 0
    adj_c = rand_adj[np.ix_(mask, mask)].astype(float)
    degree_c = degree[mask]
    n_c = adj_c.shape[0]
    d_inv = np.diag(1.0 / np.sqrt(degree_c))
    L = np.eye(n_c) - d_inv @ adj_c @ d_inv
    evals = np.linalg.eigvalsh(L)
    evals = np.clip(evals, 0, None)
    sigmas_rand = np.logspace(np.log10(0.1), np.log10(100.0), 50)
    P_rand = np.array([np.mean(np.exp(-evals * s)) for s in sigmas_rand])
    ds_rand = -2 * np.gradient(np.log(P_rand + 1e-300), np.log(sigmas_rand))

    print(f"\n  {'sigma':>10} {'CDT(2D)':>10} {'Causet(2D)':>12} {'Random':>10}")
    print("  " + "-" * 48)

    for idx in range(0, 50, 5):
        ds_cdt_val = ds_cdt[idx] if idx < len(ds_cdt) else float('nan')
        ds_cs_val = ds_cs[idx] if idx < len(ds_cs) else float('nan')
        ds_r_val = ds_rand[idx] if idx < len(ds_rand) else float('nan')
        sigma = sigmas_cdt[idx] if idx < len(sigmas_cdt) else sigmas_rand[idx]
        print(f"  {sigma:>10.3f} {ds_cdt_val:>10.2f} {ds_cs_val:>12.2f} {ds_r_val:>10.2f}")

    # Peaks
    if len(ds_cdt) > 0:
        print(f"\n  Peak d_s:  CDT={np.max(ds_cdt):.2f}  Causet={np.max(ds_cs):.2f}  Random={np.max(ds_rand):.2f}")

        # Plateau widths
        for name, ds in [("CDT", ds_cdt), ("Causet", ds_cs), ("Random", ds_rand)]:
            near2 = np.sum(np.abs(ds - 2.0) < 0.3)
            print(f"  {name} d_s≈2 plateau: {near2} points")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
