"""
Experiment 08: Deep investigation of the step k_max=1 coupling.

t_0 = 1, t_1 = 1, t_k = 0 for k >= 2

This coupling produced MM dimension ≈ 4.24 in the scan — the only
4D candidate from any CSG dynamics we've tested.

Physics: each new element can be born with at most 1 maximal ancestor.
This creates a branching/tree-like causal structure.

We need to:
1. Verify at larger N with more trials
2. Measure the spectral dimension profile
3. Check scaling: does MM dimension hold as N increases?
4. Explore variations: t_0 = a, t_1 = b, t_k = 0 for different a/b ratios
5. Test whether this is genuinely 4D or an artifact
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.general_csg import general_csg, _evaluate_couplings
from causal_sets.fast_core import spectral_dimension_fast


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 08: Deep Investigation of 4D Candidate (step k_max=1)")
    print("=" * 80)

    # Phase 1: Verify at multiple sizes
    print("\n--- Phase 1: Scaling with N ---")
    print(f"{'N':>6} {'ord_frac':>10} {'MM_dim':>10} {'chain':>8} {'chain/N^.25':>12} "
          f"{'links/N':>8}")
    print("-" * 65)

    t_step1 = [1.0, 1.0]  # t_0=1, t_1=1, rest=0
    n_trials = 5

    for n in [50, 100, 200, 300, 500]:
        r = _evaluate_couplings(n, t_step1, n_trials, rng, f"step1 N={n}")
        chain_scaled = r['longest_chain'] / (n ** 0.25)
        print(f"  {n:>5} {r['ordering_fraction']:>10.4f} {r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
              f"{r['longest_chain']:>8.1f} {chain_scaled:>12.2f} {r['links_per_element']:>8.2f}")

    # Phase 2: Explore the t_0/t_1 ratio
    print("\n--- Phase 2: Varying t_0/t_1 ratio ---")
    print(f"{'t_0':>6} {'t_1':>6} {'ord_frac':>10} {'MM_dim':>10} {'chain':>8}")
    print("-" * 50)

    n = 200
    for t0 in [0.5, 1.0, 2.0, 5.0]:
        for t1 in [0.1, 0.5, 1.0, 2.0, 5.0]:
            t = [t0, t1]
            r = _evaluate_couplings(n, t, 3, rng, f"t0={t0} t1={t1}")
            marker = " ***" if 3.5 <= r['mm_dimension'] <= 4.5 else ""
            print(f"  {t0:>5.1f} {t1:>5.1f} {r['ordering_fraction']:>10.4f} "
                  f"{r['mm_dimension']:>8.2f}±{r['mm_std']:>4.2f} "
                  f"{r['longest_chain']:>8.1f}{marker}")

    # Phase 3: Spectral dimension profile for the best candidate
    print("\n--- Phase 3: Spectral dimension of step k_max=1 ---")
    n = 300
    cs = general_csg(n, t_step1, rng=rng)
    sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=80)

    if len(sigmas) > 0:
        peak_idx = np.argmax(d_s)
        print(f"  N={n}, peak d_s = {d_s[peak_idx]:.2f} at sigma = {sigmas[peak_idx]:.3f}")
        print(f"\n  {'sigma':>10} {'d_s':>8}")
        for idx in np.linspace(3, len(sigmas) - 3, 15, dtype=int):
            marker = " <--" if abs(d_s[idx] - 4.0) < 0.3 else ""
            marker = marker or (" <-- d_s≈2" if abs(d_s[idx] - 2.0) < 0.2 else "")
            print(f"  {sigmas[idx]:>10.4f} {d_s[idx]:>8.2f}{marker}")

    # Phase 4: Compare against sprinkled 4D causet
    print("\n--- Phase 4: Comparison with sprinkled 4D causet ---")
    from causal_sets.fast_core import sprinkle_fast

    cs_sprinkled, _ = sprinkle_fast(300, dim=4, extent_t=1.0, region='diamond', rng=rng)
    r_sprinkled = {
        'ordering_fraction': cs_sprinkled.ordering_fraction(),
        'longest_chain': cs_sprinkled.longest_chain(),
        'links_per_element': int(np.sum(cs_sprinkled.link_matrix())) / 300,
    }

    from causal_sets.core import CausalSet
    from causal_sets.dimension import myrheim_meyer
    cs_old = CausalSet(300)
    cs_old.order = cs_sprinkled.order.astype(np.int8)
    r_sprinkled['mm_dimension'] = myrheim_meyer(cs_old)

    cs_old2 = CausalSet(n)
    cs_old2.order = cs.order.astype(np.int8)
    r_csg = {
        'ordering_fraction': cs.ordering_fraction(),
        'mm_dimension': myrheim_meyer(cs_old2),
        'longest_chain': cs.longest_chain(),
        'links_per_element': int(np.sum(cs.link_matrix())) / n,
    }

    print(f"  {'':>20} {'Sprinkled 4D':>15} {'CSG step1':>15}")
    print(f"  {'Ordering fraction':>20} {r_sprinkled['ordering_fraction']:>15.4f} {r_csg['ordering_fraction']:>15.4f}")
    print(f"  {'MM dimension':>20} {r_sprinkled['mm_dimension']:>15.2f} {r_csg['mm_dimension']:>15.2f}")
    print(f"  {'Longest chain':>20} {r_sprinkled['longest_chain']:>15.0f} {r_csg['longest_chain']:>15.0f}")
    print(f"  {'Links/element':>20} {r_sprinkled['links_per_element']:>15.2f} {r_csg['links_per_element']:>15.2f}")

    # Phase 5: Interval size distribution comparison
    print("\n--- Phase 5: Interval size distributions ---")
    _, sizes_csg = cs.interval_sizes_vectorized()
    _, sizes_spr = cs_sprinkled.interval_sizes_vectorized()

    for label, sizes in [("CSG step1", sizes_csg), ("Sprinkled 4D", sizes_spr)]:
        if len(sizes) == 0:
            print(f"  {label}: no intervals")
            continue
        print(f"  {label}: n_intervals={len(sizes)}, "
              f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}, "
              f"max={np.max(sizes)}, "
              f"frac_empty={np.mean(sizes == 0):.3f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
