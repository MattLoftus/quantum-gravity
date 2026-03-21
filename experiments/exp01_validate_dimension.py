"""
Experiment 01: Validate dimension estimators against known manifolds.

Sprinkle causal sets into 2D, 3D, and 4D Minkowski spacetime and verify
that the Myrheim-Meyer estimator correctly recovers the dimension.

This is our ground truth test before we apply these estimators to
dynamically grown causets.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.sprinkle import sprinkle_minkowski
from causal_sets.dimension import myrheim_meyer, _ordering_fraction_theory, spectral_dimension

def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 01: Dimension Estimator Validation")
    print("=" * 70)

    # First, print the theoretical ordering fractions for reference
    print("\nTheoretical ordering fractions f(d):")
    for d in range(1, 8):
        print(f"  d={d}: f = {_ordering_fraction_theory(d):.6f}")

    # Test across dimensions and sizes
    dims = [2, 3, 4]
    sizes = [100, 200, 500, 1000]
    n_trials = 5

    print(f"\n{'Dim':>4} {'N':>6} {'f_measured':>12} {'f_theory':>12} {'d_estimated':>12} {'error':>8}")
    print("-" * 60)

    for dim in dims:
        f_theory = _ordering_fraction_theory(dim)
        for n in sizes:
            estimates = []
            fractions = []
            for trial in range(n_trials):
                causet, coords = sprinkle_minkowski(n, dim=dim, extent_t=1.0,
                                                     extent_x=1.0, rng=rng)
                f = causet.ordering_fraction()
                d_est = myrheim_meyer(causet)
                estimates.append(d_est)
                fractions.append(f)

            d_mean = np.mean(estimates)
            f_mean = np.mean(fractions)
            error = d_mean - dim
            print(f"  {dim:>3} {n:>6} {f_mean:>12.6f} {f_theory:>12.6f} "
                  f"{d_mean:>12.4f} {error:>+8.4f}")

    # Spectral dimension test for 2D (small size due to link matrix cost)
    print("\n" + "=" * 70)
    print("SPECTRAL DIMENSION TEST (2D Minkowski, N=200)")
    print("=" * 70)

    causet, coords = sprinkle_minkowski(200, dim=2, extent_t=1.0, extent_x=1.0, rng=rng)
    sigmas, d_s = spectral_dimension(causet, sigma_range=(0.01, 100.0), n_sigma=80)

    if len(sigmas) > 0:
        # Find the plateau region
        mid = len(sigmas) // 2
        quarter = len(sigmas) // 4
        d_s_plateau = np.mean(d_s[quarter:3*quarter])
        print(f"  Spectral dimension (plateau average): {d_s_plateau:.2f}")
        print(f"  Expected: ~2.0")
        print(f"\n  sigma range -> d_s sample:")
        for idx in range(0, len(sigmas), len(sigmas) // 8):
            print(f"    sigma={sigmas[idx]:>8.3f}  d_s={d_s[idx]:>6.2f}")
    else:
        print("  (insufficient connected components for spectral dimension)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
