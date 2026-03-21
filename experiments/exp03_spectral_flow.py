"""
Experiment 03: Spectral dimension flow — the key test.

The universal prediction across CDT, asymptotic safety, causal sets, and
Horava-Lifshitz gravity is that the spectral dimension flows from ~4 at
large scales to ~2 at short scales.

This experiment:
1. Validates spectral dimension measurement on sprinkled causets (known d)
2. Measures spectral dimension flow on CSG causets at various couplings
3. Tries the originary growth model
4. Looks for the d_s=2 UV behavior
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet, sprinkle_fast, csg_fast, spectral_dimension_fast


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 03: Spectral Dimension Flow")
    print("=" * 70)

    # Phase 1: Validate on known manifolds
    print("\n--- Phase 1: Spectral dimension of sprinkled causets ---")

    for dim in [2, 3, 4]:
        n = 800
        causet, coords = sprinkle_fast(n, dim=dim, extent_t=1.0, region='diamond', rng=rng)
        sigmas, d_s = spectral_dimension_fast(causet, sigma_range=(0.005, 200.0), n_sigma=80)

        if len(sigmas) == 0:
            print(f"  d={dim}: no connected component")
            continue

        # Find the peak spectral dimension (should be near the true dimension)
        peak_idx = np.argmax(d_s)
        peak_d_s = d_s[peak_idx]
        peak_sigma = sigmas[peak_idx]

        # Average over the "plateau" region (middle third)
        third = len(sigmas) // 3
        plateau_d_s = np.mean(d_s[third:2*third])

        print(f"  d={dim}, N={n}: peak d_s = {peak_d_s:.2f} at sigma={peak_sigma:.3f}, "
              f"plateau d_s = {plateau_d_s:.2f}")

        # Detailed profile
        print(f"    {'sigma':>10} {'d_s':>8}")
        for idx in np.linspace(5, len(sigmas) - 5, 10, dtype=int):
            print(f"    {sigmas[idx]:>10.4f} {d_s[idx]:>8.2f}")

    # Phase 2: CSG at various couplings — look for dimensional flow
    print("\n--- Phase 2: CSG spectral dimension flow ---")

    n = 1000
    couplings = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    for p in couplings:
        print(f"\n  Coupling p = {p:.2f}, N = {n}")
        cs = csg_fast(n, coupling=p, rng=rng)

        sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=80)
        if len(sigmas) == 0:
            print("    No connected component")
            continue

        peak_idx = np.argmax(d_s)
        peak_d_s = d_s[peak_idx]
        peak_sigma = sigmas[peak_idx]

        # Look for UV behavior (small sigma) and IR behavior (large sigma)
        uv_region = d_s[:15]
        ir_region = d_s[-15:]

        # Find the scale where d_s is closest to 2 (UV prediction)
        closest_to_2 = np.argmin(np.abs(d_s - 2.0))

        print(f"    Peak d_s = {peak_d_s:.2f} at sigma = {peak_sigma:.4f}")
        print(f"    d_s closest to 2.0 at sigma = {sigmas[closest_to_2]:.4f}")
        print(f"    Ordering fraction: {cs.ordering_fraction():.4f}")
        print(f"    Longest chain: {cs.longest_chain()}")

        # Full profile
        print(f"    {'sigma':>10} {'d_s':>8}")
        for idx in np.linspace(3, len(sigmas) - 3, 12, dtype=int):
            marker = " <-- d_s≈2" if abs(d_s[idx] - 2.0) < 0.2 else ""
            print(f"    {sigmas[idx]:>10.4f} {d_s[idx]:>8.2f}{marker}")

    # Phase 3: Multiple trials for statistics on the peak
    print("\n--- Phase 3: Statistics on peak spectral dimension ---")
    print(f"{'coupling':>10} {'peak_d_s':>12} {'sigma_at_peak':>14} {'d_s_at_sigma=1':>16}")
    print("-" * 55)

    n = 800
    n_trials = 8

    for p in [0.03, 0.05, 0.08, 0.10, 0.15]:
        peaks = []
        peak_sigmas = []
        d_at_1 = []

        for trial in range(n_trials):
            cs = csg_fast(n, coupling=p, rng=rng)
            sigmas, d_s = spectral_dimension_fast(cs, sigma_range=(0.005, 200.0), n_sigma=80)
            if len(sigmas) == 0:
                continue
            peak_idx = np.argmax(d_s)
            peaks.append(d_s[peak_idx])
            peak_sigmas.append(sigmas[peak_idx])

            # d_s at sigma ≈ 1
            idx_1 = np.argmin(np.abs(sigmas - 1.0))
            d_at_1.append(d_s[idx_1])

        if peaks:
            print(f"{p:>10.2f} {np.mean(peaks):>8.2f} ± {np.std(peaks):>4.2f}"
                  f" {np.mean(peak_sigmas):>10.3f} ± {np.std(peak_sigmas):>4.3f}"
                  f" {np.mean(d_at_1):>10.2f} ± {np.std(d_at_1):>4.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
