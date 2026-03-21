"""
Experiment 06: Link density fluctuations as a cosmological constant proxy.

The ordering fraction saturates toward 1 in CSG, confounding the Poisson test.
Instead, use the LINK density (irreducible relations), which doesn't saturate.

Links are the "nearest neighbor" relations — more analogous to local
spacetime structure than the transitive closure.

Also test: longest chain / N^(1/d) scaling, which gives an independent
handle on the effective dimension and its fluctuations.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet, sprinkle_fast, csg_fast


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 06: Link Density Fluctuations")
    print("=" * 70)

    # Observable: number of links / N
    # For sprinkled causets, this probes local structure
    # For CSG, it measures the effective "connectivity"

    n_trials = 20

    # Phase 1: Sprinkled causets — link density scaling
    print("\n--- Phase 1: Sprinkled causets (2D, 3D, 4D) ---")
    print(f"{'dim':>4} {'N':>6} {'links/N':>10} {'std':>10} {'std*sqrt(N)':>12}")
    print("-" * 48)

    for dim in [2, 3, 4]:
        for n in [100, 200, 500, 1000]:
            link_densities = []
            for _ in range(n_trials):
                cs, _ = sprinkle_fast(n, dim=dim, extent_t=1.0, region='diamond', rng=rng)
                n_links = int(np.sum(cs.link_matrix()))
                link_densities.append(n_links / n)

            ld = np.array(link_densities)
            print(f"  {dim:>3} {n:>6} {np.mean(ld):>10.4f} {np.std(ld):>10.4f} "
                  f"{np.std(ld)*np.sqrt(n):>12.4f}")

    # Phase 2: CSG causets — link density scaling
    print("\n--- Phase 2: CSG causets ---")
    print(f"{'p':>6} {'N':>6} {'links/N':>10} {'std':>10} {'std*sqrt(N)':>12} {'chain/N':>10}")
    print("-" * 62)

    for p in [0.03, 0.05, 0.08, 0.10]:
        for n in [100, 200, 500, 1000, 2000]:
            link_densities = []
            chain_fracs = []
            for _ in range(n_trials):
                cs = csg_fast(n, coupling=p, rng=rng)
                n_links = int(np.sum(cs.link_matrix()))
                link_densities.append(n_links / n)
                chain_fracs.append(cs.longest_chain() / n)

            ld = np.array(link_densities)
            cf = np.array(chain_fracs)
            print(f"  {p:>5.2f} {n:>6} {np.mean(ld):>10.4f} {np.std(ld):>10.4f} "
                  f"{np.std(ld)*np.sqrt(n):>12.4f} {np.mean(cf):>10.4f}")

    # Phase 3: The key comparison — do fluctuations scale like Poisson?
    print("\n--- Phase 3: Poisson test on link count ---")
    print("If links follow Poisson: var(L) = <L>, so var(L/N)/mean(L/N) ~ 1/N")
    print("Equivalently: std(L) / sqrt(mean(L)) should be ~constant")

    print(f"\n{'model':>15} {'N':>6} {'mean_L':>10} {'std_L':>10} {'std/sqrt(mean)':>15} {'note':>10}")
    print("-" * 70)

    # Sprinkled 2D
    for n in [100, 200, 500, 1000]:
        link_counts = []
        for _ in range(n_trials):
            cs, _ = sprinkle_fast(n, dim=2, extent_t=1.0, region='diamond', rng=rng)
            link_counts.append(int(np.sum(cs.link_matrix())))
        lc = np.array(link_counts, dtype=float)
        ratio = np.std(lc) / np.sqrt(np.mean(lc)) if np.mean(lc) > 0 else float('nan')
        print(f"  {'Sprinkle 2D':>13} {n:>6} {np.mean(lc):>10.1f} {np.std(lc):>10.2f} "
              f"{ratio:>15.4f}")

    print()

    # CSG p=0.05
    for n in [100, 200, 500, 1000, 2000]:
        link_counts = []
        for _ in range(n_trials):
            cs = csg_fast(n, coupling=0.05, rng=rng)
            link_counts.append(int(np.sum(cs.link_matrix())))
        lc = np.array(link_counts, dtype=float)
        ratio = np.std(lc) / np.sqrt(np.mean(lc)) if np.mean(lc) > 0 else float('nan')
        print(f"  {'CSG p=0.05':>13} {n:>6} {np.mean(lc):>10.1f} {np.std(lc):>10.2f} "
              f"{ratio:>15.4f}")

    print()

    # CSG p=0.10
    for n in [100, 200, 500, 1000, 2000]:
        link_counts = []
        for _ in range(n_trials):
            cs = csg_fast(n, coupling=0.10, rng=rng)
            link_counts.append(int(np.sum(cs.link_matrix())))
        lc = np.array(link_counts, dtype=float)
        ratio = np.std(lc) / np.sqrt(np.mean(lc)) if np.mean(lc) > 0 else float('nan')
        print(f"  {'CSG p=0.10':>13} {n:>6} {np.mean(lc):>10.1f} {np.std(lc):>10.2f} "
              f"{ratio:>15.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  Constant ratio → Poisson fluctuations → Sorkin prediction holds")
    print("  Decreasing ratio → sub-Poisson → dynamics suppresses Lambda")
    print("  Increasing ratio → super-Poisson → dynamics amplifies Lambda")
    print("=" * 70)


if __name__ == '__main__':
    main()
