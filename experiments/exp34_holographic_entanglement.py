"""
Experiment 34: Holographic Entanglement Structure at the BD Transition

Paper B3: Does the BD continuum phase satisfy Ryu-Takayanagi-like
entanglement structure?

Approach:
1. For 2-orders, one permutation coordinate provides a "spatial" direction
2. Partition elements into spatial regions of varying size
3. Define "causal mutual information" from the cross-boundary relation structure
4. Check if it satisfies area-law scaling (S ~ boundary size, not volume)
5. Compare continuum vs crystalline phases

Three information-theoretic observables:
A. Cross-boundary link fraction: fraction of all links that cross the partition
B. Conditional interval entropy: how does the interval structure in region A
   depend on what's in region B?
C. Causal mutual information: I(A:B) from the joint vs marginal distributions
   of interval sizes in the two regions
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.two_orders import TwoOrder
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet


def spatial_partition(two_order, frac=0.5):
    """
    Partition a 2-order into spatial regions using the v-coordinate.
    Elements with v-rank < frac*N go to region A, rest to region B.
    Returns (A_indices, B_indices).
    """
    N = two_order.N
    # Sort by v-coordinate to get spatial ordering
    v_order = np.argsort(two_order.v)
    cut = int(frac * N)
    A = set(v_order[:cut])
    B = set(v_order[cut:])
    return A, B


def cross_boundary_links(cs, A, B):
    """Count links that cross from A to B or B to A."""
    links = cs.link_matrix()
    count = 0
    for i in A:
        for j in B:
            if links[i, j] or links[j, i]:
                count += 1
    return count


def cross_boundary_relations(cs, A, B):
    """Count all causal relations crossing the boundary."""
    count = 0
    for i in A:
        for j in B:
            if cs.order[i, j] or cs.order[j, i]:
                count += 1
    return count


def region_interval_distribution(cs, region, max_k=10):
    """
    Interval distribution restricted to a region.
    Count intervals [x,y] where BOTH x and y are in the region.
    """
    region_list = sorted(region)
    counts = {}
    for idx_i, i in enumerate(region_list):
        for idx_j, j in enumerate(region_list):
            if i < j and cs.order[i, j]:
                # Count elements between i and j that are ALSO in the region
                n_between = 0
                for k in region_list:
                    if k != i and k != j and cs.order[i, k] and cs.order[k, j]:
                        n_between += 1
                n_between = min(n_between, max_k)
                counts[n_between] = counts.get(n_between, 0) + 1
    return counts


def shannon_entropy(counts, max_k=10):
    """Shannon entropy of a count distribution."""
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0:
        return 0.0
    p = dist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def causal_mutual_information(cs, A, B, max_k=10):
    """
    Causal mutual information I(A:B).

    Define probability distributions from the interval structure:
    - P_A(n): interval size distribution within region A
    - P_B(n): interval size distribution within region B
    - P_AB(n): interval size distribution of the full causet

    Mutual information: I(A:B) = H(A) + H(B) - H(AB)
    where H is the Shannon entropy of the interval distribution.

    For an area law: I(A:B) should scale with the boundary size,
    not with the volume of A or B.
    """
    counts_A = region_interval_distribution(cs, A, max_k)
    counts_B = region_interval_distribution(cs, B, max_k)
    counts_AB = count_intervals_by_size(cs, max_size=max_k)

    H_A = shannon_entropy(counts_A, max_k)
    H_B = shannon_entropy(counts_B, max_k)
    H_AB = shannon_entropy(counts_AB, max_k)

    # Mutual information (can be negative due to finite-size effects)
    I_AB = H_A + H_B - H_AB

    return {
        'I_AB': I_AB,
        'H_A': H_A,
        'H_B': H_B,
        'H_AB': H_AB,
    }


def boundary_size(two_order, A, B):
    """
    Estimate the "boundary size" between regions A and B.
    In 2D, this is the number of elements at the boundary of the partition.
    For a v-coordinate partition, it's approximately constant (~2 in 2D).
    But we can measure it as the number of links crossing the boundary.
    """
    cs = two_order.to_causet()
    return cross_boundary_links(cs, A, B)


def main():
    rng = np.random.default_rng(42)
    N = 50
    eps = 0.12

    print("=" * 80)
    print("EXPERIMENT 34: Holographic Entanglement at the BD Transition")
    print(f"N={N}, eps={eps}")
    print("=" * 80)

    # Phase 1: Entanglement vs partition size in the continuum phase
    print("\n--- Phase 1: S(A) vs |A|/N in CONTINUUM phase (beta=0) ---")
    print("  RT prediction for 2D: S(A) ~ constant (boundary = 2 points)")
    print("  Volume law: S(A) ~ |A|")

    n_samples = 30
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  {'|A|/N':>6} {'I(A:B)':>8} {'H(A)':>7} {'H(B)':>7} {'H(AB)':>7} {'x-links':>8}")
    print("-" * 55)

    for frac in fracs:
        I_vals, HA_vals, HB_vals, HAB_vals, xlink_vals = [], [], [], [], []

        for _ in range(n_samples):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            A, B = spatial_partition(to, frac)

            mi = causal_mutual_information(cs, A, B)
            I_vals.append(mi['I_AB'])
            HA_vals.append(mi['H_A'])
            HB_vals.append(mi['H_B'])
            HAB_vals.append(mi['H_AB'])
            xlink_vals.append(cross_boundary_links(cs, A, B))

        print(f"  {frac:>6.1f} {np.mean(I_vals):>8.3f} {np.mean(HA_vals):>7.3f} "
              f"{np.mean(HB_vals):>7.3f} {np.mean(HAB_vals):>7.3f} "
              f"{np.mean(xlink_vals):>8.1f}")

    # Phase 2: Compare continuum vs crystalline
    print("\n--- Phase 2: Continuum vs Crystalline phase ---")

    beta_c = 6.64 / (N * eps ** 2)

    for beta, label in [(0, "Continuum"), (beta_c * 3, "Crystalline")]:
        print(f"\n  {label} (beta={beta:.1f}):")

        res = mcmc_corrected(N, beta=beta, eps=eps,
                              n_steps=40000, n_therm=20000,
                              record_every=50, rng=rng)

        I_half_vals = []
        xlink_half_vals = []
        H_AB_vals = []

        for cs in res['samples'][-30:]:
            # Need the TwoOrder for spatial partition — approximate with
            # a simple index-based partition (first half vs second half by label)
            A = set(range(N // 2))
            B = set(range(N // 2, N))

            mi = causal_mutual_information(cs, A, B)
            I_half_vals.append(mi['I_AB'])
            H_AB_vals.append(mi['H_AB'])
            xlink_half_vals.append(cross_boundary_links(cs, A, B))

        print(f"    I(A:B) = {np.mean(I_half_vals):.3f} ± {np.std(I_half_vals):.3f}")
        print(f"    H(AB) = {np.mean(H_AB_vals):.3f}")
        print(f"    Cross-boundary links = {np.mean(xlink_half_vals):.1f}")

    # Phase 3: Area law test — I(A:B) vs boundary size
    print("\n--- Phase 3: Area law test ---")
    print("  If I(A:B) scales with boundary (cross-links): area law")
    print("  If I(A:B) scales with min(|A|, |B|): volume law")

    # Vary partition size and measure both I and boundary
    print(f"\n  {'|A|/N':>6} {'I(A:B)':>8} {'boundary':>9} {'I/boundary':>11} {'|A|':>5}")
    print("-" * 50)

    for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        I_vals, bnd_vals = [], []
        for _ in range(40):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            A, B = spatial_partition(to, frac)
            mi = causal_mutual_information(cs, A, B)
            I_vals.append(mi['I_AB'])
            bnd_vals.append(cross_boundary_links(cs, A, B))

        I_mean = np.mean(I_vals)
        bnd_mean = np.mean(bnd_vals)
        ratio = I_mean / bnd_mean if bnd_mean > 0 else float('nan')

        print(f"  {frac:>6.1f} {I_mean:>8.3f} {bnd_mean:>9.1f} {ratio:>11.4f} {int(frac*N):>5}")

    print("\n  If I/boundary is CONSTANT: area law (RT-like)")
    print("  If I/boundary INCREASES with |A|: volume law")

    # Phase 4: Scaling with N
    print("\n--- Phase 4: Scaling with N (half partition, continuum) ---")
    print(f"  {'N':>5} {'I(A:B)':>8} {'boundary':>9} {'I/boundary':>11}")
    print("-" * 40)

    for N_test in [30, 50, 70, 90]:
        I_vals, bnd_vals = [], []
        for _ in range(30):
            to = TwoOrder(N_test, rng=rng)
            cs = to.to_causet()
            A, B = spatial_partition(to, 0.5)
            mi = causal_mutual_information(cs, A, B)
            I_vals.append(mi['I_AB'])
            bnd_vals.append(cross_boundary_links(cs, A, B))

        I_mean = np.mean(I_vals)
        bnd_mean = np.mean(bnd_vals)
        ratio = I_mean / bnd_mean if bnd_mean > 0 else float('nan')
        print(f"  {N_test:>5} {I_mean:>8.3f} {bnd_mean:>9.1f} {ratio:>11.4f}")

    print("\n  If I/boundary is constant as N grows: area law holds in continuum limit")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
