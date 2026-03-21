"""
Experiment 02: Measure emergent dimension from Classical Sequential Growth dynamics.

Key question: For what coupling values does the CSG model produce causets
that look like they came from a manifold? And what dimension?

If the CSG model can produce causets with dimension ~4 at large scales
and dimension ~2 at small scales, that would be a significant result
matching the spectral dimension flow seen in CDT and asymptotic safety.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.growth import classical_sequential_growth, originary_growth
from causal_sets.dimension import myrheim_meyer, spectral_dimension
from causal_sets.core import CausalSet

def measure_scale_dependent_dimension(causet: CausalSet, n_samples: int = 200,
                                       rng: np.random.Generator = None) -> dict:
    """
    Measure the Myrheim-Meyer dimension at different scales by sampling
    intervals of different sizes within the causet.

    Small intervals probe short-distance (UV) structure.
    Large intervals probe long-distance (IR) structure.
    If these give different dimensions, we have dimensional flow.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = causet.n

    # Collect all related pairs with their interval sizes
    pairs_by_size = {}  # interval_size -> list of (i, j) pairs
    for i in range(n):
        for j in range(i + 1, n):
            if causet.order[i, j]:
                size = causet.interval_count(i, j)
                if size not in pairs_by_size:
                    pairs_by_size[size] = []
                pairs_by_size[size].append((i, j))

    if not pairs_by_size:
        return {'scales': [], 'dimensions': []}

    # Group intervals into scale bins
    all_sizes = sorted(pairs_by_size.keys())
    max_size = max(all_sizes) if all_sizes else 0

    # Define scale bins
    if max_size < 5:
        return {'scales': [], 'dimensions': [], 'note': 'intervals too small'}

    # Bin edges: small (0-5), medium (5-20), large (20+)
    bins = [(0, 5, 'UV'), (5, 20, 'mid'), (20, max_size + 1, 'IR')]
    results = {'scales': [], 'dimensions': [], 'counts': []}

    for low, high, label in bins:
        # Collect all related pairs whose intervals fall in this size range
        pairs = []
        for size in range(low, min(high, max_size + 1)):
            if size in pairs_by_size:
                pairs.extend(pairs_by_size[size])

        if len(pairs) < 10:
            continue

        # For these intervals, build a sub-ordering-fraction
        # Count what fraction of all pairs in these intervals are related
        # Actually, use the interval sizes directly with MM formula
        # The ordering fraction for an interval of size m is related to:
        # f = R_interval / (m * (m-1) / 2)
        # But computing R_interval for each interval is expensive.

        # Simpler approach: use the distribution of interval sizes
        # The mean interval size <m> relative to the total n gives dimension info
        # via: <m>/n ~ f_d (for intervals that are a fixed fraction of the total)

        # Even simpler: just report the mean interval size for each scale
        sizes_in_bin = []
        for i, j in pairs:
            sizes_in_bin.append(causet.interval_count(i, j))
        mean_size = np.mean(sizes_in_bin)
        results['scales'].append(label)
        results['counts'].append(len(pairs))

        # Sample some intervals and compute their internal ordering fraction
        if len(pairs) > n_samples:
            sample_idx = rng.choice(len(pairs), n_samples, replace=False)
            sampled_pairs = [pairs[idx] for idx in sample_idx]
        else:
            sampled_pairs = pairs

        # For each sampled interval, compute ordering fraction within it
        dim_estimates = []
        for i, j in sampled_pairs:
            # Get elements in the interval
            interval_elements = [k for k in range(n)
                                 if k != i and k != j
                                 and causet.order[i, k] and causet.order[k, j]]
            m = len(interval_elements)
            if m < 6:
                continue

            # Count relations among interval elements
            relations = 0
            for a_idx in range(len(interval_elements)):
                for b_idx in range(a_idx + 1, len(interval_elements)):
                    a, b = interval_elements[a_idx], interval_elements[b_idx]
                    if causet.order[a, b]:
                        relations += 1

            total_pairs = m * (m - 1) // 2
            if total_pairs == 0:
                continue

            f_measured = relations / (m * (m - 1))  # MM convention: ordered pairs
            from causal_sets.dimension import _invert_ordering_fraction
            d_est = _invert_ordering_fraction(f_measured)
            if 0.5 < d_est < 15:
                dim_estimates.append(d_est)

        if dim_estimates:
            results['dimensions'].append({
                'mean': np.mean(dim_estimates),
                'std': np.std(dim_estimates),
                'n_samples': len(dim_estimates)
            })
        else:
            results['dimensions'].append(None)

    return results


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 02: Emergent Dimension from CSG Dynamics")
    print("=" * 70)

    # Phase 1: Global dimension vs coupling
    print("\n--- Phase 1: Global Myrheim-Meyer dimension vs coupling ---")
    print(f"{'coupling':>10} {'ordering_f':>12} {'MM_dim':>10} {'longest_chain':>14} {'n_links':>10}")
    print("-" * 60)

    n = 300
    n_trials = 5
    couplings = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

    for p in couplings:
        dims = []
        ofs = []
        chains = []
        links = []

        for trial in range(n_trials):
            cs = classical_sequential_growth(n, coupling=p, rng=rng)
            d = myrheim_meyer(cs)
            dims.append(d)
            ofs.append(cs.ordering_fraction())
            chains.append(cs.longest_chain())
            links.append(np.sum(cs.link_matrix()))

        print(f"{p:>10.2f} {np.mean(ofs):>12.4f} {np.mean(dims):>10.2f} ± {np.std(dims):>4.2f}"
              f" {np.mean(chains):>10.1f} {np.mean(links):>10.1f}")

    # Phase 2: Scale-dependent dimension for interesting coupling values
    print("\n--- Phase 2: Scale-dependent dimension (dimensional flow?) ---")

    interesting_couplings = [0.10, 0.20, 0.30]
    n_big = 500

    for p in interesting_couplings:
        print(f"\n  Coupling p = {p:.2f}, N = {n_big}")
        cs = classical_sequential_growth(n_big, coupling=p, rng=rng)
        result = measure_scale_dependent_dimension(cs, n_samples=100, rng=rng)

        if result.get('note'):
            print(f"    {result['note']}")
            continue

        for scale, dim_data, count in zip(result['scales'],
                                           result['dimensions'],
                                           result['counts']):
            if dim_data:
                print(f"    Scale {scale:>4}: d = {dim_data['mean']:.2f} ± {dim_data['std']:.2f}"
                      f"  (n_intervals={count}, n_measured={dim_data['n_samples']})")
            else:
                print(f"    Scale {scale:>4}: insufficient data (n_intervals={count})")

    # Phase 3: Spectral dimension for a large CSG causet
    print("\n--- Phase 3: Spectral dimension ---")
    for p in [0.15, 0.25, 0.40]:
        cs = classical_sequential_growth(300, coupling=p, rng=rng)
        sigmas, d_s = spectral_dimension(cs, sigma_range=(0.01, 50.0), n_sigma=60)
        if len(sigmas) > 0:
            # Report at several scales
            print(f"\n  Coupling p = {p:.2f}:")
            for idx in [10, 20, 30, 40, 50]:
                if idx < len(sigmas):
                    print(f"    sigma={sigmas[idx]:>8.3f}  d_s={d_s[idx]:>6.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
