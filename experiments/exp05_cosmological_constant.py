"""
Experiment 05: The Cosmological Constant from Causal Set Dynamics.

Sorkin's prediction (1987): In a causal set universe with N elements,
the cosmological constant fluctuates with amplitude:

    Lambda ~ 1/sqrt(N)

in Planck units. This correctly predicted the observed value of Lambda
BEFORE its discovery — arguably the only correct a priori prediction
of Lambda's magnitude from any quantum gravity approach.

The argument: in a causal set with N elements, the number of unborn
elements is a Poisson random variable. The fluctuation in the number
of spacetime elements within a 4-volume V is delta_N ~ sqrt(N).
Since N ~ rho * V and Lambda ~ 1/V in Planck units, the fluctuation
in Lambda is delta_Lambda ~ 1/sqrt(N).

This experiment tests whether this prediction holds in DYNAMICAL
causal set models (CSG), not just kinematic (sprinkled) ones.

If the Lambda ~ 1/sqrt(N) scaling holds in CSG, that's a confirmation
of Sorkin's argument. If it fails, that constrains which dynamics
are physically viable.

We measure Lambda via the number-volume relationship: in a sprinkled
causet, N = rho * V. Fluctuations in N at fixed V (or V at fixed N)
probe the effective cosmological constant.

For CSG causets (no embedding manifold), we use a proxy: the ratio
of the interval cardinality to the chain length raised to the d-th power
measures an effective "volume fluctuation." The key observable is:
how does the variance of interval sizes scale with the mean?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet, sprinkle_fast, csg_fast


def measure_volume_fluctuations_sprinkled(dim: int, n: int, n_trials: int,
                                           rng: np.random.Generator) -> dict:
    """
    For sprinkled causets: measure how the number of elements N fluctuates
    across trials at fixed sprinkling density and region.

    If Lambda ~ 1/sqrt(N), then delta_N/N ~ 1/sqrt(N), i.e., delta_N ~ sqrt(N).
    For a Poisson process, delta_N = sqrt(<N>) exactly.
    """
    counts = []
    for _ in range(n_trials):
        cs, _ = sprinkle_fast(n, dim=dim, extent_t=1.0, region='diamond', rng=rng)
        counts.append(cs.num_relations())

    counts = np.array(counts, dtype=float)
    mean_R = np.mean(counts)
    std_R = np.std(counts)

    return {
        'mean_N': n,
        'mean_R': mean_R,
        'std_R': std_R,
        'ratio': std_R / np.sqrt(mean_R) if mean_R > 0 else float('nan'),
        'note': 'For Poisson: std_R/sqrt(mean_R) should be ~constant'
    }


def measure_interval_fluctuations(causet: FastCausalSet, n_samples: int = 500,
                                   rng: np.random.Generator = None) -> dict:
    """
    For a given causet: measure the variance of interval sizes for pairs
    at similar chain distances.

    The Sorkin argument says that the variance of the number of elements
    in a spacetime region of volume V scales as sqrt(V) (Poisson).

    For causal sets, this means: for intervals of similar "proper time"
    (longest chain length), the variance of the interval cardinality
    should scale as sqrt(mean cardinality).

    If var(N) ~ N, that's Poisson (consistent with Lambda ~ 1/sqrt(N)).
    If var(N) ~ N^alpha with alpha != 1, that's a deviation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = causet.n
    pairs, sizes = causet.interval_sizes_vectorized()

    if len(sizes) == 0:
        return {'error': 'no related pairs'}

    # Group intervals by their size (binned)
    max_size = int(np.max(sizes))
    if max_size < 5:
        return {'error': 'intervals too small'}

    # Logarithmic bins
    bin_edges = np.logspace(np.log10(max(1, 1)), np.log10(max_size + 1), 15)
    bin_edges = np.unique(np.floor(bin_edges).astype(int))

    results = {'bin_means': [], 'bin_vars': [], 'bin_counts': []}

    for i in range(len(bin_edges) - 1):
        mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i + 1])
        if np.sum(mask) < 20:
            continue
        bin_sizes = sizes[mask].astype(float)
        results['bin_means'].append(np.mean(bin_sizes))
        results['bin_vars'].append(np.var(bin_sizes))
        results['bin_counts'].append(len(bin_sizes))

    return results


def fit_poisson_exponent(means, variances):
    """
    Fit var = A * mean^alpha.
    For Poisson: alpha = 1.
    Return alpha and its quality of fit.
    """
    means = np.array(means)
    variances = np.array(variances)

    # Filter out zeros
    mask = (means > 0) & (variances > 0)
    means = means[mask]
    variances = variances[mask]

    if len(means) < 3:
        return float('nan'), float('nan')

    # Log-log fit
    log_m = np.log(means)
    log_v = np.log(variances)
    coeffs = np.polyfit(log_m, log_v, 1)
    alpha = coeffs[0]

    # R^2
    predicted = np.polyval(coeffs, log_m)
    ss_res = np.sum((log_v - predicted) ** 2)
    ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return alpha, r_squared


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 05: Cosmological Constant from Volume Fluctuations")
    print("=" * 70)

    # Phase 1: Verify Poisson statistics in sprinkled causets
    print("\n--- Phase 1: Sprinkled causets (should be Poisson) ---")
    print("If Lambda ~ 1/sqrt(N), then delta_N ~ sqrt(N)")
    print("i.e., var(N)/mean(N) = 1 (Poisson)")

    for dim in [2, 3, 4]:
        print(f"\n  d={dim}:")
        for n in [100, 500, 1000]:
            cs, _ = sprinkle_fast(n, dim=dim, extent_t=1.0, region='diamond', rng=rng)
            result = measure_interval_fluctuations(cs, rng=rng)
            if 'error' in result:
                print(f"    N={n}: {result['error']}")
                continue
            alpha, r2 = fit_poisson_exponent(result['bin_means'], result['bin_vars'])
            print(f"    N={n}: var~mean^{alpha:.2f} (R²={r2:.3f})"
                  f"  [Poisson would be alpha=1.0]")

    # Phase 2: CSG causets — does Poisson scaling hold?
    print("\n--- Phase 2: CSG causets (test Sorkin prediction) ---")
    print("Does var(interval_size) ~ mean(interval_size)^alpha with alpha=1?")

    for p in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        n = 1000
        cs = csg_fast(n, coupling=p, rng=rng)
        result = measure_interval_fluctuations(cs, rng=rng)
        if 'error' in result:
            print(f"  p={p:.2f}: {result['error']}")
            continue
        alpha, r2 = fit_poisson_exponent(result['bin_means'], result['bin_vars'])
        print(f"  p={p:.2f}: var~mean^{alpha:.2f} (R²={r2:.3f})"
              f"  n_bins={len(result['bin_means'])}")

    # Phase 3: Scaling with causet size
    print("\n--- Phase 3: How does the total ordering fraction fluctuate with N? ---")
    print("Sorkin: delta_f / f ~ 1/sqrt(N)")

    p = 0.08
    sizes = [50, 100, 200, 500, 1000, 2000]
    n_trials = 20

    print(f"\n  CSG coupling p={p}:")
    print(f"  {'N':>6} {'mean_f':>10} {'std_f':>10} {'std_f*sqrt(N)':>14} {'note':>20}")
    print("  " + "-" * 65)

    for n in sizes:
        fracs = []
        for _ in range(n_trials):
            cs = csg_fast(n, coupling=p, rng=rng)
            fracs.append(cs.ordering_fraction())
        fracs = np.array(fracs)
        mean_f = np.mean(fracs)
        std_f = np.std(fracs)
        scaled = std_f * np.sqrt(n)
        note = "constant if Poisson" if n == sizes[0] else ""
        print(f"  {n:>6} {mean_f:>10.6f} {std_f:>10.6f} {scaled:>14.4f} {note:>20}")

    # Same for sprinkled causets
    print(f"\n  Sprinkled 2D:")
    print(f"  {'N':>6} {'mean_f':>10} {'std_f':>10} {'std_f*sqrt(N)':>14}")
    print("  " + "-" * 45)

    for n in [50, 100, 200, 500, 1000]:
        fracs = []
        for _ in range(n_trials):
            cs, _ = sprinkle_fast(n, dim=2, extent_t=1.0, region='diamond', rng=rng)
            fracs.append(cs.ordering_fraction())
        fracs = np.array(fracs)
        mean_f = np.mean(fracs)
        std_f = np.std(fracs)
        scaled = std_f * np.sqrt(n)
        print(f"  {n:>6} {mean_f:>10.6f} {std_f:>10.6f} {scaled:>14.4f}")

    print("\n" + "=" * 70)
    print("KEY QUESTION: If std_f * sqrt(N) is approximately constant as N grows,")
    print("then fluctuations are Poisson-like, supporting Sorkin's Lambda ~ 1/sqrt(N).")
    print("If it grows or shrinks, the fluctuation statistics differ from Poisson.")
    print("=" * 70)


if __name__ == '__main__':
    main()
