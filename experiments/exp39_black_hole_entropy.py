"""
Experiment 39: Bekenstein-Hawking Entropy from Causal Set Entanglement

The moonshot: sprinkle a causal set into a 2D black hole spacetime,
compute the SJ entanglement entropy of the exterior, and check if
S = A/(4G) = (perimeter of horizon)/(4l_P).

In 2D, a "black hole" is a causal diamond with a boundary that acts as a horizon.
The simplest model: a causal diamond [p, q] with a smaller diamond [p', q'] inside.
The "horizon" is the boundary of the inner diamond. The "exterior" is the
annular region between the inner and outer diamonds.

For a 2D "black hole" (Schwarzschild in 2D is trivial — just Rindler space),
the entanglement entropy across the Rindler horizon should scale as:
  S ∝ ln(L/ε)
where L is the horizon "area" (just a point in 2D — but the entropy still
has a logarithmic UV divergence that is cut off by the discreteness).

More precisely, for a 1+1D massless scalar field, the entanglement entropy
of an interval of length l in a system of size L is (Calabrese-Cardy):
  S = (c/3) ln(l/ε)
where c=1 for a scalar and ε is the UV cutoff.

On a causal set, ε ~ l_P ~ N^{-1/2} (in 2D). So:
  S = (c/3) ln(l * N^{1/2})

The test: for a fixed "horizon" position, does S scale logarithmically
with the sprinkling density (i.e., with N)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet
from causal_sets.sj_vacuum import sj_wightman_function, entanglement_entropy


def sprinkle_2d_diamond(N, extent_t=1.0, rng=None):
    """Sprinkle N points into a 2D causal diamond centered at origin."""
    if rng is None:
        rng = np.random.default_rng()

    coords_list = []
    while len(coords_list) < N:
        batch = N * 4
        t = rng.uniform(-extent_t, extent_t, batch)
        x = rng.uniform(-extent_t, extent_t, batch)
        mask = np.abs(t) + np.abs(x) <= extent_t
        for i in np.where(mask)[0]:
            coords_list.append((t[i], x[i]))
            if len(coords_list) >= N:
                break

    coords = np.array(coords_list[:N])
    # Sort by time
    coords = coords[np.argsort(coords[:, 0])]

    # Build causal order
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i + 1:, 0] - coords[i, 0]
        dx_sq = (coords[i + 1:, 1] - coords[i, 1]) ** 2
        related = dt * dt >= dx_sq
        cs.order[i, i + 1:] = related

    return cs, coords


def partition_by_horizon(coords, horizon_x=0.0):
    """
    Partition elements into "left of horizon" and "right of horizon"
    using the spatial coordinate.

    In 2D, the "horizon" at x = horizon_x divides space into two halves.
    The entanglement entropy of one half measures the correlations across
    the horizon — analogous to Bekenstein-Hawking entropy.
    """
    N = len(coords)
    interior = [i for i in range(N) if coords[i, 1] <= horizon_x]
    exterior = [i for i in range(N) if coords[i, 1] > horizon_x]
    return interior, exterior


def partition_by_nested_diamond(coords, inner_fraction=0.5):
    """
    Create a "black hole" by defining an inner causal diamond.
    Elements inside the inner diamond are the "interior" (behind the horizon).
    Elements outside are the "exterior."

    The horizon "area" in 2D is the number of boundary points of the inner diamond.
    """
    N = len(coords)
    inner_extent = inner_fraction  # inner diamond has this fraction of the outer extent

    interior = []
    exterior = []
    for i in range(N):
        if np.abs(coords[i, 0]) + np.abs(coords[i, 1]) <= inner_extent:
            interior.append(i)
        else:
            exterior.append(i)

    return interior, exterior


def main():
    rng = np.random.default_rng(42)

    print("=" * 75)
    print("EXPERIMENT 39: Black Hole Entropy from Causal Set Entanglement")
    print("=" * 75)

    # Phase 1: Entanglement entropy across a spatial boundary
    print("\n--- Phase 1: S across a spatial boundary (horizon at x=0) ---")
    print("  CFT prediction: S = (c/3) * ln(N) + const")
    print(f"  {'N':>5} {'S_horizon':>10} {'S/ln(N)':>10} {'N_left':>8} {'N_right':>8}")
    print("-" * 50)

    for N in [20, 30, 50, 70, 100]:
        S_vals = []
        for _ in range(10):
            cs, coords = sprinkle_2d_diamond(N, extent_t=1.0, rng=rng)
            W = sj_wightman_function(cs)

            interior, exterior = partition_by_horizon(coords, horizon_x=0.0)
            if len(interior) < 2 or len(exterior) < 2:
                continue

            S = entanglement_entropy(W, interior)
            S_vals.append(S)

        if S_vals:
            S_mean = np.mean(S_vals)
            print(f"  {N:>5} {S_mean:>10.3f} {S_mean/np.log(N):>10.3f}")

    # Phase 2: Nested diamond ("black hole")
    print("\n--- Phase 2: Nested diamond black hole ---")
    print("  Inner diamond = 'black hole interior'")
    print("  Entropy should scale with 'horizon area'")
    print("  In 2D, horizon is 2 points → S should be O(1) + log corrections")

    N = 60
    print(f"\n  Fixed N={N}, varying inner diamond fraction:")
    print(f"  {'inner_frac':>12} {'N_in':>6} {'N_out':>6} {'S':>8}")
    print("-" * 40)

    for inner_frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        S_vals = []
        n_in_vals, n_out_vals = [], []
        for _ in range(15):
            cs, coords = sprinkle_2d_diamond(N, extent_t=1.0, rng=rng)
            W = sj_wightman_function(cs)

            interior, exterior = partition_by_nested_diamond(coords, inner_frac)
            if len(interior) < 2 or len(exterior) < 2:
                continue

            S = entanglement_entropy(W, interior)
            S_vals.append(S)
            n_in_vals.append(len(interior))
            n_out_vals.append(len(exterior))

        if S_vals:
            print(f"  {inner_frac:>12.1f} {np.mean(n_in_vals):>6.0f} "
                  f"{np.mean(n_out_vals):>6.0f} {np.mean(S_vals):>8.3f}")

    # Phase 3: Scaling with N at fixed horizon fraction
    print("\n--- Phase 3: S vs N at fixed inner fraction 0.5 ---")
    print("  BH entropy: S = A/(4G) = const (independent of N)")
    print("  CFT entropy: S = (c/3) ln(N) + const")
    print("  Volume law: S ∝ N")
    print(f"\n  {'N':>5} {'S':>8} {'S/ln(N)':>10} {'S/N':>8}")
    print("-" * 35)

    for N in [20, 30, 40, 60, 80, 100]:
        S_vals = []
        for _ in range(10):
            cs, coords = sprinkle_2d_diamond(N, extent_t=1.0, rng=rng)
            W = sj_wightman_function(cs)
            interior, exterior = partition_by_nested_diamond(coords, 0.5)
            if len(interior) < 2 or len(exterior) < 2:
                continue
            S = entanglement_entropy(W, interior)
            S_vals.append(S)

        if S_vals:
            S_mean = np.mean(S_vals)
            print(f"  {N:>5} {S_mean:>8.3f} {S_mean/np.log(N):>10.3f} "
                  f"{S_mean/N:>8.4f}")

    # Phase 4: Does the entropy depend on the horizon "area" or the bulk volume?
    print("\n--- Phase 4: Area law test ---")
    print("  Fix N=80, vary horizon fraction")
    print("  Area law: S depends on horizon (2 points in 2D) → S ≈ const")
    print("  Volume law: S depends on min(N_in, N_out) → S varies")

    N = 80
    print(f"\n  {'inner_frac':>12} {'S':>8} {'min(Nin,Nout)':>14} {'S/min':>8}")
    print("-" * 48)

    for inner_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        S_vals, min_vals = [], []
        for _ in range(10):
            cs, coords = sprinkle_2d_diamond(N, extent_t=1.0, rng=rng)
            W = sj_wightman_function(cs)
            interior, exterior = partition_by_nested_diamond(coords, inner_frac)
            if len(interior) < 2 or len(exterior) < 2:
                continue
            S = entanglement_entropy(W, interior)
            S_vals.append(S)
            min_vals.append(min(len(interior), len(exterior)))

        if S_vals:
            S_m = np.mean(S_vals)
            min_m = np.mean(min_vals)
            print(f"  {inner_frac:>12.1f} {S_m:>8.3f} {min_m:>14.0f} "
                  f"{S_m/min_m:>8.4f}")

    print("\n  If S/min(Nin,Nout) = const: volume law")
    print("  If S = const (independent of fraction): area law")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
