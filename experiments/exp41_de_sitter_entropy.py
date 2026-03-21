"""
Experiment 41: SJ Entanglement Entropy of the de Sitter Static Patch

In 2D de Sitter spacetime, an observer has a cosmological horizon.
The Gibbons-Hawking entropy of the horizon is S_GH = A/(4G).
In 2D, the "area" of the horizon is trivial (a point), but the
entanglement entropy should still show characteristic scaling.

For a massless scalar in 2D de Sitter:
- The static patch is the region causally accessible to an observer
- The entanglement entropy between the static patch and the complement
  measures the correlations across the horizon
- In the continuum, this is UV divergent; on a causal set, the
  discreteness provides a natural cutoff

We sprinkle into 2D de Sitter (conformal to a causal diamond with
specific volume weighting) and compute the SJ entanglement entropy
of the static patch.

Key question: how does the horizon entropy scale with the
de Sitter radius (equivalently, with the cosmological constant)?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from causal_sets.fast_core import FastCausalSet


def sprinkle_de_sitter_2d(N, H=1.0, eta_range=(-2.0, -0.1), rng=None):
    """
    Sprinkle into 2D de Sitter in conformal coordinates.

    Metric: ds^2 = (1/(H^2 eta^2))(-d_eta^2 + dx^2)
    Conformal time eta < 0, approaching 0 at future infinity.

    The causal structure is conformally Minkowski, so we sprinkle
    uniformly in conformal coordinates and use flat causality.

    But the PHYSICAL volume element is dV = sqrt(-g) d^2x = (1/(H^2 eta^2)) deta dx
    So we weight by 1/eta^2 to get uniform physical sprinkling.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_range = 2 * np.pi / H
    coords_list = []

    while len(coords_list) < N:
        batch = N * 8
        eta = rng.uniform(eta_range[0], eta_range[1], batch)
        x = rng.uniform(0, x_range, batch)

        # Weight by physical volume: accept with probability proportional to 1/eta^2
        # Normalize: max weight at eta_range[1] (closest to 0)
        weight = (eta_range[1] / eta) ** 2
        accept = rng.random(batch) < weight

        for i in np.where(accept)[0]:
            coords_list.append((eta[i], x[i]))
            if len(coords_list) >= N:
                break

    coords = np.array(coords_list[:N])
    coords = coords[np.argsort(coords[:, 0])]  # sort by conformal time

    # Build causal order (conformal = Minkowski causality)
    cs = FastCausalSet(N)
    for i in range(N):
        d_eta = coords[i + 1:, 0] - coords[i, 0]  # positive (sorted)
        dx = np.abs(coords[i + 1:, 1] - coords[i, 1])
        # Periodic spatial coordinate
        dx = np.minimum(dx, x_range - dx)
        cs.order[i, i + 1:] = d_eta >= dx

    return cs, coords


def static_patch_partition(coords, observer_x=None, H=1.0):
    """
    Partition elements into the static patch of an observer at position observer_x.

    In 2D de Sitter (conformal coords), the static patch of an observer
    at x = observer_x is the set of events that can both send and receive
    signals from the observer's worldline.

    For a comoving observer at x = observer_x, the static patch at
    conformal time eta is: |x - observer_x| < |eta| (the Hubble radius
    in conformal coordinates shrinks toward the future).

    Simplified: partition by spatial distance from observer.
    Elements within the horizon radius at their conformal time are "inside."
    """
    N = len(coords)
    if observer_x is None:
        observer_x = np.mean(coords[:, 1])

    x_range = 2 * np.pi / H

    inside = []
    outside = []
    for i in range(N):
        eta_i = coords[i, 0]
        dx = abs(coords[i, 1] - observer_x)
        dx = min(dx, x_range - dx)  # periodic

        # Static patch condition: dx < |eta| (conformal Hubble radius)
        if dx < abs(eta_i):
            inside.append(i)
        else:
            outside.append(i)

    return inside, outside


def sj_wightman(cs):
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N))
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W


def entanglement_entropy(W, A):
    if len(A) < 2:
        return 0.0
    W_A = W[np.ix_(A, A)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 41: De Sitter Static Patch Entanglement Entropy")
    print("=" * 70)

    # Phase 1: Flat Minkowski comparison
    print("\n--- Phase 1: Flat 2D Minkowski baseline ---")
    from causal_sets.fast_core import sprinkle_fast

    print(f"  {'N':>5} {'S_flat':>8} {'S/ln(N)':>8}")
    print("-" * 25)
    for N in [20, 30, 50, 70]:
        S_vals = []
        for _ in range(10):
            cs, coords = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
            W = sj_wightman(cs)
            A = list(range(N // 2))
            S_vals.append(entanglement_entropy(W, A))
        print(f"  {N:>5} {np.mean(S_vals):>8.3f} {np.mean(S_vals)/np.log(N):>8.3f}")

    # Phase 2: de Sitter with varying N
    print("\n--- Phase 2: de Sitter static patch ---")
    print(f"  {'N':>5} {'N_in':>6} {'N_out':>6} {'S_dS':>8} {'S/ln(N)':>8}")
    print("-" * 40)

    for N in [20, 30, 50, 70, 100]:
        S_vals = []
        n_in_vals = []
        for _ in range(10):
            cs, coords = sprinkle_de_sitter_2d(N, H=1.0, eta_range=(-3.0, -0.1), rng=rng)
            W = sj_wightman(cs)
            inside, outside = static_patch_partition(coords, H=1.0)

            if len(inside) < 2 or len(outside) < 2:
                continue

            S = entanglement_entropy(W, inside)
            S_vals.append(S)
            n_in_vals.append(len(inside))

        if S_vals:
            print(f"  {N:>5} {np.mean(n_in_vals):>6.0f} {N-np.mean(n_in_vals):>6.0f} "
                  f"{np.mean(S_vals):>8.3f} {np.mean(S_vals)/np.log(N):>8.3f}")

    # Phase 3: Vary the Hubble parameter (cosmological constant)
    print("\n--- Phase 3: Entropy vs Hubble parameter (fixed N=50) ---")
    print("  Gibbons-Hawking: S_GH = pi/(G*H) in 2D")
    print("  If S decreases with H: consistent with S ~ 1/H (horizon area shrinks)")
    print(f"  {'H':>6} {'N_in':>6} {'S':>8}")
    print("-" * 25)

    N = 50
    for H in [0.3, 0.5, 1.0, 2.0, 3.0]:
        S_vals = []
        n_in_vals = []
        for _ in range(15):
            cs, coords = sprinkle_de_sitter_2d(N, H=H, eta_range=(-3.0, -0.1), rng=rng)
            W = sj_wightman(cs)
            inside, outside = static_patch_partition(coords, H=H)

            if len(inside) < 2 or len(outside) < 2:
                continue

            S = entanglement_entropy(W, inside)
            S_vals.append(S)
            n_in_vals.append(len(inside))

        if S_vals:
            print(f"  {H:>6.1f} {np.mean(n_in_vals):>6.0f} {np.mean(S_vals):>8.3f}")

    # Phase 4: Compare flat vs de Sitter at same N
    print("\n--- Phase 4: Flat vs de Sitter at N=50 ---")
    N = 50

    # Flat
    S_flat = []
    for _ in range(15):
        cs, _ = sprinkle_fast(N, dim=2, extent_t=1.0, region='diamond', rng=rng)
        W = sj_wightman(cs)
        S_flat.append(entanglement_entropy(W, list(range(N // 2))))

    # de Sitter
    S_dS = []
    for _ in range(15):
        cs, coords = sprinkle_de_sitter_2d(N, H=1.0, rng=rng)
        W = sj_wightman(cs)
        inside, _ = static_patch_partition(coords, H=1.0)
        if len(inside) >= 2:
            S_dS.append(entanglement_entropy(W, inside))

    print(f"  Flat Minkowski: S = {np.mean(S_flat):.3f} ± {np.std(S_flat):.3f}")
    print(f"  de Sitter (H=1): S = {np.mean(S_dS):.3f} ± {np.std(S_dS):.3f}")
    if np.mean(S_dS) < np.mean(S_flat):
        print("  --> de Sitter has LESS entanglement (horizon limits correlations)")
    else:
        print("  --> de Sitter has MORE entanglement")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
