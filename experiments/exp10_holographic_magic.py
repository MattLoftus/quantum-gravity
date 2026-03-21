"""
Experiment 10: Holographic codes with tunable magic.

Following Preskill et al. (March 2026): magic (non-stabilizer resources)
in the encoding map controls whether the bulk geometry is dynamical.

Key prediction: "proto-area entropy" (boundary entropy minus recoverable
bulk entropy) should increase monotonically with bulk entropy when magic
is present, mimicking quantum extremal surface (QES) behavior.

We test:
1. How entanglement entropy of boundary regions depends on magic content
2. Whether the Ryu-Takayanagi-like area law emerges
3. Whether magic controls the "backreaction" of bulk matter on geometry
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from holographic.tensor_network import (
    random_state_with_magic, compute_entanglement_entropy
)


def toy_holographic_model(n_boundary: int, n_bulk: int, magic: float,
                           bulk_entropy: float = 0.0,
                           rng: np.random.Generator = None) -> dict:
    """
    Simplified holographic model:
    - n_boundary qubits (the "boundary CFT")
    - n_bulk qubits (the "bulk")
    - The encoding map has tunable magic
    - bulk_entropy controls how much matter is in the bulk

    Returns entanglement properties of boundary subregions.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = n_boundary + n_bulk

    # Step 1: Create a bulk state with specified entropy
    # (maximally mixed on a subset of bulk qubits to simulate matter)
    n_excited = max(0, min(n_bulk, int(bulk_entropy)))

    # Step 2: Create the encoding map (bulk -> boundary) with tunable magic
    # For simplicity, create a random state on all qubits with controlled magic
    # The first n_bulk qubits are "bulk", the rest are "boundary"
    state = random_state_with_magic(n_total, magic_fraction=magic, rng=rng)

    # If we want bulk entropy, apply additional mixing to bulk qubits
    if n_excited > 0 and magic > 0:
        # Apply random rotations to excited bulk qubits
        dim = 2 ** n_total
        state_tensor = state.reshape([2] * n_total)
        for q in range(n_excited):
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            U = np.array([[np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2)],
                           [np.exp(-1j*phi)*np.sin(theta/2), np.cos(theta/2)]])
            new_tensor = np.zeros_like(state_tensor)
            for idx in np.ndindex(*([2] * n_total)):
                for nv in range(2):
                    idx_new = list(idx)
                    idx_new[q] = nv
                    new_tensor[tuple(idx_new)] += U[nv, idx[q]] * state_tensor[tuple(idx)]
            state_tensor = new_tensor
        state = state_tensor.reshape(dim)
        state /= np.linalg.norm(state)

    # Step 3: Measure entanglement properties of boundary subregions
    dims = [2] * n_total
    boundary_start = n_bulk

    results = {'magic': magic, 'bulk_entropy_param': bulk_entropy}

    # Entanglement entropy of half the boundary
    half = n_boundary // 2
    region_A = list(range(boundary_start, boundary_start + half))
    complement_A = [i for i in range(n_total) if i not in region_A]
    S_half = compute_entanglement_entropy(state, dims, complement_A)
    results['S_half_boundary'] = S_half

    # Entanglement entropy of boundary subregions of different sizes
    entropies = []
    for size in range(1, n_boundary + 1):
        region = list(range(boundary_start, boundary_start + size))
        complement = [i for i in range(n_total) if i not in region]
        S = compute_entanglement_entropy(state, dims, complement)
        entropies.append(S)
    results['boundary_entropies'] = entropies

    # "Proto-area entropy" = S(boundary) - S(bulk|boundary)
    # This is the entropy attributable to the "area" of the entangling surface
    # For a good holographic code, this should scale with the bulk geometry
    bulk_region = list(range(n_bulk))
    complement_bulk = list(range(n_bulk, n_total))
    S_bulk = compute_entanglement_entropy(state, dims, complement_bulk)
    results['S_bulk'] = S_bulk

    # Total entropy
    S_total = compute_entanglement_entropy(state, dims, [])
    results['S_total'] = S_total

    # Proto-area: S(half_boundary) - conditional entropy of bulk given boundary
    # Simplified: proto_area ≈ S(half_boundary) - (S_bulk - mutual_info)
    results['proto_area'] = S_half - S_bulk / 2  # rough proxy

    return results


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("EXPERIMENT 10: Holographic Codes with Tunable Magic")
    print("=" * 80)

    n_boundary = 6
    n_bulk = 4

    # Phase 1: Entanglement vs magic at fixed bulk entropy
    print(f"\n--- Phase 1: Boundary entanglement vs magic (n_bulk={n_bulk}, n_boundary={n_boundary}) ---")
    print(f"{'magic':>8} {'S_half':>10} {'S_bulk':>10} {'proto_area':>12} {'max_S_bdry':>12}")
    print("-" * 55)

    n_trials = 20
    for magic in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        s_halfs = []
        s_bulks = []
        proto_areas = []
        max_s = []

        for _ in range(n_trials):
            r = toy_holographic_model(n_boundary, n_bulk, magic=magic,
                                       bulk_entropy=0, rng=rng)
            s_halfs.append(r['S_half_boundary'])
            s_bulks.append(r['S_bulk'])
            proto_areas.append(r['proto_area'])
            max_s.append(max(r['boundary_entropies']))

        print(f"  {magic:>6.1f} {np.mean(s_halfs):>10.3f} {np.mean(s_bulks):>10.3f} "
              f"{np.mean(proto_areas):>12.3f} {np.mean(max_s):>12.3f}")

    # Phase 2: Proto-area entropy vs bulk entropy at different magic levels
    print(f"\n--- Phase 2: Proto-area vs bulk entropy (does magic enable backreaction?) ---")
    print("Preskill prediction: proto-area should increase with bulk entropy ONLY when magic > 0")

    print(f"\n{'magic':>8} {'bulk_S':>8} {'S_half':>10} {'proto_area':>12}")
    print("-" * 42)

    for magic in [0.0, 0.5, 1.0]:
        for bulk_s in [0, 1, 2, 3]:
            results_list = []
            for _ in range(n_trials):
                r = toy_holographic_model(n_boundary, n_bulk, magic=magic,
                                           bulk_entropy=bulk_s, rng=rng)
                results_list.append(r)

            mean_half = np.mean([r['S_half_boundary'] for r in results_list])
            mean_proto = np.mean([r['proto_area'] for r in results_list])
            print(f"  {magic:>6.1f} {bulk_s:>8} {mean_half:>10.3f} {mean_proto:>12.3f}")
        print()

    # Phase 3: Entanglement entropy profile vs subregion size
    print("--- Phase 3: S(A) vs |A| for boundary subregions ---")
    print("RT formula predicts: S(A) = min(|A|, |A_complement|) * log(d) for holographic codes")
    print("Page curve: S(A) increases then decreases (Page transition)")

    for magic in [0.0, 0.5, 1.0]:
        print(f"\n  magic = {magic}:")
        all_profiles = []
        for _ in range(n_trials):
            r = toy_holographic_model(n_boundary, n_bulk, magic=magic,
                                       bulk_entropy=0, rng=rng)
            all_profiles.append(r['boundary_entropies'])

        mean_profile = np.mean(all_profiles, axis=0)
        print(f"  {'|A|':>6} {'S(A)':>8} {'S_max_page':>12}")
        for size in range(n_boundary):
            # Page prediction for random states
            s_page = min(size + 1, n_boundary - size - 1) if size + 1 <= n_boundary else 0
            print(f"  {size+1:>6} {mean_profile[size]:>8.3f} {s_page:>12}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  If proto-area increases with bulk entropy only when magic > 0:")
    print("  → Magic controls geometric backreaction (Preskill's prediction)")
    print("  If S(A) follows the Page curve shape:")
    print("  → The code has holographic-like entanglement structure")
    print("=" * 80)


if __name__ == '__main__':
    main()
