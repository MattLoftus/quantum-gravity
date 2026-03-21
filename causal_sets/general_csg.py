"""
General Classical Sequential Growth (CSG) model.

The full Rideout-Sorkin dynamics with arbitrary coupling constants {t_n},
not just the single-parameter transitive percolation family.

The transition probability for adding a new element whose precursor set
has total size (past_size) and m maximal ancestors is:

    alpha = sum_{l=m}^{past_size} C(past_size-m, past_size-l) * t_l
            / sum_{j=0}^{n} C(n,j) * t_j

where n is the current causet size and {t_n} are the coupling constants.

Reference: Rideout & Sorkin, Phys. Rev. D 61, 024002 (2000), Eq. 12
"""

import numpy as np
from math import comb
from causal_sets.fast_core import FastCausalSet


def compute_denominator(n: int, t: list) -> float:
    """Compute the normalization: sum_{j=0}^{n} C(n,j) * t_j"""
    result = 0.0
    for j in range(min(n + 1, len(t))):
        result += comb(n, j) * t[j]
    return result


def compute_numerator(past_size: int, n_maximal: int, t: list) -> float:
    """
    Compute the unnormalized transition probability for a precursor set
    with past_size total ancestors and n_maximal maximal ancestors.

    sum_{l=m}^{past_size} C(past_size - m, past_size - l) * t_l
    """
    m = n_maximal
    result = 0.0
    for l in range(m, min(past_size + 1, len(t))):
        result += comb(past_size - m, past_size - l) * t[l]
    return result


def general_csg(n: int, t: list, rng: np.random.Generator = None) -> FastCausalSet:
    """
    Grow a causal set using general CSG dynamics with coupling constants t.

    At each step, we enumerate all possible "birth positions" for the new
    element — each corresponding to a different precursor set (past of the
    new element). The precursor set must be a "partial stem" (downward-closed
    subset) of the existing causet.

    For computational tractability, we use the MAXIMAL ELEMENT formulation:
    the new element's past is determined by choosing which maximal elements
    of the existing causet it will be born above. By Bell causality, the
    probability depends only on (past_size, n_maximal_in_precursor).

    For large causets, enumerating all partial stems is exponential.
    We use the approximation: independently decide for each maximal element
    whether it's in the precursor set, with probability determined by the
    coupling constants.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure t has enough entries
    t_ext = list(t) + [0.0] * max(0, n - len(t))

    causet = FastCausalSet(n)

    for new in range(1, n):
        # Find maximal elements of the current causet (elements 0..new-1)
        maximal = []
        for e in range(new):
            is_max = True
            for f in range(e + 1, new):
                if causet.order[e, f]:
                    is_max = False
                    break
            if is_max:
                maximal.append(e)

        n_max = len(maximal)

        # For each possible subset of maximal elements, compute the transition
        # probability. For tractability, limit to subsets up to size ~10.
        if n_max <= 12:
            # Exact enumeration
            probs = []
            subsets = []

            for mask in range(2 ** n_max):
                # Which maximal elements are in the precursor set?
                chosen_max = []
                for bit in range(n_max):
                    if mask & (1 << bit):
                        chosen_max.append(maximal[bit])

                m = len(chosen_max)  # number of maximal ancestors

                # Compute past_size: total number of ancestors
                # = union of the pasts of chosen maximal elements + themselves
                past = set(chosen_max)
                for cm in chosen_max:
                    for ancestor in range(cm):
                        if causet.order[ancestor, cm]:
                            past.add(ancestor)

                past_size = len(past)

                # Compute transition probability
                denom = compute_denominator(new, t_ext)
                if denom <= 0:
                    prob = 0.0
                else:
                    num = compute_numerator(past_size, m, t_ext)
                    prob = num / denom

                probs.append(max(0.0, prob))
                subsets.append(past)

            # Normalize (should already sum to 1, but numerical safety)
            total = sum(probs)
            if total <= 0:
                # Fallback: add as isolated element
                continue

            probs = [p / total for p in probs]

            # Sample
            choice = rng.choice(len(probs), p=probs)
            chosen_past = subsets[choice]

        else:
            # Approximate: for each maximal element, compute marginal
            # probability of it being an ancestor. Use the ratio of
            # transition probabilities with/without that element.
            chosen_past = set()

            # Greedy: process maximal elements one at a time
            for m_elem in maximal:
                # Past if we include m_elem
                past_with = set(chosen_past)
                past_with.add(m_elem)
                for ancestor in range(m_elem):
                    if causet.order[ancestor, m_elem]:
                        past_with.add(ancestor)

                past_without = set(chosen_past)

                # Simplified probability: use the ratio of t values
                size_with = len(past_with)
                size_without = len(past_without)
                n_max_with = sum(1 for x in past_with
                                 if not any(causet.order[x, y] for y in past_with if y != x))
                n_max_without = sum(1 for x in past_without
                                    if not any(causet.order[x, y] for y in past_without if y != x)) if past_without else 0

                num_with = compute_numerator(size_with, n_max_with, t_ext)
                num_without = compute_numerator(size_without, max(0, n_max_without), t_ext)

                total = num_with + num_without
                if total > 0:
                    p_include = num_with / total
                else:
                    p_include = 0.5

                if rng.random() < p_include:
                    chosen_past = past_with

            # Also consider: new element born with NO ancestors
            # (This is always an option and may dominate for small t_n)

        # Set causal relations
        for ancestor in chosen_past:
            causet.order[ancestor, new] = True

    return causet


def scan_coupling_space(n: int, n_couplings: int = 5, n_samples: int = 50,
                        n_trials: int = 3,
                        rng: np.random.Generator = None) -> list:
    """
    Scan the space of coupling constants looking for causets with
    4D manifold-like properties.

    Target properties for d=4:
    - Ordering fraction ≈ 0.10 (f_0(4) ≈ 0.05, so r ≈ 0.10)
    - Longest chain ~ N^(1/4)
    - Link density neither too high nor too low

    We parameterize t_k = exp(-alpha * k + beta * k^2 + ...) to explore
    different growth patterns.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []

    # Strategy 1: Power law t_k = t^k (transitive percolation) — baseline
    for t_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        t = [t_val ** k for k in range(n)]
        result = _evaluate_couplings(n, t, n_trials, rng, f"power t={t_val}")
        results.append(result)

    # Strategy 2: Exponential decay t_k = exp(-alpha * k)
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        t = [np.exp(-alpha * k) for k in range(n)]
        result = _evaluate_couplings(n, t, n_trials, rng, f"exp alpha={alpha}")
        results.append(result)

    # Strategy 3: Gaussian t_k = exp(-k^2 / (2*sigma^2))
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        t = [np.exp(-k * k / (2 * sigma * sigma)) for k in range(n)]
        result = _evaluate_couplings(n, t, n_trials, rng, f"gauss sigma={sigma}")
        results.append(result)

    # Strategy 4: Step function t_k = 1 for k <= k_max, 0 otherwise
    for k_max in [1, 2, 3, 5, 10]:
        t = [1.0 if k <= k_max else 0.0 for k in range(n)]
        result = _evaluate_couplings(n, t, n_trials, rng, f"step k_max={k_max}")
        results.append(result)

    # Strategy 5: Polynomial t_k = 1/(k+1)^p
    for p in [0.5, 1.0, 2.0, 3.0]:
        t = [1.0 / (k + 1) ** p for k in range(n)]
        result = _evaluate_couplings(n, t, n_trials, rng, f"poly p={p}")
        results.append(result)

    return results


def _evaluate_couplings(n: int, t: list, n_trials: int,
                        rng: np.random.Generator, label: str) -> dict:
    """Evaluate a set of coupling constants by growing causets and measuring observables."""
    from causal_sets.dimension import myrheim_meyer

    ofs = []
    dims = []
    chains = []
    link_counts = []

    for _ in range(n_trials):
        cs = general_csg(n, t, rng=rng)
        of = cs.ordering_fraction()
        ofs.append(of)

        # Convert to old-style CausalSet for MM dimension
        from causal_sets.core import CausalSet
        cs_old = CausalSet(n)
        cs_old.order = cs.order.astype(np.int8)
        d = myrheim_meyer(cs_old)
        dims.append(d)

        chains.append(cs.longest_chain())
        link_counts.append(int(np.sum(cs.link_matrix())))

    return {
        'label': label,
        'ordering_fraction': np.mean(ofs),
        'mm_dimension': np.mean(dims),
        'mm_std': np.std(dims),
        'longest_chain': np.mean(chains),
        'chain_ratio': np.mean(chains) / n,  # should be ~ N^(1/d-1) for d-dim
        'links_per_element': np.mean(link_counts) / n,
        'n_links': np.mean(link_counts),
    }
