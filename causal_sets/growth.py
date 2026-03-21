"""
Dynamical models for causal set growth.

The Classical Sequential Growth (CSG) model of Rideout and Sorkin grows a causal set
one element at a time, with transition probabilities constrained by discrete general
covariance (Bell causality + label invariance).
"""

import numpy as np
from .core import CausalSet, transitive_closure


def classical_sequential_growth(n: int, coupling: float = 0.0,
                                 rng: np.random.Generator = None) -> CausalSet:
    """
    Grow a causal set of n elements using the Rideout-Sorkin CSG dynamics.

    The CSG model is parameterized by coupling constants t_k (k = 0, 1, 2, ...).
    The simplest family is the "transitive percolation" model where the probability
    of a new element being to the future of an existing element is p (independent
    for each existing element).

    In the general CSG model, when adding element n+1, the probability of it
    being born to the future of a subset A of existing elements depends only
    on |A| (by label invariance / discrete general covariance).

    For the transitive percolation model:
        - Each existing element independently has probability p of being
          in the past of the new element
        - Then take the transitive closure

    The coupling parameter here controls the "percolation probability" p.
    - coupling = 0: new elements are unrelated to all existing (antichain growth)
    - coupling = 1: new element is to the future of everything (total order)
    - coupling between 0 and 1: interpolates

    Reference: Rideout & Sorkin, Phys. Rev. D 61, 024002 (2000)
    """
    if rng is None:
        rng = np.random.default_rng()

    p = coupling  # percolation probability

    causet = CausalSet(n)

    # Grow one element at a time
    for new in range(1, n):
        # For each existing element, independently decide if it precedes the new one
        for old in range(new):
            if rng.random() < p:
                causet.order[old, new] = 1

        # Enforce transitivity: if a < b and b < new, then a < new
        for old in range(new):
            if causet.order[old, new]:
                # Everything in old's past is also in new's past
                for ancestor in range(old):
                    if causet.order[ancestor, old]:
                        causet.order[ancestor, new] = 1

    return causet


def originary_growth(n: int, q: float = 0.5, rng: np.random.Generator = None) -> CausalSet:
    """
    Originary CSG model with parameter q.

    In this model, the transition probabilities are given by:
        P(new element is to the future of exactly the set A) proportional to q^|A|

    where |A| is the size of the "precursor set" — the set of existing elements
    that are maximal among those preceding the new element.

    The parameter q controls the "tendency to order":
    - q -> 0: strongly favors antichains (no causal relations)
    - q -> 1: favors more ordered structures
    - q > 1: strongly favors total orders

    This is the most general single-parameter CSG dynamics consistent with
    Bell causality and discrete general covariance.

    Reference: Varadarajan & Rideout, Phys. Rev. D 73, 104021 (2006)
    """
    if rng is None:
        rng = np.random.default_rng()

    causet = CausalSet(n)

    for new in range(1, n):
        # Identify all maximal elements (elements with no successors among existing)
        existing = list(range(new))
        maximal = []
        for e in existing:
            is_max = True
            for f in existing:
                if f != e and causet.order[e, f]:
                    is_max = False
                    break
            if is_max:
                maximal.append(e)

        # For each subset of maximal elements, compute the probability
        # that the new element is born to the future of exactly that subset
        # For efficiency with large sets, use the independent approximation:
        # each maximal element independently has probability q/(1+q) of preceding new
        p_link = q / (1.0 + q) if q < float('inf') else 1.0

        # Determine which maximal elements precede the new element
        for m in maximal:
            if rng.random() < p_link:
                causet.order[m, new] = 1
                # Also add transitive relations
                for ancestor in range(m):
                    if causet.order[ancestor, m]:
                        causet.order[ancestor, new] = 1

    return causet


def grow_sweep(n: int, p_values: np.ndarray,
               seeds: int = 10, rng: np.random.Generator = None) -> dict:
    """
    Run CSG growth for a range of coupling values and measure observables.

    Returns dict with arrays of results for each coupling value.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {
        'p': p_values,
        'ordering_fraction_mean': np.zeros(len(p_values)),
        'ordering_fraction_std': np.zeros(len(p_values)),
        'longest_chain_mean': np.zeros(len(p_values)),
        'longest_chain_std': np.zeros(len(p_values)),
        'num_links_mean': np.zeros(len(p_values)),
        'num_links_std': np.zeros(len(p_values)),
    }

    for idx, p in enumerate(p_values):
        of_samples = []
        lc_samples = []
        nl_samples = []

        for seed in range(seeds):
            cs = classical_sequential_growth(n, coupling=p, rng=rng)
            of_samples.append(cs.ordering_fraction())
            lc_samples.append(cs.longest_chain())
            nl_samples.append(np.sum(cs.link_matrix()))

        results['ordering_fraction_mean'][idx] = np.mean(of_samples)
        results['ordering_fraction_std'][idx] = np.std(of_samples)
        results['longest_chain_mean'][idx] = np.mean(lc_samples)
        results['longest_chain_std'][idx] = np.std(lc_samples)
        results['num_links_mean'][idx] = np.mean(nl_samples)
        results['num_links_std'][idx] = np.std(nl_samples)

    return results
