"""
MCMC sampling of causal sets weighted by the Benincasa-Dowker action.

Samples causets from the distribution P(C) ~ exp(-beta * S_BD(C))
using Metropolis-Hastings with local moves:
  1. Add an element (with random relations consistent with partial order)
  2. Remove a random element
  3. Flip a relation (add or remove a link, maintaining transitivity and acyclicity)

The coupling beta controls the temperature:
  - Large beta: low temperature, strongly favors low-action (manifold-like) causets
  - Small beta: high temperature, samples broadly
  - beta = 0: uniform sampling over causets

References:
  - Surya, Class. Quant. Grav. 29, 132001 (2012) — review of causet dynamics
  - Glaser & Surya, Phys. Rev. D 94, 064014 (2016) — 2D causet MCMC
"""

import numpy as np
from .fast_core import FastCausalSet
from .bd_action import bd_action_2d, count_intervals_by_size


def _ensure_transitivity(order: np.ndarray):
    """Enforce transitive closure in-place using Warshall's algorithm.
    Optimized: only processes upper triangle since order is a DAG."""
    n = order.shape[0]
    for k in range(n):
        for i in range(k):
            if order[i, k]:
                order[i, k+1:] |= order[k, k+1:]


def _check_acyclic(order: np.ndarray) -> bool:
    """Check that the order matrix is acyclic (upper triangular after relabeling)."""
    # If any diagonal element is True, there's a cycle
    if np.any(np.diag(order)):
        return False
    # Check: no i < j with both order[i,j] and order[j,i]
    sym = order & order.T
    return not np.any(sym)


def propose_add_element(causet: FastCausalSet, rng: np.random.Generator,
                        target_n: int = None) -> FastCausalSet:
    """
    Propose adding a new element to the causet.

    The new element is placed at a random position in the total order.
    It gets random causal relations with existing elements, then
    transitive closure is enforced.
    """
    n = causet.n
    new_n = n + 1

    # Create new order matrix
    new_order = np.zeros((new_n, new_n), dtype=np.bool_)
    # Insert the new element at position `pos`
    pos = rng.integers(0, new_n)

    # Copy old relations, shifting indices around the insertion point
    old_idx = list(range(pos)) + list(range(pos + 1, new_n))
    for i_new, i_old_idx in enumerate(old_idx):
        for j_new, j_old_idx in enumerate(old_idx):
            if i_new < n and j_new < n:
                new_order[i_old_idx, j_old_idx] = causet.order[i_new, j_new]

    # Add random relations for the new element
    # Elements before `pos` might precede it; elements after might follow it
    link_prob = 0.3  # probability of direct relation
    for i in range(pos):
        if rng.random() < link_prob:
            new_order[i, pos] = True
    for j in range(pos + 1, new_n):
        if rng.random() < link_prob:
            new_order[pos, j] = True

    # Enforce transitivity
    _ensure_transitivity(new_order)

    result = FastCausalSet(new_n)
    result.order = new_order
    return result


def propose_remove_element(causet: FastCausalSet, rng: np.random.Generator) -> FastCausalSet:
    """
    Propose removing a random element from the causet.
    Relations among remaining elements are preserved.
    """
    if causet.n <= 3:
        return causet  # Don't go below 3 elements

    n = causet.n
    remove_idx = rng.integers(0, n)

    # Build new order matrix without the removed element
    keep = np.ones(n, dtype=bool)
    keep[remove_idx] = False
    new_order = causet.order[np.ix_(keep, keep)]

    result = FastCausalSet(n - 1)
    result.order = new_order.astype(np.bool_)
    return result


def propose_flip_relation(causet: FastCausalSet, rng: np.random.Generator) -> FastCausalSet:
    """
    Propose flipping a link: either add a new link or remove an existing link.

    Adding a link: pick two unrelated elements and add a relation + transitive closure.
    Removing a link: pick an existing link and remove it (if it doesn't break transitivity
    in a way we don't want, we just remove the link and recompute transitive closure from links).
    """
    n = causet.n
    result = FastCausalSet(n)
    result.order = causet.order.copy()

    if rng.random() < 0.5:
        # Try to add a relation
        # Find pairs that are NOT related
        not_related = ~causet.order & ~causet.order.T & ~np.eye(n, dtype=bool)
        # Only consider upper triangle (i < j by label)
        not_related = np.triu(not_related, k=1)
        candidates = np.argwhere(not_related)

        if len(candidates) == 0:
            return result  # no-op

        idx = rng.integers(0, len(candidates))
        i, j = candidates[idx]
        result.order[i, j] = True
        # Enforce transitivity
        _ensure_transitivity(result.order)
    else:
        # Try to remove a link
        links = causet.link_matrix()
        link_pairs = np.argwhere(links)

        if len(link_pairs) == 0:
            return result

        idx = rng.integers(0, len(link_pairs))
        i, j = link_pairs[idx]

        # Remove this link: rebuild order from remaining links
        links[i, j] = False

        # Reconstruct order from links via transitive closure
        new_order = links.copy()
        _ensure_transitivity(new_order)
        result.order = new_order

    return result


def mcmc_bd_action(initial_causet: FastCausalSet, beta: float,
                   n_steps: int = 1000, target_n: int = None,
                   n_size_penalty: float = 0.0,
                   rng: np.random.Generator = None,
                   record_every: int = 10,
                   verbose: bool = False) -> dict:
    """
    Run MCMC sampling of causets weighted by exp(-beta * S_BD).

    Parameters:
      initial_causet: starting causet (e.g., from sprinkling)
      beta: inverse temperature (larger = stronger preference for low action)
      n_steps: number of MCMC steps
      target_n: preferred causet size (if set, adds a penalty |N - target_n|)
      n_size_penalty: strength of the size penalty
      rng: random number generator
      record_every: record observables every this many steps
      verbose: print progress

    Returns dict with:
      'actions': action values at recorded steps
      'sizes': causet sizes at recorded steps
      'accept_rate': overall acceptance rate
      'samples': list of (causet, action) at recorded steps
      'mm_dims': Myrheim-Meyer dimensions
    """
    if rng is None:
        rng = np.random.default_rng()

    current = initial_causet
    current_action = bd_action_2d(current)

    actions = []
    sizes = []
    samples = []
    n_accepted = 0

    move_weights = [0.2, 0.2, 0.6]  # add, remove, flip
    move_names = ['add', 'remove', 'flip']

    for step in range(n_steps):
        # Choose move type
        move_type = rng.choice(3, p=move_weights)

        if move_type == 0:
            proposed = propose_add_element(current, rng)
        elif move_type == 1:
            proposed = propose_remove_element(current, rng)
        else:
            proposed = propose_flip_relation(current, rng)

        # Compute action of proposed causet
        proposed_action = bd_action_2d(proposed)

        # Energy = beta * S_BD + size penalty
        current_energy = beta * current_action
        proposed_energy = beta * proposed_action

        if target_n is not None and n_size_penalty > 0:
            current_energy += n_size_penalty * abs(current.n - target_n)
            proposed_energy += n_size_penalty * abs(proposed.n - target_n)

        # Metropolis acceptance
        delta_E = proposed_energy - current_energy
        if delta_E <= 0 or rng.random() < np.exp(-delta_E):
            current = proposed
            current_action = proposed_action
            n_accepted += 1

        # Record
        if step % record_every == 0:
            actions.append(current_action)
            sizes.append(current.n)
            samples.append(current)

            if verbose and step % (record_every * 10) == 0:
                print(f"  Step {step}/{n_steps}: N={current.n}, S={current_action:.2f}, "
                      f"accept={n_accepted/(step+1):.3f}")

    return {
        'actions': np.array(actions),
        'sizes': np.array(sizes),
        'accept_rate': n_accepted / n_steps,
        'samples': samples,
    }
