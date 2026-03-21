"""
Benincasa-Dowker action for causal sets.

The BD action is a causal set discretization of the Einstein-Hilbert action.
For a causal set C with N elements:

  S_BD = (1/l^2) * sum_{x in C} [ alpha_0 + alpha_1*n_1(x) + alpha_2*n_2(x) + ... ]

where n_k(x) counts the number of k-element inclusive order intervals above x.

In 2D the action simplifies to:
  S_BD = N - 2*L + I_2
where N = number of elements, L = number of links, I_2 = number of order-2 intervals
(pairs x<y with exactly 1 element between them, i.e. interval size = 1).

In 4D:
  S_BD = (1/l^2) * [ alpha_0*N + alpha_1*L + alpha_2*I_2 + alpha_3*I_3 ]
  alpha_0 = 1/24, alpha_1 = -4/24, alpha_2 = 6/24, alpha_3 = -4/24

References:
  - Benincasa & Dowker, Phys. Rev. Lett. 104, 181301 (2010)
  - Dowker & Glaser, Class. Quant. Grav. 30, 195016 (2013)
"""

import numpy as np
from .fast_core import FastCausalSet


def count_links(causet: FastCausalSet) -> int:
    """Count the number of links (irreducible relations) in the causet."""
    links = causet.link_matrix()
    return int(np.sum(links))


def count_intervals_by_size(causet: FastCausalSet, max_size: int = 3):
    """
    Count intervals by their interior size.

    Returns dict: {k: count} where k is the number of interior elements.
    An interval [x,y] has interior size k if there are exactly k elements z
    with x < z < y.

    - k=0 corresponds to links
    - k=1 corresponds to order-2 intervals (I_2)
    - k=2 corresponds to order-3 intervals (I_3)
    """
    order_int = causet.order.astype(np.int32)
    # interval_matrix[i,j] = number of elements between i and j
    interval_matrix = order_int @ order_int

    counts = {}
    for k in range(max_size + 1):
        # Count pairs (i,j) with i<j, order[i,j]=True, and exactly k elements between
        mask = causet.order & (interval_matrix == k)
        # Only count upper triangle (i < j)
        counts[k] = int(np.sum(np.triu(mask, k=1)))

    return counts


def bd_action_2d(causet: FastCausalSet) -> float:
    """
    Compute the 2D Benincasa-Dowker action.

    S = N - 2*L + I_2

    where:
      N = number of elements
      L = number of links (irreducible relations)
      I_2 = number of order-2 intervals (pairs with exactly 1 element between them)

    For a flat 2D causet sprinkled into Minkowski, S_BD ~ 0 (flat space has zero action).
    For non-manifold-like causets, |S_BD| will be large.
    """
    N = causet.n
    counts = count_intervals_by_size(causet, max_size=1)
    L = counts[0]  # links = intervals with 0 interior elements
    I_2 = counts[1]  # order-2 intervals = 1 interior element

    return N - 2 * L + I_2


def bd_action_4d(causet: FastCausalSet, l_sq: float = 1.0) -> float:
    """
    Compute the 4D Benincasa-Dowker action.

    S = (1/l^2) * [ (1/24)*N - (4/24)*L + (6/24)*I_2 - (4/24)*I_3 ]

    where I_k = number of intervals with k-1 interior elements.
    """
    N = causet.n
    counts = count_intervals_by_size(causet, max_size=2)
    L = counts[0]
    I_2 = counts[1]
    I_3 = counts[2]

    return (1.0 / l_sq) * (N / 24.0 - 4 * L / 24.0 + 6 * I_2 / 24.0 - 4 * I_3 / 24.0)


def bd_action_2d_from_counts(N, L, I_2):
    """Compute 2D BD action from precomputed counts (for MCMC efficiency)."""
    return N - 2 * L + I_2


def delta_bd_action_2d(causet: FastCausalSet, old_action: float,
                       old_N: int, old_L: int, old_I2: int,
                       move_type: str, move_data: dict) -> tuple:
    """
    Compute the change in BD action incrementally after a move.

    Returns (new_action, new_N, new_L, new_I2).

    For efficiency in MCMC, we recompute from scratch (incremental updates
    are complex and error-prone for initial implementation).
    """
    counts = count_intervals_by_size(causet, max_size=1)
    new_N = causet.n
    new_L = counts[0]
    new_I2 = counts[1]
    new_action = bd_action_2d_from_counts(new_N, new_L, new_I2)
    return new_action, new_N, new_L, new_I2
