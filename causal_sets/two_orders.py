"""
2-orders: causal sets defined by the intersection of two total orders.

A 2-order on N elements is specified by two permutations U and V of {1,...,N}.
Element i precedes element j iff u_i < u_j AND v_i < v_j.

This restricts the sample space to causets that faithfully embed in 2D Minkowski
spacetime, which is exactly the right sample space for 2D causal set quantum
gravity (Surya 2012, Glaser et al. 2018).

The MCMC move is simple: pick two elements, swap one coordinate.
This is ergodic on the space of 2-orders with fixed N.

Reference: Glaser, O'Connor, Surya, Class. Quant. Grav. 35, 024002 (2018)
"""

import numpy as np
from .fast_core import FastCausalSet


class TwoOrder:
    """
    A 2-order on N elements, defined by permutations U and V.
    Element i precedes element j iff u[i] < u[j] AND v[i] < v[j].
    """

    def __init__(self, N: int, rng: np.random.Generator = None):
        self.N = N
        if rng is None:
            rng = np.random.default_rng()
        self.u = rng.permutation(N)
        self.v = rng.permutation(N)

    @classmethod
    def from_permutations(cls, u: np.ndarray, v: np.ndarray):
        obj = cls.__new__(cls)
        obj.N = len(u)
        obj.u = u.copy()
        obj.v = v.copy()
        return obj

    def copy(self):
        return TwoOrder.from_permutations(self.u, self.v)

    def to_causet(self) -> FastCausalSet:
        """Convert to a FastCausalSet (order matrix representation).
        Vectorized: O(N^2) via broadcasting instead of Python double loop."""
        cs = FastCausalSet(self.N)
        # u[i] < u[j] for all pairs (i,j) via broadcasting
        u_less = self.u[:, None] < self.u[None, :]  # (N, N) bool
        v_less = self.v[:, None] < self.v[None, :]  # (N, N) bool
        cs.order = u_less & v_less  # i precedes j iff both coordinates are less
        return cs

    def count_relations(self) -> int:
        """Count the number of causal relations (i < j pairs)."""
        count = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if (self.u[i] < self.u[j] and self.v[i] < self.v[j]) or \
                   (self.u[j] < self.u[i] and self.v[j] < self.v[i]):
                    count += 1
        return count

    def ordering_fraction(self) -> float:
        return self.count_relations() / (self.N * (self.N - 1) / 2)

    def height(self) -> int:
        """Longest chain length (proxy for proper time extent)."""
        cs = self.to_causet()
        return cs.longest_chain()


def swap_move(two_order: TwoOrder, rng: np.random.Generator) -> TwoOrder:
    """
    The Surya/Glaser MCMC move: pick two elements, swap one coordinate.

    1. Pick i, j uniformly from {0, ..., N-1} with i != j
    2. Pick coordinate c in {u, v} uniformly
    3. Swap c[i] and c[j]

    This move is ergodic on 2-orders with fixed N.
    """
    new = two_order.copy()
    i, j = rng.choice(two_order.N, size=2, replace=False)

    if rng.random() < 0.5:
        new.u[i], new.u[j] = new.u[j], new.u[i]
    else:
        new.v[i], new.v[j] = new.v[j], new.v[i]

    return new


def bd_action_2d_nonlocal(two_order: TwoOrder, epsilon: float = 1.0) -> float:
    """
    Non-local Benincasa-Dowker action for 2D.

    For epsilon = 1 (Glaser et al.):
      S = N - 2*N_0 + 4*N_1 - 2*N_2

    where N_k = number of order intervals containing exactly k elements.
    N_0 = links, N_1 = intervals with 1 interior element, etc.

    For general epsilon, the action is:
      S = 4*epsilon * (N - 2*epsilon * sum_n N_n * f_2(n, epsilon))
    where f_2(n, eps) = (1-eps)^n * [1 - 2*eps*n/(1-eps) + eps^2*n*(n-1)/(2*(1-eps)^2)]

    Reference: Glaser et al., Eq. (2.5)
    """
    cs = two_order.to_causet()
    N = cs.n

    if epsilon == 1.0:
        # Use the simplified form: S = N - 2*N_0 + 4*N_1 - 2*N_2
        from .bd_action import count_intervals_by_size
        counts = count_intervals_by_size(cs, max_size=2)
        N_0 = counts.get(0, 0)  # links
        N_1 = counts.get(1, 0)  # 1-element intervals
        N_2 = counts.get(2, 0)  # 2-element intervals
        return N - 2 * N_0 + 4 * N_1 - 2 * N_2
    else:
        # General epsilon formula
        from .bd_action import count_intervals_by_size
        max_k = min(N, 20)  # truncate the sum
        counts = count_intervals_by_size(cs, max_size=max_k)

        def f2(n, eps):
            if abs(1 - eps) < 1e-10:
                return 0.0
            r = (1 - eps) ** n
            term1 = 1.0
            term2 = -2 * eps * n / (1 - eps)
            term3 = eps ** 2 * n * (n - 1) / (2 * (1 - eps) ** 2)
            return r * (term1 + term2 + term3)

        total = 0.0
        for n in range(max_k + 1):
            if n in counts:
                total += counts[n] * f2(n, epsilon)

        return 4 * epsilon * (N - 2 * epsilon * total)


def bd_action_2d_fast(cs: FastCausalSet) -> float:
    """
    Fast computation of the epsilon=1 non-local BD action.
    S = N - 2*N_0 + 4*N_1 - 2*N_2
    Uses matrix operations.
    """
    N = cs.n
    order_int = cs.order.astype(np.int32)

    # Interval sizes: interval_matrix[i,j] = number of k with i < k < j
    interval_matrix = order_int @ order_int

    # N_0 = links: pairs with order[i,j]=True and interval_size=0
    mask_related = np.triu(cs.order, k=1)
    N_0 = int(np.sum(mask_related & (interval_matrix == 0)))
    N_1 = int(np.sum(mask_related & (interval_matrix == 1)))
    N_2 = int(np.sum(mask_related & (interval_matrix == 2)))

    return N - 2 * N_0 + 4 * N_1 - 2 * N_2


def mcmc_two_order(N: int, beta: float, epsilon: float = 1.0,
                    n_steps: int = 10000, n_thermalize: int = 5000,
                    record_every: int = 10,
                    rng: np.random.Generator = None,
                    verbose: bool = False) -> dict:
    """
    MCMC sampling of 2-orders weighted by exp(-beta * S_BD).

    Uses the Surya/Glaser methodology:
    - Fixed N (2-orders preserve element count)
    - Coordinate swap moves (ergodic on 2-orders)
    - Non-local BD action with parameter epsilon

    Returns dict with samples, actions, acceptance rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    current = TwoOrder(N, rng=rng)

    if epsilon == 1.0:
        current_cs = current.to_causet()
        current_action = bd_action_2d_fast(current_cs)
    else:
        current_action = bd_action_2d_nonlocal(current, epsilon)

    actions = []
    samples = []
    n_accepted = 0

    for step in range(n_steps):
        proposed = swap_move(current, rng)

        if epsilon == 1.0:
            proposed_cs = proposed.to_causet()
            proposed_action = bd_action_2d_fast(proposed_cs)
        else:
            proposed_action = bd_action_2d_nonlocal(proposed, epsilon)

        delta_S = beta * (proposed_action - current_action)

        if delta_S <= 0 or rng.random() < np.exp(-delta_S):
            current = proposed
            current_action = proposed_action
            if epsilon == 1.0:
                current_cs = proposed_cs
            n_accepted += 1

        if step >= n_thermalize and step % record_every == 0:
            actions.append(current_action)
            if epsilon == 1.0:
                samples.append(current_cs)
            else:
                samples.append(current.to_causet())

            if verbose and step % (record_every * 100) == 0:
                print(f"  Step {step}/{n_steps}: S={current_action:.1f}, "
                      f"S/N={current_action/N:.3f}, "
                      f"accept={n_accepted/(step+1):.3f}")

    return {
        'actions': np.array(actions),
        'samples': samples,
        'accept_rate': n_accepted / n_steps,
        'N': N,
        'beta': beta,
        'epsilon': epsilon,
    }
