"""
d-orders: causal sets defined by the intersection of d total orders.

A d-order on N elements is specified by d permutations of {1,...,N}.
Element i precedes element j iff perm_k[i] < perm_k[j] for ALL k in 1..d.

- d=2: embeds faithfully in 2D Minkowski (reproduces TwoOrder behavior)
- d=3: embeds in 3D Minkowski
- d=4: embeds in 4D Minkowski (the physically relevant case)

The MCMC move: pick two elements, swap one coordinate (uniformly chosen).
This is ergodic on the space of d-orders with fixed N.

References:
  - Brightwell & Gregory, "Structure of random discrete spacetime" (1991)
  - Glaser, O'Connor, Surya, Class. Quant. Grav. 35, 024002 (2018)
"""

import numpy as np
from .fast_core import FastCausalSet
from .bd_action import count_intervals_by_size


class DOrder:
    """
    A d-order on N elements, defined by d permutations.
    Element i precedes element j iff perms[k][i] < perms[k][j] for all k.
    """

    def __init__(self, d: int, N: int, rng: np.random.Generator = None):
        self.d = d
        self.N = N
        if rng is None:
            rng = np.random.default_rng()
        # d independent random permutations
        self.perms = [rng.permutation(N) for _ in range(d)]

    @classmethod
    def from_permutations(cls, perms: list):
        """Create a DOrder from a list of d permutation arrays."""
        obj = cls.__new__(cls)
        obj.d = len(perms)
        obj.N = len(perms[0])
        obj.perms = [p.copy() for p in perms]
        return obj

    def copy(self):
        return DOrder.from_permutations(self.perms)

    def to_causet(self) -> FastCausalSet:
        """
        Convert to a FastCausalSet (order matrix representation).
        O(N^2 * d) — for d=4, N=30 this is 3600 ops.
        """
        N = self.N
        cs = FastCausalSet(N)

        # Build a (d, N) array for vectorized comparison
        perm_arr = np.array(self.perms)  # shape (d, N)

        for i in range(N):
            for j in range(i + 1, N):
                # i precedes j if all perms have perm[k][i] < perm[k][j]
                all_less = True
                all_greater = True
                for k in range(self.d):
                    if self.perms[k][i] >= self.perms[k][j]:
                        all_less = False
                    if self.perms[k][j] >= self.perms[k][i]:
                        all_greater = False
                    if not all_less and not all_greater:
                        break
                if all_less:
                    cs.order[i, j] = True
                elif all_greater:
                    cs.order[j, i] = True

        return cs

    def to_causet_fast(self) -> FastCausalSet:
        """
        Vectorized conversion to FastCausalSet.
        Uses broadcasting — faster for moderate N.
        """
        N = self.N
        cs = FastCausalSet(N)
        perm_arr = np.array(self.perms)  # shape (d, N)

        # For all pairs (i,j): check if perm[k][i] < perm[k][j] for all k
        # perm_arr[:, :, None] shape (d, N, 1) vs perm_arr[:, None, :] shape (d, 1, N)
        # less[k, i, j] = True if perm[k][i] < perm[k][j]
        less = perm_arr[:, :, None] < perm_arr[:, None, :]  # (d, N, N)

        # i precedes j iff all d coordinates have i < j
        all_less = np.all(less, axis=0)  # (N, N)
        cs.order = all_less

        return cs

    def count_relations(self) -> int:
        """Count causal relations using vectorized method."""
        perm_arr = np.array(self.perms)
        less = perm_arr[:, :, None] < perm_arr[:, None, :]
        all_less = np.all(less, axis=0)
        return int(np.sum(np.triu(all_less, k=1)))

    def ordering_fraction(self) -> float:
        return self.count_relations() / (self.N * (self.N - 1) / 2)

    def height(self) -> int:
        """Longest chain length."""
        cs = self.to_causet_fast()
        return cs.longest_chain()


def swap_move(d_order: DOrder, rng: np.random.Generator) -> DOrder:
    """
    MCMC move for d-orders: pick two elements, swap one coordinate.

    1. Pick i, j uniformly from {0, ..., N-1} with i != j
    2. Pick coordinate k uniformly from {0, ..., d-1}
    3. Swap perms[k][i] and perms[k][j]

    This is ergodic on d-orders with fixed N and d.
    """
    new = d_order.copy()
    i, j = rng.choice(d_order.N, size=2, replace=False)
    k = rng.integers(d_order.d)
    new.perms[k][i], new.perms[k][j] = new.perms[k][j], new.perms[k][i]
    return new


def bd_action_4d_fast(cs: FastCausalSet) -> float:
    """
    Fast 4D BD action (local, epsilon=1 limit).

    S = (1/24)*N - (4/24)*L + (6/24)*I_2 - (4/24)*I_3

    where L = links (0-intervals), I_2 = 1-intervals, I_3 = 2-intervals.
    Uses matrix operations.
    """
    N = cs.n
    order_int = cs.order.astype(np.int32)

    # interval_matrix[i,j] = number of elements between i and j
    interval_matrix = order_int @ order_int

    mask_related = np.triu(cs.order, k=1)
    L = int(np.sum(mask_related & (interval_matrix == 0)))
    I_2 = int(np.sum(mask_related & (interval_matrix == 1)))
    I_3 = int(np.sum(mask_related & (interval_matrix == 2)))

    return N / 24.0 - 4 * L / 24.0 + 6 * I_2 / 24.0 - 4 * I_3 / 24.0


def interval_entropy(cs: FastCausalSet, max_k: int = 15) -> float:
    """Shannon entropy of the interval-size distribution."""
    counts = count_intervals_by_size(cs, max_size=max_k)
    dist = np.array([counts.get(k, 0) for k in range(max_k + 1)], dtype=float)
    total = np.sum(dist)
    if total == 0:
        return 0.0
    p = dist / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def mcmc_d_order(d: int, N: int, beta: float,
                 n_steps: int = 10000, n_thermalize: int = 5000,
                 record_every: int = 10,
                 rng: np.random.Generator = None,
                 verbose: bool = False) -> dict:
    """
    MCMC sampling of d-orders weighted by exp(-beta * S_BD_4D).

    For d=4, uses the 4D BD action.
    For d=2, uses the 2D BD action (epsilon=1 form).

    Returns dict with samples, actions, observables, acceptance rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    current = DOrder(d, N, rng=rng)
    current_cs = current.to_causet_fast()

    if d == 4:
        current_action = bd_action_4d_fast(current_cs)
    else:
        # 2D action for d=2
        from .two_orders import bd_action_2d_fast
        current_action = bd_action_2d_fast(current_cs)

    actions = []
    entropies = []
    ordering_fracs = []
    heights = []
    n_accepted = 0

    total_steps = n_thermalize + n_steps

    for step in range(total_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet_fast()

        if d == 4:
            proposed_action = bd_action_4d_fast(proposed_cs)
        else:
            from .two_orders import bd_action_2d_fast
            proposed_action = bd_action_2d_fast(proposed_cs)

        delta_S = beta * (proposed_action - current_action)

        if delta_S <= 0 or rng.random() < np.exp(-min(delta_S, 500)):
            current = proposed
            current_cs = proposed_cs
            current_action = proposed_action
            n_accepted += 1

        if step >= n_thermalize and (step - n_thermalize) % record_every == 0:
            actions.append(current_action)
            entropies.append(interval_entropy(current_cs))
            ordering_fracs.append(current_cs.ordering_fraction())
            heights.append(current_cs.longest_chain())

            if verbose and (step - n_thermalize) % (record_every * 100) == 0:
                print(f"  Step {step}/{total_steps}: S={current_action:.2f}, "
                      f"H={entropies[-1]:.3f}, "
                      f"ord={ordering_fracs[-1]:.3f}, "
                      f"accept={n_accepted/(step+1):.3f}")

    return {
        'actions': np.array(actions),
        'entropies': np.array(entropies),
        'ordering_fracs': np.array(ordering_fracs),
        'heights': np.array(heights),
        'accept_rate': n_accepted / total_steps,
        'N': N,
        'd': d,
        'beta': beta,
    }
