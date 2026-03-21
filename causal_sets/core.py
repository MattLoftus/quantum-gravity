"""
Core causal set data structures and operations.

A causal set (causet) is a locally finite partially ordered set (poset).
Elements represent spacetime events; the order relation encodes causality.
"""

import numpy as np
from typing import Optional


class CausalSet:
    """
    A causal set represented by its causal (adjacency) matrix.

    C[i,j] = 1 means element i precedes element j (i is in the causal past of j).
    The matrix encodes the transitive closure of the order relation.
    """

    def __init__(self, n: int = 0):
        self.n = n
        # Causal relation matrix: C[i,j] = 1 if i < j (i precedes j)
        self.order = np.zeros((n, n), dtype=np.int8) if n > 0 else np.zeros((0, 0), dtype=np.int8)

    @property
    def size(self) -> int:
        return self.n

    def precedes(self, i: int, j: int) -> bool:
        return bool(self.order[i, j])

    def num_relations(self) -> int:
        return int(np.sum(self.order))

    def ordering_fraction(self) -> float:
        """Fraction of pairs that are causally related. Related to dimension."""
        if self.n < 2:
            return 0.0
        total_pairs = self.n * (self.n - 1) / 2
        return self.num_relations() / total_pairs

    def link_matrix(self) -> np.ndarray:
        """
        Compute the link matrix L where L[i,j] = 1 if i -< j (i is linked to j),
        meaning i < j and there's no k with i < k < j.
        A link is an irreducible relation (nearest neighbor in causal order).
        """
        links = np.zeros_like(self.order)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.order[i, j]:
                    # Check if there's any k between i and j
                    is_link = True
                    for k in range(i + 1, j):
                        if self.order[i, k] and self.order[k, j]:
                            is_link = False
                            break
                    if is_link:
                        links[i, j] = 1
        return links

    def longest_chain(self) -> int:
        """
        Length of the longest chain (totally ordered subset).
        This is an estimator of the proper time extent of the causet.
        Uses dynamic programming.
        """
        if self.n == 0:
            return 0
        # dp[i] = length of longest chain ending at element i
        dp = np.ones(self.n, dtype=int)
        for j in range(self.n):
            for i in range(j):
                if self.order[i, j]:
                    dp[j] = max(dp[j], dp[i] + 1)
        return int(np.max(dp))

    def interval_count(self, i: int, j: int) -> int:
        """Count the number of elements in the Alexandrov interval [i, j]
        (elements k where i <= k <= j), excluding i and j themselves."""
        if not self.order[i, j]:
            return 0
        count = 0
        for k in range(self.n):
            if k != i and k != j and self.order[i, k] and self.order[k, j]:
                count += 1
        return count

    def all_interval_counts(self) -> list:
        """Compute interval sizes for all related pairs. Used for dimension estimation."""
        counts = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.order[i, j]:
                    counts.append(self.interval_count(i, j))
        return counts


def transitive_closure(order: np.ndarray) -> np.ndarray:
    """Compute the transitive closure of a relation matrix using Warshall's algorithm."""
    n = order.shape[0]
    closure = order.copy()
    for k in range(n):
        for i in range(n):
            if closure[i, k]:
                for j in range(n):
                    if closure[k, j]:
                        closure[i, j] = 1
    return closure
