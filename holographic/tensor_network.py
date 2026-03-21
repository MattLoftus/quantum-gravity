"""
Holographic tensor network codes.

Implements toy models of holographic quantum error correcting codes
inspired by the HaPPY (Harlow-Preskill-Pastawski-Yoshida) construction.

The key idea: a bulk spacetime geometry is encoded in a boundary quantum
state via a tensor network. The structure of the tensor network determines:
- Which bulk regions can be reconstructed from which boundary regions
  (entanglement wedge reconstruction)
- The entanglement entropy of boundary regions (Ryu-Takayanagi formula)
- Whether the geometry is dynamical (requires "magic" / non-stabilizer resources)

Reference: Cao, Cheng, Karthikeyan, Li, Preskill (arXiv:2603.13475, March 2026)
"""

import numpy as np
from typing import Optional


class TensorNetwork:
    """
    A tensor network on a graph.

    Each node (tensor) has a bond dimension d and connects to neighboring
    nodes via edges. The boundary consists of open (uncontracted) legs.

    For holographic codes:
    - Internal nodes = bulk
    - Open legs = boundary
    - The isometry from bulk to boundary is the "holographic map"
    """

    def __init__(self, n_bulk: int, n_boundary: int, bond_dim: int = 2):
        self.n_bulk = n_bulk
        self.n_boundary = n_boundary
        self.bond_dim = bond_dim
        self.total = n_bulk + n_boundary

        # Adjacency: which nodes are connected
        self.adjacency = np.zeros((self.total, self.total), dtype=bool)

        # Tensors at each bulk node (random by default)
        # Each tensor maps: (bulk_input) x (bond_legs...) -> (bond_legs...)
        # Stored as matrices for simplicity
        self.tensors = {}

    def add_edge(self, i: int, j: int):
        self.adjacency[i, j] = True
        self.adjacency[j, i] = True

    def set_random_tensors(self, rng: np.random.Generator = None):
        """Initialize with random (Haar-random) tensors — maximal magic."""
        if rng is None:
            rng = np.random.default_rng()

        for i in range(self.n_bulk):
            n_legs = int(np.sum(self.adjacency[i]))
            dim = self.bond_dim ** n_legs
            # Random unitary (Haar-distributed)
            A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
            Q, R = np.linalg.qr(A)
            # Fix phases
            D = np.diag(R)
            Q = Q @ np.diag(D / np.abs(D))
            self.tensors[i] = Q

    def set_stabilizer_tensors(self):
        """Initialize with Clifford/stabilizer tensors — zero magic."""
        for i in range(self.n_bulk):
            n_legs = int(np.sum(self.adjacency[i]))
            dim = self.bond_dim ** n_legs
            # Identity-like (stabilizer = no magic)
            self.tensors[i] = np.eye(dim, dtype=complex)


def build_happy_code(layers: int = 2, bond_dim: int = 2) -> TensorNetwork:
    """
    Build a simplified HaPPY-like code on a tree graph.

    The HaPPY code uses perfect tensors on a hyperbolic tiling.
    We use a tree (Bethe lattice) as an approximation:
    - Central bulk node connects to k children
    - Each child connects to k-1 further children
    - Leaves are boundary nodes

    This captures the key feature: the boundary dimension grows
    exponentially with the number of layers (UV/IR connection).
    """
    branching = 3  # each node has 3 children

    # Build tree
    nodes = [0]  # start with root
    edges = []
    next_id = 1

    current_layer = [0]
    n_bulk = 1

    for layer in range(layers):
        next_layer = []
        for parent in current_layer:
            for _ in range(branching):
                child = next_id
                next_id += 1
                nodes.append(child)
                edges.append((parent, child))
                next_layer.append(child)
                if layer < layers - 1:
                    n_bulk += 1
        current_layer = next_layer

    n_boundary = len(current_layer)
    n_total_bulk = n_bulk

    tn = TensorNetwork(n_total_bulk, n_boundary, bond_dim=bond_dim)
    for i, j in edges:
        tn.add_edge(i, j)

    return tn


def compute_entanglement_entropy(state: np.ndarray, subsystem_dims: list,
                                  subsystem_indices: list) -> float:
    """
    Compute von Neumann entanglement entropy of a subsystem.

    state: state vector of the full system
    subsystem_dims: dimensions of each subsystem
    subsystem_indices: which subsystems to trace over (the complement)

    Returns S = -Tr(rho_A * log(rho_A))
    """
    n = len(subsystem_dims)
    total_dim = int(np.prod(subsystem_dims))

    if len(state) != total_dim:
        return float('nan')

    # Reshape into tensor
    tensor = state.reshape(subsystem_dims)

    # Determine which indices to keep and which to trace
    keep = [i for i in range(n) if i not in subsystem_indices]
    trace_over = subsystem_indices

    if not keep or not trace_over:
        return 0.0

    # Compute reduced density matrix via SVD
    # Permute axes: (keep, trace_over)
    perm = keep + trace_over
    tensor = np.transpose(tensor, perm)

    dim_keep = int(np.prod([subsystem_dims[i] for i in keep]))
    dim_trace = int(np.prod([subsystem_dims[i] for i in trace_over]))
    matrix = tensor.reshape(dim_keep, dim_trace)

    # SVD
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-15]  # numerical cutoff

    # Entanglement entropy
    probs = s ** 2
    probs = probs / np.sum(probs)  # normalize
    entropy = -np.sum(probs * np.log2(probs + 1e-300))

    return entropy


def compute_mutual_information(state: np.ndarray, dims: list,
                                region_A: list, region_B: list) -> float:
    """
    I(A:B) = S(A) + S(B) - S(AB)
    """
    complement_A = [i for i in range(len(dims)) if i not in region_A]
    complement_B = [i for i in range(len(dims)) if i not in region_B]
    complement_AB = [i for i in range(len(dims)) if i not in region_A and i not in region_B]

    S_A = compute_entanglement_entropy(state, dims, complement_A)
    S_B = compute_entanglement_entropy(state, dims, complement_B)
    S_AB = compute_entanglement_entropy(state, dims, complement_AB)

    return S_A + S_B - S_AB


def random_state_with_magic(n_qubits: int, magic_fraction: float = 1.0,
                             rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a random state with tunable magic content.

    magic_fraction = 0: computational basis state (stabilizer, zero magic)
    magic_fraction = 1: Haar-random state (maximal magic)
    Intermediate: interpolate by applying random unitaries to a fraction of qubits

    Magic (non-stabilizer resources) is what Preskill et al. showed is needed
    for dynamical geometry in holographic codes.
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = 2 ** n_qubits

    if magic_fraction <= 0:
        # Computational basis state |0...0>
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        return state

    if magic_fraction >= 1:
        # Haar-random state
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state /= np.linalg.norm(state)
        return state

    # Partial magic: apply random unitaries to a subset of qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    n_magic_qubits = max(1, int(magic_fraction * n_qubits))

    # Apply random single-qubit rotations to the first n_magic_qubits
    for q in range(n_magic_qubits):
        # Random SU(2) element
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        U = np.array([[np.cos(theta / 2), -np.exp(1j * phi) * np.sin(theta / 2)],
                       [np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)]])

        # Apply to qubit q
        state_tensor = state.reshape([2] * n_qubits)
        # Construct the full operator: I ⊗ ... ⊗ U ⊗ ... ⊗ I
        indices = list(range(n_qubits))
        new_tensor = np.zeros_like(state_tensor)
        for idx in np.ndindex(*([2] * n_qubits)):
            idx_list = list(idx)
            for new_val in range(2):
                idx_new = list(idx)
                idx_new[q] = new_val
                new_tensor[tuple(idx_new)] += U[new_val, idx[q]] * state_tensor[tuple(idx)]
        state = new_tensor.reshape(dim)

    state /= np.linalg.norm(state)
    return state
