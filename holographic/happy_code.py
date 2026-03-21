"""
Proper HaPPY holographic code with Clifford+T magic interpolation.

Key improvement over previous version: magic is injected INTO the encoding
circuit via entangling T+CNOT layers, not just post-encoding single-qubit
rotations. This ensures intermediate magic levels produce genuinely entangled
non-stabilizer states.

Implements:
1. The [[5,1,3]] perfect tensor (five-qubit stabilizer code)
2. A compact 2-bulk, 6-boundary HaPPY code (8 qubits total)
3. Clifford+T magic interpolation with multi-qubit entangling gates

The [[5,1,3]] code is the unique 5-qubit quantum error correcting code with
distance 3. Its encoding isometry V: C^2 -> C^32 maps 1 logical qubit to
5 physical qubits, and the associated 6-index tensor is "perfect".

Reference: Pastawski, Yoshida, Harlow, Preskill, JHEP 2015
           Cao, Cheng, Karthikeyan, Li, Preskill, arXiv:2603.13475 (2026)
"""

import numpy as np
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Gate definitions
# ═══════════════════════════════════════════════════════════════════════════

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def kron_list(ops):
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def apply_gate_1q(state, gate, qubit, n_qubits):
    """Apply a single-qubit gate to a state vector."""
    psi = state.reshape([2] * n_qubits)
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(2 ** n_qubits)


def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT: |c,t> -> |c, t XOR c>."""
    psi = state.reshape([2] * n_qubits)
    result = np.zeros_like(psi)
    # Only flip target when control=1
    # control=0 part
    slc_c0 = [slice(None)] * n_qubits
    slc_c0[control] = 0
    result[tuple(slc_c0)] += psi[tuple(slc_c0)]
    # control=1 part: flip target
    slc_c1_src = [slice(None)] * n_qubits
    slc_c1_src[control] = 1
    for t_val in range(2):
        slc_src = list(slc_c1_src)
        slc_src[target] = t_val
        slc_dst = list(slc_c1_src)
        slc_dst[target] = 1 - t_val
        result[tuple(slc_dst)] += psi[tuple(slc_src)]
    return result.reshape(2 ** n_qubits)


# ═══════════════════════════════════════════════════════════════════════════
# [[5,1,3]] Perfect Tensor
# ═══════════════════════════════════════════════════════════════════════════

def build_513_encoding_isometry():
    """
    Build the encoding isometry V for the [[5,1,3]] code.

    V: C^2 -> C^32 (32x2 matrix). Maps |0> -> |0_L>, |1> -> |1_L>.

    Stabilizer generators:
        g1 = XZZXI,  g2 = IXZZX,  g3 = XIXZZ,  g4 = ZXIXZ

    Logical operators: X_L = XXXXX, Z_L = ZZZZZ
    """
    dim = 32
    stabilizers = [
        [X, Z, Z, X, I2],
        [I2, X, Z, Z, X],
        [X, I2, X, Z, Z],
        [Z, X, I2, X, Z],
    ]

    identity = np.eye(dim, dtype=complex)
    projector = identity.copy()
    for ops in stabilizers:
        g = kron_list(ops)
        projector = projector @ (identity + g) / 2

    zero_state = np.zeros(dim, dtype=complex)
    zero_state[0] = 1.0
    logical_zero = projector @ zero_state
    logical_zero /= np.linalg.norm(logical_zero)

    X_L = kron_list([X, X, X, X, X])
    logical_one = X_L @ logical_zero
    logical_one /= np.linalg.norm(logical_one)

    assert abs(np.dot(logical_zero.conj(), logical_one)) < 1e-10

    V = np.column_stack([logical_zero, logical_one])
    return V


def verify_perfect_tensor(V):
    """Verify the perfect tensor property: any 3-of-6 bipartition is max mixed."""
    tensor = V.T.reshape(2, 2, 2, 2, 2, 2)
    from itertools import combinations
    max_dev = 0.0
    for keep in combinations(range(6), 3):
        trace = [i for i in range(6) if i not in keep]
        perm = list(keep) + list(trace)
        t = np.transpose(tensor, perm).reshape(8, 8)
        rho = t @ t.conj().T
        rho /= np.trace(rho)
        dev = np.linalg.norm(rho - np.eye(8) / 8, 'fro')
        max_dev = max(max_dev, dev)
    return max_dev


# ═══════════════════════════════════════════════════════════════════════════
# Clifford+T Magic Interpolation (with entangling gates)
# ═══════════════════════════════════════════════════════════════════════════

def build_encoding_circuit(n_qubits, magic_level, depth, rng):
    """
    Build a Clifford+T encoding circuit with controlled T-gate density.

    Key difference from naive approach: every layer includes CNOT gates
    that entangle qubits, so intermediate magic levels produce genuinely
    entangled non-stabilizer states (not just product states with local
    rotations).

    Circuit structure per layer:
        1. Random CNOT between a pair of qubits (entangling)
        2. Random Clifford single-qubit gate (H or S)
        3. With probability=magic_level: T gate on a random qubit (magic)

    magic_level=0: pure Clifford circuit (stabilizer states only)
    magic_level=1: T gate every layer (maximal non-stabilizer content)
    """
    gates = []
    for _ in range(depth):
        # Step 1: Entangling CNOT (always present)
        if n_qubits >= 2:
            q1, q2 = rng.choice(n_qubits, size=2, replace=False)
            gates.append(('CNOT', (int(q1), int(q2))))

        # Step 2: Random Clifford single-qubit gate
        q = int(rng.integers(n_qubits))
        if rng.random() < 0.5:
            gates.append(('H', (q,)))
        else:
            gates.append(('S', (q,)))

        # Step 3: T gate with probability = magic_level
        if magic_level > 0 and rng.random() < magic_level:
            q = int(rng.integers(n_qubits))
            gates.append(('T', (q,)))

    return gates


def apply_circuit(state, gates, n_qubits):
    """Apply a list of gates to a state vector."""
    psi = state.copy()
    for gate_type, qubits in gates:
        if gate_type == 'H':
            psi = apply_gate_1q(psi, H_gate, qubits[0], n_qubits)
        elif gate_type == 'S':
            psi = apply_gate_1q(psi, S_gate, qubits[0], n_qubits)
        elif gate_type == 'T':
            psi = apply_gate_1q(psi, T_gate, qubits[0], n_qubits)
        elif gate_type == 'CNOT':
            psi = apply_cnot(psi, qubits[0], qubits[1], n_qubits)
    return psi


# ═══════════════════════════════════════════════════════════════════════════
# Compact HaPPY Code: 2 bulk qubits, 6 boundary qubits (8 total)
# ═══════════════════════════════════════════════════════════════════════════

class HaPPYCode:
    """
    Compact tree-based HaPPY code from [[5,1,3]] perfect tensors.

    Two-tensor chain:
        Tensor A: bulk_0 (logical) -> 5 legs (bond, b0, b1, b2, b3)
        Tensor B: (bond, bulk_1) -> 4 boundary legs (b4, b5, b6, b7)

    The bond leg from A becomes the logical input of B (via perfect tensor
    isometry property: any 2 inputs -> 4 outputs is valid).

    Total: 2 bulk qubits -> 8 boundary qubits, but we keep only 6 boundary
    qubits by tracing over the 2 "inner" legs used for the contraction.

    Actually, the chain produces 4 + 4 = 8 boundary qubits from 2 bulk qubits.
    We keep all 8 as the boundary. Total system: 8 qubits.

    Magic interpolation: after the stabilizer encoding, apply a Clifford+T
    circuit to ALL 8 boundary qubits with entangling gates. This modifies
    the encoding map itself (the encoding + magic circuit together form the
    holographic map from bulk to boundary).
    """

    def __init__(self, magic_level=0.0, rng=None, circuit_depth=10):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.magic_level = magic_level
        self.circuit_depth = circuit_depth

        # Build base [[5,1,3]] isometry
        self.V = build_513_encoding_isometry()  # 32x2
        self.T6 = self.V.T.reshape(2, 2, 2, 2, 2, 2)
        # T6[logical, p0, p1, p2, p3, p4]

        self.n_bulk = 2
        self.n_boundary = 8
        self.n_total = 8  # boundary qubits only (bulk is encoded away)

        # Pre-build the magic circuit (applied to all 8 boundary qubits)
        if magic_level > 0:
            self.magic_circuit = build_encoding_circuit(
                self.n_boundary, magic_level, circuit_depth, rng
            )
        else:
            self.magic_circuit = None

    def encode(self, bulk_state):
        """
        Encode a 2-qubit bulk state into an 8-qubit boundary state.

        bulk_state: length-4 vector (2 bulk qubits)
        Returns: length-256 vector (8 boundary qubits)

        The encoding contracts two [[5,1,3]] perfect tensors along a bond:

            result[b0..b3, b4..b7] = sum_{bond} T6[bulk0, bond, b0,b1,b2,b3]
                                                * T6[bond, bulk1, b4,b5,b6,b7]
        """
        bulk = bulk_state.reshape(2, 2)
        result = np.zeros(2 ** self.n_boundary, dtype=complex)

        for b0 in range(2):
            for b1 in range(2):
                coeff = bulk[b0, b1]
                if abs(coeff) < 1e-16:
                    continue
                # Tensor A: T6[b0, bond, p1, p2, p3, p4] -> shape (2, 2, 2, 2, 2)
                tA = self.T6[b0]  # (bond, p1, p2, p3, p4)
                # Tensor B: T6[bond, b1, q1, q2, q3, q4] -> shape (2, 2, 2, 2, 2)
                tB = self.T6[:, b1, :, :, :, :]  # (bond, q1, q2, q3, q4)
                # Contract over bond index
                tA_mat = tA.reshape(2, 16)   # (bond, 4_boundary_A)
                tB_mat = tB.reshape(2, 16)   # (bond, 4_boundary_B)
                piece = tA_mat.T @ tB_mat    # (16, 16) = (bdry_A, bdry_B)
                result += coeff * piece.reshape(256)

        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm

        # Apply magic circuit (entangling Clifford+T on all boundary qubits)
        if self.magic_circuit is not None:
            result = apply_circuit(result, self.magic_circuit, self.n_boundary)

        return result

    def encode_product_bulk(self, theta0, phi0, theta1, phi1):
        """Encode a product bulk state from Bloch sphere angles."""
        q0 = np.array([np.cos(theta0 / 2),
                        np.exp(1j * phi0) * np.sin(theta0 / 2)], dtype=complex)
        q1 = np.array([np.cos(theta1 / 2),
                        np.exp(1j * phi1) * np.sin(theta1 / 2)], dtype=complex)
        return self.encode(np.kron(q0, q1))

    def encode_bulk_with_entropy(self, entropy_param=0.0):
        """
        Encode bulk state with controlled bulk entanglement.

        entropy_param=0: |00> (product, zero bulk entropy)
        entropy_param=1: (|00>+|11>)/sqrt(2) (Bell pair, max bulk entropy)
        Intermediate: cos(a)|00> + sin(a)|11>
        """
        dim = 4
        a = entropy_param * np.pi / 4  # 0 -> 0, 1 -> pi/4
        bulk = np.zeros(dim, dtype=complex)
        bulk[0] = np.cos(a)    # |00>
        bulk[3] = np.sin(a)    # |11>
        return self.encode(bulk)


# ═══════════════════════════════════════════════════════════════════════════
# Entanglement entropy measurement
# ═══════════════════════════════════════════════════════════════════════════

def von_neumann_entropy(state_vec, n_qubits, keep_qubits):
    """
    Compute von Neumann entropy S(A) for subsystem A = keep_qubits.

    Returns entropy in bits (log base 2).
    """
    trace_qubits = [q for q in range(n_qubits) if q not in keep_qubits]
    if not keep_qubits or not trace_qubits:
        return 0.0

    psi = state_vec.reshape([2] * n_qubits)
    perm = list(keep_qubits) + list(trace_qubits)
    psi = np.transpose(psi, perm)

    dim_keep = 2 ** len(keep_qubits)
    dim_trace = 2 ** len(trace_qubits)
    mat = psi.reshape(dim_keep, dim_trace)

    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-15]
    p = s ** 2
    p = p / np.sum(p)
    return -np.sum(p * np.log2(p + 1e-300))


def mutual_information(state_vec, n_qubits, region_A, region_B):
    """I(A:B) = S(A) + S(B) - S(A union B)."""
    S_A = von_neumann_entropy(state_vec, n_qubits, region_A)
    S_B = von_neumann_entropy(state_vec, n_qubits, region_B)
    S_AB = von_neumann_entropy(state_vec, n_qubits,
                                list(set(region_A) | set(region_B)))
    return S_A + S_B - S_AB
