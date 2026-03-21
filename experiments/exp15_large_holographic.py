"""
Experiment 15: Large-scale holographic codes — testing Preskill's prediction.

Scales up from exp11's 2-bulk/8-boundary system to 3-bulk/12-boundary using
a 3-tensor chain of [[5,1,3]] perfect tensors, and also a 2-layer tree
(1 center + 3 leaves) giving 4 bulk / 12 boundary.

Key prediction from Preskill et al. (arXiv:2603.13475):
  At magic=0 (stabilizer), the "geometry" (entanglement structure) is FIXED
  regardless of bulk state. Proto-area entropy is constant.
  At magic>0, proto-area should increase monotonically with bulk entropy,
  because the non-stabilizer encoding enables geometric backreaction.

exp11 found the OPPOSITE at 8 qubits (magic=0 was MOST sensitive). Hypothesis:
the system was too small. This experiment tests whether larger system sizes
reveal the predicted Preskill effect.

Proto-area = S(A) - S_bulk_recoverable(A):
  For boundary region A covering > half the boundary, entire bulk is recoverable.
  For A < half, only the entanglement wedge portion of the bulk is recoverable.

System: /usr/bin/python3 with numpy, scipy. Uses state vectors (not density
matrices) and SVD for entropy. Memory: 2^12 = 4096 dim vectors, manageable.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from holographic.happy_code import (
    build_513_encoding_isometry, build_encoding_circuit, apply_circuit,
    apply_gate_1q, apply_cnot, von_neumann_entropy, mutual_information,
    H_gate, S_gate, T_gate, I2, X, Z, kron_list
)


# ═══════════════════════════════════════════════════════════════════════════
# 3-tensor chain: 3 bulk qubits -> 12 boundary qubits
# ═══════════════════════════════════════════════════════════════════════════

class ThreeTensorChain:
    """
    Chain of 3 [[5,1,3]] perfect tensors.

    Layout:
        Tensor A: bulk_0 -> (bond_AB, bA0, bA1, bA2, bA3)  [4 boundary]
        Tensor B: (bond_AB, bulk_1) -> (bond_BC, bB0, bB1, bB2)  [3 boundary]
           - bond_AB is logical input, bulk_1 is a second input
           - The [[5,1,3]] isometry maps 1 qubit to 5, but we can use the
             perfect tensor property: any 2-to-4 bipartition is an isometry
        Tensor C: (bond_BC, bulk_2) -> (bC0, bC1, bC2, bC3)  [4 boundary + 1 bond]
           - Same 2-to-4 isometry

    Actually, for simplicity and correctness, we use the 6-index perfect tensor
    T6[i0,i1,i2,i3,i4,i5] where any 3 indices -> other 3 is an isometry.

    Chain contraction:
        T_A[bulk0, bond_AB, bA0, bA1, bA2, bA3]  — 1 bulk, 1 bond, 4 boundary
        T_B[bond_AB, bulk1, bond_BC, bB0, bB1, bB2]  — 1 bulk, 2 bonds, 3 boundary
        T_C[bond_BC, bulk2, bC0, bC1, bC2, bC3]  — 1 bulk, 1 bond, 4 boundary

    Wait — each tensor has 6 indices. For T_B, that's bond_AB + bulk1 + bond_BC +
    3 boundary = 6. For T_A: bulk0 + bond_AB + 4 boundary = 6. T_C: bond_BC +
    bulk2 + 4 boundary = 6. Total boundary = 4 + 3 + 4 = 11.

    Better: Make T_B have 2 boundary legs instead:
        T_B[bond_AB, bulk1, bond_BC, bB0, bB1, bB2] — 6 indices, 3 boundary
    Then total boundary = 4 + 3 + 4 = 11. Still odd.

    Cleanest: use 2 bonds from T_A (leaving 3 boundary), 2 bonds from T_C
    (leaving 3 boundary), and T_B has bulk1 + 2 bonds in + 3 boundary out.
    That gives boundary = 3 + 3 + 3 = 9. Hmm.

    Let's just do the simple thing: 3 tensors in a chain with single bonds.
    T_A: bulk0 -> 5 legs. One leg is bond_AB. 4 boundary legs.
    T_B: 2 inputs (bond_AB, bulk1) -> 4 outputs (bond_BC, bB0, bB1, bB2).
         Using T6's 2-to-4 isometry property.
    T_C: 2 inputs (bond_BC, bulk2) -> 4 outputs (bC0, bC1, bC2, bC3).

    Total: 4 + 3 + 4 = 11 boundary qubits. With 3 bulk: 14 total... but we
    only track the 11 boundary. Let's do 4+3+4 = 11 boundary.

    Actually, let me reconsider. T_B has 6 indices total. 2 are bond (in/out),
    1 is bulk. That leaves 3 boundary. So 4 + 3 + 4 = 11 boundary, 3 bulk.
    The encoded state lives in 2^11 = 2048 dimensions. Very manageable.

    For 12 boundary: use T_B with only 1 bond in and 1 bond out:
    T_A[bulk0, bond_AB, b0, b1, b2, b3] — 4 boundary
    T_B[bond_AB, bulk1, bond_BC, b4, b5, b6] — 3 boundary  (6 indices total)
    T_C[bond_BC, bulk2, b7, b8, b9, b10] — 4 boundary (6 indices total)
    Total = 11 boundary. Close enough.

    OR: use a 2-layer tree for 12 boundary.
    """

    def __init__(self, magic_level=0.0, rng=None, circuit_depth=12):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.magic_level = magic_level
        self.circuit_depth = circuit_depth

        # Build [[5,1,3]] perfect tensor
        V = build_513_encoding_isometry()  # 32x2
        self.T6 = V.T.reshape(2, 2, 2, 2, 2, 2)
        # T6[logical, p0, p1, p2, p3, p4]

        self.n_bulk = 3
        self.n_boundary = 11  # 4 + 3 + 4
        self.n_total = self.n_boundary

        # Build magic circuit on boundary qubits
        if magic_level > 0:
            self.magic_circuit = build_encoding_circuit(
                self.n_boundary, magic_level, circuit_depth, rng
            )
        else:
            self.magic_circuit = None

    def encode(self, bulk_state):
        """
        Encode a 3-qubit bulk state (length 8) into 11-qubit boundary state.

        Contraction:
          result[b0..b3, b4..b6, b7..b10] =
            sum_{bond_AB, bond_BC} T6[bulk0, bond_AB, b0, b1, b2, b3]
                                 * T6[bond_AB, bulk1, bond_BC, b4, b5, b6]
                                 * T6[bond_BC, bulk2, b7, b8, b9, b10]
        """
        bulk = bulk_state.reshape(2, 2, 2)  # (bulk0, bulk1, bulk2)
        T6 = self.T6

        # Step 1: Contract T_A and T_B over bond_AB
        # T_A[bulk0, bond_AB, b0, b1, b2, b3] has shape (2,2,2,2,2,2)
        # T_B[bond_AB, bulk1, bond_BC, b4, b5, b6] has shape (2,2,2,2,2,2)
        #   but T_B only uses 6 indices total. We index T6 as:
        #   T6[bond_AB, bulk1, bond_BC, b4, b5, b6]

        # We'll build the result by iterating over bulk indices (only 8 combos)
        result = np.zeros(2 ** self.n_boundary, dtype=complex)

        for bk0 in range(2):
            for bk1 in range(2):
                for bk2 in range(2):
                    coeff = bulk[bk0, bk1, bk2]
                    if abs(coeff) < 1e-16:
                        continue

                    # T_A[bk0, :, :, :, :, :] shape (2, 2, 2, 2, 2)
                    # = (bond_AB, b0, b1, b2, b3)
                    tA = T6[bk0]  # (bond_AB, b0, b1, b2, b3)

                    # T_B[:, bk1, :, :, :, :] shape (2, 2, 2, 2, 2)
                    # = (bond_AB, bond_BC, b4, b5, b6)
                    tB = T6[:, bk1, :, :, :, :]  # (bond_AB, bond_BC, b4, b5, b6)
                    # but this is (2, 2, 2, 2, 2) with indices
                    # [index0=bond_AB, index1=bond_BC, index2=b4, index3=b5, index4=b6]

                    # T_C[:, bk2, :, :, :, :] shape (2, 2, 2, 2, 2)
                    # = (bond_BC, b7, b8, b9, b10)
                    tC = T6[:, bk2, :, :, :, :]  # (bond_BC, b7, b8, b9, b10)

                    # Contract: sum over bond_AB between tA and tB
                    # tA: (bond_AB, b0, b1, b2, b3) -> reshape (2, 16)
                    # tB: (bond_AB, bond_BC, b4, b5, b6) -> reshape (2, 8*bond_BC)
                    # Actually, let's contract step by step.

                    # AB contraction: sum_{bond_AB} tA[bond_AB, b0..b3] * tB[bond_AB, bond_BC, b4..b6]
                    # tA reshaped: (2, 16) where 16 = 2^4 boundary of A
                    # tB reshaped: (2, 2*8) = (2, 16) but we need bond_BC separate
                    # tB: (bond_AB, bond_BC, 2, 2, 2) -> (bond_AB, bond_BC * 8)
                    tA_mat = tA.reshape(2, 16)  # (bond_AB, bdryA=16)
                    tB_mat = tB.reshape(2, 2 * 8)  # (bond_AB, bond_BC*bdryB)
                    # Note: tB has 5 dims: (bond_AB=2, bond_BC=2, b4=2, b5=2, b6=2)
                    # reshape to (2, 16) where 16 = 2*2*2*2 = bond_BC * b4 * b5 * b6
                    tB_mat = tB.reshape(2, 16)

                    # AB contracted: (bdryA, bond_BC*bdryB) = (16, 16)
                    tAB = tA_mat.T @ tB_mat  # (16, 16)

                    # Now separate bond_BC from bdryB:
                    # tAB: (16, 16) = (bdryA=16, bond_BC * bdryB)
                    # bdryB has 3 qubits = 8 dims, bond_BC = 2 dim
                    # So tAB: (16, 2, 8) when reshaped
                    tAB = tAB.reshape(16, 2, 8)  # (bdryA, bond_BC, bdryB)

                    # TC contraction: sum_{bond_BC} tAB[bdryA, bond_BC, bdryB] * tC[bond_BC, bdryC]
                    tC_mat = tC.reshape(2, 16)  # (bond_BC, bdryC=16)

                    # Contract bond_BC:
                    # tAB: (16, 2, 8) -> (16*8, 2) after transpose
                    # Actually: tAB[a, bond, b] * tC[bond, c] = sum_bond
                    # Result: (bdryA=16, bdryB=8, bdryC=16) — total 16*8*16 = 2048 = 2^11 ✓

                    # tAB: (bdryA=16, bond_BC=2, bdryB=8)
                    # tC_mat: (bond_BC=2, bdryC=16)
                    # Want: sum_{bond_BC} tAB[a, bond_BC, b] * tC[bond_BC, c]
                    piece = np.einsum('aib,ic->abc', tAB, tC_mat)
                    # piece shape: (bdryA=16, bdryB=8, bdryC=16)
                    # Total = 16 * 8 * 16 = 2048 = 2^11 ✓

                    result += coeff * piece.reshape(2 ** self.n_boundary)

        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm

        # Apply magic circuit
        if self.magic_circuit is not None:
            result = apply_circuit(result, self.magic_circuit, self.n_boundary)

        return result

    def encode_bulk_with_entropy(self, entropy_param):
        """
        Encode bulk state with controlled entanglement.

        entropy_param=0: |000> (product, zero entropy)
        entropy_param=1: GHZ-like maximally entangled
        Intermediate: partial entanglement via Schmidt decomposition

        We create entanglement between all 3 bulk qubits:
        |psi> = cos(a)|000> + sin(a)|111>
        This gives max entropy = 1 bit for any bipartition of the 3 bulk qubits.
        """
        a = entropy_param * np.pi / 4
        bulk = np.zeros(8, dtype=complex)
        bulk[0] = np.cos(a)  # |000>
        bulk[7] = np.sin(a)  # |111>
        return self.encode(bulk)

    def encode_bulk_with_pairwise_entropy(self, entropy_01, entropy_02, entropy_12):
        """
        More fine-grained control: set pairwise entanglement between bulk qubits.

        Uses a parameterized 3-qubit state:
        |psi> = c0|000> + c1|011> + c2|101> + c3|110>
        where coefficients are chosen to give desired pairwise entropies.
        For simplicity, use:
        |psi> = cos(a)|000> + sin(a)|111> with a = mean_entropy * pi/4
        """
        mean_e = (entropy_01 + entropy_02 + entropy_12) / 3.0
        return self.encode_bulk_with_entropy(mean_e)

    def get_boundary_regions(self):
        """
        Return physically meaningful boundary regions corresponding to
        the three tensors.

        Tensor A contributes boundary qubits 0-3 (4 qubits)
        Tensor B contributes boundary qubits 4-6 (3 qubits)
        Tensor C contributes boundary qubits 7-10 (4 qubits)
        """
        return {
            'A': list(range(0, 4)),      # near bulk0
            'B': list(range(4, 7)),       # near bulk1
            'C': list(range(7, 11)),      # near bulk2
            'AB': list(range(0, 7)),      # near bulk0 + bulk1
            'BC': list(range(4, 11)),     # near bulk1 + bulk2
            'left_half': list(range(0, 6)),   # ~half
            'right_half': list(range(6, 11)),
            'big_left': list(range(0, 8)),    # >half
            'big_right': list(range(3, 11)),  # >half
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2-layer tree: 1 center + 3 leaf tensors -> 4 bulk, 12 boundary
# ═══════════════════════════════════════════════════════════════════════════

class TreeCode:
    """
    2-layer tree of [[5,1,3]] perfect tensors.

    Center tensor: bulk0 -> (bond1, bond2, bond3, bC0, bC1)  [2 boundary]
      - 1 bulk input + 3 bonds out + 2 boundary = 6 indices ✓

    Leaf tensor i (i=1,2,3):
      T_i[bond_i, bulk_i, bLi0, bLi1, bLi2, bLi3]  [4 boundary each]
      - 1 bond in + 1 bulk + 4 boundary = 6 indices ✓

    Total boundary: 2 + 3*4 = 14 qubits. That's 2^14 = 16384 dim.
    A bit large. Let's use 3 boundary from the center:

    Center: bulk0 -> (bond1, bond2, bond3, bC0, bC1) — no, that's only 6 indices
    with bulk0 as the logical.

    Actually: T6[bulk0, bond1, bond2, bond3, bC0, bC1] — 6 indices, 2 boundary.
    Leaves: T6[bond_i, bulk_i, bLi0, bLi1, bLi2, bLi3] — 6 indices, 4 boundary.
    Total boundary = 2 + 3*4 = 14. With 4 bulk qubits. Total system = 14 boundary.

    Let's try it — 2^14 = 16384 is fine for state vectors (~128KB).
    SVD of (128, 128) matrices is fast.

    If too slow, fall back to 2 leaves (3 bulk, 10 boundary).
    """

    def __init__(self, n_leaves=3, magic_level=0.0, rng=None, circuit_depth=12):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.magic_level = magic_level
        self.circuit_depth = circuit_depth
        self.n_leaves = n_leaves

        V = build_513_encoding_isometry()
        self.T6 = V.T.reshape(2, 2, 2, 2, 2, 2)

        self.n_bulk = 1 + n_leaves  # center + leaves
        self.n_boundary_center = 6 - 1 - n_leaves  # 6 indices - 1 bulk - n_leaves bonds
        self.n_boundary_per_leaf = 4  # 6 - 1 bond - 1 bulk
        self.n_boundary = self.n_boundary_center + n_leaves * self.n_boundary_per_leaf
        self.n_total = self.n_boundary

        if magic_level > 0:
            self.magic_circuit = build_encoding_circuit(
                self.n_boundary, magic_level, circuit_depth, rng
            )
        else:
            self.magic_circuit = None

    def encode(self, bulk_state):
        """
        Encode bulk state into boundary.

        For n_leaves=3: 4 bulk qubits -> 14 boundary qubits.
        For n_leaves=2: 3 bulk qubits -> 12 boundary qubits.

        bulk_state: length 2^n_bulk vector.
        """
        n_bulk = self.n_bulk
        n_leaves = self.n_leaves
        T6 = self.T6
        n_bdry_center = self.n_boundary_center
        n_bdry = self.n_boundary

        bulk = bulk_state.reshape([2] * n_bulk)

        result = np.zeros(2 ** n_bdry, dtype=complex)

        # Iterate over all bulk configurations (2^n_bulk = 8 or 16)
        for bulk_idx in range(2 ** n_bulk):
            bits = [(bulk_idx >> (n_bulk - 1 - k)) & 1 for k in range(n_bulk)]
            bk0 = bits[0]  # center bulk
            bk_leaves = bits[1:]  # leaf bulk qubits

            coeff = bulk[tuple(bits)]
            if abs(coeff) < 1e-16:
                continue

            # Center tensor: T6[bk0, bond1, bond2, ..., bond_{n_leaves}, bC0, ...]
            # Indices: [bk0, bond1, ..., bond_{n_leaves}, bC0, ..., bC_{n_bdry_center-1}]
            # = [bk0] + [bonds] + [center_boundary]
            # Total = 1 + n_leaves + n_bdry_center = 6
            tC = T6[bk0]  # shape: (2,2,2,2,2) — 5 remaining indices
            # First n_leaves indices are bonds, remaining are center boundary

            # Leaf tensors: T6[bond_i, bk_leaf_i, bL0, bL1, bL2, bL3]
            # For each leaf: T6[:, bk_leaf_i, :, :, :, :] shape (2, 2, 2, 2, 2)
            # = (bond_i, bL0, bL1, bL2, bL3)

            leaf_tensors = []
            for i in range(n_leaves):
                tL = T6[:, bk_leaves[i], :, :, :, :]  # (bond, bL0, bL1, bL2, bL3)
                leaf_tensors.append(tL)

            # Contract bonds between center and leaves.
            # Center: (bond1, bond2, ..., bond_n, bC...)
            # We need to sum over bond1, bond2, ..., bond_n.

            if n_leaves == 2:
                # tC: (bond1, bond2, bC0, bC1, bC2) — 5 indices, center has 3 boundary
                # Actually n_bdry_center = 6 - 1 - 2 = 3
                # tL1: (bond1, bL10, bL11, bL12, bL13) — 5 indices
                # tL2: (bond2, bL20, bL21, bL22, bL23) — 5 indices

                # Contract bond1: sum_{b1} tC[b1, bond2, bC0, bC1, bC2] * tL1[b1, bL1...]
                tL1 = leaf_tensors[0].reshape(2, 16)  # (bond1, 16)
                tL2 = leaf_tensors[1].reshape(2, 16)  # (bond2, 16)

                # tC shape: (2, 2, 2, 2, 2) = (bond1, bond2, bC0, bC1, bC2)
                # Reshape: (bond1, bond2 * 8)
                tC_r = tC.reshape(2, 2, 8)  # (bond1, bond2, center_bdry=8)

                # Contract bond1: tC_r[bond1, bond2, cb] * tL1[bond1, lb1]
                # -> (bond2, cb, lb1)
                step1 = np.einsum('abc,ad->bcd', tC_r, tL1)  # (bond2, 8, 16)

                # Contract bond2: step1[bond2, cb, lb1] * tL2[bond2, lb2]
                # -> (cb, lb1, lb2)
                step2 = np.einsum('abc,ad->bcd', step1, tL2)  # (8, 16, 16)

                # Flatten: center_bdry (8) * leaf1_bdry (16) * leaf2_bdry (16) = 2048
                # But n_bdry = 3 + 4 + 4 = 11, so 2^11 = 2048 ✓
                piece = step2.reshape(2 ** n_bdry)

            elif n_leaves == 3:
                # tC: (bond1, bond2, bond3, bC0, bC1) — 5 indices
                # n_bdry_center = 6 - 1 - 3 = 2
                # center has 2 boundary qubits (4 dims)
                tL1 = leaf_tensors[0].reshape(2, 16)
                tL2 = leaf_tensors[1].reshape(2, 16)
                tL3 = leaf_tensors[2].reshape(2, 16)

                # tC: (bond1, bond2, bond3, bC0, bC1) = (2,2,2,2,2)
                tC_r = tC.reshape(2, 2, 2, 4)  # (bond1, bond2, bond3, center_bdry)

                # Contract bond1
                step1 = np.einsum('abcd,ae->bcde', tC_r, tL1)
                # (bond2, bond3, center_bdry=4, leaf1_bdry=16)

                # Contract bond2
                step2 = np.einsum('abcd,ae->bcde', step1, tL2)
                # (bond3, center_bdry=4, leaf1_bdry=16, leaf2_bdry=16)

                # Contract bond3
                step3 = np.einsum('abcd,ae->bcde', step2, tL3)
                # (center_bdry=4, leaf1_bdry=16, leaf2_bdry=16, leaf3_bdry=16)

                # Total: 4 * 16 * 16 * 16 = 65536 = 2^16... that's wrong.
                # Wait: n_bdry = 2 + 3*4 = 14. 2^14 = 16384.
                # But 4*16*16*16 = 65536 = 2^16. Problem!

                # The issue: leaf tensor has shape (bond, bL0, bL1, bL2, bL3)
                # That's 5 dims, but T6 has 6 indices. One is the bulk (fixed).
                # So tL = T6[:, bk_leaf, :, :, :, :] has shape (2, 2, 2, 2, 2)
                # = (bond, bL0, bL1, bL2, bL3). The leaf has 4 boundary qubits
                # = 2^4 = 16 boundary dims. That's correct.
                # center: (bond1, bond2, bond3, bC0, bC1) -> 2 center boundary = 4 dims
                # Total: 4 * 16 * 16 * 16... but we need 2^14 = 16384.
                # 4 = 2^2, 16 = 2^4. Total = 2^2 * (2^4)^3 = 2^14 = 16384 ✓
                # The issue is 4*16*16*16 = 65536 ≠ 16384.
                # Oh wait, 4*16*16*16 = 4*4096 = 16384. Let me recheck.
                # 4 * 16 * 16 * 16 = 4 * 4096 = 16384. Yes, that's correct!

                piece = step3.reshape(2 ** n_bdry)

            else:
                raise ValueError(f"n_leaves={n_leaves} not supported")

            result += coeff * piece

        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result /= norm

        if self.magic_circuit is not None:
            result = apply_circuit(result, self.magic_circuit, self.n_boundary)

        return result

    def encode_bulk_with_entropy(self, entropy_param):
        """GHZ-like state with controlled entanglement."""
        a = entropy_param * np.pi / 4
        dim = 2 ** self.n_bulk
        bulk = np.zeros(dim, dtype=complex)
        bulk[0] = np.cos(a)       # |00...0>
        bulk[dim - 1] = np.sin(a)  # |11...1>
        return self.encode(bulk)

    def get_boundary_regions(self):
        """Return boundary regions organized by tensor origin."""
        nc = self.n_boundary_center
        nl = self.n_boundary_per_leaf
        regions = {
            'center': list(range(nc)),
        }
        for i in range(self.n_leaves):
            start = nc + i * nl
            regions[f'leaf{i}'] = list(range(start, start + nl))

        # Composite regions
        half = self.n_boundary // 2
        regions['left_half'] = list(range(half))
        regions['right_half'] = list(range(half, self.n_boundary))
        regions['big_left'] = list(range(half + 2))
        regions['small_left'] = list(range(half - 2))

        # Entanglement wedge regions: for tree, each leaf's boundary
        # should reconstruct that leaf's bulk qubit
        # The center boundary + one leaf should reconstruct center bulk + that leaf bulk
        for i in range(self.n_leaves):
            start = nc + i * nl
            regions[f'center+leaf{i}'] = list(range(nc)) + list(range(start, start + nl))

        return regions

    def get_entanglement_wedge_map(self):
        """
        Return which bulk qubits should be recoverable from which boundary regions.

        For the tree code:
        - leaf_i boundary (4 qubits) -> can reconstruct bulk_i (leaf's bulk qubit)
        - center boundary alone (2 qubits) -> might reconstruct bulk_0 (center)
        - >half boundary -> should reconstruct ALL bulk qubits
        """
        nc = self.n_boundary_center
        nl = self.n_boundary_per_leaf
        wedge_map = {}

        # Each leaf region should reconstruct its own bulk qubit
        for i in range(self.n_leaves):
            start = nc + i * nl
            region = list(range(start, start + nl))
            wedge_map[f'leaf{i}_bdry'] = {
                'boundary': region,
                'recoverable_bulk': [i + 1],  # bulk qubit i+1 (0-indexed, 0=center)
                'n_recoverable': 1,
            }

        # > half boundary should recover everything
        half = self.n_boundary // 2
        wedge_map['big_region'] = {
            'boundary': list(range(half + 2)),
            'recoverable_bulk': list(range(self.n_bulk)),
            'n_recoverable': self.n_bulk,
        }

        return wedge_map


# ═══════════════════════════════════════════════════════════════════════════
# Proto-area entropy measurement
# ═══════════════════════════════════════════════════════════════════════════

def compute_proto_area(state, n_boundary, region, n_recoverable_bulk):
    """
    Proto-area = S(A) - S_bulk_recoverable(A)

    S(A) = von Neumann entropy of boundary region A
    S_bulk_recoverable(A) = entropy of the bulk qubits recoverable from A

    For a perfect code:
    - If |A| > n_boundary/2, all bulk qubits are recoverable
    - If |A| corresponds to one tensor's boundary, that tensor's bulk is recoverable

    The recoverable bulk entropy is bounded by min(|A|, n_recoverable_bulk) bits.
    In practice, for our GHZ-like bulk states with entropy_param:
      S_bulk_recoverable = n_recoverable_bulk * h(entropy_param)
    where h is the binary entropy. But we compute it from the state directly.

    Actually, proto-area = S(A) - S_bulk(A) where S_bulk(A) is the entropy
    of the bulk subsystem in the entanglement wedge of A. Since we don't
    have direct access to bulk degrees of freedom in the encoded state,
    we estimate it as:

    For a holographic code: S(A) = proto_area(A) + S_bulk(wedge(A))
    So: proto_area(A) = S(A) - S_bulk(wedge(A))

    For pure bulk state: S_bulk = 0 (product) to ~1 bit (Bell pair) per qubit.
    We pass n_recoverable_bulk as the number of bulk qubits in the wedge,
    and compute the actual bulk entropy separately.
    """
    S_A = von_neumann_entropy(state, n_boundary, region)
    return S_A


def compute_bulk_entropy(bulk_state, n_bulk, subsystem):
    """Compute entropy of a subsystem of the bulk state."""
    if len(subsystem) == 0 or len(subsystem) == n_bulk:
        return 0.0
    return von_neumann_entropy(bulk_state, n_bulk, subsystem)


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_chain_experiment(n_trials=20, circuit_depth=12):
    """Run the 3-tensor chain experiment (3 bulk, 11 boundary)."""
    print("\n" + "=" * 72)
    print("PART 1: 3-Tensor Chain (3 bulk -> 11 boundary)")
    print("=" * 72)

    rng = np.random.default_rng(42)
    magic_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    bulk_entropy_params = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # ── Page curves ──────────────────────────────────────────────────────
    print(f"\n--- Page curve: S(A) vs |A| for contiguous boundary regions ---")
    print(f"    3 bulk qubits -> 11 boundary qubits, {n_trials} trials averaged")
    print(f"    Bulk state: |000> (zero entropy)")

    bulk_000 = np.zeros(8, dtype=complex)
    bulk_000[0] = 1.0

    page_curves = {}
    sizes = list(range(1, 12))
    header = f"{'magic':>7} |" + "".join(f" {s:>5}" for s in sizes)
    print(f"\n{header}")
    print("-" * len(header))

    for magic in magic_levels:
        all_curves = []
        for trial in range(n_trials):
            seed = int(rng.integers(10**9))
            code = ThreeTensorChain(magic_level=magic,
                                     rng=np.random.default_rng(seed),
                                     circuit_depth=circuit_depth)
            state = code.encode(bulk_000)
            curve = []
            for sz in sizes:
                region = list(range(sz))
                S = von_neumann_entropy(state, 11, region)
                curve.append(S)
            all_curves.append(curve)

        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        page_curves[magic] = (mean_curve, std_curve)

        row = f"  {magic:>5.2f} |" + "".join(f" {s:5.2f}" for s in mean_curve)
        print(row)

    # ── Proto-area vs bulk entropy ───────────────────────────────────────
    print(f"\n--- Proto-area (S_half) vs bulk entropy at each magic level ---")
    print(f"    Preskill: proto-area should respond to bulk entropy ONLY when magic > 0")

    regions_to_test = {
        'left_half': list(range(0, 6)),      # slightly > half
        'right_half': list(range(6, 11)),     # slightly < half
        'tensor_A': list(range(0, 4)),        # one tensor's boundary
        'tensor_AB': list(range(0, 7)),       # two tensors' boundary (> half)
    }

    for region_name, region in regions_to_test.items():
        print(f"\n  Region '{region_name}' (qubits {region}, size {len(region)}/{11}):")
        header = f"  {'magic':>7} |" + "".join(f" bk={b:.1f}" for b in bulk_entropy_params)
        print(f"  {header}")
        print("  " + "-" * len(header))

        region_data = {}
        for magic in magic_levels:
            row_data = []
            for bulk_e in bulk_entropy_params:
                s_vals = []
                for trial in range(n_trials):
                    seed = int(rng.integers(10**9))
                    code = ThreeTensorChain(magic_level=magic,
                                             rng=np.random.default_rng(seed),
                                             circuit_depth=circuit_depth)
                    state = code.encode_bulk_with_entropy(bulk_e)
                    S = von_neumann_entropy(state, 11, region)
                    s_vals.append(S)
                row_data.append(np.mean(s_vals))
            region_data[magic] = row_data

            row = f"  {magic:>7.2f} |" + "".join(f" {v:5.3f}" for v in row_data)
            print(row)

        # Sensitivity analysis
        print(f"  Sensitivity (dS/d(bulk_entropy)):")
        for magic in magic_levels:
            x = np.array(bulk_entropy_params)
            y = np.array(region_data[magic])
            slope = np.polyfit(x, y, 1)[0]
            spread = y[-1] - y[0]
            print(f"    magic={magic:.2f}: slope={slope:+.4f}, spread={spread:+.4f}")

    return page_curves


def run_tree_experiment(n_leaves, n_trials=15, circuit_depth=12):
    """Run the tree code experiment."""
    code_tmp = TreeCode(n_leaves=n_leaves, magic_level=0.0)
    n_bulk = code_tmp.n_bulk
    n_bdry = code_tmp.n_boundary

    print(f"\n{'=' * 72}")
    print(f"PART 2: Tree Code ({n_leaves} leaves, {n_bulk} bulk -> {n_bdry} boundary)")
    print(f"{'=' * 72}")

    rng = np.random.default_rng(123)
    magic_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    bulk_entropy_params = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # ── Timing check ─────────────────────────────────────────────────────
    t0 = time.time()
    test_code = TreeCode(n_leaves=n_leaves, magic_level=0.5,
                          rng=np.random.default_rng(0), circuit_depth=circuit_depth)
    test_bulk = np.zeros(2 ** n_bulk, dtype=complex)
    test_bulk[0] = 1.0
    test_state = test_code.encode(test_bulk)
    t1 = time.time()
    print(f"  Single encode time: {t1-t0:.2f}s, state dim: {len(test_state)}")
    if t1 - t0 > 5.0:
        print(f"  WARNING: encoding is slow. Reducing n_trials.")
        n_trials = max(5, n_trials // 3)

    # ── Page curves ──────────────────────────────────────────────────────
    print(f"\n--- Page curve ---")
    bulk_zero = np.zeros(2 ** n_bulk, dtype=complex)
    bulk_zero[0] = 1.0

    page_curves = {}
    sizes = list(range(1, n_bdry + 1))

    # Only print a subset of sizes to keep output manageable
    print_sizes = [1, 2, 3, n_bdry // 4, n_bdry // 2, 3 * n_bdry // 4, n_bdry - 1, n_bdry]
    print_sizes = sorted(set(s for s in print_sizes if 1 <= s <= n_bdry))
    header = f"{'magic':>7} |" + "".join(f" |A|={s:<2}" for s in print_sizes)
    print(f"\n{header}")
    print("-" * len(header))

    for magic in magic_levels:
        all_curves = []
        for trial in range(n_trials):
            seed = int(rng.integers(10**9))
            code = TreeCode(n_leaves=n_leaves, magic_level=magic,
                            rng=np.random.default_rng(seed),
                            circuit_depth=circuit_depth)
            state = code.encode(bulk_zero)
            curve = []
            for sz in sizes:
                region = list(range(sz))
                S = von_neumann_entropy(state, n_bdry, region)
                curve.append(S)
            all_curves.append(curve)

        mean_curve = np.mean(all_curves, axis=0)
        page_curves[magic] = mean_curve

        row = f"  {magic:>5.2f} |"
        for s in print_sizes:
            row += f" {mean_curve[s-1]:5.2f} "
        print(row)

    # ── Proto-area vs bulk entropy ───────────────────────────────────────
    print(f"\n--- Proto-area vs bulk entropy ---")

    regions = code_tmp.get_boundary_regions()
    test_regions = {
        'left_half': regions['left_half'],
        'big_left': regions['big_left'],
    }
    # Add leaf regions
    for i in range(n_leaves):
        test_regions[f'leaf{i}'] = regions[f'leaf{i}']
    if 'center+leaf0' in regions:
        test_regions['center+leaf0'] = regions['center+leaf0']

    proto_area_results = {}
    for region_name, region in test_regions.items():
        print(f"\n  Region '{region_name}' (qubits {region[:6]}{'...' if len(region)>6 else ''}, "
              f"size {len(region)}/{n_bdry}):")
        header = f"  {'magic':>7} |" + "".join(f" bk={b:.1f}" for b in bulk_entropy_params)
        print(f"  {header}")
        print("  " + "-" * len(header))

        region_data = {}
        for magic in magic_levels:
            row_data = []
            for bulk_e in bulk_entropy_params:
                s_vals = []
                for trial in range(n_trials):
                    seed = int(rng.integers(10**9))
                    code = TreeCode(n_leaves=n_leaves, magic_level=magic,
                                    rng=np.random.default_rng(seed),
                                    circuit_depth=circuit_depth)
                    state = code.encode_bulk_with_entropy(bulk_e)
                    S = von_neumann_entropy(state, n_bdry, region)
                    s_vals.append(S)
                row_data.append(np.mean(s_vals))
            region_data[magic] = row_data

            row = f"  {magic:>7.2f} |" + "".join(f" {v:5.3f}" for v in row_data)
            print(row)

        # Sensitivity
        print(f"  Sensitivity:")
        for magic in magic_levels:
            x = np.array(bulk_entropy_params)
            y = np.array(region_data[magic])
            slope = np.polyfit(x, y, 1)[0]
            print(f"    magic={magic:.2f}: slope={slope:+.4f}")

        proto_area_results[region_name] = region_data

    return page_curves, proto_area_results


def run_entanglement_wedge_test(n_leaves, n_trials=15, circuit_depth=12):
    """
    Test entanglement wedge structure: does magic make the wedge state-dependent?

    For a stabilizer code, the entanglement wedge of a boundary region is
    FIXED by the code structure, regardless of the bulk state.

    For a magic-enriched code, the entanglement wedge should DEPEND on the
    bulk state (geometric backreaction).

    We test this by measuring mutual information between different boundary
    regions and varying the bulk state. If I(leaf_i : leaf_j) changes with
    bulk entropy at magic>0 but NOT at magic=0, that's evidence for
    state-dependent geometry.
    """
    code_tmp = TreeCode(n_leaves=n_leaves, magic_level=0.0)
    n_bulk = code_tmp.n_bulk
    n_bdry = code_tmp.n_boundary

    print(f"\n{'=' * 72}")
    print(f"PART 3: Entanglement Wedge Structure (tree, {n_leaves} leaves)")
    print(f"{'=' * 72}")

    rng = np.random.default_rng(777)
    regions = code_tmp.get_boundary_regions()

    magic_levels = [0.0, 0.5, 1.0]
    bulk_entropy_params = [0.0, 0.5, 1.0]

    # Test mutual information between leaf boundaries
    print(f"\n--- Mutual information I(leaf_i : leaf_j) vs bulk entropy ---")
    print(f"    Preskill: at magic=0, I should be INDEPENDENT of bulk state")
    print(f"    At magic>0, I should DEPEND on bulk state (wedge shifts)")

    if n_leaves >= 2:
        leaf_pairs = [(0, 1)]
        if n_leaves >= 3:
            leaf_pairs.append((0, 2))
            leaf_pairs.append((1, 2))

        for li, lj in leaf_pairs:
            region_i = regions[f'leaf{li}']
            region_j = regions[f'leaf{lj}']
            print(f"\n  I(leaf{li} : leaf{lj}):")
            header = f"  {'magic':>7} |" + "".join(f" bk={b:.1f}" for b in bulk_entropy_params)
            print(f"  {header}")
            print("  " + "-" * len(header))

            for magic in magic_levels:
                row_data = []
                for bulk_e in bulk_entropy_params:
                    mi_vals = []
                    for trial in range(n_trials):
                        seed = int(rng.integers(10**9))
                        code = TreeCode(n_leaves=n_leaves, magic_level=magic,
                                        rng=np.random.default_rng(seed),
                                        circuit_depth=circuit_depth)
                        state = code.encode_bulk_with_entropy(bulk_e)
                        mi = mutual_information(state, n_bdry, region_i, region_j)
                        mi_vals.append(mi)
                    row_data.append(np.mean(mi_vals))

                row = f"  {magic:>7.2f} |" + "".join(f" {v:5.3f}" for v in row_data)
                print(row)

    # Test: does the "recoverable information" from a boundary region change?
    # Use conditional entropy: H(bulk | boundary_region) should be lower when
    # more bulk is in the entanglement wedge.
    # We approximate this by measuring S(region) - S(complement) which relates
    # to how much of the bulk is "seen" by the region.
    print(f"\n--- Asymmetry: S(left_half) - S(right_half) vs bulk entropy ---")
    print(f"    For a symmetric code at magic=0, this should be ~0 regardless of bulk")
    print(f"    At magic>0, asymmetry may become bulk-state-dependent")

    left = regions['left_half']
    right = regions['right_half']

    header = f"  {'magic':>7} |" + "".join(f" bk={b:.1f}" for b in bulk_entropy_params)
    print(f"  {header}")
    print("  " + "-" * len(header))

    for magic in magic_levels:
        row_data = []
        for bulk_e in bulk_entropy_params:
            asym_vals = []
            for trial in range(n_trials):
                seed = int(rng.integers(10**9))
                code = TreeCode(n_leaves=n_leaves, magic_level=magic,
                                rng=np.random.default_rng(seed),
                                circuit_depth=circuit_depth)
                state = code.encode_bulk_with_entropy(bulk_e)
                S_left = von_neumann_entropy(state, n_bdry, left)
                S_right = von_neumann_entropy(state, n_bdry, right)
                asym_vals.append(S_left - S_right)
            row_data.append(np.mean(asym_vals))

        row = f"  {magic:>7.2f} |" + "".join(f" {v:+6.3f}" for v in row_data)
        print(row)


def compute_proper_proto_area(n_leaves, n_trials=15, circuit_depth=12):
    """
    Compute the PROPER proto-area: S(A) - S_bulk(wedge(A)).

    This is the key test of the Preskill prediction.

    For boundary region A, the entanglement wedge contains certain bulk qubits.
    The proto-area is:
        proto_area(A) = S(A) - S_bulk(wedge(A))

    where S_bulk(wedge(A)) is computed directly from the bulk state.

    At magic=0: proto-area should be CONSTANT regardless of bulk state
    (the stabilizer code has fixed geometry).
    At magic>0: proto-area should INCREASE with bulk entropy
    (backreaction makes geometry respond to matter).
    """
    code_tmp = TreeCode(n_leaves=n_leaves, magic_level=0.0)
    n_bulk = code_tmp.n_bulk
    n_bdry = code_tmp.n_boundary

    print(f"\n{'=' * 72}")
    print(f"PART 4: Proper Proto-Area = S(A) - S_bulk(wedge(A))")
    print(f"{'=' * 72}")
    print(f"  This is the key Preskill test.")
    print(f"  Prediction: proto-area CONSTANT at magic=0, INCREASING at magic>0")

    rng = np.random.default_rng(999)
    magic_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    bulk_entropy_params = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # For each boundary region, define its entanglement wedge
    # (which bulk qubits are recoverable from that region)
    regions = code_tmp.get_boundary_regions()

    # Region: "big_left" (> half boundary) -> entire bulk is recoverable
    # For the tree: left_half might only get some of the bulk
    test_cases = []

    # > half boundary: full bulk recovery
    test_cases.append({
        'name': 'big_left (>half, full bulk recovery)',
        'boundary': regions['big_left'],
        'wedge_bulk': list(range(n_bulk)),  # all bulk qubits
    })

    # Half boundary
    test_cases.append({
        'name': 'left_half (half, partial bulk)',
        'boundary': regions['left_half'],
        'wedge_bulk': list(range(n_bulk // 2 + 1)),  # roughly half the bulk
    })

    # Single leaf: only that leaf's bulk qubit
    for i in range(min(2, n_leaves)):
        test_cases.append({
            'name': f'leaf{i} (single tensor, 1 bulk qubit)',
            'boundary': regions[f'leaf{i}'],
            'wedge_bulk': [i + 1],  # leaf bulk qubit
        })

    for tc in test_cases:
        region = tc['boundary']
        wedge = tc['wedge_bulk']
        name = tc['name']

        print(f"\n  --- {name} ---")
        print(f"      Boundary region: {len(region)} qubits, Wedge: bulk qubits {wedge}")

        header = f"  {'magic':>7} |" + "".join(f" bk={b:.1f}" for b in bulk_entropy_params)
        print(f"\n  S(A) [boundary entropy]:")
        print(f"  {header}")
        print("  " + "-" * len(header))

        sa_data = {}
        pa_data = {}

        for magic in magic_levels:
            sa_row = []
            pa_row = []
            for bulk_e in bulk_entropy_params:
                sa_vals = []
                pa_vals = []
                for trial in range(n_trials):
                    seed = int(rng.integers(10**9))
                    code = TreeCode(n_leaves=n_leaves, magic_level=magic,
                                    rng=np.random.default_rng(seed),
                                    circuit_depth=circuit_depth)
                    state = code.encode_bulk_with_entropy(bulk_e)
                    S_A = von_neumann_entropy(state, n_bdry, region)

                    # Compute bulk entropy of the wedge
                    a = bulk_e * np.pi / 4
                    bulk_vec = np.zeros(2 ** n_bulk, dtype=complex)
                    bulk_vec[0] = np.cos(a)
                    bulk_vec[2 ** n_bulk - 1] = np.sin(a)
                    S_bulk_wedge = von_neumann_entropy(bulk_vec, n_bulk, wedge)

                    proto_area = S_A - S_bulk_wedge
                    sa_vals.append(S_A)
                    pa_vals.append(proto_area)

                sa_row.append(np.mean(sa_vals))
                pa_row.append(np.mean(pa_vals))

            sa_data[magic] = sa_row
            pa_data[magic] = pa_row

            row = f"  {magic:>7.2f} |" + "".join(f" {v:5.3f}" for v in sa_row)
            print(row)

        print(f"\n  Proto-area = S(A) - S_bulk(wedge):")
        print(f"  {header}")
        print("  " + "-" * len(header))

        for magic in magic_levels:
            row = f"  {magic:>7.2f} |" + "".join(f" {v:5.3f}" for v in pa_data[magic])
            print(row)

        print(f"\n  Proto-area sensitivity (slope):")
        for magic in magic_levels:
            x = np.array(bulk_entropy_params)
            y = np.array(pa_data[magic])
            slope = np.polyfit(x, y, 1)[0]
            spread = y[-1] - y[0]
            marker = ""
            if magic == 0 and abs(slope) < 0.05:
                marker = " <-- FIXED geometry (Preskill: expected)"
            elif magic == 0 and abs(slope) > 0.1:
                marker = " <-- RESPONSIVE at magic=0 (contradicts Preskill)"
            elif magic > 0 and slope > 0.05:
                marker = " <-- RESPONSIVE (Preskill: expected for magic>0)"
            elif magic > 0 and abs(slope) < 0.05:
                marker = " <-- FIXED at magic>0 (contradicts Preskill)"
            print(f"    magic={magic:.2f}: slope={slope:+.4f}, "
                  f"spread={spread:+.4f}{marker}")


def main():
    t_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 15: Large-Scale Holographic Codes")
    print("Testing Preskill's Prediction at Larger System Sizes")
    print("=" * 72)
    print()
    print("Preskill et al. (arXiv:2603.13475) prediction:")
    print("  - At magic=0 (stabilizer): geometry is FIXED, proto-area constant")
    print("  - At magic>0: geometry is DYNAMICAL, proto-area responds to bulk entropy")
    print()
    print("exp11 found the OPPOSITE at 8 qubits (2 bulk, 8 boundary).")
    print("Hypothesis: system was too small. Testing at 11-14 boundary qubits.")

    # ── Part 1: 3-tensor chain ───────────────────────────────────────────
    print(f"\nStarting 3-tensor chain (3 bulk, 11 boundary)...")
    t0 = time.time()
    chain_page = run_chain_experiment(n_trials=15, circuit_depth=12)
    t1 = time.time()
    print(f"\n  Chain experiment completed in {t1-t0:.1f}s")

    # ── Part 2: Tree code with 2 leaves ──────────────────────────────────
    # 3 bulk, 11 boundary (manageable)
    print(f"\nStarting tree code (2 leaves: 3 bulk, 11 boundary)...")
    t0 = time.time()
    tree_page_2, tree_proto_2 = run_tree_experiment(n_leaves=2, n_trials=15,
                                                      circuit_depth=12)
    t1 = time.time()
    print(f"\n  Tree (2-leaf) completed in {t1-t0:.1f}s")

    # ── Part 3: Entanglement wedge test ──────────────────────────────────
    print(f"\nStarting entanglement wedge test...")
    t0 = time.time()
    run_entanglement_wedge_test(n_leaves=2, n_trials=10, circuit_depth=12)
    t1 = time.time()
    print(f"\n  Wedge test completed in {t1-t0:.1f}s")

    # ── Part 4: Proper proto-area ────────────────────────────────────────
    print(f"\nStarting proper proto-area computation...")
    t0 = time.time()
    compute_proper_proto_area(n_leaves=2, n_trials=15, circuit_depth=12)
    t1 = time.time()
    print(f"\n  Proto-area computation completed in {t1-t0:.1f}s")

    # ── Part 5: Try 3-leaf tree (4 bulk, 14 boundary) if time allows ────
    elapsed = time.time() - t_start
    if elapsed < 300:  # less than 5 minutes so far
        print(f"\n  Elapsed: {elapsed:.0f}s. Attempting 3-leaf tree (4 bulk, 14 boundary)...")
        t0 = time.time()
        try:
            tree_page_3, tree_proto_3 = run_tree_experiment(n_leaves=3, n_trials=8,
                                                              circuit_depth=10)
            t1 = time.time()
            print(f"\n  Tree (3-leaf) completed in {t1-t0:.1f}s")

            compute_proper_proto_area(n_leaves=3, n_trials=8, circuit_depth=10)
        except Exception as e:
            print(f"\n  3-leaf tree failed: {e}")
    else:
        print(f"\n  Elapsed: {elapsed:.0f}s. Skipping 3-leaf tree to save time.")

    # ── Summary ──────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"EXPERIMENT 15 SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  Total runtime: {t_total:.1f}s")
    print(f"\n  System sizes tested:")
    print(f"    - 3-tensor chain: 3 bulk -> 11 boundary (2048-dim Hilbert space)")
    print(f"    - 2-leaf tree: 3 bulk -> 11 boundary (2048-dim)")
    print(f"    - 3-leaf tree: 4 bulk -> 14 boundary (16384-dim) [if attempted]")
    print(f"\n  Key question: Does the Preskill prediction emerge at larger sizes?")
    print(f"    - At magic=0: is proto-area CONSTANT (fixed geometry)?")
    print(f"    - At magic>0: does proto-area INCREASE with bulk entropy (backreaction)?")
    print(f"\n  Compare to exp11 (2 bulk, 8 boundary):")
    print(f"    - exp11 found magic=0 was MOST sensitive (opposite of Preskill)")
    print(f"    - If exp15 shows the SAME pattern at 11-14 qubits, the effect is robust")
    print(f"    - If exp15 shows the PREDICTED pattern, the Preskill effect is real")
    print(f"      but requires >8 qubits to manifest")
    print(f"\n{'=' * 72}")


if __name__ == '__main__':
    main()
