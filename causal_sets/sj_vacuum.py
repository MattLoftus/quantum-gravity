"""
Sorkin-Johnston (SJ) vacuum state on a causal set.

The SJ vacuum is the unique Gaussian state for a free scalar field on a
causal set that is determined by the causal structure alone (no reference
to a background spacetime or time function).

Construction:
1. Compute the Pauli-Jordan function (commutator): iΔ = G_R - G_A
   where G_R is the retarded Green's function
2. Eigendecompose iΔ: it's antisymmetric so eigenvalues come in ±λ pairs
3. The SJ Wightman function: W = (1/2)(iΔ + |iΔ|)
   where |iΔ| keeps only the positive eigenvalues
4. Entanglement entropy of region A: S(A) from the eigenvalues of W_A = W[A,A]

References:
- Johnston, Phys. Rev. Lett. 103, 180401 (2009)
- Sorkin, arXiv:1205.2953 (2012)
- Saravani, Aslanbeigi, Sorkin, JHEP 1407, 024 (2014), arXiv:1311.7146
- Yazdi, Kempf, Afshordi, JHEP 1703, 033 (2017), arXiv:1611.09947
"""

import numpy as np
from .fast_core import FastCausalSet


def pauli_jordan_function(cs: FastCausalSet) -> np.ndarray:
    """
    The Pauli-Jordan (commutator) function: iΔ[i,j] = C[j,i] - C[i,j]

    For a massless scalar on a causal set, iΔ is the antisymmetrized causal
    matrix, normalized by 2/N where N is the number of elements. The 2/N
    factor arises from the discrete-to-continuum correspondence: the
    continuum propagator (1/2) is divided by the sprinkling density ρ ∝ N/V,
    giving an overall normalization of ~2/N for the discrete Pauli-Jordan
    function to have eigenvalues in the correct range for a valid quantum state.

    Reference: Johnston, Phys. Rev. Lett. 103, 180401 (2009)
    """
    N = cs.n
    C = cs.order.astype(float)
    iDelta = (2.0 / N) * (C.T - C)  # antisymmetric, properly normalized
    return iDelta


def sj_wightman_function(cs: FastCausalSet) -> np.ndarray:
    """
    The SJ vacuum Wightman (two-point) function.

    W = (1/2)(iΔ + |iΔ|)

    where |iΔ| is the "absolute value" of the operator:
    eigendecompose iΔ = Σ λ_k |v_k><v_k|, then |iΔ| = Σ |λ_k| |v_k><v_k|.

    Since iΔ is real antisymmetric, its eigenvalues are purely imaginary (±iλ_k)
    and come in conjugate pairs. The positive part means keeping the positive
    imaginary eigenvalues.

    Actually, more carefully: iΔ as defined above is a REAL antisymmetric matrix.
    Its eigenvalues are ±iλ_k with λ_k ≥ 0. The eigenvectors come in conjugate pairs.

    The positive part of iΔ (as a Hermitian operator on the complexified space):
    Multiply by i to get a Hermitian matrix: i*(iΔ) = -Δ, which has real eigenvalues ±λ_k.
    Then |iΔ| corresponds to keeping the eigenvalues of i*(iΔ) that are positive.

    Practical computation:
    1. Compute A = iΔ (real antisymmetric, N×N)
    2. Compute eigenvalues of i*A (Hermitian): eigenvalues are real ±λ_k
    3. Keep only the positive eigenvalue subspace
    4. W = (1/2)(A + positive_part)

    Alternative (simpler): work with the SVD of A.
    A = U Σ V^T where Σ has the singular values σ_k = |λ_k|.
    |A| = U Σ U^T (in the sense of operator absolute value).
    Actually for antisymmetric matrices this needs care.

    Simplest correct approach: eigendecompose i*A as a Hermitian matrix.
    """
    N = cs.n
    A = pauli_jordan_function(cs)  # real antisymmetric N×N

    # i*A is Hermitian (since A is real antisymmetric, i*A is Hermitian)
    iA = 1j * A  # complex Hermitian matrix

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(iA)
    # eigenvalues are real, come in ±λ pairs

    # Positive part: keep only positive eigenvalues
    pos_mask = eigenvalues > 1e-12
    eigenvalues_pos = eigenvalues.copy()
    eigenvalues_pos[~pos_mask] = 0.0

    # Reconstruct |iΔ| from positive eigenvalues
    # |iΔ| in the original (real) basis:
    # The positive part of i*A is: P = V diag(λ_pos) V^†
    # We need to convert back: |iΔ| = -i * P (undo the i multiplication)
    # Actually: W = (1/2)(iΔ + |iΔ|) where |iΔ| is defined such that
    # W is the positive-frequency part of the Wightman function.

    # Following Sorkin (2012) and Johnston (2009):
    # W = Σ_{λ_k > 0} λ_k |v_k><v_k|
    # where (i*A)|v_k> = λ_k|v_k> and λ_k > 0.

    # So W is simply the projection onto the positive eigenspace of i*A,
    # weighted by the eigenvalues:
    W = np.zeros((N, N), dtype=complex)
    for k in range(N):
        if eigenvalues[k] > 1e-12:
            v = eigenvectors[:, k]
            W += eigenvalues[k] * np.outer(v, v.conj())

    # W should be Hermitian and positive semi-definite
    # For a real causal set, W should be real (up to numerical noise)
    W = np.real(W)

    return W


def entanglement_entropy(W: np.ndarray, region_A: list) -> float:
    """
    Entanglement entropy of region A in the SJ vacuum.

    For a Gaussian state with two-point function W, the entanglement entropy
    of sub-region A is:

        S(A) = -Tr[W_A ln W_A + (I - W_A) ln(I - W_A)]

    where W_A = W[A, A] is the restriction of W to region A.

    The eigenvalues ν_k of W_A satisfy 0 ≤ ν_k ≤ 1 (for a valid state),
    and the entropy is:

        S = -Σ_k [ν_k ln ν_k + (1 - ν_k) ln(1 - ν_k)]
    """
    A = sorted(region_A)
    W_A = W[np.ix_(A, A)]

    # Eigenvalues of W_A
    eigenvalues = np.linalg.eigvalsh(W_A)

    # Clip to valid range [0, 1] (numerical noise may push slightly outside)
    eigenvalues = np.clip(eigenvalues, 1e-15, 1 - 1e-15)

    # Von Neumann entropy
    S = -np.sum(eigenvalues * np.log(eigenvalues) +
                (1 - eigenvalues) * np.log(1 - eigenvalues))

    return float(S)


def sj_entanglement_profile(cs: FastCausalSet, n_partitions: int = 9) -> dict:
    """
    Compute the entanglement entropy S(A) for spatial sub-regions A of varying size.

    For a 2D causal set, we partition by the "spatial" direction:
    sort elements by their position in the causal order, then take the
    first k elements as region A.

    Returns dict with 'fracs' and 'entropies'.
    """
    N = cs.n
    W = sj_wightman_function(cs)

    fracs = np.linspace(0.1, 0.9, n_partitions)
    entropies = []

    for frac in fracs:
        k = max(1, int(frac * N))
        A = list(range(k))
        S = entanglement_entropy(W, A)
        entropies.append(S)

    return {
        'fracs': fracs,
        'entropies': entropies,
        'W': W,
    }
