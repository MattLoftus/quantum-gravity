"""
Experiment 36: Three tests to push B3 from 7 to 8

1. Sorkin-Yazdi truncation → does c = 1?
2. Monogamy of mutual information I₃ ≤ 0 (holographic signature)
3. 4D SJ entanglement: S ~ N^{1/2} (dimensional crossover)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder
from causal_sets.fast_core import FastCausalSet


def sj_wightman_truncated(cs, truncation=None):
    """
    SJ Wightman function with optional Sorkin-Yazdi mode truncation.

    truncation=None: keep all positive eigenvalues (our current approach)
    truncation='SY': cut eigenvalues below sqrt(N)/(4*pi)
    truncation=float: cut eigenvalues below this value
    """
    N = cs.n
    C = cs.order.astype(float)
    A = 0.5 * (C.T - C)  # antisymmetric Pauli-Jordan (no 2/N factor — use raw)

    H = 1j * A
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues = eigenvalues.real

    if truncation == 'SY':
        # Sorkin-Yazdi: cut below sqrt(N)/(4*pi)
        cutoff = np.sqrt(N) / (4 * np.pi)
        mask = eigenvalues > cutoff
    elif truncation is not None:
        mask = eigenvalues > truncation
    else:
        mask = eigenvalues > 1e-10

    if not np.any(mask):
        return np.zeros((N, N))

    W = (eigenvectors[:, mask] @ np.diag(eigenvalues[mask]) @
         eigenvectors[:, mask].conj().T).real

    # Normalize so eigenvalues of W are in [0, 1]
    w_eigs = np.linalg.eigvalsh(W)
    max_eig = w_eigs.max()
    if max_eig > 1.0:
        W = W / max_eig

    return W


def entanglement_entropy(W, region_A):
    A = sorted(region_A)
    W_A = W[np.ix_(A, A)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def mutual_information(W, A, B):
    """I(A:B) = S(A) + S(B) - S(A∪B)"""
    S_A = entanglement_entropy(W, A)
    S_B = entanglement_entropy(W, B)
    S_AB = entanglement_entropy(W, list(set(A) | set(B)))
    return S_A + S_B - S_AB


def tripartite_information(W, A, B, C):
    """I₃(A:B:C) = I(A:B) + I(A:C) - I(A:BC)
    Holographic: I₃ ≤ 0 (monogamy of mutual information)"""
    I_AB = mutual_information(W, A, B)
    I_AC = mutual_information(W, A, C)
    I_ABC = mutual_information(W, A, list(set(B) | set(C)))
    return I_AB + I_AC - I_ABC


def main():
    rng = np.random.default_rng(42)

    print("=" * 75)
    print("EXPERIMENT 36: Three Holographic Tests")
    print("=" * 75)

    # ================================================================
    # TEST 1: Sorkin-Yazdi truncation → c = 1?
    # ================================================================
    print("\n" + "=" * 75)
    print("TEST 1: Central charge with Sorkin-Yazdi truncation")
    print("Prediction: c = 1 for a single free scalar (S = (c/3) ln N)")
    print("=" * 75)

    print(f"\n  {'N':>5} {'S_raw':>8} {'c_raw':>6} {'S_SY':>8} {'c_SY':>6} {'S_2/N':>8} {'c_2/N':>6}")
    print("-" * 55)

    for N in [15, 20, 30, 40, 50, 60, 70]:
        S_raw_vals, S_SY_vals, S_2N_vals = [], [], []

        for _ in range(15):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            A = list(range(N // 2))

            # Raw (no truncation, no normalization)
            W_raw = sj_wightman_truncated(cs, truncation=None)
            S_raw_vals.append(entanglement_entropy(W_raw, A))

            # Sorkin-Yazdi truncation
            W_SY = sj_wightman_truncated(cs, truncation='SY')
            S_SY_vals.append(entanglement_entropy(W_SY, A))

            # Our 2/N normalization (from sj_vacuum.py)
            C = cs.order.astype(float)
            iDelta = (2.0 / N) * (C.T - C)
            H = 1j * iDelta
            evals, evecs = np.linalg.eigh(H)
            evals = evals.real
            pos = evals > 1e-12
            W_2N = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
            S_2N_vals.append(entanglement_entropy(W_2N, A))

        S_raw = np.mean(S_raw_vals)
        S_SY = np.mean(S_SY_vals)
        S_2N = np.mean(S_2N_vals)
        c_raw = 3 * S_raw / np.log(N)
        c_SY = 3 * S_SY / np.log(N)
        c_2N = 3 * S_2N / np.log(N)

        print(f"  {N:>5} {S_raw:>8.3f} {c_raw:>6.2f} {S_SY:>8.3f} {c_SY:>6.2f} "
              f"{S_2N:>8.3f} {c_2N:>6.2f}")

    print("\n  Target: c_SY → 1.0 as N → ∞")

    # ================================================================
    # TEST 2: Monogamy of mutual information I₃ ≤ 0
    # ================================================================
    print("\n" + "=" * 75)
    print("TEST 2: Monogamy of mutual information (holographic signature)")
    print("Holographic theories: I₃(A:B:C) ≤ 0 for all tripartitions")
    print("=" * 75)

    N = 40

    # Test on random 2-orders (continuum phase)
    print(f"\n  Continuum phase (random 2-orders, N={N}):")
    I3_cont = []
    for _ in range(30):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W = sj_wightman_truncated(cs, truncation='SY')

        # Tripartition into thirds by v-coordinate
        v_order = np.argsort(to.v)
        third = N // 3
        A = list(v_order[:third])
        B = list(v_order[third:2*third])
        C = list(v_order[2*third:])

        I3 = tripartite_information(W, A, B, C)
        I3_cont.append(I3)

    n_neg = sum(1 for x in I3_cont if x <= 0)
    print(f"    I₃ = {np.mean(I3_cont):.4f} ± {np.std(I3_cont):.4f}")
    print(f"    I₃ ≤ 0 in {n_neg}/{len(I3_cont)} realizations ({100*n_neg/len(I3_cont):.0f}%)")
    print(f"    {'MONOGAMOUS (holographic)' if n_neg > len(I3_cont)*0.8 else 'NOT MONOGAMOUS'}")

    # Test on BD crystalline phase
    print(f"\n  Crystalline phase (BD MCMC, β=5βc, N={N}):")
    from causal_sets.two_orders_v2 import mcmc_corrected
    eps = 0.12
    beta_c = 6.64 / (N * eps ** 2)

    res = mcmc_corrected(N, beta=beta_c * 5, eps=eps,
                          n_steps=30000, n_therm=15000,
                          record_every=100, rng=rng)

    I3_cryst = []
    for cs in res['samples'][-30:]:
        W = sj_wightman_truncated(cs, truncation='SY')
        A = list(range(N // 3))
        B = list(range(N // 3, 2 * N // 3))
        C = list(range(2 * N // 3, N))
        I3 = tripartite_information(W, A, B, C)
        I3_cryst.append(I3)

    n_neg_c = sum(1 for x in I3_cryst if x <= 0)
    print(f"    I₃ = {np.mean(I3_cryst):.4f} ± {np.std(I3_cryst):.4f}")
    print(f"    I₃ ≤ 0 in {n_neg_c}/{len(I3_cryst)} realizations ({100*n_neg_c/len(I3_cryst):.0f}%)")
    print(f"    {'MONOGAMOUS (holographic)' if n_neg_c > len(I3_cryst)*0.8 else 'NOT MONOGAMOUS'}")

    # Multiple tripartitions for robustness
    print(f"\n  Multiple tripartitions (continuum, N={N}):")
    all_I3 = []
    for _ in range(30):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W = sj_wightman_truncated(cs, truncation='SY')

        # Several random tripartitions
        for _ in range(5):
            perm = rng.permutation(N)
            A = list(perm[:N//3])
            B = list(perm[N//3:2*N//3])
            C = list(perm[2*N//3:])
            I3 = tripartite_information(W, A, B, C)
            all_I3.append(I3)

    n_neg_all = sum(1 for x in all_I3 if x <= 0)
    print(f"    I₃ = {np.mean(all_I3):.4f} ± {np.std(all_I3):.4f}")
    print(f"    I₃ ≤ 0 in {n_neg_all}/{len(all_I3)} ({100*n_neg_all/len(all_I3):.0f}%)")

    # ================================================================
    # TEST 3: 4D dimensional crossover
    # ================================================================
    print("\n" + "=" * 75)
    print("TEST 3: Dimensional crossover — 2D (ln N) vs 4D (N^{1/2})")
    print("=" * 75)

    from causal_sets.d_orders import DOrder

    print(f"\n  2D (2-orders):")
    print(f"  {'N':>5} {'S(N/2)':>8} {'S/ln(N)':>8} {'S/N^0.5':>8}")
    print("  " + "-" * 35)

    for N in [15, 20, 30, 40, 50]:
        S_vals = []
        for _ in range(10):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            W = sj_wightman_truncated(cs, truncation='SY')
            A = list(range(N // 2))
            S_vals.append(entanglement_entropy(W, A))
        S = np.mean(S_vals)
        print(f"  {N:>5} {S:>8.3f} {S/np.log(N):>8.3f} {S/np.sqrt(N):>8.3f}")

    print(f"\n  4D (4-orders):")
    print(f"  {'N':>5} {'S(N/2)':>8} {'S/ln(N)':>8} {'S/N^0.5':>8}")
    print("  " + "-" * 35)

    for N in [10, 15, 20, 25, 30]:
        S_vals = []
        for _ in range(10):
            do = DOrder(d=4, N=N, rng=rng)
            cs = do.to_causet_fast()
            W = sj_wightman_truncated(cs, truncation='SY')
            A = list(range(N // 2))
            S_vals.append(entanglement_entropy(W, A))
        S = np.mean(S_vals)
        print(f"  {N:>5} {S:>8.3f} {S/np.log(N):>8.3f} {S/np.sqrt(N):>8.3f}")

    print("\n  If 2D: S/ln(N) → constant (area law + log correction for 1+1D CFT)")
    print("  If 4D: S/N^{1/2} → constant (area law for 3+1D)")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
