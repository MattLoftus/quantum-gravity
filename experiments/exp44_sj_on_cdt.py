"""
Experiment 44: SJ Vacuum on CDT — Deep Investigation

Initial finding: CDT has c=1.55, causet has c=3.59. Different universality class.

This experiment:
1. Scale with N — does c converge to 1 for CDT? (free scalar prediction)
2. CDT monogamy test — does CDT satisfy I₃ ≤ 0?
3. CDT spectral gap — gapless or gapped?
4. CDT vs causet ER=EPR — does CDT show the same entanglement-connectivity correlation?
5. Different CDT couplings (lambda2) — does c depend on the cosmological constant?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from cdt.triangulation import mcmc_cdt
from causal_sets.fast_core import FastCausalSet, sprinkle_fast

rng = np.random.default_rng(42)


def cdt_to_causet(volume_profile):
    """Convert CDT volume profile to a causal set.
    Elements at time t precede elements at time t' > t."""
    T = len(volume_profile)
    N = int(np.sum(volume_profile))

    cs = FastCausalSet(N)
    offsets = np.zeros(T + 1, dtype=int)
    for t in range(T):
        offsets[t + 1] = offsets[t] + int(volume_profile[t])

    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            for i1 in range(int(volume_profile[t1])):
                for i2 in range(int(volume_profile[t2])):
                    cs.order[offsets[t1] + i1, offsets[t2] + i2] = True

    return cs


def sj_wightman(cs):
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), np.array([])
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals[pos]


def entanglement_entropy(W, region):
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def mutual_info(W, A, B):
    return (entanglement_entropy(W, A) + entanglement_entropy(W, B)
            - entanglement_entropy(W, list(set(A) | set(B))))


def tripartite_info(W, A, B, C):
    return (mutual_info(W, A, B) + mutual_info(W, A, C)
            - mutual_info(W, A, list(set(B) | set(C))))


def main():
    print("=" * 75)
    print("EXPERIMENT 44: SJ Vacuum on CDT — Deep Investigation")
    print("=" * 75)

    # Phase 1: c_eff scaling with N
    print("\n--- Phase 1: Central charge vs N ---")
    print(f"  {'Source':>12} {'N':>5} {'S(N/2)':>8} {'c_eff':>8}")
    print("-" * 40)

    for N_target in [40, 60, 80, 100, 120]:
        # CDT
        T = max(8, int(np.sqrt(N_target)))
        s_init = max(3, N_target // T)
        samples = mcmc_cdt(T=T, s_init=s_init, lambda2=0.0, mu=0.01,
                            n_steps=10000, target_volume=N_target, rng=rng)

        S_cdt = []
        N_actual_cdt = []
        for vp in samples[-8:]:
            cs = cdt_to_causet(vp.astype(int))
            if cs.n > 200 or cs.n < 10:
                continue
            W, _ = sj_wightman(cs)
            A = list(range(cs.n // 2))
            S_cdt.append(entanglement_entropy(W, A))
            N_actual_cdt.append(cs.n)

        # Causet
        S_cs = []
        for _ in range(8):
            cs, _ = sprinkle_fast(N_target, dim=2, region='diamond', rng=rng)
            W, _ = sj_wightman(cs)
            S_cs.append(entanglement_entropy(W, list(range(N_target // 2))))

        if S_cdt:
            N_avg = np.mean(N_actual_cdt)
            c_cdt = 3 * np.mean(S_cdt) / np.log(N_avg) if N_avg > 1 else 0
            print(f"  {'CDT':>12} {N_avg:>5.0f} {np.mean(S_cdt):>8.3f} {c_cdt:>8.3f}")

        c_cs = 3 * np.mean(S_cs) / np.log(N_target)
        print(f"  {'Causet':>12} {N_target:>5} {np.mean(S_cs):>8.3f} {c_cs:>8.3f}")
        print(flush=True)

    # Phase 2: CDT monogamy test
    print("\n--- Phase 2: Monogamy test on CDT ---")

    T, s_init = 15, 6
    N_target = T * s_init
    samples = mcmc_cdt(T=T, s_init=s_init, lambda2=0.0, mu=0.01,
                        n_steps=10000, target_volume=N_target, rng=rng)

    I3_cdt = []
    I3_causet = []

    for vp in samples[-20:]:
        cs = cdt_to_causet(vp.astype(int))
        N = cs.n
        if N < 12 or N > 150:
            continue
        W, _ = sj_wightman(cs)
        third = N // 3
        A = list(range(third))
        B = list(range(third, 2 * third))
        C = list(range(2 * third, N))
        I3 = tripartite_info(W, A, B, C)
        I3_cdt.append(I3)

    for _ in range(20):
        from causal_sets.two_orders import TwoOrder
        to = TwoOrder(N_target, rng=rng)
        cs = to.to_causet()
        W, _ = sj_wightman(cs)
        third = N_target // 3
        A = list(range(third))
        B = list(range(third, 2 * third))
        C = list(range(2 * third, N_target))
        I3 = tripartite_info(W, A, B, C)
        I3_causet.append(I3)

    n_neg_cdt = sum(1 for x in I3_cdt if x <= 0)
    n_neg_cs = sum(1 for x in I3_causet if x <= 0)

    print(f"  CDT:    I₃ = {np.mean(I3_cdt):.4f} ± {np.std(I3_cdt):.4f}, "
          f"P(I₃≤0) = {100 * n_neg_cdt / max(1, len(I3_cdt)):.0f}%")
    print(f"  Causet: I₃ = {np.mean(I3_causet):.4f} ± {np.std(I3_causet):.4f}, "
          f"P(I₃≤0) = {100 * n_neg_cs / max(1, len(I3_causet)):.0f}%")

    # Phase 3: CDT spectral gap
    print("\n--- Phase 3: Spectral gap ---")

    for source_name, make_cs in [
        ("CDT", lambda: cdt_to_causet(samples[-1].astype(int))),
        ("Causet", lambda: sprinkle_fast(N_target, dim=2, region='diamond', rng=rng)[0]),
    ]:
        gaps = []
        for _ in range(10):
            cs = make_cs()
            if cs.n > 200 or cs.n < 10:
                continue
            C = cs.order.astype(float)
            Delta = 0.5 * (C.T - C)
            H = 1j * Delta
            evals = np.linalg.eigvalsh(H).real
            pos = evals[evals > 1e-10]
            if len(pos) > 0:
                gaps.append(pos.min())

        N_eff = cs.n
        if gaps:
            print(f"  {source_name:>8} (N~{N_eff}): gap = {np.mean(gaps):.4f}, "
                  f"gap*N = {np.mean(gaps) * N_eff:.2f}")

    # Phase 4: ER=EPR comparison
    print("\n--- Phase 4: ER=EPR on CDT vs Causet ---")

    def er_epr_r(cs, W):
        N = cs.n
        order_int = cs.order.astype(np.int32)
        w_vals, conn_vals = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if not cs.order[i, j] and not cs.order[j, i]:
                    w_vals.append(abs(W[i, j]))
                    cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                    cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                    conn_vals.append(cp + cf)
        if len(w_vals) < 10:
            return float('nan')
        return np.corrcoef(w_vals, conn_vals)[0, 1]

    for source_name in ["CDT", "Causet"]:
        r_vals = []
        for _ in range(10):
            if source_name == "CDT":
                vp = samples[rng.integers(len(samples) - 20, len(samples))]
                cs = cdt_to_causet(vp.astype(int))
            else:
                to = TwoOrder(N_target, rng=rng)
                cs = to.to_causet()

            if cs.n > 150 or cs.n < 10:
                continue
            W, _ = sj_wightman(cs)
            r = er_epr_r(cs, W)
            if not np.isnan(r):
                r_vals.append(r)

        if r_vals:
            print(f"  {source_name:>8}: ER=EPR r = {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}")

    # Phase 5: CDT c vs cosmological constant
    print("\n--- Phase 5: c_eff vs cosmological constant (CDT, N~90) ---")
    print(f"  {'lambda2':>8} {'c_eff':>8}")
    print("-" * 20)

    for lambda2 in [-0.1, 0.0, 0.1, 0.3, 0.5]:
        T, s_init = 15, 6
        samples_l = mcmc_cdt(T=T, s_init=s_init, lambda2=lambda2, mu=0.01,
                              n_steps=8000, target_volume=90, rng=rng)
        S_vals = []
        N_vals = []
        for vp in samples_l[-8:]:
            cs = cdt_to_causet(vp.astype(int))
            if cs.n > 150 and cs.n < 10:
                continue
            W, _ = sj_wightman(cs)
            S_vals.append(entanglement_entropy(W, list(range(cs.n // 2))))
            N_vals.append(cs.n)

        if S_vals:
            N_avg = np.mean(N_vals)
            c = 3 * np.mean(S_vals) / np.log(N_avg) if N_avg > 1 else 0
            print(f"  {lambda2:>8.1f} {c:>8.3f}")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == '__main__':
    main()
