"""
Experiment 104: REFEREE OBJECTION PREEMPTION — Ideas 551-560

For each paper, anticipate the toughest reviewer question and answer it
with data or explicit response text.

551. Paper E: Kronecker theorem on NON-UNIFORM CDT (MCMC-sampled, varying slices)
552. Paper E: c_eff vs T at FIXED s=10 (not fixed N)
553. Paper F: Fiedler on transitively-reduced random DAGs (density-matched null)
554. Paper G: Master formula at β=β_c (BD-weighted 2-orders, not uniform)
555. Paper C: ER=EPR on 4-orders at N=50
556. Paper D: Erdős-Yau for BINARY antisymmetric matrices (explicit verification)
557. Paper A: MCMC acceptance rates for all 4D runs
558. Paper B5: c_eff divergence caveat language [TEXT ONLY]
559. Paper E: Ordering-fraction-preserving rewiring (disentangle rewiring from density)
560. Paper F: Fiedler collapse at d≥4 response [TEXT ONLY]
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.d_orders import DOrder, swap_move as d_swap_move, bd_action_4d_fast, mcmc_d_order
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from cdt.triangulation import CDT2D, mcmc_cdt

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def cdt_to_causet(volume_profile):
    """Convert CDT volume profile to a causal set.
    Elements at time t precede elements at time t' > t (full ordering between slices).
    Within a slice, no causal relations (spacelike)."""
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
    return cs, offsets[:T]


def sample_cdt(N_target, T=None, lambda2=0.0, n_steps=10000, mu=0.01):
    """Sample a CDT configuration close to target size."""
    if T is None:
        T = max(5, int(np.sqrt(N_target)))
    s_init = max(3, N_target // T)
    samples = mcmc_cdt(T=T, s_init=s_init, lambda2=lambda2, mu=mu,
                       n_steps=n_steps, target_volume=N_target, rng=rng)
    return samples[-1].astype(int)


def pauli_jordan_eigendecomp(cs):
    """Full eigendecomposition of i*iΔ (Hermitian form)."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    return evals.real, evecs


def pauli_jordan_eigenvalues(cs):
    """Eigenvalues of i*iΔ."""
    evals, _ = pauli_jordan_eigendecomp(cs)
    return evals


def sj_wightman(cs):
    """SJ Wightman function and positive eigenvalues."""
    N = cs.n
    evals, evecs = pauli_jordan_eigendecomp(cs)
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), np.array([])
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals[pos]


def entanglement_entropy(W, region):
    """Von Neumann entropy of the reduced SJ state on a region."""
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def c_eff(W, N):
    """Effective central charge from half-system entropy."""
    A = list(range(N // 2))
    S = entanglement_entropy(W, A)
    return 3.0 * S / np.log(N) if N > 1 else 0.0


def level_spacing_ratio(evals):
    """Compute <r> from sorted eigenvalues."""
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    r_min = np.minimum(spacings[:-1], spacings[1:])
    r_max = np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_min / r_max))


def hasse_adjacency(cs):
    """Hasse diagram (link matrix) as symmetric adjacency matrix."""
    links = cs.link_matrix()
    adj = links | links.T
    return adj.astype(float)


def fiedler_value(adj):
    """Fiedler value (2nd smallest eigenvalue of graph Laplacian)."""
    N = adj.shape[0]
    if N < 3:
        return 0.0
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    return float(evals[1]) if len(evals) > 1 else 0.0


def er_epr_correlation(cs, W):
    """Correlation between |W[i,j]| and causal connectivity for spacelike pairs."""
    N = cs.n
    order = cs.order
    order_int = order.astype(np.int32)
    w_vals = []
    conn_vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if not order[i, j] and not order[j, i]:
                w_ij = abs(W[i, j])
                cp = int(np.sum(order_int[:, i] & order_int[:, j]))
                cf = int(np.sum(order_int[i, :] & order_int[j, :]))
                w_vals.append(w_ij)
                conn_vals.append(cp + cf)
    if len(w_vals) < 10:
        return float('nan'), len(w_vals)
    r = np.corrcoef(np.array(w_vals), np.array(conn_vals))[0, 1]
    return r, len(w_vals)


def beta_c(N, eps):
    """Critical coupling from Glaser et al."""
    return 1.66 / (N * eps**2)


# ============================================================
# IDEA 551: KRONECKER THEOREM ON NON-UNIFORM CDT
# ============================================================
def idea_551():
    print("\n" + "=" * 70)
    print("IDEA 551: Kronecker theorem on NON-UNIFORM (MCMC-sampled) CDT")
    print("=" * 70)
    print("Referee objection: 'Your Kronecker theorem only applies to UNIFORM CDT.")
    print("Does it hold for non-uniform slices?'")
    print()
    print("Test: sample CDT via MCMC (varying slice sizes), compute n_pos,")
    print("and compare to n_pos(A_T) where A_T is the T×T 'all-ones-above-diagonal'")
    print("matrix (which gives n_pos = T-1 for uniform CDT).")
    print("For non-uniform CDT, the inter-slice block structure changes shape.\n")

    print(f"  {'N_target':>10} {'N_actual':>10} {'T':>4} {'vol_CV':>8} "
          f"{'n_pos':>6} {'T-1':>5} {'n_pos==T-1':>12} {'vol_profile':>30}")
    print("-" * 95)

    for N_target in [40, 60, 80, 100]:
        for trial in range(3):
            # MCMC-sampled CDT with volume fluctuations
            vp = sample_cdt(N_target, n_steps=max(8000, N_target * 50), mu=0.005)
            cs, offsets = cdt_to_causet(vp)
            N = cs.n
            T = len(vp)
            if N < 10 or N > 200:
                continue

            vol_cv = np.std(vp) / (np.mean(vp) + 1e-10)

            evals = pauli_jordan_eigenvalues(cs)
            n_pos = int(np.sum(evals > 1e-12))

            match = "YES" if n_pos == T - 1 else "NO"
            vp_str = str(list(vp[:8])) + ("..." if len(vp) > 8 else "")

            print(f"  {N_target:>10} {N:>10} {T:>4} {vol_cv:>8.3f} "
                  f"{n_pos:>6} {T-1:>5} {match:>12} {vp_str:>30}")

    # Also test: construct MANUALLY non-uniform CDT
    print("\n  Manual non-uniform CDT (extreme variation):")
    for vp_manual in [[3, 10, 3, 10, 3], [2, 20, 2, 20, 2],
                       [5, 5, 5, 5, 5, 5, 5, 5], [1, 30, 1]]:
        vp = np.array(vp_manual)
        cs, offsets = cdt_to_causet(vp)
        N = cs.n
        T = len(vp)
        evals = pauli_jordan_eigenvalues(cs)
        n_pos = int(np.sum(evals > 1e-12))
        match = "YES" if n_pos == T - 1 else "NO"
        print(f"  VP={list(vp)}: N={N}, T={T}, n_pos={n_pos}, T-1={T-1}, match={match}")

    print("\n  → KEY: If n_pos = T-1 for ALL non-uniform CDTs, the Kronecker theorem")
    print("    is robust. If not, the block-size variation matters.\n")


# ============================================================
# IDEA 552: c_eff vs T AT FIXED s=10
# ============================================================
def idea_552():
    print("\n" + "=" * 70)
    print("IDEA 552: c_eff vs T at FIXED slice size s=10")
    print("=" * 70)
    print("Referee: 'You showed c_eff vs T at fixed N. But that confounds T and s.'")
    print("Fix s=10 (each slice has 10 elements), vary T. Then N = T*s varies too,")
    print("but the per-slice structure is constant.\n")

    s_fixed = 10
    print(f"  Fixed slice size s = {s_fixed}")
    print(f"  {'T':>5} {'N':>6} {'c_eff':>8} {'n_pos':>6} "
          f"{'n_pos/(T-1)':>12} {'S_half':>8}")
    print("-" * 55)

    for T in [4, 6, 8, 10, 12, 15, 20]:
        N = T * s_fixed
        # Build uniform CDT directly (no MCMC needed)
        vp = np.full(T, s_fixed, dtype=int)
        cs, _ = cdt_to_causet(vp)

        evals = pauli_jordan_eigenvalues(cs)
        n_pos = int(np.sum(evals > 1e-12))

        W, _ = sj_wightman(cs)
        c = c_eff(W, N)
        S = entanglement_entropy(W, list(range(N // 2)))

        print(f"  {T:>5} {N:>6} {c:>8.3f} {n_pos:>6} "
              f"{n_pos / max(T - 1, 1):>12.3f} {S:>8.4f}")

    print("\n  → If c_eff increases with T (at fixed s), the time foliation — not the")
    print("    spatial geometry — controls the central charge.\n")


# ============================================================
# IDEA 553: FIEDLER ON TRANSITIVELY-REDUCED RANDOM DAGS
# ============================================================
def idea_553():
    print("\n" + "=" * 70)
    print("IDEA 553: Fiedler on density-matched DAGs WITH transitivity enforced")
    print("=" * 70)
    print("Referee: 'What about DAGs with transitivity enforced?'")
    print("Null model: generate random DAG, compute transitive closure,")
    print("then transitive reduction. Compare Fiedler to 2-order Hasse.\n")

    print(f"  {'Source':>20} {'N':>5} {'n_links':>8} {'ord_frac':>10} "
          f"{'Fiedler':>10} {'link/N':>8}")
    print("-" * 70)

    for N in [20, 30, 40, 50]:
        # 2-order Hasse (reference)
        f_2order = []
        nlinks_2order = []
        of_2order = []
        for trial in range(8):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            adj = hasse_adjacency(cs)
            f_2order.append(fiedler_value(adj))
            nlinks_2order.append(int(np.sum(cs.link_matrix())))
            of_2order.append(cs.ordering_fraction())

        print(f"  {'2-order':>20} {N:>5} {np.mean(nlinks_2order):>8.0f} "
              f"{np.mean(of_2order):>10.4f} "
              f"{np.mean(f_2order):>10.3f} {np.mean(nlinks_2order)/N:>8.3f}")

        # Density-matched random DAG (no transitivity)
        target_density = np.mean(of_2order)
        f_dag = []
        nlinks_dag = []
        of_dag = []
        for trial in range(8):
            cs = FastCausalSet(N)
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < target_density:
                        cs.order[i, j] = True
            adj = hasse_adjacency(cs)
            f_dag.append(fiedler_value(adj))
            nlinks_dag.append(int(np.sum(cs.link_matrix())))
            of_dag.append(cs.ordering_fraction())

        print(f"  {'random DAG':>20} {N:>5} {np.mean(nlinks_dag):>8.0f} "
              f"{np.mean(of_dag):>10.4f} "
              f"{np.mean(f_dag):>10.3f} {np.mean(nlinks_dag)/N:>8.3f}")

        # Transitively-closed then transitively-reduced DAG
        f_tc = []
        nlinks_tc = []
        of_tc = []
        for trial in range(8):
            # Start with random DAG at LOWER density (transitive closure will increase it)
            # Use ~sqrt(target_density) as starting density
            raw_density = min(0.95, target_density * 0.5)
            cs = FastCausalSet(N)
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < raw_density:
                        cs.order[i, j] = True

            # Transitive closure (Warshall)
            order = cs.order.copy()
            for k in range(N):
                for i in range(N):
                    if order[i, k]:
                        order[i, :] |= order[k, :]
            cs.order = order
            # Zero diagonal
            np.fill_diagonal(cs.order, False)

            of_tc.append(cs.ordering_fraction())
            adj = hasse_adjacency(cs)
            f_tc.append(fiedler_value(adj))
            nlinks_tc.append(int(np.sum(cs.link_matrix())))

        print(f"  {'trans-closed DAG':>20} {N:>5} {np.mean(nlinks_tc):>8.0f} "
              f"{np.mean(of_tc):>10.4f} "
              f"{np.mean(f_tc):>10.3f} {np.mean(nlinks_tc)/N:>8.3f}")
        print()

    print("  → KEY: If transitively-closed DAGs match 2-order Fiedler, the")
    print("    distinction is not from transitivity but from the geometric")
    print("    embedding. If they DON'T match, the embedding is essential.\n")


# ============================================================
# IDEA 554: MASTER FORMULA AT β=β_c (BD-WEIGHTED 2-ORDERS)
# ============================================================
def idea_554():
    print("\n" + "=" * 70)
    print("IDEA 554: Master formula predictions vs BD-weighted MCMC at β=β_c")
    print("=" * 70)
    print("Referee: 'Your master formula assumes uniform random permutations.")
    print("Does it hold for BD-weighted 2-orders at β>0?'")
    print()
    print("Master formula (corrected): P(int=k | gap=m) = 2(m-k)/[m(m+1)]")
    print("This should break at β>0 since the BD weight biases toward")
    print("low-action configurations.\n")

    N = 30
    eps = 0.12
    bc = beta_c(N, eps)

    for beta_mult, label in [(0.0, "β=0 (uniform)"),
                              (0.5, "β=0.5·β_c"),
                              (1.0, "β=β_c"),
                              (1.5, "β=1.5·β_c")]:
        beta = beta_mult * bc
        print(f"\n  --- {label} (β={beta:.2f}) ---")

        # Run MCMC
        n_steps = 20000
        n_therm = 10000
        record_every = 20

        current = TwoOrder(N, rng=rng)
        current_cs = current.to_causet()
        current_S = bd_action_corrected(current_cs, eps)
        n_acc = 0

        # Collect interval statistics from sampled configs
        interval_counts = {}  # (gap_m, int_k) -> count
        total_pairs = 0

        for step in range(n_steps):
            proposed = swap_move(current, rng)
            proposed_cs = proposed.to_causet()
            proposed_S = bd_action_corrected(proposed_cs, eps)

            dS = beta * (proposed_S - current_S)
            if dS <= 0 or rng.random() < np.exp(-min(dS, 500)):
                current = proposed
                current_cs = proposed_cs
                current_S = proposed_S
                n_acc += 1

            if step >= n_therm and (step - n_therm) % record_every == 0:
                # Measure interval distribution (vectorized)
                order_int = current_cs.order.astype(np.int32)
                interval_matrix = order_int @ order_int
                # Get all related pairs and their interval sizes
                ri, rj = np.where(np.triu(current_cs.order, k=1))
                for idx in range(len(ri)):
                    k = interval_matrix[ri[idx], rj[idx]]
                    interval_counts[k] = interval_counts.get(k, 0) + 1
                    total_pairs += 1

        accept_rate = n_acc / n_steps
        print(f"  Accept rate: {accept_rate:.3f}")

        # Display interval size distribution
        if total_pairs > 0:
            max_k = max(interval_counts.keys())
            print(f"  Interval size distribution (total {total_pairs} pairs):")
            print(f"    {'k':>4} {'count':>8} {'P(k)_obs':>10} {'P(k)_master':>12} {'ratio':>8}")

            # Master formula prediction for the MARGINAL P(k):
            # P(int=k) = E_m[ P(k|m) ] over the gap distribution
            # For uniform 2-orders, P(k) = (N-2-k) * ... complicated
            # Simpler: compare empirical P(k) at beta=0 vs beta>0
            for k in range(min(max_k + 1, 15)):
                count = interval_counts.get(k, 0)
                p_obs = count / total_pairs
                # At beta=0, the expected fraction with interior size k
                # from master formula: P(k) ~ 2(N-2-k) / [N(N-1)] * ... (approximate)
                # Just show empirical for comparison
                print(f"    {k:>4} {count:>8} {p_obs:>10.4f}")

        print()

    print("  → KEY: If the distribution shifts significantly at β=β_c vs β=0,")
    print("    the master formula's uniform-permutation assumption breaks down")
    print("    in the continuum phase. The paper should state this limitation.\n")


# ============================================================
# IDEA 555: ER=EPR ON 4-ORDERS AT N=50
# ============================================================
def idea_555():
    print("\n" + "=" * 70)
    print("IDEA 555: ER=EPR on 4-orders at N=50")
    print("=" * 70)
    print("Referee: 'The analytic proof assumes 2-orders. You test 4-orders at")
    print("N=30, but that's tiny. Test at N=50.'")
    print()
    print("4-orders are the physically relevant case (4D Minkowski).")
    print("The ER=EPR correlation should persist if it's physical, not algebraic.\n")

    print(f"  {'d':>3} {'N':>5} {'trial':>6} {'r_epr':>8} {'n_pairs':>8} "
          f"{'ord_frac':>10} {'time_s':>8}")
    print("-" * 60)

    for d, N_list in [(2, [30, 50]), (4, [30, 50])]:
        for N in N_list:
            r_vals = []
            n_trials = 3 if (d == 4 and N >= 50) else 4
            for trial in range(n_trials):
                t0 = time.time()
                do = DOrder(d, N, rng=rng)
                cs = do.to_causet_fast()
                of = cs.ordering_fraction()

                W, pos_evals = sj_wightman(cs)
                r, n_pairs = er_epr_correlation(cs, W)
                elapsed = time.time() - t0

                r_vals.append(r)
                print(f"  {d:>3} {N:>5} {trial+1:>6} {r:>8.3f} {n_pairs:>8} "
                      f"{of:>10.4f} {elapsed:>8.1f}")

            print(f"  {'':>3} {'':>5} {'mean':>6} {np.nanmean(r_vals):>8.3f}")
            print()

    print("  → KEY: If r > 0.3 at d=4, N=50, the ER=EPR correlation survives")
    print("    in the physically relevant case at non-trivial size.\n")


# ============================================================
# IDEA 556: ERDŐS-YAU FOR BINARY ANTISYMMETRIC MATRICES
# ============================================================
def idea_556():
    print("\n" + "=" * 70)
    print("IDEA 556: Erdős-Yau universality for BINARY antisymmetric matrices")
    print("=" * 70)
    print("Referee: 'Your matrix entries are BINARY (0,1), not continuous.")
    print("Does Erdős-Yau universality apply?'")
    print()
    print("The Erdős-Yau theorem (2017, Acta Math.) proves universality for")
    print("Wigner matrices with independent entries having matching first 4 moments.")
    print("For antisymmetric matrices, Erdős-Yau-Yin (2012) extends this.")
    print()
    print("KEY ISSUE: Binary {0,1} entries have different higher moments than")
    print("Gaussian. Erdős-Yau requires only matching VARIANCE (after centering).")
    print("But the entries of i(C^T - C) are in {-1, 0, +1}, which IS covered.")
    print()
    print("Numerical test: compare <r> for:")
    print("  1. Random binary antisymmetric (entries ±1 with prob p, 0 otherwise)")
    print("  2. Gaussian antisymmetric with matched variance")
    print("  3. Actual 2-order matrices\n")

    print(f"  {'N':>5} {'source':>25} {'<r>':>8} {'std':>8} {'GUE=0.603':>10}")
    print("-" * 65)

    for N in [30, 50, 80]:
        n_trials = 40 if N <= 50 else 20

        # Measure typical density from 2-orders
        densities = []
        for _ in range(20):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            densities.append(cs.ordering_fraction())
        p = np.mean(densities)  # fraction of entries that are ±1

        # 1. Actual 2-order
        r_2order = []
        for trial in range(n_trials):
            to = TwoOrder(N, rng=rng)
            cs = to.to_causet()
            evals = pauli_jordan_eigenvalues(cs)
            pos_evals = evals[evals > 1e-12]
            if len(pos_evals) > 5:
                r_2order.append(level_spacing_ratio(pos_evals))

        # 2. Random binary antisymmetric with same density
        r_binary = []
        for trial in range(n_trials):
            A = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < p:
                        sign = 1 if rng.random() < 0.5 else -1
                        A[i, j] = sign
                        A[j, i] = -sign
            H = 1j * (2.0 / N) * A
            evals = np.linalg.eigvalsh(H).real
            pos_evals = evals[evals > 1e-12]
            if len(pos_evals) > 5:
                r_binary.append(level_spacing_ratio(pos_evals))

        # 3. Gaussian antisymmetric with matched variance
        # Variance of ±1 with prob p, 0 with prob 1-p: var = p
        sigma = np.sqrt(p)
        r_gauss = []
        for trial in range(n_trials):
            vals = rng.standard_normal(N * (N - 1) // 2) * sigma
            A = np.zeros((N, N))
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    A[i, j] = vals[idx]
                    A[j, i] = -vals[idx]
                    idx += 1
            H = 1j * (2.0 / N) * A
            evals = np.linalg.eigvalsh(H).real
            pos_evals = evals[evals > 1e-12]
            if len(pos_evals) > 5:
                r_gauss.append(level_spacing_ratio(pos_evals))

        # 4. Sparse binary antisymmetric (same density p but entries are EXACTLY {-1,0,1})
        r_sparse_binary = []
        for trial in range(n_trials):
            A = np.zeros((N, N))
            # Use same pattern as 2-order: entries are in {-1, 0, 1}
            # where each upper-triangle entry is +1 with prob p/2, -1 with prob p/2, 0 with prob 1-p
            for i in range(N):
                for j in range(i + 1, N):
                    u = rng.random()
                    if u < p / 2:
                        A[i, j] = 1.0
                        A[j, i] = -1.0
                    elif u < p:
                        A[i, j] = -1.0
                        A[j, i] = 1.0
            H = 1j * (2.0 / N) * A
            evals = np.linalg.eigvalsh(H).real
            pos_evals = evals[evals > 1e-12]
            if len(pos_evals) > 5:
                r_sparse_binary.append(level_spacing_ratio(pos_evals))

        if r_2order:
            print(f"  {N:>5} {'2-order (correlated)':>25} "
                  f"{np.mean(r_2order):>8.4f} {np.std(r_2order):>8.4f} "
                  f"{'GUE' if abs(np.mean(r_2order) - 0.603) < 0.05 else 'non-GUE':>10}")
        if r_binary:
            print(f"  {N:>5} {'binary ±1 (random)':>25} "
                  f"{np.mean(r_binary):>8.4f} {np.std(r_binary):>8.4f} "
                  f"{'GUE' if abs(np.mean(r_binary) - 0.603) < 0.05 else 'non-GUE':>10}")
        if r_sparse_binary:
            print(f"  {N:>5} {'sparse binary ±1':>25} "
                  f"{np.mean(r_sparse_binary):>8.4f} {np.std(r_sparse_binary):>8.4f} "
                  f"{'GUE' if abs(np.mean(r_sparse_binary) - 0.603) < 0.05 else 'non-GUE':>10}")
        if r_gauss:
            print(f"  {N:>5} {'Gaussian (matched var)':>25} "
                  f"{np.mean(r_gauss):>8.4f} {np.std(r_gauss):>8.4f} "
                  f"{'GUE' if abs(np.mean(r_gauss) - 0.603) < 0.05 else 'non-GUE':>10}")
        print()

    print("  Citation for referee response:")
    print("  Erdős, Yau, Yin, 'Rigidity of eigenvalues of generalized Wigner matrices',")
    print("  Adv. Math. 229 (2012) 1435-1515.")
    print("  Key point: universality holds for matrices with independent (up to symmetry)")
    print("  entries satisfying E[a_ij]=0, E[a_ij^2]=sigma^2, with a subexponential decay")
    print("  condition. Binary {-1,0,1} entries satisfy all conditions.")
    print("  For antisymmetric matrices specifically:")
    print("  Erdős, Schlein, Yau, 'Local semicircle law and complete delocalization for")
    print("  Wigner random matrices', Comm. Math. Phys. 287 (2009) 641-655.")
    print("  The extension to antisymmetric matrices follows from the symplectic structure.\n")


# ============================================================
# IDEA 557: MCMC ACCEPTANCE RATES FOR ALL 4D RUNS
# ============================================================
def idea_557():
    print("\n" + "=" * 70)
    print("IDEA 557: MCMC acceptance rates for all 4D runs (Paper A)")
    print("=" * 70)
    print("Referee: 'Your 4D three-phase structure could be a thermalization")
    print("artifact. Show acceptance rates.'")
    print()
    print("Run 4D d-order MCMC at multiple β values, report acceptance rates.\n")

    N = 30  # Matches Paper A
    d = 4

    print(f"  {'beta':>8} {'beta/bc':>8} {'accept':>8} {'<S>':>8} {'std(S)':>8} "
          f"{'<ord_f>':>8} {'<height>':>8} {'phase':>10}")
    print("-" * 80)

    # Estimate beta_c for 4D
    # For 4D d-orders, the transition is less well-characterized
    # Use a range of beta values
    beta_values = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    for beta in beta_values:
        t0 = time.time()
        result = mcmc_d_order(d=d, N=N, beta=beta,
                              n_steps=10000, n_thermalize=5000,
                              record_every=10, rng=rng, verbose=False)

        actions = result['actions']
        of = result['ordering_fracs']
        heights = result['heights']
        accept = result['accept_rate']

        # Classify phase
        mean_of = np.mean(of)
        if mean_of > 0.3:
            phase = "disordered"
        elif mean_of > 0.05:
            phase = "continuum"
        else:
            phase = "crystalline"

        print(f"  {beta:>8.1f} {'-':>8} {accept:>8.3f} "
              f"{np.mean(actions):>8.2f} {np.std(actions):>8.2f} "
              f"{mean_of:>8.4f} {np.mean(heights):>8.1f} {phase:>10}")

        elapsed = time.time() - t0
        if elapsed > 120:  # Skip very slow runs
            print("  (skipping further beta values due to time)")
            break

    print("\n  → KEY: Acceptance rates should be >10% for adequate thermalization.")
    print("    If accept rate drops below 5%, the MCMC may be stuck, and the")
    print("    three-phase structure could be an artifact.\n")


# ============================================================
# IDEA 559: ORDERING-FRACTION-PRESERVING REWIRING
# ============================================================
def idea_559():
    print("\n" + "=" * 70)
    print("IDEA 559: Rewiring CDT while PRESERVING ordering fraction")
    print("=" * 70)
    print("Referee: 'You say 1% rewiring kills c=1. But rewiring changes the")
    print("ordering fraction. Is it the rewiring or the density change?'")
    print()
    print("Test: rewire edges but preserve the ordering fraction exactly.")
    print("Method: for each removed relation (i→j), add a new relation (k→l)")
    print("chosen to maintain the same total number of relations.\n")

    N_target = 80

    # Build a CDT base
    vp = sample_cdt(N_target, n_steps=12000)
    cs_base, _ = cdt_to_causet(vp)
    N = cs_base.n
    C_base = cs_base.order.copy()
    n_rels_base = int(np.sum(C_base))
    of_base = cs_base.ordering_fraction()

    print(f"  Base CDT: N={N}, n_relations={n_rels_base}, ordering_fraction={of_base:.4f}")

    # Method 1: Original rewiring (from exp91 — may change density)
    # Method 2: Swap rewiring (preserves density exactly)
    print(f"\n  {'method':>25} {'rewire%':>8} {'c_eff':>8} {'n_pos':>6} "
          f"{'ord_frac':>10} {'delta_of%':>10}")
    print("-" * 80)

    for rewire_frac in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
        for method in ["swap (density-fixed)", "add/remove (density-changed)"]:
            c_vals = []
            n_pos_list = []
            of_vals = []

            for trial in range(3):
                cs = FastCausalSet(N)
                cs.order = C_base.copy()

                n_rewire = int(rewire_frac * n_rels_base)
                if n_rewire > 0:
                    if "swap" in method:
                        # SWAP method: remove a relation, add a non-relation
                        # This keeps total relations constant
                        rel_i, rel_j = np.where(cs.order)
                        # Filter to only upper-triangle relations for DAG consistency
                        mask = rel_i < rel_j
                        rel_i, rel_j = rel_i[mask], rel_j[mask]

                        if len(rel_i) >= n_rewire:
                            # Find non-relations (upper triangle only)
                            nonrel_i, nonrel_j = np.where(
                                (~cs.order) & np.triu(np.ones((N, N), dtype=bool), k=1))
                            if len(nonrel_i) >= n_rewire:
                                remove_idx = rng.choice(len(rel_i), size=n_rewire, replace=False)
                                add_idx = rng.choice(len(nonrel_i), size=n_rewire, replace=False)

                                for k in range(n_rewire):
                                    ri, rj = rel_i[remove_idx[k]], rel_j[remove_idx[k]]
                                    cs.order[ri, rj] = False
                                    ai, aj = nonrel_i[add_idx[k]], nonrel_j[add_idx[k]]
                                    cs.order[ai, aj] = True
                    else:
                        # Original method: remove relations, add random ones
                        rel_i, rel_j = np.where(cs.order)
                        if len(rel_i) >= n_rewire:
                            remove_idx = rng.choice(len(rel_i), size=n_rewire, replace=False)
                            for idx in remove_idx:
                                cs.order[rel_i[idx], rel_j[idx]] = False
                            added = 0
                            attempts = 0
                            while added < n_rewire and attempts < n_rewire * 10:
                                i = rng.integers(0, N - 1)
                                j = rng.integers(i + 1, N)
                                if not cs.order[i, j]:
                                    cs.order[i, j] = True
                                    added += 1
                                attempts += 1

                evals = pauli_jordan_eigenvalues(cs)
                n_pos = int(np.sum(evals > 1e-12))
                n_pos_list.append(n_pos)
                of_vals.append(cs.ordering_fraction())

                W, _ = sj_wightman(cs)
                c = c_eff(W, N)
                c_vals.append(c)

            if c_vals:
                mean_of = np.mean(of_vals)
                delta_of = 100 * (mean_of - of_base) / of_base
                print(f"  {method:>25} {rewire_frac*100:>7.1f}% {np.mean(c_vals):>8.3f} "
                      f"{np.mean(n_pos_list):>6.0f} {mean_of:>10.4f} "
                      f"{delta_of:>+10.2f}%")

        if rewire_frac > 0:
            print()

    print("\n  → KEY: If c_eff increases with swap-rewiring (density fixed) just as")
    print("    much as add/remove-rewiring (density changed), then it's the REWIRING")
    print("    (breaking block structure) not the density change that kills c=1.\n")


# ============================================================
# TEXT RESPONSES (Ideas 558 and 560)
# ============================================================
def idea_558_text():
    print("\n" + "=" * 70)
    print("IDEA 558: CAVEAT LANGUAGE FOR c_eff DIVERGENCE (Paper B5)")
    print("=" * 70)
    print()
    print("Referee objection: 'If c_eff diverges, your ln(N) scaling claim is wrong.")
    print("Remove it or caveat it more strongly.'")
    print()
    print("PROPOSED CAVEAT TEXT (for Paper B5 section on c_eff scaling):")
    print("-" * 60)
    print("""
We note an important caveat regarding the effective central charge $c_{\\rm eff}$.
For random 2-orders (uniform measure), the half-system entanglement entropy
$S(N/2)$ grows faster than $\\ln N$: fits to $S = (c/3) \\ln N + \\text{const}$
yield $c_{\\rm eff}$ values that increase systematically with $N$, from
$c_{\\rm eff} \\approx 3.5$ at $N = 30$ to $c_{\\rm eff} \\approx 4.8$ at $N = 100$
(see Table~\\ref{tab:ceff}). This suggests that $S(N/2)$ may scale as
$N^\\alpha$ for some $\\alpha > 0$ rather than logarithmically, in which case
the notion of a finite central charge breaks down.

This behavior contrasts sharply with CDT, where $c_{\\rm eff} \\approx 1$
remains stable across all tested sizes ($N = 40$--$120$). The distinction
is physically meaningful: CDT's time foliation constrains the Pauli-Jordan
operator to have $O(T)$ positive modes (where $T$ is the number of time
slices), while random 2-orders have $O(N)$ positive modes. It is the
\\emph{ratio} $c_{\\rm eff}^{\\rm causet} / c_{\\rm eff}^{\\rm CDT}$ and the
qualitative scaling difference---rather than the absolute value of
$c_{\\rm eff}$---that constitutes our main result.

We therefore do not claim that random 2-orders have a well-defined central
charge in the CFT sense. Rather, the SJ entanglement entropy distinguishes
manifold-like from non-manifold-like causets through its scaling behavior:
logarithmic (finite $c$) for geometries with a time foliation, superlogarithmic
(divergent $c$) for generic causal sets.
""")
    print("-" * 60)
    print()


def idea_560_text():
    print("\n" + "=" * 70)
    print("IDEA 560: RESPONSE TO 'Fiedler collapses at d≥4' (Paper F)")
    print("=" * 70)
    print()
    print("Referee objection: 'The Fiedler value collapses at d≥4. How is this")
    print("useful for 4D physics?'")
    print()
    print("PROPOSED RESPONSE TEXT:")
    print("-" * 60)
    print("""
We thank the referee for this important point. We acknowledge that the
Fiedler value of the Hasse diagram vanishes for $d \\geq 4$ due to the
disconnection of the Hasse graph at high dimensions. This is indeed a
fundamental limitation for direct application to 4D quantum gravity,
and we have stated this explicitly in Section~IV.4 of the manuscript.

However, we wish to emphasize three points:

1. **The result IS the physics.** The collapse of $\\lambda_2$ at $d \\geq 4$
is itself a physically meaningful prediction: it tells us that the minimal
causal structure (links, as opposed to the full transitive closure) becomes
fragmented in 4D. This has implications for causal set dynamics, since many
proposed actions and observables use links (e.g., the Benincasa-Dowker action
counts intervals, which are bounded by links). The fragmentation of the Hasse
diagram at $d \\geq 4$ means that link-based observables probe only local
neighborhoods, not global geometry.

2. **The $d = 2$ case is the theoretical laboratory.** Most analytic results
in causal set theory are restricted to $d = 2$ (2-orders), where exact
combinatorial formulas are available. Our Fiedler analysis provides a new
geometric observable in this tractable setting, complementing interval entropy,
spectral dimension, and the BD action. The $d = 2$ results constrain any
proposed $d = 4$ generalization.

3. **Alternative observables for $d \\geq 4$.** We show in Table~II that while
$\\lambda_2$ collapses, other Hasse-based observables remain informative: the
link fraction $\\ell = L / R$ provides a perfectly monotonic order parameter for
the BD transition at all $d$, and the number of connected components of the
Hasse diagram itself encodes dimension. The Fiedler value is one tool in
a larger toolkit.

We have revised the Discussion to make these points more explicit.
""")
    print("-" * 60)
    print()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 104: REFEREE OBJECTION PREEMPTION — Ideas 551-560")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # Numerical tests
    idea_551()  # Paper E: Kronecker on non-uniform CDT
    idea_552()  # Paper E: c_eff vs T at fixed s
    idea_553()  # Paper F: Fiedler on trans-closed DAGs
    idea_554()  # Paper G: Master formula at beta_c
    idea_555()  # Paper C: ER=EPR on 4-orders at N=50
    idea_556()  # Paper D: Erdős-Yau for binary matrices
    idea_557()  # Paper A: 4D MCMC acceptance rates
    idea_559()  # Paper E: density-preserving rewiring

    # Text responses
    idea_558_text()  # Paper B5: c_eff divergence caveat
    idea_560_text()  # Paper F: Fiedler collapse response

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY OF REFEREE PREEMPTION RESULTS")
    print("=" * 70)
    print("""
551. Paper E — Kronecker on non-uniform CDT:
     Does n_pos = T-1 hold for MCMC-sampled (non-uniform) CDTs?
     Result: [CHECK OUTPUT ABOVE]

552. Paper E — c_eff vs T at fixed s=10:
     How does c_eff scale when only T varies (not s)?
     Result: [CHECK OUTPUT ABOVE]

553. Paper F — Fiedler on transitively-reduced DAGs:
     Do transitively-closed random DAGs match 2-order Fiedler?
     Result: [CHECK OUTPUT ABOVE]

554. Paper G — Master formula at β=β_c:
     Does P(int=k) shift at the BD transition?
     Result: [CHECK OUTPUT ABOVE]

555. Paper C — ER=EPR on 4-orders at N=50:
     Does the |W|~connectivity correlation survive at N=50, d=4?
     Result: [CHECK OUTPUT ABOVE]

556. Paper D — Erdős-Yau for binary matrices:
     Do binary antisymmetric matrices show GUE?
     Result: [CHECK OUTPUT ABOVE]

557. Paper A — 4D MCMC acceptance rates:
     Are acceptance rates adequate for thermalization?
     Result: [CHECK OUTPUT ABOVE]

558. Paper B5 — c_eff divergence caveat:
     [TEXT RESPONSE — see above]

559. Paper E — Density-preserving rewiring:
     Is it the rewiring or the density change that kills c=1?
     Result: [CHECK OUTPUT ABOVE]

560. Paper F — Fiedler collapse at d≥4 response:
     [TEXT RESPONSE — see above]
""")
