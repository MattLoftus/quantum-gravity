"""
Experiment 49: Large-N Scaling of SJ Vacuum Physics

The key bottleneck for paper quality is system size. At N=30-70, all results
are contaminated by finite-size effects. This experiment scales to N=500+
using optimized NumPy on Apple M4 Pro.

Three measurements at each N:
  (a) SJ entanglement entropy: S(N/2) vs ln(N) → effective central charge c
      CFT predicts S = (c/3) ln(N). Small-N gives c≈3; continuum → c=1?
  (b) ER=EPR correlation: |W[i,j]| vs causal connectivity for spacelike pairs
      r=0.88 at N=50, random DAG control r=0.71. Does r increase at large N?
  (c) Quantum chaos: level spacing ratio ⟨r⟩ of SJ spectrum
      GUE predicts ⟨r⟩≈0.5996. Got 0.56 at N=50. Stable at large N?

Timing: eigh at N=500 ≈ 33s on M4 Pro. Budget ~30 min total.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
import time
from causal_sets.two_orders import TwoOrder
from causal_sets.fast_core import FastCausalSet

rng = np.random.default_rng(42)


# ─── SJ vacuum computation (vectorized) ────────────────────────────────

def sj_wightman(cs):
    """Compute SJ Wightman function W and return (W, eigenvalues, eigenvectors)."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)  # Pauli-Jordan function
    H = 1j * Delta  # Hermitian
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real

    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), evals

    # Vectorized W construction
    evecs_pos = evecs[:, pos]
    evals_pos = evals[pos]
    W = (evecs_pos * evals_pos[np.newaxis, :]) @ evecs_pos.conj().T
    W = W.real

    # Normalize so eigenvalues ∈ [0,1]
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max

    return W, evals


# ─── Measurement (a): Entanglement entropy ──────────────────────────────

def entanglement_entropy(W, region):
    """Von Neumann entropy of the reduced state on region."""
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def spatial_partition_v(two_order, frac):
    """Partition by v-coordinate (spatial direction)."""
    v_order = np.argsort(two_order.v)
    cut = max(1, int(frac * two_order.N))
    return list(v_order[:cut])


# ─── Measurement (b): ER=EPR correlation (vectorized) ───────────────────

def er_epr_correlation_vectorized(cs, W):
    """
    Compute correlation between |W[i,j]| and causal connectivity
    for spacelike-separated pairs. Fully vectorized for large N.
    """
    N = cs.n
    order = cs.order
    order_int = order.astype(np.int32)

    # Find spacelike pairs: neither i<j nor j<i
    # Use upper triangle only
    i_idx, j_idx = np.triu_indices(N, k=1)
    causal_mask = order[i_idx, j_idx] | order[j_idx, i_idx]
    spacelike = ~causal_mask

    if np.sum(spacelike) < 10:
        return float('nan'), 0

    i_sl = i_idx[spacelike]
    j_sl = j_idx[spacelike]

    # |W[i,j]| for spacelike pairs
    w_vals = np.abs(W[i_sl, j_sl])

    # Connectivity: common past + common future
    # common_past[i,j] = Σ_k order[k,i] & order[k,j] = (order^T @ order)[i,j]
    # But order^T[a,b] = order[b,a], so (C^T C)[i,j] = Σ_k C[k,i]*C[k,j]
    # = number of common ancestors
    CtC = order_int.T @ order_int  # common past matrix
    CCt = order_int @ order_int.T  # common future matrix

    conn_vals = CtC[i_sl, j_sl] + CCt[i_sl, j_sl]
    conn_vals = conn_vals.astype(float)

    if np.std(w_vals) < 1e-15 or np.std(conn_vals) < 1e-15:
        return float('nan'), len(i_sl)

    r = np.corrcoef(w_vals, conn_vals)[0, 1]
    return r, len(i_sl)


def random_dag_control(N, ordering_frac_target=0.5, rng=None):
    """Generate a random DAG with similar ordering fraction as a 2-order."""
    if rng is None:
        rng = np.random.default_rng()

    # Random DAG: for each pair (i,j) with i<j, add edge with probability p
    # where p is chosen to match the ordering fraction
    p = ordering_frac_target
    cs = FastCausalSet(N)
    for i in range(N):
        edges = rng.random(N - i - 1) < p
        cs.order[i, i+1:] = edges

    # Transitive closure
    order_int = cs.order.astype(np.int32)
    for k in range(N):
        cs.order |= (cs.order[:, k:k+1] & cs.order[k:k+1, :])

    return cs


# ─── Measurement (c): Level spacing ratio ───────────────────────────────

def level_spacing_ratio(evals):
    """
    Compute mean level spacing ratio ⟨r⟩ from eigenvalues.
    Sort positive eigenvalues, compute consecutive spacings,
    r_i = min(δ_i, δ_{i+1}) / max(δ_i, δ_{i+1}).
    GUE: ⟨r⟩ ≈ 0.5996, GOE: ⟨r⟩ ≈ 0.5307, Poisson: ⟨r⟩ ≈ 0.3863.
    """
    pos_evals = np.sort(evals[evals > 1e-12])
    if len(pos_evals) < 4:
        return float('nan')

    spacings = np.diff(pos_evals)
    if len(spacings) < 2:
        return float('nan')

    r_vals = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(r_vals))


# ─── Main experiment ────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("EXPERIMENT 49: Large-N Scaling of SJ Vacuum Physics")
    print("=" * 80)

    N_values = [50, 100, 200, 300, 500]
    # Fewer trials at large N to stay within time budget
    trials_per_N = {50: 5, 100: 5, 200: 3, 300: 3, 500: 3}

    # Storage for all results
    results = {}

    print(f"\n{'N':>5} | {'trials':>6} | {'S(N/2)':>8} {'c_eff':>8} | "
          f"{'r_ER':>8} {'r_DAG':>8} | {'<r>_lsr':>8} | {'time':>6}")
    print("-" * 80)

    for N in N_values:
        n_trials = trials_per_N[N]
        t_start = time.time()

        S_vals = []
        r_er_vals = []
        r_dag_vals = []
        lsr_vals = []

        for trial in range(n_trials):
            trial_rng = np.random.default_rng(42 + trial * 1000 + N)

            # Generate 2-order and causet
            to = TwoOrder(N, rng=trial_rng)
            cs = to.to_causet()

            # Compute SJ vacuum
            W, evals = sj_wightman(cs)

            # (a) Entanglement entropy at half-partition
            region_A = spatial_partition_v(to, 0.5)
            S = entanglement_entropy(W, region_A)
            S_vals.append(S)

            # (b) ER=EPR correlation
            r_er, n_pairs = er_epr_correlation_vectorized(cs, W)
            if not np.isnan(r_er):
                r_er_vals.append(r_er)

            # Random DAG control
            dag_cs = random_dag_control(N, ordering_frac_target=cs.ordering_fraction(), rng=trial_rng)
            W_dag, evals_dag = sj_wightman(dag_cs)
            r_dag, _ = er_epr_correlation_vectorized(dag_cs, W_dag)
            if not np.isnan(r_dag):
                r_dag_vals.append(r_dag)

            # (c) Level spacing ratio
            lsr = level_spacing_ratio(evals)
            if not np.isnan(lsr):
                lsr_vals.append(lsr)

        elapsed = time.time() - t_start

        # Compute effective central charge: S = (c/3) ln(N) → c = 3*S/ln(N)
        S_mean = np.mean(S_vals)
        c_eff = 3.0 * S_mean / np.log(N)

        r_er_mean = np.mean(r_er_vals) if r_er_vals else float('nan')
        r_dag_mean = np.mean(r_dag_vals) if r_dag_vals else float('nan')
        lsr_mean = np.mean(lsr_vals) if lsr_vals else float('nan')

        results[N] = {
            'S_vals': S_vals, 'S_mean': S_mean, 'c_eff': c_eff,
            'r_er_vals': r_er_vals, 'r_er_mean': r_er_mean,
            'r_dag_vals': r_dag_vals, 'r_dag_mean': r_dag_mean,
            'lsr_vals': lsr_vals, 'lsr_mean': lsr_mean,
            'elapsed': elapsed,
        }

        print(f"{N:>5} | {n_trials:>6} | {S_mean:>8.3f} {c_eff:>8.3f} | "
              f"{r_er_mean:>8.3f} {r_dag_mean:>8.3f} | {lsr_mean:>8.4f} | "
              f"{elapsed:>5.0f}s", flush=True)

    # ─── Detailed analysis ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # (a) Central charge scaling
    print("\n--- (a) Effective Central Charge c = 3*S(N/2)/ln(N) ---")
    print("CFT prediction for 2D massless scalar: c = 1")
    print("Previous small-N result: c ≈ 3")
    print()
    print(f"  {'N':>5} {'S(N/2)':>10} {'±':>3} {'c_eff':>8} {'±':>3}")
    print("  " + "-" * 40)
    for N in N_values:
        r = results[N]
        S_std = np.std(r['S_vals']) if len(r['S_vals']) > 1 else 0
        c_vals = [3.0 * s / np.log(N) for s in r['S_vals']]
        c_std = np.std(c_vals) if len(c_vals) > 1 else 0
        print(f"  {N:>5} {r['S_mean']:>10.4f} {S_std:>5.3f} {r['c_eff']:>8.3f} {c_std:>5.3f}")

    # Fit S = a * ln(N) + b
    ln_N = np.array([np.log(N) for N in N_values])
    S_means = np.array([results[N]['S_mean'] for N in N_values])
    if len(N_values) > 2:
        coeffs = np.polyfit(ln_N, S_means, 1)
        c_fit = 3.0 * coeffs[0]
        print(f"\n  Linear fit S = {coeffs[0]:.4f} * ln(N) + {coeffs[1]:.4f}")
        print(f"  → c_eff from fit = {c_fit:.3f}")
        print(f"  Interpretation: {'approaching c=1' if c_fit < 2 else 'still far from c=1'}")

    # (b) ER=EPR
    print("\n--- (b) ER=EPR Correlation Scaling ---")
    print("Question: does r(causet) diverge from r(random DAG) at large N?")
    print()
    print(f"  {'N':>5} {'r_causet':>10} {'±':>5} {'r_DAG':>10} {'±':>5} {'gap':>8}")
    print("  " + "-" * 50)
    for N in N_values:
        r = results[N]
        r_er_std = np.std(r['r_er_vals']) if len(r['r_er_vals']) > 1 else 0
        r_dag_std = np.std(r['r_dag_vals']) if len(r['r_dag_vals']) > 1 else 0
        gap = r['r_er_mean'] - r['r_dag_mean']
        print(f"  {N:>5} {r['r_er_mean']:>10.4f} {r_er_std:>5.3f} "
              f"{r['r_dag_mean']:>10.4f} {r_dag_std:>5.3f} {gap:>8.4f}")

    # (c) Level spacing ratio
    print("\n--- (c) Quantum Chaos: Level Spacing Ratio ---")
    print("GUE: ⟨r⟩ ≈ 0.5996, GOE: ⟨r⟩ ≈ 0.5307, Poisson: ⟨r⟩ ≈ 0.3863")
    print()
    print(f"  {'N':>5} {'⟨r⟩':>10} {'±':>5} {'class':>12}")
    print("  " + "-" * 35)
    for N in N_values:
        r = results[N]
        lsr_std = np.std(r['lsr_vals']) if len(r['lsr_vals']) > 1 else 0
        lsr = r['lsr_mean']
        if lsr > 0.57:
            cls = "GUE"
        elif lsr > 0.50:
            cls = "GOE"
        elif lsr > 0.42:
            cls = "intermediate"
        else:
            cls = "Poisson"
        print(f"  {N:>5} {lsr:>10.4f} {lsr_std:>5.4f} {cls:>12}")

    # ─── Additional: entanglement profile at largest N ────────────────────
    print("\n--- Bonus: Entanglement profile at largest N ---")
    N_big = N_values[-1]
    to = TwoOrder(N_big, rng=np.random.default_rng(999))
    cs = to.to_causet()
    print(f"  Computing SJ vacuum for N={N_big}...", flush=True)
    t0 = time.time()
    W, evals = sj_wightman(cs)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    print(f"\n  {'frac':>6} {'|A|':>5} {'S(A)':>8}")
    print("  " + "-" * 25)
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        region_A = spatial_partition_v(to, frac)
        S = entanglement_entropy(W, region_A)
        print(f"  {frac:>6.1f} {len(region_A):>5} {S:>8.3f}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY & INTERPRETATION")
    print("=" * 80)

    c_final = results[N_values[-1]]['c_eff']
    r_er_final = results[N_values[-1]]['r_er_mean']
    r_dag_final = results[N_values[-1]]['r_dag_mean']
    lsr_final = results[N_values[-1]]['lsr_mean']

    print(f"""
  (a) Central charge: c_eff = {c_final:.3f} at N={N_values[-1]}
      {'CONVERGING toward c=1' if c_final < 2 else 'NOT converging toward c=1'}
      (c=1 is the CFT prediction for massless scalar in 2D)

  (b) ER=EPR: r_causet = {r_er_final:.3f}, r_DAG = {r_dag_final:.3f}
      Gap = {r_er_final - r_dag_final:.3f}
      {'Gap GROWING with N → genuine ER=EPR signal' if (r_er_final - r_dag_final) > (results[N_values[0]]['r_er_mean'] - results[N_values[0]]['r_dag_mean']) + 0.02 else 'Gap NOT growing → may be generic graph property'}

  (c) Quantum chaos: ⟨r⟩ = {lsr_final:.4f} at N={N_values[-1]}
      GUE=0.5996, GOE=0.5307, Poisson=0.3863
      {'Consistent with GUE (quantum chaotic)' if lsr_final > 0.55 else 'Not cleanly GUE — ' + ('GOE-like' if lsr_final > 0.50 else 'intermediate/Poisson')}

  Total compute time: {sum(results[N]['elapsed'] for N in N_values):.0f}s
""")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
