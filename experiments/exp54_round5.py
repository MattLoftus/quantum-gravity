"""
Experiment 54: Ideas 86-95 — Round 5 of 8+ Search

Ten genuinely new ideas exploiting what makes causets FUNDAMENTALLY different:
- Lorentzian signature (not just any partial order)
- Partial order axioms (transitivity, acyclicity)
- Embedding into Minkowski spacetime
- The BD action / partition function structure
- Connection to general relativity

KEY INSIGHT: Focus on ANALYTIC / STRUCTURAL results that random graphs cannot reproduce.

Ideas:
86. Longest chain scaling exponent (Ulam problem on causets vs random permutations)
87. Causal interval dimension spectrum (multifractal analysis of I(x,y) sizes)
88. Eigenvalue density of iΔ: analytic prediction from Marchenko-Pastur-like law
89. BD action gap: crystalline vs continuum action density gap scaling
90. Myrheim-Meyer dimension from SJ eigenvalues (spectral dim estimator)
91. Causal diamond entanglement entropy vs area (Bekenstein-Hawking from SJ)
92. Number variance Σ²(L) of PJ eigenvalues (more sensitive than <r>)
93. Longest antichain (spatial width) scaling at the phase transition
94. Retarded Green's function pole structure (propagator mass gap)
95. Spectral zeta function ζ(s) = Σ λ_k^{-s} of iΔ (analytic structure)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size
import time

rng = np.random.default_rng(42)


def sj_full(cs):
    """Return W and eigendecomposition of i*Delta."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), evals, evecs
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals, evecs


def entanglement_entropy(W, region):
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def level_spacing_ratio(evals):
    """Average level spacing ratio for positive eigenvalues."""
    pos = sorted(evals[evals > 1e-12])
    if len(pos) < 4:
        return 0.0
    spacings = np.diff(pos)
    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        if max(s1, s2) > 1e-15:
            ratios.append(min(s1, s2) / max(s1, s2))
    return float(np.mean(ratios)) if ratios else 0.0


def make_random_dag(N, density, rng):
    """Random DAG with given ordering fraction as null model."""
    cs = FastCausalSet(N)
    perm = rng.permutation(N)
    order = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < density:
                order[perm[i], perm[j]] = True
    # Transitively close
    for k in range(N):
        for i in range(N):
            if order[i, k]:
                for j in range(N):
                    if order[k, j]:
                        order[i, j] = True
    cs.order = order
    return cs


def longest_chain(cs):
    """Longest chain in the causal set."""
    N = cs.n
    dp = np.ones(N, dtype=int)
    # Need a topological order — use the order matrix
    # Simple DP: for each element, longest chain ending there
    for j in range(N):
        preds = np.where(cs.order[:, j])[0]
        if len(preds) > 0:
            dp[j] = np.max(dp[preds]) + 1
    return int(np.max(dp))


def longest_antichain(cs):
    """Longest antichain (maximum set of pairwise unrelated elements).
    This is NP-hard in general but for small N we can use Dilworth's theorem:
    max antichain = N - max matching in comparability graph.
    For simplicity, use a greedy approximation."""
    N = cs.n
    # Greedy: pick elements that are unrelated to all previously picked
    related = cs.order | cs.order.T
    antichain = []
    remaining = list(range(N))
    rng_local = np.random.default_rng(0)
    rng_local.shuffle(remaining)
    for elem in remaining:
        is_ok = True
        for a in antichain:
            if related[elem, a]:
                is_ok = False
                break
        if is_ok:
            antichain.append(elem)
    return len(antichain)


def mcmc_with_twoorders(N, beta, eps, n_steps=20000, n_therm=10000,
                          record_every=50, rng=None):
    """MCMC that stores TwoOrder objects."""
    if rng is None:
        rng = np.random.default_rng()
    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    samples_to = []
    actions = []
    n_acc = 0
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
        if step >= n_therm and step % record_every == 0:
            actions.append(current_S)
            samples_to.append(current.copy())
    return {
        'actions': np.array(actions),
        'samples': samples_to,
        'acceptance': n_acc / n_steps,
    }


# ================================================================
print("=" * 80)
print("IDEA 86: LONGEST CHAIN SCALING (Ulam Problem on Causets)")
print("=" * 80)
print("""
For a random permutation of N, the longest increasing subsequence has length
~ 2*sqrt(N) (Ulam's problem, proved by Baik-Deift-Johansson 1999).
For a 2-order, the longest chain IS the longest common subsequence of u and v.
Random 2-orders: longest chain ~ c * sqrt(N) with c = 2 (Ulam).
Question: Does the BD continuum phase change this scaling? And the crystalline?
If yes, this distinguishes phases via a WELL-KNOWN combinatorial quantity.
The Ulam scaling c=2 is a deep result connected to Tracy-Widom distribution.
""")

t0 = time.time()

# Test at multiple N
N_values = [30, 50, 80, 120, 200]
n_trials = 30

results_86 = {'random': {}, 'dag': {}}

for N in N_values:
    chains_random = []
    chains_dag = []
    for _ in range(n_trials):
        # Random 2-order (continuum phase at beta=0)
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        chains_random.append(longest_chain(cs))

        # Random DAG null with matched density
        f = np.sum(cs.order) / (N * (N - 1))
        cs_dag = make_random_dag(N, f * 2, rng)  # density param for random DAG
        chains_dag.append(longest_chain(cs_dag))

    results_86['random'][N] = (np.mean(chains_random), np.std(chains_random))
    results_86['dag'][N] = (np.mean(chains_dag), np.std(chains_dag))
    print(f"  N={N:>4}: chain_2order = {np.mean(chains_random):.1f} ± {np.std(chains_random):.1f}, "
          f"chain_DAG = {np.mean(chains_dag):.1f} ± {np.std(chains_dag):.1f}, "
          f"ratio_2order = {np.mean(chains_random)/np.sqrt(N):.3f}")

# Fit: chain ~ c * N^alpha
Ns = np.array(N_values)
chains_2o = np.array([results_86['random'][N][0] for N in N_values])
chains_dag = np.array([results_86['dag'][N][0] for N in N_values])

log_N = np.log(Ns)
log_c_2o = np.log(chains_2o)
log_c_dag = np.log(chains_dag)

slope_2o, intercept_2o, r_2o, _, _ = stats.linregress(log_N, log_c_2o)
slope_dag, intercept_dag, r_dag, _, _ = stats.linregress(log_N, log_c_dag)

print(f"\n  2-order: chain ~ N^{slope_2o:.3f} (r={r_2o:.4f}), c = {np.exp(intercept_2o):.3f}")
print(f"  Ulam prediction: chain ~ 2*N^0.5, so exponent = 0.5")
print(f"  Random DAG: chain ~ N^{slope_dag:.3f} (r={r_dag:.4f})")
print(f"  Time: {time.time()-t0:.1f}s")

# Now test at BD phase transition
print("\n  --- BD Phase Transition ---")
N_bd = 50
eps = 0.12
beta_c = 1.66 / (N_bd * eps**2)
betas_bd = [0, 0.5*beta_c, 0.9*beta_c, 1.1*beta_c, 1.5*beta_c, 2.0*beta_c]

for beta in betas_bd:
    if beta == 0:
        chains_bd = []
        for _ in range(20):
            to = TwoOrder(N_bd, rng=rng)
            cs = to.to_causet()
            chains_bd.append(longest_chain(cs))
        print(f"  beta/beta_c={0:.2f}: chain = {np.mean(chains_bd):.1f} ± {np.std(chains_bd):.1f}")
    else:
        result = mcmc_with_twoorders(N_bd, beta, eps, n_steps=15000, n_therm=7500,
                                      record_every=100, rng=rng)
        if result['samples']:
            chains_bd = [longest_chain(s.to_causet()) for s in result['samples'][-10:]]
            print(f"  beta/beta_c={beta/beta_c:.2f}: chain = {np.mean(chains_bd):.1f} ± {np.std(chains_bd):.1f}, "
                  f"acc={result['acceptance']:.1%}")

score_86 = "TBD"
print(f"\n  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 87: INTERVAL SIZE DISTRIBUTION (Multifractal Analysis)")
print("=" * 80)
print("""
For a sprinkled 2D causal set, the interval I(x,y) has |I| ~ tau^2 * rho
where tau is the proper time. The distribution P(|I|) encodes the geometry.
For 2D Minkowski: P(|I| = k) should follow a specific power law.
For random DAGs: P(|I|) is different (Poisson-like).
Question: Can we distinguish causets from random DAGs by the SHAPE of P(|I|)?
If P(|I|) has a power-law tail with a causet-specific exponent, that's analytic.
""")

t0 = time.time()

N = 100
n_trials = 20

# Collect interval size distributions
intervals_2order = []
intervals_dag = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    order_int = cs.order.astype(np.int32)
    # Interval sizes for all related pairs
    int_matrix = order_int @ order_int
    related_pairs = np.where(np.triu(cs.order, k=1))
    sizes = int_matrix[related_pairs]
    intervals_2order.extend(sizes.tolist())

    # Random DAG null
    f = np.sum(cs.order) / (N * (N - 1))
    cs_dag = make_random_dag(N, f * 2, rng)
    order_int_d = cs_dag.order.astype(np.int32)
    int_matrix_d = order_int_d @ order_int_d
    related_d = np.where(np.triu(cs_dag.order, k=1))
    sizes_d = int_matrix_d[related_d]
    intervals_dag.extend(sizes_d.tolist())

intervals_2order = np.array(intervals_2order)
intervals_dag = np.array(intervals_dag)

# Compare distributions
print(f"  2-order intervals: n={len(intervals_2order)}, mean={np.mean(intervals_2order):.2f}, "
      f"max={np.max(intervals_2order)}, std={np.std(intervals_2order):.2f}")
print(f"  DAG intervals:     n={len(intervals_dag)}, mean={np.mean(intervals_dag):.2f}, "
      f"max={np.max(intervals_dag)}, std={np.std(intervals_dag):.2f}")

# Fraction at each size
max_size = min(30, max(np.max(intervals_2order), np.max(intervals_dag)))
print(f"\n  {'k':>3} | {'P_2order(k)':>11} | {'P_DAG(k)':>11} | {'ratio':>8}")
print("  " + "-" * 45)
for k in range(max_size + 1):
    p_2o = np.mean(intervals_2order == k)
    p_dag = np.mean(intervals_dag == k)
    ratio = p_2o / p_dag if p_dag > 0 else float('inf')
    if p_2o > 0.001 or p_dag > 0.001:
        print(f"  {k:>3} | {p_2o:>11.4f} | {p_dag:>11.4f} | {ratio:>8.3f}")

# Power law fit for tail (k >= 2)
sizes_for_fit = intervals_2order[intervals_2order >= 2]
if len(sizes_for_fit) > 10:
    # MLE for power law exponent
    alpha_mle = 1 + len(sizes_for_fit) / np.sum(np.log(sizes_for_fit / 1.5))
    print(f"\n  2-order tail exponent (MLE, k>=2): alpha = {alpha_mle:.3f}")

sizes_for_fit_d = intervals_dag[intervals_dag >= 2]
if len(sizes_for_fit_d) > 10:
    alpha_mle_d = 1 + len(sizes_for_fit_d) / np.sum(np.log(sizes_for_fit_d / 1.5))
    print(f"  DAG tail exponent (MLE, k>=2): alpha = {alpha_mle_d:.3f}")

# KS test
if len(intervals_2order) > 0 and len(intervals_dag) > 0:
    ks_stat, ks_p = stats.ks_2samp(intervals_2order, intervals_dag)
    print(f"\n  KS test (2-order vs DAG): D={ks_stat:.4f}, p={ks_p:.2e}")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 88: EIGENVALUE DENSITY OF iΔ — Analytic Prediction")
print("=" * 80)
print("""
For a random antisymmetric matrix from GOE, the eigenvalue density follows
the Wigner semicircle law: rho(lambda) = (2/pi*R^2) * sqrt(R^2 - lambda^2).
For the Pauli-Jordan operator iΔ on a random 2-order, is the density semicircular?
If NOT, what IS it? The deviation from semicircle encodes the causal structure.
This could give an ANALYTIC formula for the spectral density of iΔ.
""")

t0 = time.time()

N_vals = [50, 100, 200]
n_trials_per = 30

for N in N_vals:
    all_evals_2o = []
    all_evals_antisym = []

    for _ in range(n_trials_per):
        # 2-order
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        C = cs.order.astype(float)
        Delta = (2.0 / N) * (C.T - C)
        H = 1j * Delta
        evals = np.linalg.eigvalsh(H)
        # Normalize by sqrt(N) to compare with semicircle
        all_evals_2o.extend((evals / np.sqrt(np.mean(evals**2))).tolist())

        # Random antisymmetric null
        A = rng.standard_normal((N, N))
        A = (A - A.T) / 2
        A *= np.sqrt(2.0 / N)  # Match variance
        evals_a = np.linalg.eigvalsh(1j * A)
        all_evals_antisym.extend((evals_a / np.sqrt(np.mean(evals_a**2))).tolist())

    evals_2o = np.array(all_evals_2o)
    evals_antisym = np.array(all_evals_antisym)

    # Compute histogram and compare
    bins = np.linspace(-3, 3, 61)
    hist_2o, _ = np.histogram(evals_2o, bins=bins, density=True)
    hist_antisym, _ = np.histogram(evals_antisym, bins=bins, density=True)

    # Semicircle
    bin_centers = (bins[:-1] + bins[1:]) / 2
    R = np.sqrt(2)  # for standard normalization
    semicircle = np.where(np.abs(bin_centers) < R,
                          (2 / (np.pi * R**2)) * np.sqrt(R**2 - bin_centers**2), 0)

    # L2 distance from semicircle
    L2_2o = np.sqrt(np.sum((hist_2o - semicircle)**2) * (bins[1] - bins[0]))
    L2_antisym = np.sqrt(np.sum((hist_antisym - semicircle)**2) * (bins[1] - bins[0]))

    # KL divergence (smoothed)
    h2o_smooth = hist_2o + 1e-10
    ha_smooth = hist_antisym + 1e-10
    h2o_smooth /= h2o_smooth.sum()
    ha_smooth /= ha_smooth.sum()
    KL = np.sum(h2o_smooth * np.log(h2o_smooth / ha_smooth))

    print(f"  N={N}: L2(2-order vs semicircle) = {L2_2o:.4f}, "
          f"L2(antisym vs semicircle) = {L2_antisym:.4f}, "
          f"KL(2-order || antisym) = {KL:.6f}")

    # Check: kurtosis of eigenvalue distribution (semicircle has kurtosis = 2)
    kurt_2o = stats.kurtosis(evals_2o, fisher=False)
    kurt_antisym = stats.kurtosis(evals_antisym, fisher=False)
    print(f"        kurtosis: 2-order={kurt_2o:.3f}, antisym={kurt_antisym:.3f}, "
          f"semicircle=2.000")

    # Check: fraction of eigenvalues beyond semicircle edge
    frac_outside_2o = np.mean(np.abs(evals_2o) > R)
    frac_outside_antisym = np.mean(np.abs(evals_antisym) > R)
    print(f"        frac |λ|>R: 2-order={frac_outside_2o:.4f}, antisym={frac_outside_antisym:.4f}")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 89: BD ACTION GAP — Crystalline vs Continuum Action Density")
print("=" * 80)
print("""
At a first-order transition, the TWO phases have different action densities.
The ACTION GAP = S_cryst/N - S_cont/N should be N-independent (or have
a specific scaling). This is a QUANTITATIVE prediction about the BD partition
function that could be compared with analytic calculations.
Key: Does the gap scale with N? Does it depend on epsilon?
""")

t0 = time.time()

eps = 0.12
N_values_bd = [30, 40, 50, 60, 70]

print(f"  eps = {eps}")
print(f"  {'N':>4} | {'beta_c':>7} | {'S/N (cont)':>11} | {'S/N (cryst)':>12} | {'gap':>8}")
print("  " + "-" * 55)

gaps = []
for N in N_values_bd:
    beta_c = 1.66 / (N * eps**2)

    # Continuum phase (beta = 0.5 * beta_c)
    result_cont = mcmc_with_twoorders(N, 0.5 * beta_c, eps,
                                       n_steps=15000, n_therm=7500,
                                       record_every=50, rng=rng)

    # Crystalline phase (beta = 2.0 * beta_c)
    result_cryst = mcmc_with_twoorders(N, 2.0 * beta_c, eps,
                                        n_steps=15000, n_therm=7500,
                                        record_every=50, rng=rng)

    S_cont = np.mean(result_cont['actions']) / N
    S_cryst = np.mean(result_cryst['actions']) / N
    gap = S_cryst - S_cont
    gaps.append(gap)

    print(f"  {N:>4} | {beta_c:>7.2f} | {S_cont:>11.4f} | {S_cryst:>12.4f} | {gap:>+8.4f}")

# Fit gap vs N
gaps = np.array(gaps)
Ns_bd = np.array(N_values_bd)

if len(gaps) > 2:
    slope_gap, intercept_gap, r_gap, p_gap, _ = stats.linregress(Ns_bd, gaps)
    print(f"\n  Gap scaling: gap = {slope_gap:.5f}*N + {intercept_gap:.4f} (r={r_gap:.3f}, p={p_gap:.3e})")
    print(f"  If slope ≈ 0: gap is N-independent → genuine first-order transition")
    print(f"  If slope ≠ 0: gap changes with N → could be crossover or finite-size effect")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 90: SPECTRAL DIMENSION FROM SJ EIGENVALUES")
print("=" * 80)
print("""
The spectral dimension from the LAPLACIAN is unreliable on causets (link density artifact).
But the SJ operator iΔ IS intrinsic to the causal structure, not the link graph.
Define a spectral dimension from iΔ's eigenvalue density:
  d_spec = -2 * d(ln N(E)) / d(ln E)
where N(E) = integrated density of states.
For d-dimensional spacetime: N(E) ~ E^{d/2} → d_spec = d.
Does the SJ spectral dimension correctly give d=2 for 2-orders?
""")

t0 = time.time()

N_vals = [50, 100, 200]
n_trials_per = 20

for N in N_vals:
    d_specs_2o = []
    d_specs_dag = []

    for _ in range(n_trials_per):
        # 2-order
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        C = cs.order.astype(float)
        Delta = (2.0 / N) * (C.T - C)
        H = 1j * Delta
        evals = np.sort(np.linalg.eigvalsh(H))
        pos_evals = evals[evals > 1e-12]

        if len(pos_evals) > 5:
            # Integrated density of states: N(E) = number of eigenvalues <= E
            # Use middle 50% to avoid edge effects
            n_pos = len(pos_evals)
            lo, hi = n_pos // 4, 3 * n_pos // 4
            log_E = np.log(pos_evals[lo:hi])
            log_N_E = np.log(np.arange(lo, hi) + 1)
            slope, _, r, _, _ = stats.linregress(log_E, log_N_E)
            d_spec = 2 * slope  # N(E) ~ E^{d/2} → slope = d/2
            d_specs_2o.append(d_spec)

        # Random DAG
        f = np.sum(cs.order) / (N * (N - 1))
        cs_dag = make_random_dag(N, f * 2, rng)
        C_d = cs_dag.order.astype(float)
        Delta_d = (2.0 / N) * (C_d.T - C_d)
        H_d = 1j * Delta_d
        evals_d = np.sort(np.linalg.eigvalsh(H_d))
        pos_evals_d = evals_d[evals_d > 1e-12]

        if len(pos_evals_d) > 5:
            n_pos = len(pos_evals_d)
            lo, hi = n_pos // 4, 3 * n_pos // 4
            log_E = np.log(pos_evals_d[lo:hi])
            log_N_E = np.log(np.arange(lo, hi) + 1)
            slope, _, r, _, _ = stats.linregress(log_E, log_N_E)
            d_specs_dag.append(2 * slope)

    print(f"  N={N}: d_spec(2-order) = {np.mean(d_specs_2o):.3f} ± {np.std(d_specs_2o):.3f}, "
          f"d_spec(DAG) = {np.mean(d_specs_dag):.3f} ± {np.std(d_specs_dag):.3f}")

print(f"  Target: d_spec = 2.0 for 2D Minkowski")
print(f"  If 2-order gives ~2 but DAG doesn't → SJ spectral dimension works!")
print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 91: CAUSAL DIAMOND ENTANGLEMENT vs 'AREA'")
print("=" * 80)
print("""
Bekenstein-Hawking: S = A/(4G). In 2D, 'area' of a causal diamond boundary = 2 points.
For a massless scalar in 2D: S(A) = (c/3) * ln(|A|/a) + const.
Take causal diamonds of varying size (by proper time height) within a 2-order.
Compute S(diamond) from SJ vacuum. Does it scale as ln(volume)?
Key: This uses the GEOMETRIC structure of the 2-order (the diamond),
not just an arbitrary bipartition.
""")

t0 = time.time()

N = 100
n_trials_per = 15

for trial in range(n_trials_per if trial == 0 else 0):  # Just analyze first trial carefully
    pass

# Run the actual analysis
S_vs_vol = []
for trial in range(n_trials_per):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Find causal diamonds of varying size
    # Pick a central element and grow diamonds around it
    # Diamond: all elements in I(past, future) for various past/future separations
    # Use the 2-order coordinates to define diamonds
    t_coords = (to.u + to.v) / (2.0 * N)
    x_coords = (to.v - to.u) / (2.0 * N)

    # Pick center near middle
    center_t = 0.5
    center_x = 0.0

    for radius in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        # Elements inside the causal diamond centered at (center_t, center_x) with radius
        # In 2D Minkowski: diamond is |t - center_t| + |x - center_x| < radius
        inside = []
        outside = []
        for i in range(N):
            if abs(t_coords[i] - center_t) + abs(x_coords[i] - center_x) < radius:
                inside.append(i)
            else:
                outside.append(i)

        if len(inside) >= 3 and len(outside) >= 3:
            S = entanglement_entropy(W, inside)
            S_vs_vol.append((len(inside), S))

if S_vs_vol:
    vols, entropies = zip(*S_vs_vol)
    vols, entropies = np.array(vols), np.array(entropies)

    # Fit S = a * ln(vol) + b
    valid = vols > 0
    log_vols = np.log(vols[valid])
    S_vals = entropies[valid]
    if len(log_vols) > 5:
        slope, intercept, r_val, p_val, _ = stats.linregress(log_vols, S_vals)
        print(f"  S = {slope:.3f} * ln(vol) + {intercept:.3f}")
        print(f"  r = {r_val:.4f}, p = {p_val:.2e}")
        print(f"  CFT prediction: slope = c/3 = 1/3 for c=1")
        print(f"  Our c_eff = {3*slope:.3f}")

    # Null: random bipartitions of same sizes
    S_null = []
    for vol_target in sorted(set(vols)):
        region_rand = list(rng.choice(N, size=int(vol_target), replace=False))
        S_r = entanglement_entropy(W, region_rand)
        S_null.append((int(vol_target), S_r))

    if S_null:
        vols_n, S_n = zip(*S_null)
        vols_n, S_n = np.array(vols_n), np.array(S_n)
        valid_n = vols_n > 0
        if np.sum(valid_n) > 3:
            slope_n, _, r_n, _, _ = stats.linregress(np.log(vols_n[valid_n]), S_n[valid_n])
            print(f"  Null (random regions): slope = {slope_n:.3f}, c_eff = {3*slope_n:.3f}, r = {r_n:.4f}")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 92: NUMBER VARIANCE Σ²(L) OF PJ EIGENVALUES")
print("=" * 80)
print("""
The number variance Σ²(L) = Var(n(E, E+L)) is a more sensitive test of
spectral statistics than <r>. For:
  - Poisson: Σ²(L) = L (linear)
  - GOE: Σ²(L) ~ (2/π²)(ln(2πL) + γ + 1) (logarithmic)
  - GUE: Σ²(L) ~ (1/π²)(ln(2πL) + γ + 1) (logarithmic, half GOE)
The number variance tests LONG-RANGE spectral correlations, while <r> is local.
If causets show logarithmic number variance → strong RMT universality claim.
""")

t0 = time.time()

N = 100
n_trials = 30
L_values = np.linspace(0.5, 15, 30)

# Collect unfolded eigenvalues
all_unfolded_2o = []
all_unfolded_antisym = []

for trial in range(n_trials):
    # 2-order
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals = np.sort(np.linalg.eigvalsh(H))
    pos_evals = evals[evals > 1e-12]

    if len(pos_evals) > 10:
        # Unfold: map to uniform density
        # Use cumulative empirical distribution
        n_pos = len(pos_evals)
        unfolded = np.arange(1, n_pos + 1, dtype=float)  # Simplest unfolding
        all_unfolded_2o.append(unfolded)

    # Random antisymmetric
    A = rng.standard_normal((N, N))
    A = (A - A.T) / 2
    A *= np.sqrt(2.0 / N)
    evals_a = np.sort(np.linalg.eigvalsh(1j * A))
    pos_evals_a = evals_a[evals_a > 1e-12]

    if len(pos_evals_a) > 10:
        n_pos = len(pos_evals_a)
        unfolded = np.arange(1, n_pos + 1, dtype=float)
        all_unfolded_antisym.append(unfolded)

# Better: use the actual eigenvalue positions with proper unfolding
# Redo with polynomial unfolding
all_sigma2_2o = np.zeros(len(L_values))
all_sigma2_antisym = np.zeros(len(L_values))
n_good_2o = 0
n_good_antisym = 0

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    evals = np.sort(np.linalg.eigvalsh(1j * Delta))
    pos = evals[evals > 1e-12]

    if len(pos) < 15:
        continue

    # Polynomial unfolding (degree 5)
    x = np.arange(len(pos), dtype=float)
    try:
        coeffs = np.polyfit(pos, x, deg=min(5, len(pos) - 1))
        unfolded = np.polyval(coeffs, pos)
    except:
        continue

    # Number variance for each L
    for li, L in enumerate(L_values):
        counts = []
        for start in np.linspace(unfolded[0], unfolded[-1] - L, 20):
            n_in = np.sum((unfolded >= start) & (unfolded < start + L))
            counts.append(n_in)
        if counts:
            all_sigma2_2o[li] += np.var(counts)
    n_good_2o += 1

    # Random antisymmetric
    A = rng.standard_normal((N, N))
    A = (A - A.T) / 2
    A *= np.sqrt(2.0 / N)
    evals_a = np.sort(np.linalg.eigvalsh(1j * A))
    pos_a = evals_a[evals_a > 1e-12]

    if len(pos_a) < 15:
        continue

    try:
        coeffs = np.polyfit(pos_a, np.arange(len(pos_a), dtype=float), deg=min(5, len(pos_a) - 1))
        unfolded_a = np.polyval(coeffs, pos_a)
    except:
        continue

    for li, L in enumerate(L_values):
        counts = []
        for start in np.linspace(unfolded_a[0], unfolded_a[-1] - L, 20):
            n_in = np.sum((unfolded_a >= start) & (unfolded_a < start + L))
            counts.append(n_in)
        if counts:
            all_sigma2_antisym[li] += np.var(counts)
    n_good_antisym += 1

if n_good_2o > 0:
    all_sigma2_2o /= n_good_2o
if n_good_antisym > 0:
    all_sigma2_antisym /= n_good_antisym

# Compare with RMT predictions
print(f"  Computed from {n_good_2o} 2-orders, {n_good_antisym} antisymmetric matrices")
print(f"\n  {'L':>6} | {'Σ²(2-order)':>12} | {'Σ²(antisym)':>12} | {'Σ²(Poisson)':>12} | {'Σ²(GUE)':>8}")
print("  " + "-" * 60)
gamma_em = 0.5772
for li in range(0, len(L_values), 3):
    L = L_values[li]
    sigma2_poisson = L
    sigma2_gue = (1/np.pi**2) * (np.log(max(2*np.pi*L, 0.01)) + gamma_em + 1) if L > 0.1 else L
    print(f"  {L:>6.2f} | {all_sigma2_2o[li]:>12.4f} | {all_sigma2_antisym[li]:>12.4f} | "
          f"{sigma2_poisson:>12.4f} | {sigma2_gue:>8.4f}")

# Is it logarithmic?
valid_L = L_values > 1.0
if np.any(valid_L):
    log_L = np.log(L_values[valid_L])
    s2_vals = all_sigma2_2o[valid_L]
    if np.std(s2_vals) > 0:
        slope_s2, intercept_s2, r_s2, _, _ = stats.linregress(log_L, s2_vals)
        print(f"\n  Σ²(L) vs ln(L) fit (L>1): slope = {slope_s2:.4f}, r = {r_s2:.4f}")
        print(f"  GUE prediction: slope = 1/π² ≈ {1/np.pi**2:.4f}")
        print(f"  GOE prediction: slope = 2/π² ≈ {2/np.pi**2:.4f}")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 93: LONGEST ANTICHAIN AT THE PHASE TRANSITION")
print("=" * 80)
print("""
The longest antichain = maximum set of pairwise spacelike-separated elements.
In a continuum d-dim spacetime: width ~ N^{(d-1)/d}. In 2D: width ~ N^{1/2}.
In a total order (crystalline): width = 1.
At the BD phase transition: what happens to the antichain width?
Does it jump discontinuously (first-order) or show critical scaling?
The antichain width is a GEOMETRIC observable that can't be faked by link density.
""")

t0 = time.time()

N_bd = 50
eps = 0.12
beta_c = 1.66 / (N_bd * eps**2)

betas_scan = np.linspace(0, 2.5 * beta_c, 12)

print(f"  N = {N_bd}, eps = {eps}, beta_c = {beta_c:.2f}")
print(f"\n  {'beta/beta_c':>12} | {'width':>6} | {'height':>7} | {'f_order':>7} | {'S/N':>7}")
print("  " + "-" * 55)

for beta in betas_scan:
    if beta == 0:
        widths = []
        heights = []
        f_orders = []
        s_ns = []
        for _ in range(15):
            to = TwoOrder(N_bd, rng=rng)
            cs = to.to_causet()
            widths.append(longest_antichain(cs))
            heights.append(longest_chain(cs))
            f_orders.append(np.sum(cs.order) / (N_bd * (N_bd - 1)))
            s_ns.append(bd_action_corrected(cs, eps) / N_bd)
        print(f"  {0:>12.2f} | {np.mean(widths):>6.1f} | {np.mean(heights):>7.1f} | "
              f"{np.mean(f_orders):>7.3f} | {np.mean(s_ns):>7.3f}")
    else:
        result = mcmc_with_twoorders(N_bd, beta, eps, n_steps=12000, n_therm=6000,
                                      record_every=100, rng=rng)
        if result['samples']:
            samples = result['samples'][-8:]
            widths = [longest_antichain(s.to_causet()) for s in samples]
            heights = [longest_chain(s.to_causet()) for s in samples]
            f_orders = [np.sum(s.to_causet().order) / (N_bd * (N_bd - 1)) for s in samples]
            s_ns = [bd_action_corrected(s.to_causet(), eps) / N_bd for s in samples]
            print(f"  {beta/beta_c:>12.2f} | {np.mean(widths):>6.1f} | {np.mean(heights):>7.1f} | "
                  f"{np.mean(f_orders):>7.3f} | {np.mean(s_ns):>7.3f}")

# Null check: random 2-order antichain scaling
print(f"\n  --- Random 2-order antichain scaling ---")
for N in [30, 50, 80, 120, 200]:
    ws = []
    for _ in range(20):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        ws.append(longest_antichain(cs))
    print(f"  N={N:>4}: width = {np.mean(ws):.1f} ± {np.std(ws):.1f}, "
          f"width/sqrt(N) = {np.mean(ws)/np.sqrt(N):.3f}")

print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 94: RETARDED GREEN'S FUNCTION — Mass Gap from SJ")
print("=" * 80)
print("""
The retarded Green's function G_R(x,y) = theta(x>y) * Delta(x,y) on a causal set.
In Fourier space for a MASSIVE scalar: G_R(k) ~ 1/(k^2 - m^2 + i*epsilon).
Even for a MASSLESS scalar on a causal set, discreteness could induce an
effective mass gap m_eff ~ 1/l_P.
Extract: from the SJ propagator, compute the 'time-domain' correlator
G(tau) = <W(t+tau, x) W(t, x)> averaged over positions.
The exponential decay rate gives the mass gap.
If m_eff ~ 1/sqrt(N) → discreteness-induced mass gap scales correctly.
""")

t0 = time.time()

N_vals = [50, 100, 200]
n_trials = 15

for N in N_vals:
    mass_gaps = []
    mass_gaps_null = []

    for _ in range(n_trials):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        W, evals, _ = sj_full(cs)

        # Time coordinate
        t_coords = (to.u + to.v) / (2.0 * N)

        # Compute time-separated correlator
        # G(tau) = mean of |W[i,j]| for pairs with t_j - t_i ≈ tau
        n_bins = 10
        tau_bins = np.linspace(0.05, 0.45, n_bins)
        G_tau = np.zeros(n_bins)
        G_count = np.zeros(n_bins)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dt = abs(t_coords[j] - t_coords[i])
                bin_idx = np.searchsorted(tau_bins, dt) - 1
                if 0 <= bin_idx < n_bins:
                    G_tau[bin_idx] += abs(W[i, j])
                    G_count[bin_idx] += 1

        valid_bins = G_count > 5
        if np.sum(valid_bins) > 4:
            G_avg = G_tau[valid_bins] / G_count[valid_bins]
            taus = tau_bins[valid_bins]

            # Fit exponential decay: G(tau) ~ A * exp(-m * tau)
            log_G = np.log(G_avg + 1e-15)
            slope, intercept, r_val, _, _ = stats.linregress(taus, log_G)
            m_eff = -slope  # mass gap
            mass_gaps.append(m_eff)

        # Null: random antisymmetric matrix
        A = rng.standard_normal((N, N))
        A = (A - A.T) / 2
        A *= np.sqrt(2.0 / N)
        evals_null = np.sort(np.linalg.eigvalsh(1j * A))
        pos_null = evals_null > 1e-12
        if np.any(pos_null):
            evecs_null = np.linalg.eigh(1j * A)[1]
            W_null = (evecs_null[:, pos_null] @ np.diag(evals_null[pos_null]) @
                      evecs_null[:, pos_null].conj().T).real
            G_tau_n = np.zeros(n_bins)
            G_count_n = np.zeros(n_bins)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    dt = abs(t_coords[j] - t_coords[i])
                    bin_idx = np.searchsorted(tau_bins, dt) - 1
                    if 0 <= bin_idx < n_bins:
                        G_tau_n[bin_idx] += abs(W_null[i, j])
                        G_count_n[bin_idx] += 1

            valid_n = G_count_n > 5
            if np.sum(valid_n) > 4:
                G_avg_n = G_tau_n[valid_n] / G_count_n[valid_n]
                taus_n = tau_bins[valid_n]
                log_G_n = np.log(G_avg_n + 1e-15)
                slope_n, _, _, _, _ = stats.linregress(taus_n, log_G_n)
                mass_gaps_null.append(-slope_n)

    if mass_gaps:
        print(f"  N={N:>4}: m_eff(2-order) = {np.mean(mass_gaps):.3f} ± {np.std(mass_gaps):.3f}, "
              f"m_eff(null) = {np.mean(mass_gaps_null):.3f} ± {np.std(mass_gaps_null):.3f}, "
              f"m*sqrt(N) = {np.mean(mass_gaps)*np.sqrt(N):.3f}")

print(f"\n  If m_eff*sqrt(N) ≈ const → mass gap scales as 1/sqrt(N) (discreteness-induced)")
print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
print("\n" + "=" * 80)
print("IDEA 95: SPECTRAL ZETA FUNCTION ζ(s) = Σ λ_k^{-s}")
print("=" * 80)
print("""
The spectral zeta function encodes ALL spectral information.
For a d-dimensional manifold: ζ(s) has a pole at s = d/2 (Weyl's law).
Compute ζ(s) for the positive eigenvalues of iΔ on 2-orders.
If there's a pole structure at s = 1 (= d/2 for d=2), this recovers the
spacetime dimension from the SJ spectrum — a deep analytic connection.
Also: the residue at the pole gives the volume (Weyl's formula).
Compare with random antisymmetric matrices (should NOT have pole at s=1).
""")

t0 = time.time()

N_vals = [50, 100, 200]
s_values = np.linspace(0.3, 3.0, 50)
n_trials = 20

for N in N_vals:
    zeta_2o = np.zeros(len(s_values))
    zeta_antisym = np.zeros(len(s_values))
    n_good = 0

    for _ in range(n_trials):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        C = cs.order.astype(float)
        Delta = (2.0 / N) * (C.T - C)
        evals = np.sort(np.linalg.eigvalsh(1j * Delta))
        pos = evals[evals > 1e-12]

        if len(pos) < 5:
            continue

        for si, s in enumerate(s_values):
            zeta_2o[si] += np.sum(pos ** (-s))

        # Random antisymmetric
        A = rng.standard_normal((N, N))
        A = (A - A.T) / 2
        A *= np.sqrt(2.0 / N)
        evals_a = np.sort(np.linalg.eigvalsh(1j * A))
        pos_a = evals_a[evals_a > 1e-12]

        if len(pos_a) < 5:
            continue

        for si, s in enumerate(s_values):
            zeta_antisym[si] += np.sum(pos_a ** (-s))

        n_good += 1

    if n_good > 0:
        zeta_2o /= n_good
        zeta_antisym /= n_good

        # Find where zeta diverges most rapidly (pole location)
        # Take derivative of log(zeta)
        log_zeta_2o = np.log(zeta_2o + 1e-30)
        log_zeta_antisym = np.log(zeta_antisym + 1e-30)

        # Numerical derivative
        d_log_zeta_2o = np.gradient(log_zeta_2o, s_values)
        d_log_zeta_antisym = np.gradient(log_zeta_antisym, s_values)

        # Most negative derivative = steepest growth = near pole
        pole_idx_2o = np.argmin(d_log_zeta_2o)
        pole_idx_antisym = np.argmin(d_log_zeta_antisym)

        print(f"  N={N}: pole(2-order) at s ≈ {s_values[pole_idx_2o]:.2f}, "
              f"pole(antisym) at s ≈ {s_values[pole_idx_antisym]:.2f}")
        print(f"        ζ(1) = {zeta_2o[np.argmin(np.abs(s_values - 1))]:.2f} (2-order), "
              f"{zeta_antisym[np.argmin(np.abs(s_values - 1))]:.2f} (antisym)")
        print(f"        ζ(0.5) = {zeta_2o[np.argmin(np.abs(s_values - 0.5))]:.2f} (2-order), "
              f"{zeta_antisym[np.argmin(np.abs(s_values - 0.5))]:.2f} (antisym)")

        # Check ratio zeta_2order / zeta_antisym at different s
        ratio = zeta_2o / (zeta_antisym + 1e-30)
        # If the pole structure differs, the ratio will diverge or vanish at certain s
        max_ratio_idx = np.argmax(np.abs(np.log(ratio + 1e-30)))
        print(f"        Max log(ratio) at s = {s_values[max_ratio_idx]:.2f}, "
              f"ratio = {ratio[max_ratio_idx]:.3f}")

print(f"\n  Weyl prediction for 2D: pole at s = d/2 = 1")
print(f"  If 2-order has pole at s=1 but random matrix doesn't → dimension recovery!")
print(f"  [Time: {time.time()-t0:.1f}s]")


# ================================================================
# FINAL SCORING
# ================================================================
print("\n\n")
print("=" * 80)
print("FINAL SCORING — ALL 10 IDEAS")
print("=" * 80)
print("""
Scoring criteria:
- 8+: Genuinely novel, passes all null tests, analytic component, connects to
      known result or physical prediction. Would be accepted at a good journal.
- 6-7: Interesting observation, passes null test, but either not novel enough
      or lacks analytic depth.
- 4-5: Moderate interest, partially passes null test, or result is expected.
- 1-3: Null model reproduces it, N-dependent, or trivially explainable.

NOTE: Being BRUTALLY HONEST. Most ideas are 3-5. Only 8+ if truly outstanding.
""")

scores = {
    86: ("Longest chain scaling (Ulam)", 4,
         "Exponent 0.385 (not 0.5 as Ulam predicts). Interesting deviation but: "
         "(a) this is likely due to the longest COMMON subsequence being different "
         "from longest INCREASING subsequence (LCS vs LIS), (b) no change across BD "
         "transition — chain length barely moves, (c) the DAG has a DIFFERENT exponent "
         "(0.56) but that's trivially expected since DAGs have higher density. "
         "NOT novel enough for a paper."),

    87: ("Interval size distribution", 4,
         "2-orders and DAGs have statistically different P(|I|) distributions (KS p=0). "
         "But this is EXPECTED: 2-orders have a specific embedding geometry that constrains "
         "interval sizes. The power law exponents (1.55 vs 1.37) differ but the interpretation "
         "is straightforward — not surprising or deep. Already implicit in Myrheim-Meyer work."),

    88: ("Eigenvalue density of iDelta", 5,
         "iDelta on 2-orders is dramatically NON-semicircular — kurtosis grows linearly "
         "with N (15→28→56), indicating extreme heavy tails. The density diverges from "
         "the random antisymmetric result (kurtosis=2, semicircular). This is genuinely "
         "interesting: the causal structure forces a non-Wigner spectral density. "
         "BUT: (a) we don't have an analytic formula for what the density IS, "
         "(b) the growing kurtosis suggests the density doesn't converge to any fixed shape, "
         "(c) without a prediction, this is 'the density is different' — not a paper."),

    89: ("BD action gap", 3,
         "The gap S_cryst - S_cont is NEGATIVE and small (~0.005) and shrinks with N. "
         "This is NOT a clean first-order gap — it's consistent with the transition being "
         "a crossover at these small sizes, or the MCMC not thermalizing properly in the "
         "crystalline phase (acceptance rates still ~80%). Not useful."),

    90: ("SJ spectral dimension", 4,
         "d_spec ~ 1.3 for 2-orders, ~1.6 for DAGs. Neither gives d=2. "
         "The SJ eigenvalue density does NOT follow the Weyl law N(E)~E^{d/2}. "
         "The 2-order gives LOWER d_spec than the DAG, opposite of what we'd want. "
         "The SJ operator is not a Laplacian, so Weyl's law doesn't directly apply. "
         "Interesting failure — but a failure."),

    91: ("Causal diamond entropy vs area", 3,
         "S ~ 2.1*ln(vol) giving c_eff=6.4. But the NULL (random regions) gives "
         "almost identical scaling (c_eff=6.9, r=0.94). The geometric diamond structure "
         "provides NO advantage over random bipartitions. This is the same c_eff divergence "
         "problem seen in exp49. The SJ vacuum on random 2-orders has S growing too fast. "
         "KILLED by null model."),

    92: ("Number variance Sigma^2(L)", 5,
         "Sigma^2 grows linearly for 2-orders (slope 1.27 vs ln(L)) — much faster than "
         "GUE (slope 0.10) or GOE (slope 0.20). The random antisymmetric matrices DO show "
         "logarithmic growth (consistent with GUE). So 2-orders are LESS correlated than "
         "pure random matrices — the causal structure partially decorrelates eigenvalues. "
         "This CONTRADICTS idea that causets have RMT statistics! The <r> test said GUE-like "
         "but Sigma^2 says NOT GUE. This is actually interesting as a CAVEAT to the "
         "random-matrix paper — <r> captures short-range correlations (which are GUE-like) "
         "but long-range correlations are weaker. Worth noting but undermines rather than "
         "strengthens the RMT result."),

    93: ("Antichain width at transition", 4,
         "Width/sqrt(N) ~ 1.09 for random 2-orders — consistent with sqrt(N) scaling. "
         "Across the BD transition: NO clear jump in width (7-10, noisy). "
         "The greedy antichain algorithm is approximate (not exact Dilworth), adding noise. "
         "The transition at N=50 with short MCMC is probably not thermalized. "
         "Not a useful result."),

    94: ("Mass gap from SJ propagator", 5,
         "m_eff ~ 2.2-2.5 for 2-orders (nonzero!), while null gives ~0 (no gap). "
         "This is genuinely interesting: discreteness induces a mass gap in the propagator. "
         "BUT: m_eff*sqrt(N) is NOT constant (16→25→32), growing roughly as N^0.35. "
         "So the scaling is not 1/sqrt(N) as hoped, but weaker — maybe 1/N^0.15. "
         "The mass gap is real but the scaling with N is unclear and the N-dependence "
         "makes it hard to extract a physical prediction. Needs larger N to pin down."),

    95: ("Spectral zeta function", 5,
         "The ratio zeta_2order/zeta_antisym peaks near s=1 (N=50,100) and grows with N. "
         "This is consistent with 2-orders having more small eigenvalues than random matrices "
         "(heavier tail), which amplifies zeta at larger s. Both have poles at s~0.3 "
         "(same location — not different!). The ratio growing with N at s~1 might indicate "
         "a difference in the spectral dimension, but it's not a clean pole at s=d/2=1. "
         "Without an analytic prediction, this is descriptive rather than predictive."),
}

print()
for idea_num in sorted(scores.keys()):
    name, score, comment = scores[idea_num]
    print(f"  IDEA {idea_num}: {name}")
    print(f"  SCORE: {score}/10")
    print(f"  {comment}")
    print()

# Highlight the most interesting findings
print("=" * 80)
print("SUMMARY: MOST INTERESTING FINDINGS")
print("=" * 80)
print("""
1. IDEA 88 (eigenvalue density): The kurtosis of iDelta eigenvalues DIVERGES
   with N (15→28→56), meaning the spectral density has increasingly heavy tails.
   This is NOT seen for random antisymmetric matrices (kurtosis=2, stable).
   The causal structure forces a non-Wigner spectral density with a specific
   N-dependent shape. This could lead to an analytic prediction if we can
   derive the density from the 2-order structure.

2. IDEA 92 (number variance): The number variance reveals that while <r> is
   GUE-like (short-range level repulsion), the LONG-RANGE correlations are
   much weaker than GUE. Sigma^2 grows linearly, not logarithmically.
   This is an important CAVEAT to the random-matrix paper — the universality
   is only SHORT-RANGE. The long-range decorrelation is caused by the causal
   structure breaking the full random matrix symmetry.

3. IDEA 94 (mass gap): The SJ propagator on 2-orders has a nonzero mass gap
   (~2.3) while random matrices give zero. Discreteness genuinely induces a
   mass in the scalar field propagator. The N-scaling needs more data.

4. IDEA 95 (spectral zeta): The ratio zeta_2order/zeta_random grows with N
   near s=1, suggesting a spectral asymmetry that strengthens with system size.

NONE of these reach 8+. The best (88, 92, 94) are 5/10 — interesting
observations without analytic depth or clean physical predictions.

HONEST ASSESSMENT: After 95 ideas, the GUE quantum chaos result (7.5/10)
remains the best finding. The search for 8+ may require:
- A genuine THEOREM (not just computation)
- A completely different approach (e.g., exact solution for small N)
- A connection to a specific published conjecture
- Working at the phase transition with much larger N and better MCMC
""")

print("\nDone.")
