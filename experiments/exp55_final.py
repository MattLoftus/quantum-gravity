"""
Experiment 55: Final Five Ideas (96-100) — The Last Shot at 8+

After 95 ideas, the best result is GUE quantum chaos at 7.5/10.
The 7.5 ceiling is structural: at toy scale (N=30-70), everything is
either generic, N-dependent, or density-explained.

These 5 ideas go deeper:
  96. CAUSAL MATRIX ALGEBRA: The causal order C defines a partial order algebra.
      Test whether the spectrum of C+C^T (the "undirected causal matrix")
      matches the Marchenko-Pastur distribution (random matrix universal)
      vs a Tracy-Widom edge — and whether the DEVIATION from MP encodes
      dimension. This is about C itself, not W.

  97. DIAMOND COUNTING GENERATING FUNCTION: The number of k-element intervals
      N_k is a causet observable. Its generating function Z(q) = sum_k N_k q^k
      is an analytic function. In the continuum, Z(q) has a known form related
      to the spacetime volume. Test if Z(q) has ZEROS on the unit circle
      (Lee-Yang type) and whether their distribution distinguishes causets
      from random DAGs.

  98. DYNAMICAL GROWTH: Use the classical sequential growth (CSG) model to
      GROW causets element by element, and track the SJ vacuum at each step.
      Does the entanglement entropy grow as log(N) (CFT) or as N^alpha?
      The DYNAMICS of entropy growth — not just its final value — may
      distinguish causet quantum gravity from random processes.

  99. LONGEST ANTICHAIN vs SQRT(N): Dilworth's theorem says the minimum number
      of chains covering a poset equals the longest antichain. For a causet
      from 2D Minkowski, the longest antichain ~ sqrt(N) (it's a spacelike
      hypersurface). For a random DAG, it's different. Test whether
      antichain_length / sqrt(N) converges to a UNIVERSAL CONSTANT that
      depends only on dimension — an exact result.

 100. WILD CARD — SPECTRAL GAP OF THE LINK MATRIX AS A MASS GAP: The link
      matrix L (Hasse diagram) is the adjacency matrix of the causet's
      nearest-neighbor graph. Its spectral gap (lambda_2 - lambda_1) is
      the "mass gap" of the graph Laplacian, related to expansion/connectivity.
      In a causet from flat spacetime, this should scale as 1/N (gapless, as
      expected for flat space). For de Sitter / curved space, it should be
      O(1). Test the scaling and compare to random DAGs.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size

rng = np.random.default_rng(42)


def sj_vacuum(cs):
    """Compute SJ Wightman function W."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real
    pos = evals > 1e-12
    if not np.any(pos):
        return np.zeros((N, N)), evals
    W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
    w_max = np.linalg.eigvalsh(W).max()
    if w_max > 1:
        W = W / w_max
    return W, evals


def entanglement_entropy(W, region):
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def random_dag(N, density, rng):
    """Generate a random DAG with given relation density (for null tests)."""
    cs = FastCausalSet(N)
    # Random upper-triangular matrix with given density
    mask = rng.random((N, N)) < density
    cs.order = np.triu(mask, k=1)
    # Enforce transitivity: transitive closure
    changed = True
    order_int = cs.order.astype(np.int32)
    while changed:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs


def longest_antichain_greedy(cs):
    """
    Find a long antichain using a greedy algorithm.
    Not guaranteed optimal, but good approximation for our purposes.
    We use Dilworth: max antichain = min chain cover = height of the poset
    viewed appropriately. Actually, for efficiency, we'll use a greedy approach.
    """
    N = cs.n
    order = cs.order
    # Two elements are comparable if order[i,j] or order[j,i]
    comparable = order | order.T

    # Greedy: repeatedly add elements that are incomparable to all current
    antichain = []
    # Try elements in random order for robustness
    indices = list(range(N))
    for i in indices:
        ok = True
        for j in antichain:
            if comparable[i, j]:
                ok = False
                break
        if ok:
            antichain.append(i)
    return len(antichain)


def longest_antichain_exact(cs):
    """
    Compute exact longest antichain using Dilworth's theorem:
    max antichain = N - max matching in the comparability graph.

    Actually, Dilworth says max antichain = min # chains to cover.
    For a DAG, this equals N minus the maximum matching in a bipartite graph.
    We use the Konig-Egervary theorem: in a DAG with N nodes,
    max antichain = N - (maximum matching in the comparability bipartite graph).

    We build the bipartite graph: left nodes L_i and right nodes R_i.
    Edge from L_i to R_j if order[i,j]. Maximum matching via augmenting paths.
    """
    N = cs.n

    # Bipartite matching via augmenting paths (Hungarian-like)
    match_left = [-1] * N   # match_left[i] = j means L_i matched to R_j
    match_right = [-1] * N  # match_right[j] = i

    def augment(u, visited):
        for v in range(N):
            if cs.order[u, v] and not visited[v]:
                visited[v] = True
                if match_right[v] == -1 or augment(match_right[v], visited):
                    match_left[u] = v
                    match_right[v] = u
                    return True
        return False

    matching = 0
    for i in range(N):
        visited = [False] * N
        if augment(i, visited):
            matching += 1

    return N - matching


# ================================================================
print("=" * 78)
print("IDEA 96: CAUSAL MATRIX SPECTRUM — MARCHENKO-PASTUR vs DIMENSION")
print("Does the spectrum of C+C^T deviate from random matrix universality")
print("in a way that encodes spacetime dimension?")
print("=" * 78)

# The causal order matrix C is a {0,1} matrix. C+C^T is symmetric.
# For a truly random {0,1} upper triangular matrix, the empirical spectral
# distribution follows Marchenko-Pastur. But C from a causet has GEOMETRIC
# structure — the deviation from MP should encode the dimension.

N_sizes = [40, 60, 80, 100]
n_trials = 15

print("\n--- Part A: Spectral density of C+C^T ---")
for N in N_sizes:
    tracy_widom_stats = []  # largest eigenvalue, standardized
    bulk_kurtosis = []

    tw_null = []
    bulk_kurtosis_null = []

    for trial in range(n_trials):
        # Causet from 2D Minkowski
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        C = cs.order.astype(float)
        S = C + C.T  # symmetric adjacency of comparability graph
        eigs = np.linalg.eigvalsh(S)

        # Tracy-Widom: standardize largest eigenvalue
        # For Wishart: lambda_max ~ N^{2/3} * TW
        # Approximate: (lambda_max - mean) / std
        e_std = (eigs[-1] - np.mean(eigs)) / (np.std(eigs) + 1e-10)
        tracy_widom_stats.append(e_std)

        # Bulk kurtosis (deviation from semicircle/MP)
        e_centered = (eigs - np.mean(eigs)) / (np.std(eigs) + 1e-10)
        kurt = float(stats.kurtosis(e_centered))
        bulk_kurtosis.append(kurt)

        # Null: random DAG with matched density
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.7, rng)  # approximate density match
        C_null = cs_null.order.astype(float)
        S_null = C_null + C_null.T
        eigs_null = np.linalg.eigvalsh(S_null)
        e_std_null = (eigs_null[-1] - np.mean(eigs_null)) / (np.std(eigs_null) + 1e-10)
        tw_null.append(e_std_null)
        e_c_null = (eigs_null - np.mean(eigs_null)) / (np.std(eigs_null) + 1e-10)
        bulk_kurtosis_null.append(float(stats.kurtosis(e_c_null)))

    tw_mean = np.mean(tracy_widom_stats)
    tw_null_mean = np.mean(tw_null)
    kurt_mean = np.mean(bulk_kurtosis)
    kurt_null_mean = np.mean(bulk_kurtosis_null)

    t_tw, p_tw = stats.ttest_ind(tracy_widom_stats, tw_null)
    t_kurt, p_kurt = stats.ttest_ind(bulk_kurtosis, bulk_kurtosis_null)

    print(f"\n  N={N}:")
    print(f"    TW edge stat: causet={tw_mean:.3f}, null={tw_null_mean:.3f}, p={p_tw:.4f}")
    print(f"    Bulk kurtosis: causet={kurt_mean:.3f}, null={kurt_null_mean:.3f}, p={p_kurt:.4f}")

# Part B: Does the spectral gap of S scale differently?
print("\n--- Part B: Spectral gap scaling ---")
for N in N_sizes:
    gaps = []
    gaps_null = []
    for trial in range(n_trials):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        C = cs.order.astype(float)
        S = C + C.T
        eigs = sorted(np.linalg.eigvalsh(S))
        # Gap between two largest eigenvalues
        gap = eigs[-1] - eigs[-2]
        gaps.append(gap / N)  # normalized

        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.7, rng)
        C_null = cs_null.order.astype(float)
        S_null = C_null + C_null.T
        eigs_null = sorted(np.linalg.eigvalsh(S_null))
        gaps_null.append((eigs_null[-1] - eigs_null[-2]) / N)

    print(f"  N={N}: gap/N causet={np.mean(gaps):.4f}±{np.std(gaps):.4f}, "
          f"null={np.mean(gaps_null):.4f}±{np.std(gaps_null):.4f}")

print("\n  ASSESSMENT: Looking for geometric information in the spectral")
print("  deviation from random matrix universality.")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 97: INTERVAL GENERATING FUNCTION — LEE-YANG ZEROS")
print("Does Z(q) = sum_k N_k q^k have zeros whose distribution")
print("distinguishes causets from random DAGs?")
print("=" * 78)

# The interval generating function encodes the complete set of
# interval counts {N_k}. Its zeros in the complex plane may show
# universal structure (like Lee-Yang zeros in stat mech).

N = 60
n_trials = 12

print("\n--- Zeros of the interval generating function Z(q) ---")

zero_radii_causet = []
zero_radii_null = []
zero_angles_causet = []
zero_angles_null = []

for trial in range(n_trials):
    # Causet
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    counts = count_intervals_by_size(cs, max_size=15)
    coeffs = [counts.get(k, 0) for k in range(16)]

    if max(coeffs) > 0:
        # Polynomial: Z(q) = sum_k N_k * q^k
        # Find roots of the polynomial
        poly_coeffs = coeffs[::-1]  # numpy wants highest degree first
        if len([c for c in poly_coeffs if c != 0]) > 1:
            roots = np.roots(poly_coeffs)
            radii = np.abs(roots)
            angles = np.angle(roots)
            # Keep only roots with |z| < 10 (finite)
            finite = radii < 10
            if np.any(finite):
                zero_radii_causet.extend(radii[finite].tolist())
                zero_angles_causet.extend(angles[finite].tolist())

    # Null: random DAG
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    counts_null = count_intervals_by_size(cs_null, max_size=15)
    coeffs_null = [counts_null.get(k, 0) for k in range(16)]

    if max(coeffs_null) > 0:
        poly_null = coeffs_null[::-1]
        if len([c for c in poly_null if c != 0]) > 1:
            roots_null = np.roots(poly_null)
            radii_null = np.abs(roots_null)
            angles_null = np.angle(roots_null)
            finite_null = radii_null < 10
            if np.any(finite_null):
                zero_radii_null.extend(radii_null[finite_null].tolist())
                zero_angles_null.extend(angles_null[finite_null].tolist())

if zero_radii_causet and zero_radii_null:
    # Test: do zeros cluster at |z|=1 (Lee-Yang circle theorem)?
    causet_near_unit = np.mean([1 for r in zero_radii_causet if 0.8 < r < 1.2]) / len(zero_radii_causet)
    null_near_unit = np.mean([1 for r in zero_radii_null if 0.8 < r < 1.2]) / len(zero_radii_null)

    print(f"  Causet: {len(zero_radii_causet)} zeros, mean |z|={np.mean(zero_radii_causet):.3f}, "
          f"fraction near unit circle={causet_near_unit:.3f}")
    print(f"  Null:   {len(zero_radii_null)} zeros, mean |z|={np.mean(zero_radii_null):.3f}, "
          f"fraction near unit circle={null_near_unit:.3f}")

    # KS test on radii distribution
    ks_stat, ks_p = stats.ks_2samp(zero_radii_causet, zero_radii_null)
    print(f"  KS test on |z| distributions: D={ks_stat:.3f}, p={ks_p:.4f}")

    # Angular distribution
    ks_ang, ks_p_ang = stats.ks_2samp(zero_angles_causet, zero_angles_null)
    print(f"  KS test on arg(z) distributions: D={ks_ang:.3f}, p={ks_p_ang:.4f}")

    # Mean radius
    t_rad, p_rad = stats.ttest_ind(zero_radii_causet, zero_radii_null)
    print(f"  Mean radius: causet={np.mean(zero_radii_causet):.3f}, "
          f"null={np.mean(zero_radii_null):.3f}, p={p_rad:.4f}")
else:
    print("  Insufficient zeros found.")

print("\n  ASSESSMENT: Testing if the interval generating function has")
print("  Lee-Yang-type structure that encodes geometric information.")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 98: DYNAMICAL ENTROPY GROWTH — CSG-LIKE SEQUENTIAL GROWTH")
print("Does S_ent(n) grow as log(n) when building the causet element by element?")
print("=" * 78)

# Instead of computing entropy on a fixed causet, GROW the causet
# one element at a time and track how S_ent evolves.
# In a CFT, S ~ (c/3) ln(L). If we're growing a discretization,
# the entropy growth rate should be related to c.

N_final = 60
n_trials = 8

print("\n--- Sequential growth entropy trajectory ---")

growth_entropies = []  # (n_steps, n_trials)
growth_entropies_null = []

for trial in range(n_trials):
    # Strategy: start with a full 2-order of size N_final,
    # then include elements one at a time (in natural order of u+v)
    to = TwoOrder(N_final, rng=rng)
    cs_full = to.to_causet()

    # Order elements by "time" coordinate (u+v)/2
    time_order = np.argsort(to.u + to.v)

    entropies = []
    for n_included in range(10, N_final + 1, 2):
        elements = time_order[:n_included]
        # Build sub-causet
        sub_cs = FastCausalSet(n_included)
        sub_cs.order = cs_full.order[np.ix_(elements, elements)]
        W, _ = sj_vacuum(sub_cs)
        # Half-partition
        region = list(range(n_included // 2))
        S = entanglement_entropy(W, region)
        entropies.append((n_included, S))

    growth_entropies.append(entropies)

    # Null: random DAG grown similarly
    null_entropies = []
    for n_included in range(10, N_final + 1, 2):
        cs_null = random_dag(n_included, 0.25, rng)
        W_null, _ = sj_vacuum(cs_null)
        region = list(range(n_included // 2))
        S_null = entanglement_entropy(W_null, region)
        null_entropies.append((n_included, S_null))

    growth_entropies_null.append(null_entropies)

# Analyze the growth exponent: fit S = a * ln(n) + b
print("\n  Fitting S(n) = a * ln(n) + b:")
n_vals = np.array([e[0] for e in growth_entropies[0]])
ln_n = np.log(n_vals)

# Causet growth curves
slopes_causet = []
r2_causet = []
for trial_data in growth_entropies:
    s_vals = np.array([e[1] for e in trial_data])
    if np.std(s_vals) > 1e-10:
        slope, intercept, r_val, _, _ = stats.linregress(ln_n, s_vals)
        slopes_causet.append(slope)
        r2_causet.append(r_val**2)

# Null growth curves
slopes_null = []
r2_null = []
n_vals_null = np.array([e[0] for e in growth_entropies_null[0]])
ln_n_null = np.log(n_vals_null)
for trial_data in growth_entropies_null:
    s_vals = np.array([e[1] for e in trial_data])
    if np.std(s_vals) > 1e-10:
        slope, intercept, r_val, _, _ = stats.linregress(ln_n_null, s_vals)
        slopes_null.append(slope)
        r2_null.append(r_val**2)

if slopes_causet and slopes_null:
    print(f"  Causet: slope = {np.mean(slopes_causet):.3f} ± {np.std(slopes_causet):.3f}, "
          f"R² = {np.mean(r2_causet):.3f}")
    print(f"  Null:   slope = {np.mean(slopes_null):.3f} ± {np.std(slopes_null):.3f}, "
          f"R² = {np.mean(r2_null):.3f}")

    # CFT prediction: slope = c/3 where c is central charge
    c_eff = 3 * np.mean(slopes_causet)
    print(f"  Effective central charge from growth: c_eff = {c_eff:.3f}")
    print(f"  (CFT prediction for free scalar: c = 1)")

    # Also fit power law: S = a * n^alpha
    print("\n  Fitting S(n) = a * n^alpha (power law):")
    alphas_causet = []
    alphas_null = []
    for trial_data in growth_entropies:
        s_vals = np.array([e[1] for e in trial_data])
        if np.all(s_vals > 0):
            slope, _, r_val, _, _ = stats.linregress(ln_n, np.log(s_vals))
            alphas_causet.append(slope)
    for trial_data in growth_entropies_null:
        s_vals = np.array([e[1] for e in trial_data])
        if np.all(s_vals > 0):
            slope, _, r_val, _, _ = stats.linregress(ln_n_null, np.log(s_vals))
            alphas_null.append(slope)

    if alphas_causet and alphas_null:
        print(f"  Causet: alpha = {np.mean(alphas_causet):.3f} ± {np.std(alphas_causet):.3f}")
        print(f"  Null:   alpha = {np.mean(alphas_null):.3f} ± {np.std(alphas_null):.3f}")
        t_alpha, p_alpha = stats.ttest_ind(alphas_causet, alphas_null)
        print(f"  t-test on alpha: t={t_alpha:.2f}, p={p_alpha:.4f}")
else:
    print("  Insufficient data for analysis.")

print("\n  ASSESSMENT: The entropy growth TRAJECTORY during sequential building")
print("  encodes dynamical information absent from static measurements.")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 99: LONGEST ANTICHAIN / SQRT(N) — UNIVERSAL CONSTANT")
print("Does max_antichain / sqrt(N) converge to a dimension-dependent constant?")
print("=" * 78)

# For a Poisson sprinkling into [0,1]^d Minkowski, the longest antichain
# (a maximal set of mutually spacelike elements) should scale as N^{(d-1)/d}.
# In d=2: longest antichain ~ sqrt(N).
# The ratio antichain/sqrt(N) should converge to a universal constant.
# This constant would be a NEW prediction of causal set theory.

N_sizes_ac = [30, 50, 70, 100, 150]
n_trials_ac = 20

print("\n--- Longest antichain scaling ---")
results_causet = {N: [] for N in N_sizes_ac}
results_null = {N: [] for N in N_sizes_ac}

for N in N_sizes_ac:
    for trial in range(n_trials_ac):
        # Causet from 2D Minkowski (2-order)
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        ac = longest_antichain_exact(cs)
        results_causet[N].append(ac / np.sqrt(N))

        # Null: random DAG with matched density
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.6, rng)
        ac_null = longest_antichain_exact(cs_null)
        results_null[N].append(ac_null / np.sqrt(N))

print(f"  {'N':>5}  {'causet AC/√N':>15}  {'null AC/√N':>15}  {'ratio':>8}  {'p-val':>8}")
print(f"  {'─'*5}  {'─'*15}  {'─'*15}  {'─'*8}  {'─'*8}")

for N in N_sizes_ac:
    c_mean = np.mean(results_causet[N])
    c_std = np.std(results_causet[N])
    n_mean = np.mean(results_null[N])
    n_std = np.std(results_null[N])
    t_val, p_val = stats.ttest_ind(results_causet[N], results_null[N])
    print(f"  {N:>5}  {c_mean:>8.3f}±{c_std:.3f}  {n_mean:>8.3f}±{n_std:.3f}  "
          f"{c_mean/(n_mean+1e-10):>8.3f}  {p_val:>8.4f}")

# Check convergence: does the causet value stabilize?
means = [np.mean(results_causet[N]) for N in N_sizes_ac]
cv_of_means = np.std(means) / (np.mean(means) + 1e-10)
print(f"\n  Coefficient of variation across N: {cv_of_means:.3f}")
print(f"  {'CONVERGING' if cv_of_means < 0.1 else 'NOT YET CONVERGED'}")

# Exact scaling exponent: fit AC = a * N^alpha
ac_means = np.array([np.mean(results_causet[N]) * np.sqrt(N) for N in N_sizes_ac])
log_N = np.log(N_sizes_ac)
log_AC = np.log(ac_means)
slope, intercept, r_val, _, _ = stats.linregress(log_N, log_AC)
print(f"\n  Power law fit: AC ~ N^{slope:.3f} (R² = {r_val**2:.3f})")
print(f"  Expected for 2D Minkowski: AC ~ N^0.500")
print(f"  Deviation: {abs(slope - 0.5):.3f}")

# Same for null
ac_means_null = np.array([np.mean(results_null[N]) * np.sqrt(N) for N in N_sizes_ac])
log_AC_null = np.log(ac_means_null + 1e-10)
slope_null, _, r_null, _, _ = stats.linregress(log_N, log_AC_null)
print(f"  Null power law: AC ~ N^{slope_null:.3f}")

print("\n  ASSESSMENT: Testing for a universal antichain-to-size ratio.")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 100: WILD CARD — LINK MATRIX SPECTRAL GAP AS MASS GAP")
print("Does the spectral gap of the graph Laplacian encode curvature/flatness?")
print("=" * 78)

# The link matrix L is the adjacency matrix of the Hasse diagram.
# The graph Laplacian is D - L (where D = degree matrix).
# Its smallest nonzero eigenvalue (Fiedler value) measures connectivity.
# For flat spacetime: the causet is "volume-filling" and the Fiedler value
# should scale as ~1/N^{2/d} (diffusion on a d-dimensional lattice).
# For a random graph: Fiedler value ~ O(1).
# This would be a MASS GAP indicator.

N_sizes_lg = [30, 50, 70, 100]
n_trials_lg = 15

print("\n--- Fiedler value (algebraic connectivity) scaling ---")

fiedler_causet = {N: [] for N in N_sizes_lg}
fiedler_null = {N: [] for N in N_sizes_lg}
degree_stats = {N: [] for N in N_sizes_lg}

for N in N_sizes_lg:
    for trial in range(n_trials_lg):
        # Causet
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        L = cs.link_matrix()

        # Make undirected adjacency: A = L + L^T
        A = (L | L.T).astype(float)
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        Laplacian = D - A

        eigs_lap = sorted(np.linalg.eigvalsh(Laplacian))
        # Fiedler value = second smallest eigenvalue (first is always 0)
        fiedler = eigs_lap[1] if len(eigs_lap) > 1 else 0
        fiedler_causet[N].append(fiedler)
        degree_stats[N].append(np.mean(degrees))

        # Null: random DAG
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.6, rng)
        L_null = cs_null.link_matrix()
        A_null = (L_null | L_null.T).astype(float)
        D_null = np.diag(A_null.sum(axis=1))
        Lap_null = D_null - A_null
        eigs_null = sorted(np.linalg.eigvalsh(Lap_null))
        fiedler_null[N].append(eigs_null[1] if len(eigs_null) > 1 else 0)

print(f"  {'N':>5}  {'causet λ₂':>12}  {'null λ₂':>12}  {'ratio':>8}  {'p-val':>8}  {'deg':>6}")
print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}")

for N in N_sizes_lg:
    c_mean = np.mean(fiedler_causet[N])
    c_std = np.std(fiedler_causet[N])
    n_mean = np.mean(fiedler_null[N])
    n_std = np.std(fiedler_null[N])
    t_val, p_val = stats.ttest_ind(fiedler_causet[N], fiedler_null[N])
    d_mean = np.mean(degree_stats[N])
    print(f"  {N:>5}  {c_mean:>7.3f}±{c_std:.3f}  {n_mean:>7.3f}±{n_std:.3f}  "
          f"{c_mean/(n_mean+1e-10):>8.3f}  {p_val:>8.4f}  {d_mean:>6.1f}")

# Scaling analysis: λ₂ ~ N^{-gamma}
log_N = np.log(N_sizes_lg)
log_fiedler = np.log([np.mean(fiedler_causet[N]) for N in N_sizes_lg])
slope_f, _, r_f, _, _ = stats.linregress(log_N, log_fiedler)
print(f"\n  Fiedler scaling: λ₂ ~ N^{slope_f:.3f} (R² = {r_f**2:.3f})")
print(f"  Expected for 2D: λ₂ ~ N^{-1.0} (diffusion on 2D lattice)")

log_fiedler_null = np.log([np.mean(fiedler_null[N]) + 1e-10 for N in N_sizes_lg])
slope_fn, _, r_fn, _, _ = stats.linregress(log_N, log_fiedler_null)
print(f"  Null scaling: λ₂ ~ N^{slope_fn:.3f}")

# Check: does λ₂ * N converge (gapless) or λ₂ → const (gapped)?
products = [np.mean(fiedler_causet[N]) * N for N in N_sizes_lg]
cv_prod = np.std(products) / (np.mean(products) + 1e-10)
print(f"\n  λ₂ * N values: {[f'{p:.2f}' for p in products]}")
print(f"  CV of λ₂*N: {cv_prod:.3f}")
print(f"  {'GAPLESS (flat space)' if cv_prod < 0.2 else 'NOT clearly gapless'}")


# ================================================================
# FINAL SCORING
# ================================================================
print("\n\n" + "=" * 78)
print("FINAL SCORING — IDEAS 96-100")
print("=" * 78)

print("""
Scoring criteria:
  - Novelty: Has this been done before? (0-3)
  - Rigor: Clean signal, survives null test? (0-3)
  - Depth: Connection to fundamental physics? (0-2)
  - Audience: Who cares? (0-2)
  Total out of 10.

IDEA 96 — Causal Matrix Spectrum: 5.0/10
  Novelty: 2 — Spectral analysis of C+C^T is somewhat new for causets.
  Rigor: 1 — Strong separation from null (p<0.0001), BUT the kurtosis grows
    with N for both causet and null — the difference is quantitative, not
    qualitative. The spectral gap/N stabilizes for causets (~0.35) but this
    is just the ordering fraction effect. No dimension encoding beyond density.
  Depth: 1 — Connection to RMT is real but shallow.
  Audience: 1 — RMT people might glance at it.

IDEA 97 — Lee-Yang Zeros of Interval Generating Function: 5.5/10
  Novelty: 2.5 — Nobody has looked at the zeros of this polynomial. Fresh idea.
  Rigor: 1 — KS test shows the radii distributions differ (p<0.0001), but
    the effect is just that causet intervals are more spread out (mean |z|=1.22
    vs 1.08). The zeros do NOT cluster on the unit circle (fraction ~0.006
    for both). No Lee-Yang structure. Angular distribution identical. The
    distinguishing power comes from density again.
  Depth: 1 — Lee-Yang connection is suggestive but doesn't materialize.
  Audience: 1 — Stat mech crowd might find it curious.

IDEA 98 — Dynamical Entropy Growth: 4.5/10
  Novelty: 1.5 — Sequential growth + SJ entropy is new-ish.
  Rigor: 0.5 — Causet slope 1.49±0.11 vs null 1.44±0.04 — INDISTINGUISHABLE
    within error bars (p=0.24). c_eff=4.5, far from c=1. The growth dynamics
    do NOT distinguish causets from random DAGs. Both follow ~n^0.5 power law.
    This is a null result.
  Depth: 1.5 — Dynamical approach is conceptually right.
  Audience: 1 — Only causet specialists.

IDEA 99 — Longest Antichain / sqrt(N) Universal Constant: 7.0/10
  Novelty: 2 — The antichain scaling is known (Erdos-Szekeres / Ulam), but
    the specific ratio for 2-orders (causets) is not well-characterized. The
    AC/sqrt(N) ~ 1.7 convergence is clean.
  Rigor: 2 — STRONG results:
    * Causet AC/sqrt(N) converges to ~1.7 (CV=4.5% — genuinely converging)
    * Null AC/sqrt(N) DECREASES with N (0.576 at N=150 vs 1.06 at N=30)
    * p < 0.0001 at all sizes
    * BUT: the measured exponent is 0.58, not 0.50. This 16% deviation from
      the theoretical N^0.5 means we're either (a) not yet at large enough N,
      or (b) 2-orders don't quite give the expected scaling.
    * The convergence of AC/sqrt(N) is real but the exponent discrepancy
      prevents this from being an "exact result" paper.
  Depth: 1.5 — Connects to Dilworth/Erdos-Szekeres, has geometric meaning.
  Audience: 1.5 — Combinatorics + causal set people.

IDEA 100 — Link Matrix Spectral Gap: 6.0/10
  Novelty: 2 — Graph Laplacian of the Hasse diagram is unstudied in causets.
  Rigor: 2 — Fiedler value INCREASES with N for causets (0.54 to 0.99),
    DECREASES for null (0.34 to 0.04). Huge separation. But the scaling
    lambda_2 ~ N^0.51 is WRONG for flat space — should be N^{-1} for
    gapless. The causet Hasse diagram becomes MORE connected at large N,
    which is the opposite of what a mass gap argument needs. The increasing
    Fiedler value just reflects that 2-order link graphs are good expanders
    — interesting but not a mass gap.
  Depth: 1 — Connection to mass gap is loose.
  Audience: 1 — Graph theory + causet people.

SUMMARY:
  Best of the final 5: Idea 99 (antichain scaling) at 7.0/10.
  None reaches 8+.
  The 7.5 ceiling (GUE quantum chaos) remains the best result of all 100 ideas.

  WHY 8+ IS STRUCTURALLY IMPOSSIBLE AT TOY SCALE:
  1. Density dominance: Most observables that distinguish causets from
     random DAGs do so via ordering fraction, not geometry.
  2. N-dependence: At N=30-150, finite-size corrections are larger than
     the signals we're looking for.
  3. No exact predictions: Causal set theory makes few quantitative
     predictions at finite N — most results are asymptotic or qualitative.
  4. The SJ vacuum at small N gives c_eff ~ 3-4, not c=1. Until we can
     reach N ~ 10^3-10^4, we cannot test continuum predictions.

  THE REAL FINDING (after 100 ideas):
  The GUE quantum chaos result at 7.5 is genuine and robust. The SJ vacuum
  on random 2-orders has level spacing statistics consistent with GUE
  (the universality class of systems with broken time-reversal symmetry).
  This persists to N=500. It is the single most paper-worthy finding.

  To get to 8+, one would need:
  (a) A theoretical explanation for WHY causet SJ spectrum is GUE, or
  (b) Computation at N > 1000 showing convergence to exact GUE value, or
  (c) Connection between the GUE class and a physical prediction (e.g.,
      scrambling time, Lyapunov exponent).
""")
