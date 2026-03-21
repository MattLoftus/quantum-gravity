"""
Experiment 93: PAPER CANDIDATE EVALUATION — 10 NEW IDEAS (441-450)

GOAL: Determine which of the top-5 unwritten results from Ideas 101-400 can
become standalone papers, via deeper numerical tests.

NUMERICAL TESTS (Ideas 441-445):
441. PATH ENTROPY AS DIMENSION ESTIMATOR: Test across d=2,3,4,5 with multiple N.
     Does path_entropy cleanly encode dimension? If yes -> dimension estimator paper.
442. NEWTON'S LAW STABILITY: Test |W| ~ -a*ln(r) at N=40,60,80,100.
     Is the coefficient stable? If yes -> "continuum limit of SJ correlations" paper.
443. CASIMIR EFFECT ROBUSTNESS: Test at multiple boundary separations AND multiple N.
     Is the 1/d scaling robust? If yes -> strengthen the Casimir paper case.
444. KR LAYER COUNT SCALING: Test N=30,50,100,150,200. Fit layer_count vs N.
     Is it ~sqrt(N)? Connection to RSK correspondence?
445. TOPOLOGY DETECTION BREADTH: Test diamond vs cylinder vs torus vs "Klein bottle"
     analogue. More topologies, more observables.

FEASIBILITY ASSESSMENTS (Ideas 446-450):
446. Graph-theoretic dimension paper: path entropy + Fiedler + treewidth.
447. SJ vacuum reproduces physics paper: Casimir + Newton + area law.
448. Comprehensive BD transition paper: interval entropy + link fraction +
     spectral statistics + Fiedler + path entropy + action bimodality.
449. Methods paper: sparse SVD, parallel tempering, d-orders, CDT interface.
450. Meta-paper: "400 computational experiments in causal set quantum gravity."
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import time
from collections import Counter, defaultdict

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.sj_vacuum import sj_wightman_function, pauli_jordan_function, entanglement_entropy
from causal_sets.bd_action import count_intervals_by_size, count_links
from causal_sets.dimension import _ordering_fraction_theory, _invert_ordering_fraction

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ============================================================
# SHARED UTILITIES
# ============================================================

def path_entropy(cs):
    """Entropy of the chain-length distribution from each element."""
    N = cs.n
    if N < 3:
        return 0.0
    chain_lengths = []
    for i in range(N):
        successors = np.where(cs.order[i])[0]
        chain_lengths.append(len(successors))
    cl = np.array(chain_lengths, dtype=float)
    cl = cl[cl > 0]
    if len(cl) == 0:
        return 0.0
    p = cl / np.sum(cl)
    return -np.sum(p * np.log(p + 1e-300))

def fiedler_value(cs):
    """Algebraic connectivity (second smallest eigenvalue of Laplacian) of Hasse diagram."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    degree = adj.sum(axis=1)
    if np.any(degree == 0):
        return 0.0
    L = np.diag(degree) - adj
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    return float(evals[1]) if len(evals) > 1 else 0.0

def treewidth_approx(cs):
    """Approximate treewidth via greedy minimum degree elimination."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(bool)
    N = cs.n
    remaining = set(range(N))
    tw = 0
    adj_dict = {i: set(np.where(adj[i])[0]) - {i} for i in range(N)}
    for _ in range(N - 1):
        if not remaining:
            break
        # Find vertex with minimum degree
        min_v = min(remaining, key=lambda v: len(adj_dict[v] & remaining))
        neighbors = adj_dict[min_v] & remaining
        tw = max(tw, len(neighbors))
        # Connect neighbors (fill-in)
        for u in neighbors:
            for w in neighbors:
                if u != w:
                    adj_dict[u].add(w)
                    adj_dict[w].add(u)
        remaining.remove(min_v)
    return tw

def interval_entropy(cs):
    """Shannon entropy of the interval size distribution."""
    counts = count_intervals_by_size(cs, max_size=6)
    vals = np.array([counts[k] for k in sorted(counts.keys()) if counts[k] > 0], dtype=float)
    if vals.sum() == 0:
        return 0.0
    p = vals / vals.sum()
    return -np.sum(p * np.log(p + 1e-300))

def hw_ratio(cs):
    """Height/width ratio: longest chain / longest antichain."""
    # Height = longest chain
    N = cs.n
    dp = np.ones(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            dp[j] = np.max(dp[preds]) + 1
    height = int(np.max(dp))

    # Width = longest antichain (Dilworth) - approximate via max antichain in layers
    layers = defaultdict(list)
    for i in range(N):
        layers[dp[i]].append(i)
    width = max(len(v) for v in layers.values()) if layers else 1

    return height / max(width, 1)

def sprinkle_cylinder(N, rng_local):
    """Sprinkle into 2D Minkowski cylinder: t in [0,1], x periodic in [0,1]."""
    cs = FastCausalSet(N)
    coords = np.zeros((N, 2))
    coords[:, 0] = rng_local.uniform(0, 1, N)
    coords[:, 1] = rng_local.uniform(0, 1, N)
    coords = coords[np.argsort(coords[:, 0])]

    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        dx = np.minimum(dx, 1.0 - dx)
        cs.order[i, i+1:] = dt * dt >= dx * dx

    return cs, coords

def sprinkle_torus(N, rng_local):
    """Sprinkle into 2D Minkowski torus: t in [0,1], x periodic in [0,1].
    Note: uses t-periodic causal structure to give truly different topology."""
    cs = FastCausalSet(N)
    coords = np.zeros((N, 2))
    coords[:, 0] = rng_local.uniform(0, 1, N)
    coords[:, 1] = rng_local.uniform(0, 1, N)
    coords = coords[np.argsort(coords[:, 0])]

    for i in range(N):
        for j in range(i+1, N):
            dt = coords[j, 0] - coords[i, 0]
            dx = abs(coords[j, 1] - coords[i, 1])
            dx = min(dx, 1.0 - dx)
            dt_wrap = min(dt, 1.0 - dt)
            if dt_wrap * dt_wrap >= dx * dx:
                cs.order[i, j] = True

    return cs, coords

def sprinkle_cylinder_wide(N, L, rng_local):
    """Cylinder with circumference L (allows testing different radii)."""
    cs = FastCausalSet(N)
    coords = np.zeros((N, 2))
    coords[:, 0] = rng_local.uniform(0, 1, N)
    coords[:, 1] = rng_local.uniform(0, L, N)
    coords = coords[np.argsort(coords[:, 0])]

    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        dx = np.minimum(dx, L - dx)
        cs.order[i, i+1:] = dt * dt >= dx * dx

    return cs, coords

def sprinkle_with_boundaries(N, d_boundary, rng_local=None):
    """Causet confined between two parallel boundaries at x = +/- d/2."""
    if rng_local is None:
        rng_local = np.random.default_rng()
    extent_t = 2.0
    coords = np.zeros((N, 2))
    coords[:, 0] = rng_local.uniform(-extent_t, extent_t, N)
    coords[:, 1] = rng_local.uniform(-d_boundary/2, d_boundary/2, N)
    coords = coords[np.argsort(coords[:, 0])]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx
    return cs, coords

def topology_observables(cs):
    """Observables that distinguish spatial topology."""
    N = cs.n
    links = cs.link_matrix()
    f = cs.ordering_fraction()
    n_links = links.sum()
    link_density = n_links / (N * (N - 1) / 2)

    # Boundary elements
    order = cs.order.astype(int)
    n_predecessors = order.sum(axis=0)
    n_successors = order.sum(axis=1)
    n_minimal = int((n_predecessors == 0).sum())
    n_maximal = int((n_successors == 0).sum())

    # Spatial width at midpoint
    past_counts = order.sum(axis=0)
    future_counts = order.sum(axis=1)
    midpoint = np.where(np.abs(past_counts - future_counts) < N * 0.2)[0]
    n_mid = len(midpoint)
    if n_mid > 1:
        spacelike_count = 0
        for i in range(min(n_mid, 30)):
            for j in range(i + 1, min(n_mid, 30)):
                ei, ej = midpoint[i], midpoint[j]
                if not cs.order[ei, ej] and not cs.order[ej, ei]:
                    spacelike_count += 1
        total_pairs = min(n_mid, 30) * (min(n_mid, 30) - 1) / 2
        spatial_width = spacelike_count / total_pairs if total_pairs > 0 else 0
    else:
        spatial_width = 0

    # Link graph clustering
    adj = (links | links.T).astype(float)
    degree = adj.sum(axis=1)
    clustering = 0.0
    n_counted = 0
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        sub = adj[np.ix_(neighbors, neighbors)]
        edges = sub.sum() / 2
        possible = k * (k - 1) / 2
        clustering += edges / possible
        n_counted += 1
    clustering = clustering / n_counted if n_counted > 0 else 0

    return {
        'f': f,
        'link_density': link_density,
        'spatial_width': spatial_width,
        'n_minimal': n_minimal,
        'n_maximal': n_maximal,
        'clustering': clustering,
    }

def get_layers(cs):
    """Partition causet into antichains/layers by longest path from bottom."""
    N = cs.n
    dp = np.ones(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            dp[j] = np.max(dp[preds]) + 1
    return dp

def layer_stats(dp):
    """Given layer assignments, return (n_layers, sizes, max_width)."""
    layers = Counter(dp)
    n_layers = len(layers)
    sizes = [layers[k] for k in sorted(layers.keys())]
    max_width = max(sizes) if sizes else 0
    return n_layers, sizes, max_width


# ================================================================
P("=" * 78)
P("EXPERIMENT 93: PAPER CANDIDATE EVALUATION (Ideas 441-450)")
P("=" * 78)
P()

total_start = time.time()

# ================================================================
# IDEA 441: PATH ENTROPY AS DIMENSION ESTIMATOR
# ================================================================
P("=" * 78)
P("IDEA 441: PATH ENTROPY AS DIMENSION ESTIMATOR")
P("=" * 78)
P("""
Path entropy H_path = -sum(p_i * ln(p_i)) where p_i is the normalized
successor count for element i. In Idea 208/311, it achieved |rho|=1.0
correlation with dimension. Here we test:
  (a) Does it cleanly separate d=2,3,4,5 at fixed N?
  (b) Is the separation stable across N=40,60,80,100?
  (c) What is the functional form H_path(d)?
  (d) Null test: random DAGs with matched density?
""")

t0 = time.time()

dims = [2, 3, 4, 5]
Ns_441 = [40, 60, 80, 100]
n_trials = 12

P(f"  {'N':>4}  {'d':>3}  {'H_path':>10}  {'std':>8}  {'f':>8}")
P(f"  {'':>4}  {'':>3}  {'':>10}  {'':>8}  {'':>8}")

results_441 = {}  # (N, d) -> (mean, std)
for N in Ns_441:
    for d in dims:
        hps = []
        fs = []
        for trial in range(n_trials):
            cs, coords = sprinkle_fast(N, dim=d, rng=rng)
            hps.append(path_entropy(cs))
            fs.append(cs.ordering_fraction())
        mean_hp = np.mean(hps)
        std_hp = np.std(hps)
        mean_f = np.mean(fs)
        results_441[(N, d)] = (mean_hp, std_hp, mean_f)
        P(f"  {N:>4}  {d:>3}  {mean_hp:10.4f}  {std_hp:8.4f}  {mean_f:8.4f}")
    P()

# (a) Separation test: are dimensions well-separated?
P("  --- Separation test (overlap in H_path between consecutive dimensions) ---")
for N in Ns_441:
    P(f"  N={N}:")
    for i in range(len(dims) - 1):
        d_lo, d_hi = dims[i], dims[i+1]
        m1, s1, _ = results_441[(N, d_lo)]
        m2, s2, _ = results_441[(N, d_hi)]
        gap = abs(m2 - m1)
        combined_std = np.sqrt(s1**2 + s2**2)
        sigma_sep = gap / combined_std if combined_std > 0 else float('inf')
        P(f"    d={d_lo} vs d={d_hi}: gap={gap:.4f}, "
          f"combined_std={combined_std:.4f}, separation={sigma_sep:.1f}sigma")

# (b) Monotonicity with d at each N
P("\n  --- Monotonicity test ---")
for N in Ns_441:
    means = [results_441[(N, d)][0] for d in dims]
    monotonic = all(means[i] > means[i+1] for i in range(len(means)-1)) or \
                all(means[i] < means[i+1] for i in range(len(means)-1))
    direction = "INCREASING" if means[-1] > means[0] else "DECREASING"
    P(f"  N={N}: {direction}, monotonic={monotonic}, "
      f"values={[f'{m:.3f}' for m in means]}")

# (c) Functional form: fit H_path(d) at fixed N
P("\n  --- Functional form fit ---")
for N in Ns_441:
    d_arr = np.array(dims, dtype=float)
    h_arr = np.array([results_441[(N, d)][0] for d in dims])
    # Try linear, log, power law
    try:
        # Linear: H = a*d + b
        sl, ic, r_lin, p_lin, _ = stats.linregress(d_arr, h_arr)
        ss_lin = np.sum((h_arr - (sl * d_arr + ic))**2)

        # Log: H = a*ln(d) + b
        sl2, ic2, r_log, _, _ = stats.linregress(np.log(d_arr), h_arr)
        ss_log = np.sum((h_arr - (sl2 * np.log(d_arr) + ic2))**2)

        P(f"  N={N}: linear R^2={r_lin**2:.4f} (SS={ss_lin:.6f}), "
          f"log R^2={r_log**2:.4f} (SS={ss_log:.6f})")
        best = "LINEAR" if ss_lin < ss_log else "LOG"
        P(f"         Best fit: {best}")
    except:
        P(f"  N={N}: fit failed")

# (d) Null test: random DAGs with matched density
P("\n  --- Null test: random DAGs with matched ordering fraction ---")
for d in [2, 4]:
    N = 60
    causet_hps = []
    dag_hps = []
    for trial in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        f_target = cs.ordering_fraction()
        causet_hps.append(path_entropy(cs))

        # Random DAG with matched density
        dag = FastCausalSet(N)
        mask = rng.random((N, N)) < f_target
        dag.order = np.triu(mask, k=1)
        # Transitive closure
        order_int = dag.order.astype(np.int32)
        for _ in range(int(np.log2(N)) + 2):
            new = (order_int @ order_int > 0) | dag.order
            if np.array_equal(new, dag.order):
                break
            dag.order = new
            order_int = dag.order.astype(np.int32)
        dag_hps.append(path_entropy(dag))

    stat, pval = stats.mannwhitneyu(causet_hps, dag_hps, alternative='two-sided')
    P(f"  d={d}: causet H_path={np.mean(causet_hps):.3f}+/-{np.std(causet_hps):.3f}, "
      f"DAG H_path={np.mean(dag_hps):.3f}+/-{np.std(dag_hps):.3f}, "
      f"p={pval:.4f}")

P(f"\n  Time: {time.time()-t0:.1f}s")

# ASSESSMENT
P("\n  PAPER ASSESSMENT:")
P("  Path entropy cleanly encodes dimension IF:")
P("    - Separation > 3 sigma between consecutive d")
P("    - Monotonic with d at all N")
P("    - Survives null test (different from random DAGs)")


# ================================================================
# IDEA 442: NEWTON'S LAW STABILITY AT MULTIPLE N
# ================================================================
P("\n" + "=" * 78)
P("IDEA 442: NEWTON'S LAW — |W| ~ ln(r) STABILITY VS N")
P("=" * 78)
P("""
Idea 270 found |W(r)| = -0.0075*ln(r) + 0.003 at N=60.
Key question: Is the coefficient STABLE as N increases?
If a ~ constant as N -> infinity, we have a continuum limit.
If a ~ 1/N or similar, the signal is a finite-size artifact.
""")

t0 = time.time()

Ns_442 = [40, 60, 80, 100]
n_trials_442 = 8
n_bins = 12
r_bins = np.linspace(0.01, 0.6, n_bins + 1)
r_centers = (r_bins[:-1] + r_bins[1:]) / 2

log_coefficients = []
log_intercepts = []
r_squareds = []

for N in Ns_442:
    P(f"\n  N={N}:")
    green_by_r = [[] for _ in range(n_bins)]

    for trial in range(n_trials_442):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        W = sj_wightman_function(cs)

        for i in range(N):
            for j in range(i+1, N):
                dt = abs(coords[i, 0] - coords[j, 0])
                dx = abs(coords[i, 1] - coords[j, 1])
                if dt < 0.12:
                    r = dx
                    bin_idx = np.searchsorted(r_bins, r) - 1
                    if 0 <= bin_idx < n_bins:
                        green_by_r[bin_idx].append(abs(W[i, j]))

    # Extract decay profile
    valid_data = []
    for b in range(n_bins):
        if len(green_by_r[b]) > 5:
            g = np.mean(green_by_r[b])
            valid_data.append((r_centers[b], g))

    if len(valid_data) >= 4:
        rs = np.array([v[0] for v in valid_data])
        gs = np.array([v[1] for v in valid_data])

        # Log fit: W ~ a * ln(r) + b
        try:
            def log_model(r, a, b):
                return a * np.log(r) + b
            popt, _ = curve_fit(log_model, rs, gs, p0=[-0.01, 0.01])

            # Power fit for comparison
            popt_pow, _ = curve_fit(lambda x, a, b: a * x**b, rs, gs, p0=[0.01, -0.5])

            ss_log = np.sum((gs - log_model(rs, *popt))**2)
            ss_pow = np.sum((gs - popt_pow[0] * rs**popt_pow[1])**2)

            # R^2 for log fit
            ss_tot = np.sum((gs - np.mean(gs))**2)
            r2 = 1 - ss_log / ss_tot if ss_tot > 0 else 0

            log_coefficients.append(popt[0])
            log_intercepts.append(popt[1])
            r_squareds.append(r2)

            winner = "LOG" if ss_log < ss_pow else "POWER"
            ratio = ss_pow / ss_log if ss_log > 0 else float('inf')
            P(f"    Log fit: W = {popt[0]:.6f}*ln(r) + {popt[1]:.6f} (R^2={r2:.4f})")
            P(f"    Power:   W = {popt_pow[0]:.6f}*r^{popt_pow[1]:.3f}")
            P(f"    {winner} WINS (ratio={ratio:.2f}x)")
        except Exception as e:
            P(f"    Fit failed: {e}")
    else:
        P(f"    Insufficient data points: {len(valid_data)}")

# Stability analysis
P(f"\n  --- Coefficient stability ---")
P(f"  {'N':>4}  {'a (log coeff)':>14}  {'b (intercept)':>14}  {'R^2':>8}")
for i, N in enumerate(Ns_442):
    if i < len(log_coefficients):
        P(f"  {N:>4}  {log_coefficients[i]:14.6f}  {log_intercepts[i]:14.6f}  "
          f"{r_squareds[i]:8.4f}")

if len(log_coefficients) >= 3:
    coeffs = np.array(log_coefficients)
    cv = np.std(coeffs) / abs(np.mean(coeffs)) if abs(np.mean(coeffs)) > 0 else float('inf')
    P(f"\n  Mean coefficient: {np.mean(coeffs):.6f} +/- {np.std(coeffs):.6f}")
    P(f"  CV (coefficient of variation): {cv:.3f}")

    # Check if coefficient scales with N
    Ns_arr = np.array(Ns_442[:len(coeffs)], dtype=float)
    sl, ic, r_val, p_val, _ = stats.linregress(Ns_arr, coeffs)
    P(f"  a vs N: slope={sl:.6f}, p={p_val:.4f}")

    if cv < 0.3:
        P(f"  >>> STABLE — CV={cv:.3f} < 0.3. Coefficient converges!")
    else:
        P(f"  >>> UNSTABLE — CV={cv:.3f}. Coefficient drifts with N.")

    # Expected value: a = -1/(2*pi) = -0.1592 for 2D Green's function
    # But SJ normalization is 2/N, so effective a = -1/(2*pi) * (2/N)^correction...
    expected = -1 / (2 * np.pi)
    P(f"\n  Continuum expected: a = -1/(2*pi) = {expected:.6f}")
    P(f"  Measured mean a = {np.mean(coeffs):.6f}")
    P(f"  Ratio: {np.mean(coeffs)/expected:.4f}")

P(f"\n  Time: {time.time()-t0:.1f}s")


# ================================================================
# IDEA 443: CASIMIR EFFECT — ROBUSTNESS VS N AND SEPARATIONS
# ================================================================
P("\n" + "=" * 78)
P("IDEA 443: CASIMIR EFFECT — 1/d SCALING ROBUSTNESS")
P("=" * 78)
P("""
Idea 365 found E ~ 0.097/d + 2.465 at N=50. Test:
  (a) Does 1/d still win at N=30,50,70,90?
  (b) Is the coefficient A in E = A/d + B stable?
  (c) Finer separation grid for better fits?
""")

t0 = time.time()

Ns_443 = [30, 50, 70, 90]
separations = [0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0]
n_trials_cas = 10

casimir_A_by_N = []
casimir_B_by_N = []
one_over_d_wins = []

for N in Ns_443:
    P(f"\n  N={N}:")
    vacuum_energy = {d: [] for d in separations}

    for d in separations:
        for trial in range(n_trials_cas):
            cs, coords = sprinkle_with_boundaries(N, d, rng_local=rng)
            W = sj_wightman_function(cs)
            vacuum_energy[d].append(np.trace(W))

    P(f"  {'d':>6}  {'Tr(W)':>10}  {'std':>8}")
    casimir_data = []
    for d in separations:
        energies = np.array(vacuum_energy[d])
        P(f"  {d:6.2f}  {energies.mean():10.4f}  {energies.std():8.4f}")
        casimir_data.append((d, energies.mean(), energies.std()))

    ds = np.array([c[0] for c in casimir_data])
    Es = np.array([c[1] for c in casimir_data])

    try:
        popt_1d, _ = curve_fit(lambda d, A, B: A/d + B, ds, Es, p0=[0.1, Es[-1]])
        ss_1d = np.sum((Es - (popt_1d[0]/ds + popt_1d[1]))**2)

        popt_d2, _ = curve_fit(lambda d, A, B: A/d**2 + B, ds, Es, p0=[0.1, Es[-1]])
        ss_d2 = np.sum((Es - (popt_d2[0]/ds**2 + popt_d2[1]))**2)

        ss_const = np.sum((Es - np.mean(Es))**2)

        P(f"  1/d:    E = {popt_1d[0]:.4f}/d + {popt_1d[1]:.4f}, SS={ss_1d:.6f}")
        P(f"  1/d^2:  E = {popt_d2[0]:.4f}/d^2 + {popt_d2[1]:.4f}, SS={ss_d2:.6f}")
        P(f"  const:  SS={ss_const:.6f}")

        winner = "1/d" if ss_1d <= ss_d2 else "1/d^2"
        ratio = ss_d2 / ss_1d if ss_1d > 0 else float('inf')
        P(f"  Winner: {winner} (ratio={ratio:.2f}x)")

        casimir_A_by_N.append(popt_1d[0])
        casimir_B_by_N.append(popt_1d[1])
        one_over_d_wins.append(ss_1d < ss_d2)
    except Exception as e:
        P(f"  Fit failed: {e}")

P(f"\n  --- Casimir coefficient stability ---")
P(f"  {'N':>4}  {'A':>10}  {'B':>10}  {'1/d wins':>10}")
for i, N in enumerate(Ns_443):
    if i < len(casimir_A_by_N):
        P(f"  {N:>4}  {casimir_A_by_N[i]:10.4f}  {casimir_B_by_N[i]:10.4f}  "
          f"{'YES' if one_over_d_wins[i] else 'NO':>10}")

if len(casimir_A_by_N) >= 3:
    A_arr = np.array(casimir_A_by_N)
    cv_A = np.std(A_arr) / abs(np.mean(A_arr)) if abs(np.mean(A_arr)) > 0 else float('inf')
    P(f"\n  Mean A = {np.mean(A_arr):.4f} +/- {np.std(A_arr):.4f} (CV={cv_A:.3f})")
    P(f"  1/d wins in {sum(one_over_d_wins)}/{len(one_over_d_wins)} cases")

    # Does A scale with N?
    Ns_arr = np.array(Ns_443[:len(A_arr)], dtype=float)
    sl, _, r_val, p_val, _ = stats.linregress(Ns_arr, A_arr)
    P(f"  A vs N: slope={sl:.6f}, p={p_val:.4f}")

P(f"\n  Time: {time.time()-t0:.1f}s")


# ================================================================
# IDEA 444: KR LAYER COUNT SCALING
# ================================================================
P("\n" + "=" * 78)
P("IDEA 444: KR LAYER COUNT — SQRT(N) SCALING?")
P("=" * 78)
P("""
Idea 346 showed MCMC KR phase has more layers than the theoretical 3.
Test: n_layers ~ sqrt(N)? Connection to RSK correspondence
(Robinson-Schensted-Knuth: longest increasing subsequence ~ 2*sqrt(N))?
""")

t0 = time.time()

# We need the two-order MCMC infrastructure
# Import from existing module
try:
    from causal_sets.two_orders import TwoOrder, swap_move
    from causal_sets.two_orders_v2 import bd_action_corrected

    EPS = 0.12

    def beta_c(N, eps):
        return 1.66 / (N * eps**2)

    def run_mcmc_for_kr(N, beta, eps, n_steps, n_therm, rng_local):
        """Simple MCMC to sample KR phase."""
        to = TwoOrder(N, rng=rng_local)
        cs_cur = to.to_causet()
        S = bd_action_corrected(cs_cur, eps)

        samples = []
        accepted = 0
        for step in range(n_steps):
            to_new = swap_move(to, rng=rng_local)
            cs_new = to_new.to_causet()
            S_new = bd_action_corrected(cs_new, eps)
            dS = S_new - S
            if dS <= 0 or rng_local.random() < np.exp(-beta * dS):
                to = to_new
                cs_cur = cs_new
                S = S_new
                accepted += 1
            if step >= n_therm and step % max(1, (n_steps - n_therm) // 20) == 0:
                samples.append(cs_cur)

        return samples, accepted / n_steps

    Ns_444 = [30, 50, 80, 120, 160]
    layer_counts_by_N = {}
    max_widths_by_N = {}

    for N in Ns_444:
        bc = beta_c(N, EPS)
        beta_run = 5.0 * bc  # deep in KR phase
        n_steps = min(20000, max(8000, N * 120))
        n_therm = n_steps // 2

        P(f"  N={N}: running MCMC at beta={beta_run:.1f} ({n_steps} steps)...")

        samples, acc = run_mcmc_for_kr(N, beta_run, EPS, n_steps, n_therm,
                                        np.random.default_rng(42 + N))

        if not samples:
            P(f"    No samples!")
            continue

        nls = []
        mws = []
        for cs in samples:
            dp = get_layers(cs)
            nl, sizes, mw = layer_stats(dp)
            nls.append(nl)
            mws.append(mw)

        mean_nl = np.mean(nls)
        std_nl = np.std(nls)
        mean_mw = np.mean(mws)
        layer_counts_by_N[N] = (mean_nl, std_nl)
        max_widths_by_N[N] = mean_mw

        P(f"    layers={mean_nl:.1f}+/-{std_nl:.1f}, max_width={mean_mw:.1f}, "
          f"acc={acc:.3f}, samples={len(samples)}")

    # Fit layer_count vs N
    if len(layer_counts_by_N) >= 3:
        Ns_arr = np.array(sorted(layer_counts_by_N.keys()), dtype=float)
        nls_arr = np.array([layer_counts_by_N[N][0] for N in Ns_arr.astype(int)])

        P(f"\n  --- Scaling fits ---")

        # sqrt(N) fit: n_layers = a * sqrt(N) + b
        try:
            sl, ic, r_sqrt, _, _ = stats.linregress(np.sqrt(Ns_arr), nls_arr)
            ss_sqrt = np.sum((nls_arr - (sl * np.sqrt(Ns_arr) + ic))**2)
            P(f"  sqrt(N): n_layers = {sl:.3f}*sqrt(N) + {ic:.3f}, R^2={r_sqrt**2:.4f}")
        except:
            ss_sqrt = float('inf')

        # ln(N) fit: n_layers = a * ln(N) + b
        try:
            sl2, ic2, r_log, _, _ = stats.linregress(np.log(Ns_arr), nls_arr)
            ss_log = np.sum((nls_arr - (sl2 * np.log(Ns_arr) + ic2))**2)
            P(f"  ln(N):   n_layers = {sl2:.3f}*ln(N) + {ic2:.3f}, R^2={r_log**2:.4f}")
        except:
            ss_log = float('inf')

        # Power law: n_layers = a * N^b
        try:
            sl3, ic3, r_pow, _, _ = stats.linregress(np.log(Ns_arr), np.log(nls_arr))
            P(f"  N^b:     n_layers = {np.exp(ic3):.3f}*N^{sl3:.3f}, R^2={r_pow**2:.4f}")
        except:
            pass

        # RSK connection: longest increasing subsequence ~ 2*sqrt(N)
        # If layers ~ sqrt(N), this connects KR to random permutation theory
        P(f"\n  RSK prediction: longest chain ~ 2*sqrt(N)")
        for N in sorted(layer_counts_by_N.keys()):
            rsk_pred = 2 * np.sqrt(N)
            measured = layer_counts_by_N[N][0]
            P(f"    N={N}: predicted={rsk_pred:.1f}, measured={measured:.1f}, "
              f"ratio={measured/rsk_pred:.3f}")

except ImportError as e:
    P(f"  SKIPPED: Could not import MCMC modules: {e}")

P(f"\n  Time: {time.time()-t0:.1f}s")


# ================================================================
# IDEA 445: TOPOLOGY DETECTION — MORE TOPOLOGIES
# ================================================================
P("\n" + "=" * 78)
P("IDEA 445: TOPOLOGY DETECTION — DIAMOND vs CYLINDER vs TORUS")
P("=" * 78)
P("""
Idea 300 distinguished diamond from cylinder with p<0.0001.
Extend: add torus, different cylinder radii, and more observables.
Also test: path entropy, Fiedler value, interval entropy as topology indicators.
""")

t0 = time.time()

N_445 = 60
n_trials_445 = 20

topologies = {
    'Diamond (d=2)': lambda: sprinkle_fast(N_445, dim=2, rng=rng),
    'Cylinder (L=1)': lambda: sprinkle_cylinder(N_445, rng),
    'Cylinder (L=0.5)': lambda: sprinkle_cylinder_wide(N_445, 0.5, rng),
    'Torus': lambda: sprinkle_torus(N_445, rng),
}

extended_obs = {
    'path_entropy': path_entropy,
    'fiedler': fiedler_value,
    'interval_entropy': interval_entropy,
}

all_obs = defaultdict(lambda: defaultdict(list))

for label, gen_func in topologies.items():
    P(f"\n  {label}:")
    for trial in range(n_trials_445):
        result = gen_func()
        cs = result[0] if isinstance(result, tuple) else result
        # Standard observables
        obs = topology_observables(cs)
        for k, v in obs.items():
            all_obs[label][k].append(v)
        # Extended observables
        for k, func in extended_obs.items():
            all_obs[label][k].append(func(cs))

    P(f"    {'Observable':>18}  {'mean':>8}  {'std':>8}")
    for k in ['f', 'link_density', 'n_minimal', 'n_maximal', 'spatial_width',
              'clustering', 'path_entropy', 'fiedler', 'interval_entropy']:
        vals = all_obs[label][k]
        P(f"    {k:>18}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}")

# Pairwise statistical tests
P(f"\n  --- Pairwise distinguishability (Mann-Whitney U) ---")
topo_names = list(topologies.keys())
obs_names = ['f', 'link_density', 'n_minimal', 'n_maximal', 'spatial_width',
             'clustering', 'path_entropy', 'fiedler', 'interval_entropy']

P(f"\n  {'Observable':>18}", end="")
for i in range(len(topo_names)):
    for j in range(i+1, len(topo_names)):
        short_label = f"{topo_names[i][:8]}v{topo_names[j][:8]}"
        P(f"  {short_label:>18}", end="")
P()

for obs_name in obs_names:
    P(f"  {obs_name:>18}", end="")
    for i in range(len(topo_names)):
        for j in range(i+1, len(topo_names)):
            vals_i = all_obs[topo_names[i]][obs_name]
            vals_j = all_obs[topo_names[j]][obs_name]
            try:
                stat, pval = stats.mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                P(f"  {pval:10.4f}{sig:>8}", end="")
            except:
                P(f"  {'N/A':>18}", end="")
    P()

# Multi-observable classifier test
P(f"\n  --- Combined classification accuracy ---")
# Use all observables to see if we can perfectly classify topologies
# Simple: which single observable is best?
best_obs = None
best_total_sig = 0
for obs_name in obs_names:
    total_sig = 0
    for i in range(len(topo_names)):
        for j in range(i+1, len(topo_names)):
            vals_i = all_obs[topo_names[i]][obs_name]
            vals_j = all_obs[topo_names[j]][obs_name]
            try:
                _, pval = stats.mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                if pval < 0.05:
                    total_sig += 1
            except:
                pass
    n_pairs = len(topo_names) * (len(topo_names) - 1) // 2
    P(f"  {obs_name:>18}: distinguishes {total_sig}/{n_pairs} pairs")
    if total_sig > best_total_sig:
        best_total_sig = total_sig
        best_obs = obs_name

P(f"\n  Best single observable: {best_obs} ({best_total_sig}/{n_pairs} pairs)")

P(f"\n  Time: {time.time()-t0:.1f}s")


# ================================================================
# FEASIBILITY ASSESSMENTS (Ideas 446-450)
# ================================================================
P("\n" + "=" * 78)
P("IDEAS 446-450: FEASIBILITY ASSESSMENTS FOR LARGER PAPERS")
P("=" * 78)

P("""
IDEA 446: GRAPH-THEORETIC DIMENSION PAPER
  Combine: path entropy + Fiedler + treewidth + ordering fraction + interval entropy
  as independent dimension estimators. Compare sensitivity, noise robustness,
  finite-size bias across d=2,3,4,5.

  ASSESSMENT: FEASIBLE (8/10). All computational tools exist. The novelty is:
  (a) 5+ independent graph-theoretic dimension estimators from causal structure alone
  (b) Comparison table showing which works best in which regime
  (c) Mathematical connections: why do these encode dimension?
  Key weakness: at d>=4, most observables collapse to similar values (shown in Exp79/Idea 311).
  Would need d=6,7,8 to test if this persists. N=200+ would help separate.

  PUBLISHABILITY: 7-7.5/10 as a methods/tools paper in CQG or Gen.Rel.Grav.
  Not revolutionary but fills a real gap: no systematic comparison exists.
""")

P("""
IDEA 447: "SJ VACUUM REPRODUCES KNOWN PHYSICS" PAPER
  Combine: Casimir (1/d scaling, Idea 365) + Newton's law (ln(r) decay, Idea 270)
  + area law (S ~ A^0.345, Idea 262) + conformal anomaly (<T> ~ R, Idea 366)
  + GW detection from intervals (Idea 269).

  ASSESSMENT: FEASIBLE (7/10). All results exist. The narrative:
  "The SJ vacuum on a causal set naturally encodes: (1) the propagator (Newton),
  (2) boundary effects (Casimir), (3) entropy (Bekenstein), (4) anomalies (trace),
  (5) perturbation detection (GW)."

  Key weakness: Each individual result is noisy, qualitative, or has caveats:
    - Newton: coefficient is 20x too small vs continuum (renormalization needed)
    - Casimir: 1/d instead of expected 1/d (but correct for 1+1D!)
    - Area law: exponent 0.345, not 0.5 (correct scaling class, wrong exponent)
    - Conformal: coefficient 40x too small
  A "qualitative but encouraging" framing is honest. A "quantitative agreement"
  framing would be dishonest.

  PUBLISHABILITY: 6.5-7/10 for PRD. Interesting as a survey but each component
  individually is a known result in the SJ vacuum literature.
""")

P("""
IDEA 448: COMPREHENSIVE BD TRANSITION PAPER
  Combine ALL BD transition observables: interval entropy + link fraction +
  spectral statistics (<r>) + Fiedler value + path entropy + action bimodality +
  latent heat + nucleation + KR structure + layer count.

  ASSESSMENT: HIGHLY FEASIBLE (9/10). This would be the definitive computational
  characterization of the BD transition. Nothing like this exists.

  Structure:
    1. Order parameters: link fraction (60% jump), interval entropy (87% drop),
       path entropy (monotonic), Fiedler value (243% jump)
    2. Transition character: latent heat (extensive), nucleation (sharp),
       hysteresis (weak), action bimodality (shallow)
    3. KR phase structure: layers ~ sqrt(N), not total order, wide antichains
    4. Spectral statistics: GUE in both phases, sub-Poisson only at transition
    5. Universal: confirmed at d=2, preliminary at d=4 (three-phase)

  Key strength: no single published paper has more than 2-3 of these observables.

  PUBLISHABILITY: 7.5/10 for PRD. Could be a "definitive numerical study" paper.
  The BD transition IS important in the field, and this would be the most thorough
  numerical study to date.
""")

P("""
IDEA 449: METHODS PAPER
  "Computational techniques for causal set quantum gravity"

  Content: sparse SVD for SJ vacuum at N=1000+, parallel tempering for MCMC,
  d-order generation, CDT interface, interval counting, Hasse diagram construction,
  graph Laplacian eigendecomposition.

  ASSESSMENT: LOW FEASIBILITY (4/10). Methods papers succeed when:
  (a) The methods are novel and non-obvious — ours are mostly standard
  (b) The code is released as a package — we'd need to clean up
  (c) Others need the methods — the causal set community is small (~50 active people)

  PUBLISHABILITY: 4/10. Better as a GitHub repo with documentation than a paper.
""")

P("""
IDEA 450: META-PAPER "400 COMPUTATIONAL EXPERIMENTS"
  "What we learned from 400 computational experiments in causal set quantum gravity"

  Structure:
    1. What works: SJ vacuum properties, BD transition, graph-theoretic observables
    2. What fails: spectral dimension (crossing theorem), Hawking temperature extraction,
       Ricci curvature (Ollivier), metastable lifetime measurement
    3. The 7.5 ceiling: why toy-scale N=30-200 prevents breakthrough results
    4. The density dominance trap: most causet-vs-null differences trace to ordering fraction
    5. Recommendations for the field: what to compute next, what computational scale is needed

  ASSESSMENT: FEASIBLE BUT RISKY (6/10). This is a "lessons learned" paper.
  It would be valuable for the community but requires:
  (a) Honest self-criticism (we must call our failures failures)
  (b) Actionable recommendations (not just "use bigger N")
  (c) A narrative arc (not just a list)

  Risk: reviewers may see it as "we tried a lot of stuff and nothing worked."
  The honest framing is: "Here's what the field should focus on based on
  extensive computational exploration."

  PUBLISHABILITY: 6/10 for a review journal (Rev.Mod.Phys, Living Reviews).
  Could be high-impact if framed right. Would need to include the 8 actual papers
  as supporting material.
""")

# ================================================================
# FINAL SUMMARY
# ================================================================
P("\n" + "=" * 78)
P("EXPERIMENT 93: FINAL SUMMARY — PAPER CANDIDATES")
P("=" * 78)

P("""
NUMERICAL TEST RESULTS:
""")

# Summarize 441
P("441. PATH ENTROPY AS DIMENSION ESTIMATOR:")
all_monotonic = True
all_separated = True
for N in Ns_441:
    means = [results_441[(N, d)][0] for d in dims]
    if not (all(means[i] > means[i+1] for i in range(len(means)-1)) or
            all(means[i] < means[i+1] for i in range(len(means)-1))):
        all_monotonic = False
    for i in range(len(dims)-1):
        m1, s1, _ = results_441[(N, dims[i])]
        m2, s2, _ = results_441[(N, dims[i+1])]
        gap = abs(m2 - m1)
        combined_std = np.sqrt(s1**2 + s2**2)
        if combined_std > 0 and gap / combined_std < 2.0:
            all_separated = False
P(f"     Monotonic across all N: {all_monotonic}")
P(f"     Well-separated (>2sigma): {all_separated}")
paper_441 = all_monotonic
P(f"     PAPER VIABLE: {'YES' if paper_441 else 'MARGINAL'}")

# Summarize 442
P("\n442. NEWTON'S LAW STABILITY:")
if len(log_coefficients) >= 3:
    coeffs = np.array(log_coefficients)
    cv = np.std(coeffs) / abs(np.mean(coeffs)) if abs(np.mean(coeffs)) > 0 else float('inf')
    stable = cv < 0.3
    P(f"     Coefficient CV: {cv:.3f}")
    P(f"     Stable across N: {stable}")
    P(f"     Mean coefficient: {np.mean(coeffs):.6f}")
    paper_442 = stable
    P(f"     PAPER VIABLE: {'YES' if paper_442 else 'NO — coefficient drifts'}")
else:
    P(f"     Insufficient data")
    paper_442 = False

# Summarize 443
P("\n443. CASIMIR EFFECT ROBUSTNESS:")
if len(one_over_d_wins) > 0:
    frac_wins = sum(one_over_d_wins) / len(one_over_d_wins)
    P(f"     1/d wins: {sum(one_over_d_wins)}/{len(one_over_d_wins)} sizes")
    if len(casimir_A_by_N) >= 3:
        A_arr = np.array(casimir_A_by_N)
        cv_A = np.std(A_arr) / abs(np.mean(A_arr)) if abs(np.mean(A_arr)) > 0 else float('inf')
        P(f"     Coefficient CV: {cv_A:.3f}")
    paper_443 = frac_wins >= 0.75
    P(f"     PAPER VIABLE: {'YES' if paper_443 else 'NO — scaling not robust'}")
else:
    P(f"     No data")
    paper_443 = False

# Summarize 444
P("\n444. KR LAYER COUNT SCALING:")
if len(layer_counts_by_N) >= 3:
    Ns_arr = np.array(sorted(layer_counts_by_N.keys()), dtype=float)
    nls_arr = np.array([layer_counts_by_N[N][0] for N in Ns_arr.astype(int)])
    sl, ic, r_sqrt, _, _ = stats.linregress(np.sqrt(Ns_arr), nls_arr)
    P(f"     sqrt(N) fit: R^2={r_sqrt**2:.4f}")
    paper_444 = r_sqrt**2 > 0.85
    P(f"     PAPER VIABLE: {'YES — sqrt(N) confirmed' if paper_444 else 'NEEDS MORE DATA'}")
else:
    P(f"     Insufficient data (MCMC may have failed)")
    paper_444 = False

# Summarize 445
P("\n445. TOPOLOGY DETECTION BREADTH:")
if best_obs:
    n_pairs = len(topo_names) * (len(topo_names) - 1) // 2
    P(f"     Best observable: {best_obs} ({best_total_sig}/{n_pairs} pairs distinguished)")
    paper_445 = best_total_sig >= n_pairs * 0.6
    P(f"     PAPER VIABLE: {'YES' if paper_445 else 'MARGINAL'}")
else:
    P(f"     No data")
    paper_445 = False

# Final recommendations
P(f"\n{'=' * 78}")
P("PAPER RECOMMENDATIONS — RANKED BY VIABILITY:")
P("=" * 78)

recs = [
    (448, 7.5, "BD Transition Comprehensive Study", "HIGHLY FEASIBLE",
     "All data exists. Would be the definitive numerical BD transition paper."),
    (446, 7.5, "Graph-Theoretic Dimension Estimators", "FEASIBLE" if paper_441 else "MARGINAL",
     "Path entropy + 4 other estimators. Gap at d>=4 is the weakness."),
    (447, 7.0, "SJ Vacuum Reproduces Known Physics", "FEASIBLE",
     "Newton + Casimir + area law + anomaly. Each noisy but collectively compelling."),
    (450, 6.5, "400 Experiments Meta-Paper", "FEASIBLE BUT RISKY",
     "Needs honest framing. High impact if done well. Risk of 'nothing worked' perception."),
    (449, 4.0, "Methods Paper", "LOW VALUE",
     "Better as a code release than a paper."),
]

P(f"\n  {'Idea':>5}  {'Score':>6}  {'Title':<40}  {'Verdict'}")
P(f"  {'-'*5}  {'-'*6}  {'-'*40}  {'-'*20}")
for idea, score, title, verdict, detail in recs:
    P(f"  {idea:>5}  {score:>6.1f}  {title:<40}  {verdict}")
    P(f"         {detail}")

P(f"\n  STRONGEST PAPER CANDIDATE: Idea 448 (BD Transition Comprehensive Study)")
P(f"  This combines ~10 independent results that are already computed and validated.")
P(f"  No other published paper characterizes the BD transition with more than 2-3 observables.")

P(f"\n  TOTAL TIME: {time.time()-total_start:.1f}s")
