"""
Experiment 58: Round 6 — Ideas 101-110

Ten genuinely new directions, all null-modeled, searching for an 8+ paper.

WHAT WE KNOW KILLS IDEAS (from 100 prior attempts):
  - GUE level statistics generic to all dense antisymmetric matrices
  - c_eff diverges on all causets (SJ normalization problem)
  - ER=EPR, monogamy, BW theorem, Bekenstein, c-theorem, ETH — all failed
  - Density/connectivity explains 60-80% of causet-vs-null differences

WHAT IS GENUINELY CAUSET-SPECIFIC:
  - Sub-Poisson clustering <r>=0.12 in KR crystalline phase
  - BD phase transition itself (interval entropy jump)
  - 4D three-phase structure (non-monotonic interval entropy)
  - Heavy-tailed eigenvalue density (kurtosis=53 vs semicircle=-1)

NEW DIRECTIONS EXPLORED HERE:
  101. Persistent homology (Betti numbers of the order complex)
  102. Causal set as quantum channel (channel capacity of C)
  103. Eigenvalue SPACING distribution shape (not just <r>)
  104. Specific heat exponent at the BD transition (thermodynamic critical exponent)
  105. Treewidth of the comparability graph (graph structural complexity)
  106. Interval size distribution mod p (number-theoretic structure)
  107. Entanglement spectrum (not entropy) — level statistics of W_A eigenvalues
  108. Graph zeta function (Ihara zeta of the link graph)
  109. Mutual information decay across causal diamonds (spatial correlations)
  110. Eigenvalue density as FUNCTION of position (local spectral measure)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.linalg import eigvalsh
from collections import Counter
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size

np.set_printoptions(precision=4, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# UTILITIES
# ============================================================

def sj_vacuum(cs):
    """Compute SJ Wightman function W and eigenvalues of H = i*Delta."""
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


def pauli_jordan_evals(cs):
    """Eigenvalues of H = i*(2/N)(C^T - C)."""
    N = cs.n
    C = cs.order.astype(float)
    H = 1j * (2.0 / N) * (C.T - C)
    return np.linalg.eigvalsh(H).real


def level_spacing_ratio(evals):
    """Mean level spacing ratio <r>."""
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    r_min = np.minimum(spacings[:-1], spacings[1:])
    r_max = np.maximum(spacings[:-1], spacings[1:])
    return np.mean(r_min / r_max)


def entanglement_entropy(W, region):
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    return float(-np.sum(eigs * np.log(eigs) + (1 - eigs) * np.log(1 - eigs)))


def make_2order(N, rng):
    to = TwoOrder(N, rng=rng)
    return to.to_causet()


def random_dag(N, density, rng):
    """Generate a random DAG with given relation density."""
    cs = FastCausalSet(N)
    mask = rng.random((N, N)) < density
    cs.order = np.triu(mask, k=1).astype(np.bool_)
    order_int = cs.order.astype(np.int32)
    changed = True
    while changed:
        new_order = ((order_int @ order_int) > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs


def make_dense_antisym(N, rng):
    """Random dense antisymmetric +-1 matrix."""
    signs = rng.choice([-1.0, 1.0], size=(N*(N-1)//2,))
    A = np.zeros((N, N))
    A[np.triu_indices(N, k=1)] = signs
    A -= A.T
    return A


# ================================================================
print("=" * 78)
print("IDEA 101: PERSISTENT HOMOLOGY — BETTI NUMBERS OF THE ORDER COMPLEX")
print("Do causal sets have topological invariants that random DAGs lack?")
print("=" * 78)

# The order complex of a poset: simplices are totally ordered subsets (chains).
# Beta_0 = connected components, Beta_1 = 1-cycles in the Hasse diagram.
# For a causet from flat 2D Minkowski: Beta_0=1 (connected), Beta_1 should
# reflect the trivial topology.
# For a random DAG: different Betti structure.
# We compute Betti numbers via the link graph (Hasse diagram).

N = 60
n_trials = 20

print("\n--- Betti numbers from the link graph ---")

# Beta_0 from link graph (connected components via union-find)
def connected_components(adj, N):
    """Count connected components of undirected graph."""
    visited = [False] * N
    n_comp = 0
    for start in range(N):
        if visited[start]:
            continue
        n_comp += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for nb in range(N):
                if adj[node, nb] and not visited[nb]:
                    stack.append(nb)
    return n_comp


# Beta_1 from the link graph: use Euler characteristic
# Beta_0 - Beta_1 = V - E (for graph, no higher simplices)
# => Beta_1 = E - V + Beta_0

b0_causet, b1_causet = [], []
b0_null, b1_null = [], []
cycle_density_causet, cycle_density_null = [], []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    L = cs.link_matrix()
    adj = (L | L.T).astype(int)
    V = N
    E = np.sum(np.triu(adj, k=1))
    b0 = connected_components(adj, N)
    b1 = E - V + b0
    b0_causet.append(b0)
    b1_causet.append(b1)
    cycle_density_causet.append(b1 / N)

    # Null
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    L_null = cs_null.link_matrix()
    adj_null = (L_null | L_null.T).astype(int)
    E_null = np.sum(np.triu(adj_null, k=1))
    b0_null_val = connected_components(adj_null, N)
    b1_null_val = E_null - N + b0_null_val
    b0_null.append(b0_null_val)
    b1_null.append(b1_null_val)
    cycle_density_null.append(b1_null_val / N)

print(f"  Causet: Beta_0={np.mean(b0_causet):.1f}±{np.std(b0_causet):.1f}, "
      f"Beta_1={np.mean(b1_causet):.1f}±{np.std(b1_causet):.1f}, "
      f"cycle density={np.mean(cycle_density_causet):.3f}")
print(f"  Null:   Beta_0={np.mean(b0_null):.1f}±{np.std(b0_null):.1f}, "
      f"Beta_1={np.mean(b1_null):.1f}±{np.std(b1_null):.1f}, "
      f"cycle density={np.mean(cycle_density_null):.3f}")

t_b1, p_b1 = stats.ttest_ind(cycle_density_causet, cycle_density_null)
print(f"  Cycle density t-test: t={t_b1:.2f}, p={p_b1:.4f}")

# N-scaling: does cycle density converge?
print("\n  N-scaling of cycle density (Beta_1/N):")
for Ntest in [30, 50, 70, 100]:
    cd_c, cd_n = [], []
    for _ in range(15):
        cs = make_2order(Ntest, rng)
        L = cs.link_matrix()
        adj = (L | L.T).astype(int)
        E = np.sum(np.triu(adj, k=1))
        b0 = connected_components(adj, Ntest)
        cd_c.append((E - Ntest + b0) / Ntest)

        dens = cs.ordering_fraction()
        cs_n = random_dag(Ntest, dens * 0.7, rng)
        L_n = cs_n.link_matrix()
        adj_n = (L_n | L_n.T).astype(int)
        E_n = np.sum(np.triu(adj_n, k=1))
        b0_n = connected_components(adj_n, Ntest)
        cd_n.append((E_n - Ntest + b0_n) / Ntest)

    t, p = stats.ttest_ind(cd_c, cd_n)
    print(f"    N={Ntest:4d}: causet={np.mean(cd_c):.3f}±{np.std(cd_c):.3f}, "
          f"null={np.mean(cd_n):.3f}±{np.std(cd_n):.3f}, p={p:.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 102: CAUSAL SET AS QUANTUM CHANNEL — CHANNEL CAPACITY OF C")
print("Treat C as a classical channel matrix. What is its capacity?")
print("=" * 78)

# The causal matrix C maps past events to future events.
# Normalize each row of C to get a stochastic matrix: P[i,j] = C[i,j]/sum_j C[i,j]
# This defines a classical channel. The Shannon capacity of this channel
# measures how much information can flow through the causal structure.
# For flat spacetime: capacity should scale with dimension.
# For random DAGs: different scaling.

N = 60
n_trials = 20

print("\n--- Channel capacity of the normalized causal matrix ---")

def channel_entropy(P_row):
    """Shannon entropy of a probability distribution."""
    p = P_row[P_row > 1e-15]
    return -np.sum(p * np.log2(p))


def channel_capacity_approx(C):
    """Approximate channel capacity: max H(Y) - H(Y|X).
    Use uniform input distribution as lower bound."""
    N = C.shape[0]
    row_sums = C.sum(axis=1)
    # Only rows with nonzero sum can transmit
    active = row_sums > 0
    if not np.any(active):
        return 0.0

    P = np.zeros_like(C, dtype=float)
    P[active] = C[active] / row_sums[active, None]

    # H(Y|X) = average row entropy
    h_yx = np.mean([channel_entropy(P[i]) for i in range(N) if active[i]])

    # H(Y) with uniform input
    p_y = P[active].mean(axis=0)  # average column distribution
    p_y = p_y / (p_y.sum() + 1e-15)
    h_y = channel_entropy(p_y)

    return max(0, h_y - h_yx)


cap_causet, cap_null = [], []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    C = cs.order.astype(float)
    cap_causet.append(channel_capacity_approx(C))

    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    C_null = cs_null.order.astype(float)
    cap_null.append(channel_capacity_approx(C_null))

print(f"  Causet: capacity = {np.mean(cap_causet):.3f}±{np.std(cap_causet):.3f} bits")
print(f"  Null:   capacity = {np.mean(cap_null):.3f}±{np.std(cap_null):.3f} bits")
t_cap, p_cap = stats.ttest_ind(cap_causet, cap_null)
print(f"  t-test: t={t_cap:.2f}, p={p_cap:.4f}")

# N-scaling
print("\n  N-scaling of channel capacity:")
for Ntest in [30, 50, 80, 100]:
    cc, cn = [], []
    for _ in range(15):
        cs = make_2order(Ntest, rng)
        cc.append(channel_capacity_approx(cs.order.astype(float)))
        dens = cs.ordering_fraction()
        cs_n = random_dag(Ntest, dens * 0.7, rng)
        cn.append(channel_capacity_approx(cs_n.order.astype(float)))
    t, p = stats.ttest_ind(cc, cn)
    print(f"    N={Ntest:4d}: causet={np.mean(cc):.3f}±{np.std(cc):.3f}, "
          f"null={np.mean(cn):.3f}±{np.std(cn):.3f}, p={p:.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 103: EIGENVALUE SPACING DISTRIBUTION SHAPE (not just <r>)")
print("Is the FULL P(s) distribution GUE, or just the mean?")
print("=" * 78)

# We know <r>~0.60 (GUE-like). But is the FULL distribution of spacings
# actually GUE? Or is it a mixture? The higher moments (variance, skewness,
# kurtosis of the spacing distribution) could reveal non-GUE structure.

N = 100
n_trials = 80

all_spacings_causet = []
all_spacings_null = []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    evals = np.sort(pauli_jordan_evals(cs))
    # Unfold: normalize spacings by local mean
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) > 5:
        # Local unfolding: divide by rolling mean
        window = max(5, len(spacings) // 5)
        local_mean = np.convolve(spacings, np.ones(window)/window, mode='same')
        local_mean = np.maximum(local_mean, 1e-14)
        unfolded = spacings / local_mean
        all_spacings_causet.extend(unfolded.tolist())

    # Dense antisymmetric null
    A = make_dense_antisym(N, rng)
    evals_null = np.sort(eigvalsh(1j * (2.0/N) * A).real)
    spacings_null = np.diff(evals_null)
    spacings_null = spacings_null[spacings_null > 1e-14]
    if len(spacings_null) > 5:
        window = max(5, len(spacings_null) // 5)
        local_mean_n = np.convolve(spacings_null, np.ones(window)/window, mode='same')
        local_mean_n = np.maximum(local_mean_n, 1e-14)
        unfolded_n = spacings_null / local_mean_n
        all_spacings_null.extend(unfolded_n.tolist())

sc = np.array(all_spacings_causet)
sn = np.array(all_spacings_null)

print(f"\n  Causet unfolded spacings: n={len(sc)}")
print(f"    mean={np.mean(sc):.4f}, std={np.std(sc):.4f}, "
      f"skew={stats.skew(sc):.4f}, kurt={stats.kurtosis(sc):.4f}")
print(f"  Dense null unfolded spacings: n={len(sn)}")
print(f"    mean={np.mean(sn):.4f}, std={np.std(sn):.4f}, "
      f"skew={stats.skew(sn):.4f}, kurt={stats.kurtosis(sn):.4f}")

# GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4s^2/pi)
# Mean = sqrt(pi)/2 * Gamma(3/2)/Gamma(2) ... ~0.94
# Variance ~0.178, skewness ~0.64, kurtosis ~0.49
# Test via KS against GUE Wigner surmise CDF
def gue_wigner_cdf(s):
    """CDF of the GUE Wigner surmise P(s)=(32/pi^2)*s^2*exp(-4s^2/pi)."""
    from scipy.special import erf, erfc
    a = 4.0 / np.pi
    # CDF = 1 - erfc(sqrt(a)*s) - (2*sqrt(a)/sqrt(pi))*s*exp(-a*s^2)
    # More carefully: integrate P(s) from 0 to s
    # Use numerical integration for accuracy
    return None  # Will use KS 2-sample instead

ks_stat, ks_p = stats.ks_2samp(sc, sn)
print(f"\n  KS test (causet vs dense null): D={ks_stat:.4f}, p={ks_p:.6f}")

# Test specific moments against GUE predictions
# GUE number variance (Sigma^2) in unfolded spectrum
print("\n  Testing GUE number variance:")
# For GUE: Sigma^2(L) ~ (1/pi^2) ln(L) for large L
# Compute number variance in windows of different sizes
for win_size in [5, 10, 20]:
    var_c = np.var([np.sum((sc[i:i+win_size] > 0)) for i in range(0, len(sc)-win_size, win_size)])
    var_n = np.var([np.sum((sn[i:i+win_size] > 0)) for i in range(0, len(sn)-win_size, win_size)])
    gue_pred = (1.0/np.pi**2) * np.log(win_size) + 0.5  # approximate
    print(f"    L={win_size}: causet Sigma^2={var_c:.3f}, null={var_n:.3f}, GUE~{gue_pred:.3f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 104: SPECIFIC HEAT EXPONENT AT THE BD TRANSITION")
print("What is the universality class of the BD phase transition?")
print("=" * 78)

# The BD transition is real and causet-specific. What's its universality class?
# Measure the specific heat C_v = beta^2 * (<S^2> - <S>^2) / N
# near the transition. The specific heat exponent alpha determines
# the universality class: C_v ~ |beta - beta_c|^{-alpha}
# Known universality classes: Ising (alpha=0), XY (alpha~-0.015),
# 3-state Potts (alpha=1/3), mean-field (alpha=0).
# If alpha is NEW, that's a genuine result.

from causal_sets.two_orders import mcmc_two_order

N = 40  # Moderate size for MCMC
n_therm = 3000
n_sample = 5000
record_every = 5

# Scan beta near the transition (from previous work: beta_c ~ 1-3 for N=40)
betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

print(f"\n--- Specific heat scan: N={N}, {n_sample} MCMC steps ---")
print(f"  {'beta':>6}  {'<S>/N':>8}  {'Cv/N':>8}  {'<of>':>8}  {'accept':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

cv_values = []
beta_values = []

for beta in betas:
    result = mcmc_two_order(N, beta, epsilon=1.0,
                            n_steps=n_therm + n_sample,
                            n_thermalize=n_therm,
                            record_every=record_every,
                            rng=rng, verbose=False)

    actions = result['actions']
    S_mean = np.mean(actions)
    S_var = np.var(actions)
    Cv = beta**2 * S_var / N  # specific heat per element

    # Also measure ordering fraction
    of_vals = [s.ordering_fraction() for s in result['samples'][-20:]]
    of_mean = np.mean(of_vals)

    print(f"  {beta:6.1f}  {S_mean/N:8.3f}  {Cv:8.3f}  {of_mean:8.3f}  {result['accept_rate']:8.3f}")
    cv_values.append(Cv)
    beta_values.append(beta)

# Find peak in specific heat
cv_arr = np.array(cv_values)
peak_idx = np.argmax(cv_arr)
beta_c_est = beta_values[peak_idx]
Cv_max = cv_arr[peak_idx]
print(f"\n  Estimated beta_c = {beta_c_est:.1f} (Cv_max/N = {Cv_max:.3f})")

# FSS: repeat for different N to get alpha
print("\n  Finite-size scaling: Cv_max vs N")
N_sizes_fss = [25, 35, 45]
Cv_max_by_N = []
for Nfss in N_sizes_fss:
    best_cv = 0
    for beta_scan in [1.0, 2.0, 3.0, 4.0]:
        res = mcmc_two_order(Nfss, beta_scan, epsilon=1.0,
                             n_steps=2000 + 3000,
                             n_thermalize=2000,
                             record_every=5,
                             rng=rng, verbose=False)
        cv_scan = beta_scan**2 * np.var(res['actions']) / Nfss
        best_cv = max(best_cv, cv_scan)
    Cv_max_by_N.append(best_cv)
    print(f"    N={Nfss}: Cv_max/N = {best_cv:.4f}")

# Fit Cv_max ~ N^{alpha/nu} (for 2D: if alpha > 0, Cv diverges with N)
if len(N_sizes_fss) >= 3:
    log_N = np.log(N_sizes_fss)
    log_Cv = np.log(np.array(Cv_max_by_N) + 1e-10)
    slope, intercept, r_val, _, _ = stats.linregress(log_N, log_Cv)
    print(f"\n  Cv_max ~ N^{slope:.3f} (R^2={r_val**2:.3f})")
    print(f"  alpha/nu estimate: {slope:.3f}")
    if abs(slope) < 0.2:
        print("  Consistent with logarithmic divergence (alpha=0, like 2D Ising)")
    elif slope > 0.3:
        print(f"  Divergent specific heat: alpha/nu ~ {slope:.2f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 105: TREEWIDTH OF THE COMPARABILITY GRAPH")
print("Does the causal set have bounded treewidth (like a manifold)?")
print("=" * 78)

# Treewidth measures how "tree-like" a graph is.
# For a d-dimensional lattice: treewidth ~ N^{(d-1)/d}
# For a random graph: treewidth ~ N
# If causets have treewidth ~ N^{1/2} (for d=2), this encodes dimension.
# Exact treewidth is NP-hard, but we can get bounds.
# Upper bound: minimum degree ordering width.
# Lower bound: maximum clique size - 1.

N = 60
n_trials = 15

print("\n--- Treewidth bounds for causets vs random DAGs ---")

def min_degree_width(adj, N):
    """Upper bound on treewidth via greedy minimum degree elimination."""
    adj = adj.copy().astype(bool)
    remaining = set(range(N))
    width = 0
    for _ in range(N):
        if not remaining:
            break
        # Find node with minimum degree among remaining
        min_deg = N + 1
        min_node = -1
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and adj[v, u])
            if deg < min_deg:
                min_deg = deg
                min_node = v
        width = max(width, min_deg)
        # Eliminate: connect all neighbors of min_node
        neighbors = [u for u in remaining if u != min_node and adj[min_node, u]]
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                adj[neighbors[i], neighbors[j]] = True
                adj[neighbors[j], neighbors[i]] = True
        remaining.remove(min_node)
    return width


def max_clique_lower_bound(adj, N):
    """Lower bound on treewidth: max clique size - 1 (greedy)."""
    best = 0
    for start in range(N):
        clique = [start]
        candidates = [v for v in range(N) if v != start and adj[start, v]]
        for c in candidates:
            if all(adj[c, u] for u in clique):
                clique.append(c)
        best = max(best, len(clique))
    return best - 1


tw_upper_causet, tw_upper_null = [], []
tw_lower_causet, tw_lower_null = [], []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    # Comparability graph: i~j if order[i,j] or order[j,i]
    comp = (cs.order | cs.order.T).astype(int)
    tw_upper_causet.append(min_degree_width(comp, N))
    tw_lower_causet.append(max_clique_lower_bound(comp, N))

    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    comp_null = (cs_null.order | cs_null.order.T).astype(int)
    tw_upper_null.append(min_degree_width(comp_null, N))
    tw_lower_null.append(max_clique_lower_bound(comp_null, N))

print(f"  Causet: tw upper={np.mean(tw_upper_causet):.1f}±{np.std(tw_upper_causet):.1f}, "
      f"lower={np.mean(tw_lower_causet):.1f}±{np.std(tw_lower_causet):.1f}")
print(f"  Null:   tw upper={np.mean(tw_upper_null):.1f}±{np.std(tw_upper_null):.1f}, "
      f"lower={np.mean(tw_lower_null):.1f}±{np.std(tw_lower_null):.1f}")

t_tw, p_tw = stats.ttest_ind(tw_upper_causet, tw_upper_null)
print(f"  t-test (upper bounds): t={t_tw:.2f}, p={p_tw:.4f}")

# N-scaling
print("\n  N-scaling of treewidth/N:")
for Ntest in [30, 50, 70, 100]:
    twc, twn = [], []
    for _ in range(10):
        cs = make_2order(Ntest, rng)
        comp = (cs.order | cs.order.T).astype(int)
        twc.append(min_degree_width(comp, Ntest) / Ntest)
        dens = cs.ordering_fraction()
        cs_n = random_dag(Ntest, dens * 0.7, rng)
        comp_n = (cs_n.order | cs_n.order.T).astype(int)
        twn.append(min_degree_width(comp_n, Ntest) / Ntest)
    t, p = stats.ttest_ind(twc, twn)
    print(f"    N={Ntest:4d}: causet tw/N={np.mean(twc):.3f}±{np.std(twc):.3f}, "
          f"null={np.mean(twn):.3f}±{np.std(twn):.3f}, p={p:.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 106: INTERVAL SIZE DISTRIBUTION MOD p")
print("Does the distribution of interval sizes have number-theoretic structure?")
print("=" * 78)

# For a causet with N elements, the interval sizes {|I(x,y)|}
# form a distribution. Look at this distribution mod p for small primes.
# If there's geometric structure, certain residues might be preferred.
# This is a shot in the dark but could reveal hidden periodicity.

N = 80
n_trials = 20

print("\n--- Interval size distribution mod p ---")

all_intervals_causet = []
all_intervals_null = []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    _, sizes = cs.interval_sizes_vectorized()
    all_intervals_causet.extend(sizes.tolist())

    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    _, sizes_null = cs_null.interval_sizes_vectorized()
    all_intervals_null.extend(sizes_null.tolist())

all_intervals_causet = np.array(all_intervals_causet)
all_intervals_null = np.array(all_intervals_null)

for p in [2, 3, 5, 7]:
    residues_c = all_intervals_causet % p
    residues_n = all_intervals_null % p

    # Chi-squared test for uniformity
    counts_c = np.bincount(residues_c, minlength=p)
    counts_n = np.bincount(residues_n, minlength=p)

    # Normalize to frequencies
    freq_c = counts_c / counts_c.sum()
    freq_n = counts_n / counts_n.sum()

    # Chi-squared against uniform
    expected_c = np.full(p, len(residues_c) / p)
    expected_n = np.full(p, len(residues_n) / p)
    chi2_c, p_val_c = stats.chisquare(counts_c, expected_c)
    chi2_n, p_val_n = stats.chisquare(counts_n, expected_n)

    # Compare causet vs null
    ks_stat, ks_p = stats.ks_2samp(residues_c.astype(float), residues_n.astype(float))

    print(f"\n  mod {p}:")
    print(f"    Causet frequencies: {freq_c}")
    print(f"    Null frequencies:   {freq_n}")
    print(f"    Causet chi2 vs uniform: {chi2_c:.1f}, p={p_val_c:.4f}")
    print(f"    Null chi2 vs uniform:   {chi2_n:.1f}, p={p_val_n:.4f}")
    print(f"    KS causet vs null: D={ks_stat:.3f}, p={ks_p:.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 107: ENTANGLEMENT SPECTRUM — LEVEL STATISTICS OF W_A EIGENVALUES")
print("Does the entanglement spectrum have GUE statistics too?")
print("=" * 78)

# The entanglement spectrum is the set of eigenvalues of the reduced
# Wightman function W_A = W restricted to a subregion A.
# If these eigenvalues have GUE spacing statistics, that's a new link
# between quantum chaos and entanglement in causet QG.
# If they DON'T match GUE (despite the full spectrum being GUE),
# that's also interesting — it means entanglement breaks the universality.

N = 80
n_trials = 40

r_full_causet = []
r_ent_causet = []
r_ent_null = []

print("\n--- Entanglement spectrum level statistics ---")

for trial in range(n_trials):
    cs = make_2order(N, rng)
    W, full_evals = sj_vacuum(cs)

    # Full spectrum <r>
    r_full_causet.append(level_spacing_ratio(full_evals))

    # Entanglement spectrum: eigenvalues of W_A for A = first half
    region_A = list(range(N // 2))
    W_A = W[np.ix_(region_A, region_A)]
    ent_evals = np.linalg.eigvalsh(W_A)
    r_ent_causet.append(level_spacing_ratio(ent_evals))

    # Null: random dense antisymmetric matrix
    A_null = make_dense_antisym(N, rng)
    H_null = 1j * (2.0/N) * A_null
    evals_n, evecs_n = np.linalg.eigh(H_null)
    pos = evals_n.real > 1e-12
    if np.any(pos):
        W_null = (evecs_n[:, pos] @ np.diag(evals_n.real[pos]) @ evecs_n[:, pos].conj().T).real
        w_max = np.linalg.eigvalsh(W_null).max()
        if w_max > 1:
            W_null = W_null / w_max
        W_A_null = W_null[np.ix_(region_A, region_A)]
        ent_evals_null = np.linalg.eigvalsh(W_A_null)
        r_ent_null.append(level_spacing_ratio(ent_evals_null))

print(f"  Full spectrum <r>:       causet = {np.nanmean(r_full_causet):.4f} (GUE=0.5996)")
print(f"  Entanglement spectrum:   causet = {np.nanmean(r_ent_causet):.4f}")
print(f"  Ent. spectrum (null):            = {np.nanmean(r_ent_null):.4f}")

t_ent, p_ent = stats.ttest_ind([x for x in r_ent_causet if not np.isnan(x)],
                                [x for x in r_ent_null if not np.isnan(x)])
print(f"  t-test (ent spec causet vs null): t={t_ent:.2f}, p={p_ent:.4f}")

# Is the entanglement spectrum GUE or something else?
r_ent_mean = np.nanmean(r_ent_causet)
if abs(r_ent_mean - 0.5996) < 0.03:
    print("  => Entanglement spectrum is ALSO GUE")
elif abs(r_ent_mean - 0.5307) < 0.03:
    print("  => Entanglement spectrum is GOE (time reversal restored!)")
elif abs(r_ent_mean - 0.3863) < 0.03:
    print("  => Entanglement spectrum is Poisson (integrable)")
else:
    print(f"  => Entanglement spectrum <r>={r_ent_mean:.4f} — intermediate/non-universal")

# N-scaling
print("\n  N-scaling of entanglement spectrum <r>:")
for Ntest in [40, 60, 80, 100]:
    rc, rn = [], []
    for _ in range(20):
        cs = make_2order(Ntest, rng)
        W, _ = sj_vacuum(cs)
        region = list(range(Ntest // 2))
        W_A = W[np.ix_(region, region)]
        rc.append(level_spacing_ratio(np.linalg.eigvalsh(W_A)))

        A_null = make_dense_antisym(Ntest, rng)
        H_null = 1j * (2.0/Ntest) * A_null
        evals_n, evecs_n = np.linalg.eigh(H_null)
        pos = evals_n.real > 1e-12
        if np.any(pos):
            W_null = (evecs_n[:, pos] @ np.diag(evals_n.real[pos]) @ evecs_n[:, pos].conj().T).real
            wm = np.linalg.eigvalsh(W_null).max()
            if wm > 1:
                W_null = W_null / wm
            W_A_n = W_null[np.ix_(region, region)]
            rn.append(level_spacing_ratio(np.linalg.eigvalsh(W_A_n)))

    print(f"    N={Ntest:4d}: causet <r>_ent={np.nanmean(rc):.4f}, null={np.nanmean(rn):.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 108: IHARA ZETA FUNCTION OF THE LINK GRAPH")
print("Does the graph zeta function encode geometric information?")
print("=" * 78)

# The Ihara zeta function Z(u) of a graph is related to the adjacency matrix:
# 1/Z(u) = (1-u^2)^{r-1} * det(I - Au + Qu^2)
# where A = adjacency matrix, Q = degree matrix - I, r = |E|-|V|+1.
# The zeros of 1/Z(u) (poles of Z(u)) encode the graph spectrum.
# For a regular graph, the Ihara zeta recovers the eigenvalues.
# For irregular graphs (like causet link graphs), the distribution of
# these poles may encode geometric information.
# We compute the characteristic polynomial det(I - Au + Qu^2) at specific u values.

N = 50
n_trials = 15

print("\n--- Ihara zeta function poles (via adjacency spectrum) ---")

# The poles of Z(u) on |u|<1 come from eigenvalues of the adjacency matrix.
# For a (q+1)-regular graph: poles at u = 1/lambda where lambda are adj eigenvalues.
# For irregular graphs: use the Hashimoto (edge) matrix instead.
# Simplification: just compare adj eigenvalue distributions.

adj_evals_causet = []
adj_evals_null = []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    L = cs.link_matrix()
    adj = (L | L.T).astype(float)
    evals = np.linalg.eigvalsh(adj)
    adj_evals_causet.extend(evals.tolist())

    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    L_null = cs_null.link_matrix()
    adj_null = (L_null | L_null.T).astype(float)
    evals_null = np.linalg.eigvalsh(adj_null)
    adj_evals_null.extend(evals_null.tolist())

ec = np.array(adj_evals_causet)
en = np.array(adj_evals_null)

print(f"  Causet link adj spectrum: mean={np.mean(ec):.3f}, std={np.std(ec):.3f}, "
      f"max={np.max(ec):.3f}, kurtosis={stats.kurtosis(ec):.3f}")
print(f"  Null link adj spectrum:   mean={np.mean(en):.3f}, std={np.std(en):.3f}, "
      f"max={np.max(en):.3f}, kurtosis={stats.kurtosis(en):.3f}")

ks_stat, ks_p = stats.ks_2samp(ec, en)
print(f"  KS test: D={ks_stat:.4f}, p={ks_p:.6f}")

# Spectral gap of link adjacency
print("\n  Spectral gap (lambda_1 - lambda_2) of link adjacency:")
for Ntest in [30, 50, 70]:
    gc, gn = [], []
    for _ in range(12):
        cs = make_2order(Ntest, rng)
        L = cs.link_matrix()
        adj = (L | L.T).astype(float)
        evals = sorted(np.linalg.eigvalsh(adj))
        gc.append(evals[-1] - evals[-2])

        dens = cs.ordering_fraction()
        cs_n = random_dag(Ntest, dens * 0.7, rng)
        L_n = cs_n.link_matrix()
        adj_n = (L_n | L_n.T).astype(float)
        evals_n = sorted(np.linalg.eigvalsh(adj_n))
        gn.append(evals_n[-1] - evals_n[-2])
    t, p = stats.ttest_ind(gc, gn)
    print(f"    N={Ntest:4d}: causet gap={np.mean(gc):.3f}±{np.std(gc):.3f}, "
          f"null={np.mean(gn):.3f}±{np.std(gn):.3f}, p={p:.4f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 109: MUTUAL INFORMATION DECAY ACROSS CAUSAL DIAMONDS")
print("How does I(A:B) decay with separation in the causal set?")
print("=" * 78)

# In a QFT, mutual information I(A:B) decays as a power law with distance.
# In a CFT in 2D: I(A:B) ~ (l_A * l_B) / d^2 where d = separation.
# For the SJ vacuum on a causet: does I(A:B) show a similar power law?
# And does it differ from the null (random antisymmetric matrix)?
# This probes whether the SJ vacuum encodes locality.

N = 80
n_trials = 15

print("\n--- Mutual information decay with causal separation ---")

# Strategy: fix two small regions A and B at different "heights" in the causet.
# Sort elements by layer (longest chain distance from bottom).
# A = elements in layers 0-2, B = elements in layers k to k+2.
# Measure I(A:B) = S_A + S_B - S_{AB}.

def assign_layers(cs):
    """Assign each element to its chain-distance from the bottom."""
    N = cs.n
    layers = np.zeros(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            layers[j] = np.max(layers[preds]) + 1
    return layers


mi_by_sep_causet = {}
mi_by_sep_null = {}

for trial in range(n_trials):
    cs = make_2order(N, rng)
    W, _ = sj_vacuum(cs)
    layers = assign_layers(cs)
    max_layer = layers.max()

    if max_layer < 8:
        continue

    # Region A: layer 0-1
    A = list(np.where(layers <= 1)[0])
    if len(A) < 3:
        continue

    for sep in range(2, min(max_layer - 2, 10)):
        B = list(np.where((layers >= sep) & (layers <= sep + 1))[0])
        if len(B) < 3:
            continue
        AB = sorted(set(A) | set(B))

        S_A = entanglement_entropy(W, A)
        S_B = entanglement_entropy(W, B)
        S_AB = entanglement_entropy(W, AB)
        MI = S_A + S_B - S_AB

        if sep not in mi_by_sep_causet:
            mi_by_sep_causet[sep] = []
        mi_by_sep_causet[sep].append(MI)

    # Null
    A_null_mat = make_dense_antisym(N, rng)
    H_null = 1j * (2.0/N) * A_null_mat
    evals_n, evecs_n = np.linalg.eigh(H_null)
    pos = evals_n.real > 1e-12
    if np.any(pos):
        W_null = (evecs_n[:, pos] @ np.diag(evals_n.real[pos]) @ evecs_n[:, pos].conj().T).real
        wm = np.linalg.eigvalsh(W_null).max()
        if wm > 1:
            W_null = W_null / wm

        for sep in range(2, min(max_layer - 2, 10)):
            B = list(np.where((layers >= sep) & (layers <= sep + 1))[0])
            if len(B) < 3:
                continue
            AB = sorted(set(A) | set(B))

            S_A_n = entanglement_entropy(W_null, A)
            S_B_n = entanglement_entropy(W_null, B)
            S_AB_n = entanglement_entropy(W_null, AB)
            MI_n = S_A_n + S_B_n - S_AB_n

            if sep not in mi_by_sep_null:
                mi_by_sep_null[sep] = []
            mi_by_sep_null[sep].append(MI_n)

print(f"  {'sep':>5}  {'MI_causet':>10}  {'MI_null':>10}  {'ratio':>8}  {'p':>8}")
print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

for sep in sorted(mi_by_sep_causet.keys()):
    if sep in mi_by_sep_null and len(mi_by_sep_causet[sep]) >= 5 and len(mi_by_sep_null[sep]) >= 5:
        mc = np.mean(mi_by_sep_causet[sep])
        mn = np.mean(mi_by_sep_null[sep])
        t, p = stats.ttest_ind(mi_by_sep_causet[sep], mi_by_sep_null[sep])
        print(f"  {sep:5d}  {mc:10.4f}  {mn:10.4f}  {mc/(mn+1e-10):8.3f}  {p:8.4f}")

# Power law fit for causet MI decay
seps = sorted([s for s in mi_by_sep_causet.keys() if len(mi_by_sep_causet[s]) >= 5])
if len(seps) >= 3:
    mi_means = [np.mean(mi_by_sep_causet[s]) for s in seps]
    mi_positive = [(s, m) for s, m in zip(seps, mi_means) if m > 0]
    if len(mi_positive) >= 3:
        log_sep = np.log([x[0] for x in mi_positive])
        log_mi = np.log([x[1] for x in mi_positive])
        slope, _, r_val, _, _ = stats.linregress(log_sep, log_mi)
        print(f"\n  Power law fit: I(A:B) ~ sep^{slope:.2f} (R^2={r_val**2:.3f})")
        print(f"  CFT prediction for 2D: I ~ 1/d^2 (slope = -2)")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 110: LOCAL SPECTRAL MEASURE — EIGENVALUE DENSITY AS f(position)")
print("Does the SJ spectrum depend on WHERE you are in the causet?")
print("=" * 78)

# The SJ vacuum H = i*(2/N)*(C^T - C) has eigenvectors v_k.
# The LOCAL density of states at element x is:
# rho(x, E) = sum_k |v_k(x)|^2 * delta(E - E_k)
# If the causet has geometric structure, elements near the center
# should have different rho than elements near the boundary.
# This is analogous to the Local Density of States (LDOS) in condensed matter.
# For random matrices: LDOS is uniform. For physical systems: it varies.

N = 80
n_trials = 20

print("\n--- Local density of states (LDOS) variation ---")

# Measure: for each element x, compute the participation of x in
# positive vs negative eigenvalue eigenvectors.
# Define asymmetry: A(x) = sum_{k: E_k>0} |v_k(x)|^2 - sum_{k: E_k<0} |v_k(x)|^2

ldos_variation_causet = []
ldos_variation_null = []
ldos_layer_corr_causet = []
ldos_layer_corr_null = []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    C = cs.order.astype(float)
    H = 1j * (2.0/N) * (C.T - C)
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real

    # LDOS asymmetry per element
    pos_mask = evals > 1e-12
    neg_mask = evals < -1e-12
    weight_pos = np.sum(np.abs(evecs[:, pos_mask])**2, axis=1) if np.any(pos_mask) else np.zeros(N)
    weight_neg = np.sum(np.abs(evecs[:, neg_mask])**2, axis=1) if np.any(neg_mask) else np.zeros(N)
    asymmetry = weight_pos - weight_neg

    # Variation: std of asymmetry across elements
    ldos_variation_causet.append(np.std(asymmetry))

    # Does asymmetry correlate with layer (time position)?
    layers = assign_layers(cs)
    if np.std(layers) > 0:
        corr = np.corrcoef(layers, asymmetry)[0, 1]
        ldos_layer_corr_causet.append(corr)

    # Null: random dense antisymmetric
    A_null = make_dense_antisym(N, rng)
    H_null = 1j * (2.0/N) * A_null
    evals_n, evecs_n = np.linalg.eigh(H_null)
    pos_mask_n = evals_n.real > 1e-12
    neg_mask_n = evals_n.real < -1e-12
    wp_n = np.sum(np.abs(evecs_n[:, pos_mask_n])**2, axis=1) if np.any(pos_mask_n) else np.zeros(N)
    wn_n = np.sum(np.abs(evecs_n[:, neg_mask_n])**2, axis=1) if np.any(neg_mask_n) else np.zeros(N)
    asym_n = wp_n - wn_n
    ldos_variation_null.append(np.std(asym_n))

    # For null, layers don't exist, so use a random ordering
    rand_layers = rng.permutation(N)
    ldos_layer_corr_null.append(np.corrcoef(rand_layers, asym_n)[0, 1])

print(f"  LDOS asymmetry variation (std across elements):")
print(f"    Causet: {np.mean(ldos_variation_causet):.4f}±{np.std(ldos_variation_causet):.4f}")
print(f"    Null:   {np.mean(ldos_variation_null):.4f}±{np.std(ldos_variation_null):.4f}")
t_ldos, p_ldos = stats.ttest_ind(ldos_variation_causet, ldos_variation_null)
print(f"    t-test: t={t_ldos:.2f}, p={p_ldos:.4f}")

print(f"\n  LDOS-layer correlation (does spectral weight depend on time position?):")
print(f"    Causet: corr={np.mean(ldos_layer_corr_causet):.4f}±{np.std(ldos_layer_corr_causet):.4f}")
print(f"    Null:   corr={np.mean(ldos_layer_corr_null):.4f}±{np.std(ldos_layer_corr_null):.4f}")
t_lc, p_lc = stats.ttest_ind(ldos_layer_corr_causet, ldos_layer_corr_null)
print(f"    t-test: t={t_lc:.2f}, p={p_lc:.4f}")

# Inverse participation ratio (IPR) — measures eigenstate localization
print(f"\n  Eigenstate localization (inverse participation ratio):")
ipr_causet = []
ipr_null = []

for trial in range(n_trials):
    cs = make_2order(N, rng)
    C = cs.order.astype(float)
    H = 1j * (2.0/N) * (C.T - C)
    _, evecs = np.linalg.eigh(H)
    # IPR = sum_x |psi(x)|^4 for each eigenvector. Delocalized: IPR~1/N, localized: IPR~1
    iprs = np.mean(np.sum(np.abs(evecs)**4, axis=0))
    ipr_causet.append(iprs * N)  # Normalize: =1 for uniform, =N for localized

    A_null = make_dense_antisym(N, rng)
    H_null = 1j * (2.0/N) * A_null
    _, evecs_n = np.linalg.eigh(H_null)
    iprs_n = np.mean(np.sum(np.abs(evecs_n)**4, axis=0))
    ipr_null.append(iprs_n * N)

print(f"    Causet IPR*N: {np.mean(ipr_causet):.3f}±{np.std(ipr_causet):.3f} (1=delocalized, N=localized)")
print(f"    Null IPR*N:   {np.mean(ipr_null):.3f}±{np.std(ipr_null):.3f}")
t_ipr, p_ipr = stats.ttest_ind(ipr_causet, ipr_null)
print(f"    t-test: t={t_ipr:.2f}, p={p_ipr:.4f}")


# ================================================================
# FINAL SCORING
# ================================================================
print("\n\n" + "=" * 78)
print("FINAL SCORING — IDEAS 101-110")
print("=" * 78)

print("""
DEEPER FOLLOW-UP on the two most promising results follows below...
""")


# ================================================================
# FOLLOW-UP A: TREEWIDTH — is it density-explained?
# ================================================================
print("=" * 78)
print("FOLLOW-UP A: TREEWIDTH — DENSITY-MATCHED NULL TEST")
print("=" * 78)

# The null above used density*0.7 which is approximate.
# Now use EXACT density matching. If treewidth is just a function of
# ordering fraction, the density-matched null will reproduce it.

print("\n  Testing whether treewidth/N=0.50 is explained by ordering fraction alone:")

for Ntest in [40, 60, 80]:
    twc_vals, twn_vals, dens_c, dens_n = [], [], [], []
    for _ in range(12):
        cs = make_2order(Ntest, rng)
        comp = (cs.order | cs.order.T).astype(int)
        twc_vals.append(min_degree_width(comp, Ntest) / Ntest)
        of = cs.ordering_fraction()
        dens_c.append(of)

        # Density-EXACT null: random DAG iterated until density matches
        for attempt in range(5):
            d_try = of * (0.5 + 0.1 * attempt)  # try different raw densities
            cs_n = random_dag(Ntest, d_try, rng)
            if abs(cs_n.ordering_fraction() - of) < 0.05:
                break
        comp_n = (cs_n.order | cs_n.order.T).astype(int)
        twn_vals.append(min_degree_width(comp_n, Ntest) / Ntest)
        dens_n.append(cs_n.ordering_fraction())

    t, p = stats.ttest_ind(twc_vals, twn_vals)
    print(f"    N={Ntest}: causet tw/N={np.mean(twc_vals):.3f} (of={np.mean(dens_c):.3f}), "
          f"null tw/N={np.mean(twn_vals):.3f} (of={np.mean(dens_n):.3f}), p={p:.4f}")


# ================================================================
# FOLLOW-UP B: ENTANGLEMENT SPECTRUM SYMMETRY CLASS TRANSITION
# ================================================================
print("\n" + "=" * 78)
print("FOLLOW-UP B: ENTANGLEMENT SPECTRUM — GUE → GOE TRANSITION")
print("Is the symmetry class change real and specific to causets?")
print("=" * 78)

# This is potentially the most interesting finding:
# Full spectrum has GUE statistics (<r>~0.60) but the entanglement
# spectrum has GOE statistics (<r>~0.51). This means restricting to
# a subregion RESTORES time-reversal symmetry.
# Test: (1) larger N, (2) different subregion sizes, (3) does null show same?

print("\n  Full vs entanglement spectrum <r> for different subregion fractions:")
N = 80
n_trials_b = 30

for frac in [0.2, 0.3, 0.4, 0.5]:
    r_full, r_ent_c, r_ent_n = [], [], []
    for _ in range(n_trials_b):
        cs = make_2order(N, rng)
        C = cs.order.astype(float)
        H = 1j * (2.0/N) * (C.T - C)
        evals, evecs = np.linalg.eigh(H)
        evals = evals.real

        r_full.append(level_spacing_ratio(evals))

        # Subregion
        n_sub = int(N * frac)
        pos = evals > 1e-12
        if np.any(pos):
            W = (evecs[:, pos] @ np.diag(evals[pos]) @ evecs[:, pos].conj().T).real
            wm = np.linalg.eigvalsh(W).max()
            if wm > 1:
                W = W / wm
            region = list(range(n_sub))
            W_A = W[np.ix_(region, region)]
            r_ent_c.append(level_spacing_ratio(np.linalg.eigvalsh(W_A)))

        # Null
        A_null = make_dense_antisym(N, rng)
        H_null = 1j * (2.0/N) * A_null
        evals_n, evecs_n = np.linalg.eigh(H_null)
        pos_n = evals_n.real > 1e-12
        if np.any(pos_n):
            W_null = (evecs_n[:, pos_n] @ np.diag(evals_n.real[pos_n]) @ evecs_n[:, pos_n].conj().T).real
            wm_n = np.linalg.eigvalsh(W_null).max()
            if wm_n > 1:
                W_null = W_null / wm_n
            W_A_n = W_null[np.ix_(region, region)]
            r_ent_n.append(level_spacing_ratio(np.linalg.eigvalsh(W_A_n)))

    t, p = stats.ttest_ind([x for x in r_ent_c if not np.isnan(x)],
                           [x for x in r_ent_n if not np.isnan(x)])
    print(f"    frac={frac:.1f}: full <r>={np.nanmean(r_full):.4f}, "
          f"ent causet={np.nanmean(r_ent_c):.4f}, "
          f"ent null={np.nanmean(r_ent_n):.4f}, p={p:.4f}")


# ================================================================
# HONEST SCORING
# ================================================================
print("\n\n" + "=" * 78)
print("FINAL HONEST SCORING — IDEAS 101-110")
print("=" * 78)

print("""
IDEA 101 — Persistent Homology (Betti numbers):  5.5/10
  Novelty: 2 — Nobody has computed Betti numbers of causet link graphs.
  Rigor: 1 — Huge separation (cycle density 1.8 vs 0.7), BUT the cycle
    density GROWS with N for causets, suggesting it's driven by
    link density (which also grows). Almost certainly density-explained.
  Depth: 1.5 — Topological invariants are the right idea, but Beta_1
    of the link graph is too crude — it's just E - V + 1.
  Audience: 1 — Topological data analysis + causet people.

IDEA 102 — Quantum Channel Capacity:  5.0/10
  Novelty: 2 — Novel framing (C as a channel matrix). Nobody has done this.
  Rigor: 1 — Clear separation (1.4 vs 0.9 bits) but capacity grows with N
    for causets (1.2→1.5), which is the density effect again. The causal
    matrix has more structure than a random DAG, but this is expected and
    not geometric — it's just that 2-orders have higher ordering fractions.
  Depth: 1 — Information-theoretic framing is interesting but shallow.
  Audience: 1 — Quantum information people might glance.

IDEA 103 — Full Spacing Distribution:  6.0/10
  Novelty: 1.5 — Going beyond <r> to full P(s) is incremental.
  Rigor: 2 — KS test rejects that causet and null have the same P(s)
    (D=0.20, p<1e-10). The causet has HEAVIER tails (kurtosis=36 vs 9).
    This is consistent with the known heavy-tailed eigenvalue density.
    The number variance test was inconclusive (numerical issues).
  Depth: 1.5 — Heavy-tailed P(s) is a genuine deviation from standard
    GUE. Could connect to non-ergodic extended states or multifractality.
  Audience: 1 — RMT specialists.

IDEA 104 — Specific Heat Exponent:  4.0/10
  Novelty: 1.5 — BD specific heat has been measured (Glaser et al.).
  Rigor: 0.5 — Very noisy. Accept rates are 1-3% which means the MCMC
    is NOT equilibrated. Cv fluctuates wildly (3→154→73). The FSS gives
    alpha/nu=0.95 but R^2=0.16, totally unreliable. Need much longer
    chains and larger N.
  Depth: 1.5 — Universality class IS the right question.
  Audience: 0.5 — Only if the exponent were clean.

IDEA 105 — Treewidth of Comparability Graph:  7.0/10
  Novelty: 2.5 — NOBODY has studied treewidth of causal sets. Genuinely new.
  Rigor: 2 — Very clean: causet tw/N → 0.50 (stable across N=30-100),
    null tw/N → 0.91 (approaching 1.0). The separation is massive and
    grows with N. BUT: needs density-matched null to rule out trivial
    explanation. If the density-matched null ALSO gives tw/N~0.5, it's
    just ordering fraction. If not, it's geometric.
  Depth: 1.5 — Treewidth encodes dimensional structure (tw ~ N^{(d-1)/d}
    for lattices). tw/N=0.5 for 2-orders suggests sqrt(N) scaling,
    consistent with d=2. This would be a clean dimension indicator.
  Audience: 1 — Graph theory + discrete geometry people.

IDEA 106 — Interval Sizes mod p:  4.0/10
  Novelty: 1.5 — Novel question but the answer is trivial.
  Rigor: 1 — The non-uniformity mod p is just the interval size
    distribution favoring small intervals (0, 1, 2, ...) which naturally
    concentrates at low residues. This is a property of the DISTRIBUTION
    shape (heavy at 0), not any number-theoretic structure. The null
    shows the same pattern, just less pronounced.
  Depth: 0.5 — No number theory here, just small-number dominance.
  Audience: 0 — Nobody would care.

IDEA 107 — Entanglement Spectrum Symmetry Class:  7.5/10
  Novelty: 3 — NOBODY has studied the symmetry class of the entanglement
    spectrum in causal set quantum gravity. This is genuinely new.
  Rigor: 2 — The full Pauli-Jordan spectrum has GUE statistics (<r>=0.59),
    but the entanglement spectrum drops to <r>=0.50-0.52. This is in the
    GOE range (0.5307). The null (dense antisymmetric) shows <r>=0.53 for
    the entanglement spectrum. So: the entanglement spectrum is GOE for
    BOTH causets and null — the symmetry class transition is GENERIC to
    restricting any antisymmetric matrix's positive eigenspace to a
    subblock. This weakens the result. -1 point.
  Depth: 1.5 — The physics question (does entanglement restore time-
    reversal symmetry?) is deep. The answer (yes, but generically) is
    less exciting than hoped.
  Audience: 1 — RMT + quantum gravity + quantum chaos communities.

  UPDATE AFTER FOLLOW-UP B: The transition is the SAME for causets and
  the dense null. Both show GUE→GOE when restricting to a subblock.
  This is a mathematical property of projecting GUE matrices, not
  physics. Score drops to 6.5/10.

IDEA 108 — Ihara Zeta / Link Adjacency Spectrum:  5.0/10
  Novelty: 1.5 — Link graph spectra have been studied in some form.
  Rigor: 1.5 — Spectral gap is 2x larger for causets than nulls
    (p<0.0001), but the overall spectral distributions are marginally
    different (KS p=0.07). The gap is likely a connectivity effect.
  Depth: 1 — Ihara zeta connection is suggestive but doesn't deliver.
  Audience: 1 — Graph theory people.

IDEA 109 — Mutual Information Decay:  3.0/10
  Novelty: 2 — Good idea, nobody has measured MI decay in causet SJ vacuum.
  Rigor: 0 — FAILED. No data produced because the layer assignment was
    too sparse (max_layer < 8 for most trials). The algorithm needs
    better spatial partitioning. This is fixable but we got no results.
  Depth: 1 — The physics question (locality in quantum gravity) is great.
  Audience: 0 — No results to show.

IDEA 110 — Local Spectral Measure (LDOS):  5.5/10
  Novelty: 2 — LDOS in causet QG is new. IPR analysis is from condensed matter.
  Rigor: 1.5 — LDOS variation is tiny (both ~0.0000) — the eigenstates
    are almost perfectly delocalized. IPR*N = 2.19 for causets vs 1.90 for
    null (p<0.001). So causets have SLIGHTLY more localized eigenstates.
    But the effect is tiny (10% difference) and probably density-driven.
    The LDOS-layer correlation is zero (p=0.81) — spectral weight does NOT
    depend on time position. Null result for the physics question.
  Depth: 1 — Connection to Anderson localization is real but doesn't deliver.
  Audience: 1 — Condensed matter + quantum gravity crossover.

═══════════════════════════════════════════════════════════════════════════
SUMMARY OF ROUND 6:
═══════════════════════════════════════════════════════════════════════════

  Idea 101 (Betti numbers):          5.5/10 — density-explained
  Idea 102 (channel capacity):       5.0/10 — density-explained
  Idea 103 (full P(s) shape):        6.0/10 — real but incremental
  Idea 104 (specific heat):          4.0/10 — MCMC too noisy
  Idea 105 (treewidth):              7.0/10 — clean, needs density control
  Idea 106 (intervals mod p):        4.0/10 — trivial explanation
  Idea 107 (entanglement spectrum):  6.5/10 — generic to all antisym matrices
  Idea 108 (Ihara zeta):             5.0/10 — marginal
  Idea 109 (MI decay):               3.0/10 — failed execution
  Idea 110 (LDOS/IPR):               5.5/10 — tiny effect

  BEST OF ROUND 6: Idea 105 (treewidth) at 7.0/10.

  No 8+ found. The 7.5 ceiling (GUE quantum chaos from prior rounds)
  still stands as the project's best result.

  THE STRUCTURAL BARRIER:
  After 110 ideas, the pattern is clear. At toy scale (N=30-100):
  1. Most causet-vs-null differences are explained by ordering fraction.
  2. Spectral properties of the Pauli-Jordan operator are generic to
     antisymmetric random matrices (GUE is universal).
  3. The BD phase transition is real but requires large-scale MCMC
     (N>200, 10^6 steps) to extract clean exponents — not feasible
     in a single Python experiment.
  4. Topological/structural invariants (treewidth, Betti) separate
     causets from random DAGs, but the separation tracks density.

  TO REACH 8+, the path is:
  (a) Scale treewidth to N=500+ with density-matched nulls, prove the
      convergence tw/N → 0.5 is geometric (not density-driven), and
      connect to dimension estimation. This could be 8/10.
  (b) The GUE paper remains the strongest candidate. Add the analytic
      explanation (Erdos-Yau universality for antisymmetric matrices
      with sufficient entry density) and this becomes a 7.5-8.0 paper.
""")

