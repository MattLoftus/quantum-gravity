"""
Experiment 59: Ideas 111-120 — Round 7 of 8+ Search

FOCUS: Under-explored areas that go beyond spectral statistics of the SJ vacuum.

111. BD ACTION FLUCTUATIONS: Statistical properties of S_BD[C] across the 2-order
     ensemble. Distribution shape, variance scaling with N, higher cumulants.
     Null: random DAG with matched density.

112. MCMC AUTOCORRELATION & DYNAMICAL CRITICAL EXPONENT: Under BD-weighted MCMC,
     how many steps does it take for the ordering fraction to decorrelate?
     Is there critical slowing down near the phase transition (z exponent)?

113. HASSE DIAGRAM GRAPH INVARIANTS: Diameter, girth (shortest cycle in undirected
     link graph), vertex connectivity, chromatic number (greedy approx).
     These are fundamentally different from the transitive closure properties.

114. CROSS-DIMENSIONAL SCALING (d=2,3,4,5): How does ordering fraction, link
     density, BD action density, and longest chain scale with d?
     Is there a dimension-dependent phase structure?

115. INFORMATION-THEORETIC PROPERTIES OF C: Compressibility of the causal matrix
     (gzip ratio), row-wise mutual information, effective rank.
     Null: random upper-triangular matrix with matched density.

116. PERCOLATION ON THE LINK GRAPH: Remove links with probability (1-p).
     At what p does a giant connected component form? Is this threshold
     different for causets vs random DAGs? Relation to BD transition?

117. AUTOMORPHISM GROUP SIZE: How many approximate symmetries does the causal set
     have? Measured by counting elements with identical causal neighborhoods.
     How does this change across the BD transition?

118. CAUSAL MATRIX AS QUANTUM CHANNEL: Treat C as defining a quantum channel
     (Choi matrix). Compute channel capacity bounds. How does capacity
     scale with N and relate to geometry?

119. SPECTRAL PROPERTIES OF THE HASSE LAPLACIAN: The graph Laplacian of the
     link graph (not the full order). Spectral gap, Cheeger constant bound,
     Fiedler vector structure. Compare to random DAG link graph.

120. CAUSAL MATRIX SINGULAR VALUE DECOMPOSITION: The SVD of C (not the symmetric
     i*Delta). Singular value distribution, condition number, effective rank.
     What does the SVD reveal about the causal structure?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected, mcmc_corrected
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from causal_sets.d_orders import DOrder, bd_action_4d_fast
import time
import zlib

rng = np.random.default_rng(42)
np.set_printoptions(precision=4, suppress=True, linewidth=120)


# ============================================================
# UTILITIES
# ============================================================

def make_2order_causet(N, rng):
    to = TwoOrder(N, rng=rng)
    return to.to_causet()

def random_dag_matched(N, target_of, rng):
    """Random upper-triangular boolean matrix with matched ordering fraction."""
    n_pairs = N * (N - 1) // 2
    target_rels = int(target_of * n_pairs)
    cs = FastCausalSet(N)
    # Random upper-triangular with matched density (no transitivity — raw DAG)
    mask = np.zeros(n_pairs, dtype=bool)
    mask[:target_rels] = True
    rng.shuffle(mask)
    cs.order[np.triu_indices(N, k=1)] = mask
    # Transitive closure
    order_int = cs.order.astype(np.int32)
    for _ in range(int(np.ceil(np.log2(N))) + 1):
        new_order = (order_int @ order_int > 0) | cs.order
        if np.array_equal(new_order, cs.order):
            break
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs

def level_spacing_ratio(evals):
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


# ============================================================
# IDEA 111: BD ACTION FLUCTUATIONS
# ============================================================
print("=" * 70)
print("IDEA 111: BD ACTION FLUCTUATIONS ACROSS THE 2-ORDER ENSEMBLE")
print("=" * 70)
print()
print("Study the statistical distribution of S_BD for random 2-orders.")
print("Null: random DAG with matched ordering fraction.")
print()

for N in [20, 30, 50, 70]:
    n_trials = 200 if N <= 50 else 100
    actions_2o = []
    actions_null = []

    for trial in range(n_trials):
        # 2-order
        cs = make_2order_causet(N, rng)
        s = bd_action_2d(cs)
        actions_2o.append(s)
        of = cs.ordering_fraction()

        # Null: random DAG with matched density
        cs_null = random_dag_matched(N, of, rng)
        s_null = bd_action_2d(cs_null)
        actions_null.append(s_null)

    a2 = np.array(actions_2o)
    an = np.array(actions_null)

    print(f"N={N:3d}:")
    print(f"  2-order:  mean={np.mean(a2):8.2f}  std={np.std(a2):7.2f}  "
          f"skew={stats.skew(a2):6.3f}  kurt={stats.kurtosis(a2):6.3f}")
    print(f"  Null DAG: mean={np.mean(an):8.2f}  std={np.std(an):7.2f}  "
          f"skew={stats.skew(an):6.3f}  kurt={stats.kurtosis(an):6.3f}")
    # KS test
    ks_stat, ks_p = stats.ks_2samp(a2, an)
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4e}")
    # Variance scaling: Var(S) ~ N^alpha
    if N == 20:
        var_ref = np.var(a2)
        N_ref = N
    print(f"  Var(S)/N = {np.var(a2)/N:.4f}  Var(S)/N^2 = {np.var(a2)/N**2:.6f}")
    print()

# Variance scaling
print("Variance scaling with N:")
Ns_var = [15, 20, 30, 40, 50, 60]
vars_2o = []
for N in Ns_var:
    acts = []
    n_t = 200 if N <= 40 else 100
    for _ in range(n_t):
        cs = make_2order_causet(N, rng)
        acts.append(bd_action_2d(cs))
    vars_2o.append(np.var(acts))
    print(f"  N={N:3d}: Var(S)={np.var(acts):.2f}")

# Fit power law
log_N = np.log(Ns_var)
log_var = np.log(vars_2o)
slope, intercept = np.polyfit(log_N, log_var, 1)
print(f"\n  Power law fit: Var(S) ~ N^{slope:.2f}")
print(f"  (If alpha=2: extensive fluctuations; if alpha=1: sub-extensive)")
print()


# ============================================================
# IDEA 112: MCMC AUTOCORRELATION & DYNAMICAL CRITICAL EXPONENT
# ============================================================
print("=" * 70)
print("IDEA 112: MCMC AUTOCORRELATION NEAR THE BD TRANSITION")
print("=" * 70)
print()
print("Measure autocorrelation time of ordering fraction under MCMC")
print("at different beta values near beta_c.")
print()

N = 30
eps = 0.12
beta_c_approx = 1.66 / (N * eps**2)  # ~ 3.84

betas_test = [0.0, 1.0, 2.0, 3.0, beta_c_approx, 5.0, 8.0, 15.0]
print(f"N={N}, eps={eps}, beta_c_approx={beta_c_approx:.2f}")
print(f"{'beta':>8s}  {'<of>':>8s}  {'std(of)':>8s}  {'tau_int':>8s}  {'accept':>8s}")
print("-" * 50)

for beta in betas_test:
    # Run MCMC
    n_steps = 20000
    n_therm = 5000
    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    of_trace = []
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
        if step >= n_therm:
            of_trace.append(current_cs.ordering_fraction())

    of_trace = np.array(of_trace)
    mean_of = np.mean(of_trace)
    std_of = np.std(of_trace)
    acc_rate = n_acc / n_steps

    # Integrated autocorrelation time
    of_centered = of_trace - mean_of
    n_lag = min(len(of_centered) // 2, 2000)
    autocorr = np.correlate(of_centered, of_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
    # Integrated: sum until first negative
    tau_int = 0.5
    for lag in range(1, n_lag):
        if autocorr[lag] < 0:
            break
        tau_int += autocorr[lag]

    print(f"{beta:8.2f}  {mean_of:8.4f}  {std_of:8.4f}  {tau_int:8.1f}  {acc_rate:8.3f}")

print()
print("If tau_int diverges near beta_c => critical slowing down (z > 0)")
print()


# ============================================================
# IDEA 113: HASSE DIAGRAM GRAPH INVARIANTS
# ============================================================
print("=" * 70)
print("IDEA 113: HASSE DIAGRAM GRAPH INVARIANTS")
print("=" * 70)
print()
print("Study the link graph (Hasse diagram), NOT the transitive closure.")
print("Metrics: diameter, avg degree, clustering coeff, connected components.")
print()

def hasse_graph_stats(cs):
    """Compute graph statistics of the Hasse diagram (undirected link graph)."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(np.int32)
    N = cs.n
    degree = np.sum(adj, axis=1)

    # Connected components via BFS
    visited = np.zeros(N, dtype=bool)
    components = []
    for start in range(N):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nbr in range(N):
                if adj[node, nbr] and not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        components.append(comp)

    largest_comp = max(len(c) for c in components)

    # Diameter of largest component (BFS from several nodes)
    largest_comp_nodes = max(components, key=len)
    if len(largest_comp_nodes) <= 1:
        diameter = 0
    else:
        diameter = 0
        # Sample a few start nodes
        sample_starts = largest_comp_nodes[:min(10, len(largest_comp_nodes))]
        for start in sample_starts:
            dist = np.full(N, -1)
            dist[start] = 0
            queue = [start]
            while queue:
                node = queue.pop(0)
                for nbr in range(N):
                    if adj[node, nbr] and dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            max_d = np.max(dist[dist >= 0])
            if max_d > diameter:
                diameter = max_d

    # Clustering coefficient (triangle ratio)
    if np.sum(degree * (degree - 1)) == 0:
        clustering = 0.0
    else:
        adj_f = adj.astype(float)
        triangles = np.trace(adj_f @ adj_f @ adj_f) / 6.0
        triplets = np.sum(degree * (degree - 1)) / 2.0
        clustering = 3 * triangles / triplets if triplets > 0 else 0.0

    # Girth (shortest cycle) — check triangles first, then 4-cycles
    has_triangle = triangles > 0
    girth = 3 if has_triangle else 0
    if not has_triangle:
        # Check 4-cycles: (A^2)_{ij} > 0 and adj[i,j] = 0 for some i<j
        adj2 = adj_f @ adj_f
        for i in range(N):
            for j in range(i+1, N):
                if adj2[i,j] >= 2 and adj[i,j] == 0:
                    girth = 4
                    break
            if girth == 4:
                break
        if girth == 0:
            girth = float('inf')  # Tree-like

    return {
        'avg_degree': np.mean(degree),
        'max_degree': np.max(degree),
        'n_components': len(components),
        'largest_comp_frac': largest_comp / N,
        'diameter': diameter,
        'clustering': clustering,
        'girth': girth,
        'n_links': int(np.sum(links)),
    }

for N in [30, 50, 70]:
    n_trials = 60 if N <= 50 else 30
    stats_2o = {k: [] for k in ['avg_degree', 'diameter', 'clustering',
                                  'girth', 'largest_comp_frac', 'n_links']}
    stats_null = {k: [] for k in stats_2o}

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        s = hasse_graph_stats(cs)
        for k in stats_2o:
            stats_2o[k].append(s[k])

        # Null: random DAG
        of = cs.ordering_fraction()
        cs_null = random_dag_matched(N, of, rng)
        s_null = hasse_graph_stats(cs_null)
        for k in stats_null:
            stats_null[k].append(s_null[k])

    print(f"N={N}:")
    for k in stats_2o:
        m2 = np.mean(stats_2o[k])
        mn = np.mean(stats_null[k])
        s2 = np.std(stats_2o[k])
        sn = np.std(stats_null[k])
        # Effect size
        pooled_std = np.sqrt((s2**2 + sn**2) / 2) if (s2 + sn) > 0 else 1
        d_cohen = abs(m2 - mn) / pooled_std if pooled_std > 0 else 0
        sig = "*" if d_cohen > 0.8 else ("~" if d_cohen > 0.5 else "")
        print(f"  {k:20s}  2-order: {m2:8.3f}({s2:.3f})  null: {mn:8.3f}({sn:.3f})  d={d_cohen:.2f} {sig}")
    print()


# ============================================================
# IDEA 114: CROSS-DIMENSIONAL SCALING (d=2,3,4,5)
# ============================================================
print("=" * 70)
print("IDEA 114: CROSS-DIMENSIONAL SCALING")
print("=" * 70)
print()
print("How do causet properties scale with dimension d for d-orders?")
print()

N = 30  # Keep small for d=5
n_trials = 50

print(f"N={N}, {n_trials} trials per dimension")
print(f"{'d':>3s}  {'ord_frac':>10s}  {'link_dens':>10s}  {'height':>8s}  "
      f"{'BD_2d':>8s}  {'longest_ac':>10s}")
print("-" * 60)

for d in [2, 3, 4, 5]:
    of_vals, ld_vals, h_vals, bd_vals, ac_vals = [], [], [], [], []

    for trial in range(n_trials):
        do = DOrder(d, N, rng=rng)
        cs = do.to_causet_fast()
        of = cs.ordering_fraction()
        of_vals.append(of)

        links = cs.link_matrix()
        n_links = int(np.sum(links))
        ld_vals.append(n_links / (N * (N - 1) / 2))

        h = cs.longest_chain()
        h_vals.append(h)

        bd = bd_action_2d(cs)
        bd_vals.append(bd)

        # Greedy antichain length
        order = cs.order
        comparable = order | order.T
        antichain = []
        for i in range(N):
            ok = True
            for j in antichain:
                if comparable[i, j]:
                    ok = False
                    break
            if ok:
                antichain.append(i)
        ac_vals.append(len(antichain))

    print(f"{d:3d}  {np.mean(of_vals):10.4f}  {np.mean(ld_vals):10.4f}  "
          f"{np.mean(h_vals):8.2f}  {np.mean(bd_vals):8.2f}  {np.mean(ac_vals):10.2f}")

print()
print("Theoretical: ord_frac for d-order ~ 1/d! for large d (fraction of")
print("pairs ordered by ALL d coordinates simultaneously).")
print("For d=2: ~1/3, d=3: ~1/4, d=4: ~1/5, ... (actually ~d!/(2d)! type)")
print()

# Null: random DAGs with matched density for each d
print("Null comparison (link density):")
print(f"{'d':>3s}  {'2-order links':>14s}  {'null links':>12s}  {'ratio':>8s}")
print("-" * 45)
for d in [2, 3, 4, 5]:
    ld_2o, ld_null = [], []
    for trial in range(n_trials):
        do = DOrder(d, N, rng=rng)
        cs = do.to_causet_fast()
        links = cs.link_matrix()
        n_links = int(np.sum(links))
        ld_2o.append(n_links)
        of = cs.ordering_fraction()

        cs_null = random_dag_matched(N, of, rng)
        links_null = cs_null.link_matrix()
        ld_null.append(int(np.sum(links_null)))

    print(f"{d:3d}  {np.mean(ld_2o):14.1f}  {np.mean(ld_null):12.1f}  "
          f"{np.mean(ld_2o)/(np.mean(ld_null)+1e-10):8.3f}")
print()


# ============================================================
# IDEA 115: INFORMATION-THEORETIC PROPERTIES OF C
# ============================================================
print("=" * 70)
print("IDEA 115: INFORMATION-THEORETIC PROPERTIES OF CAUSAL MATRIX C")
print("=" * 70)
print()
print("Compressibility, effective rank, row mutual information.")
print()

for N in [30, 50, 70]:
    n_trials = 60 if N <= 50 else 30

    compress_2o, compress_null = [], []
    erank_2o, erank_null = [], []
    row_mi_2o, row_mi_null = [], []

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        C = cs.order.astype(np.uint8)
        of = cs.ordering_fraction()

        # Compressibility: gzip ratio
        raw = C.tobytes()
        compressed = zlib.compress(raw, level=9)
        compress_2o.append(len(compressed) / len(raw))

        # Effective rank (from SVD)
        C_float = C.astype(float)
        svd_vals = np.linalg.svd(C_float, compute_uv=False)
        svd_vals = svd_vals[svd_vals > 1e-10]
        p_sv = svd_vals / np.sum(svd_vals)
        eff_rank = np.exp(-np.sum(p_sv * np.log(p_sv + 1e-15)))
        erank_2o.append(eff_rank)

        # Row mutual information: avg MI between pairs of rows
        # (using binary entries as probability distributions)
        mi_samples = []
        for _ in range(20):
            i, j = rng.choice(N, 2, replace=False)
            row_i = C[i, :]
            row_j = C[j, :]
            # Joint distribution of (row_i[k], row_j[k]) over k
            joint = np.zeros((2, 2))
            for a in range(2):
                for b in range(2):
                    joint[a, b] = np.mean((row_i == a) & (row_j == b))
            p_i = np.sum(joint, axis=1)
            p_j = np.sum(joint, axis=0)
            mi = 0.0
            for a in range(2):
                for b in range(2):
                    if joint[a, b] > 0 and p_i[a] > 0 and p_j[b] > 0:
                        mi += joint[a, b] * np.log(joint[a, b] / (p_i[a] * p_j[b]))
            mi_samples.append(mi)
        row_mi_2o.append(np.mean(mi_samples))

        # Null: random upper-triangular with matched density
        cs_null = FastCausalSet(N)
        n_pairs = N * (N - 1) // 2
        mask = rng.random(n_pairs) < of
        cs_null.order[np.triu_indices(N, k=1)] = mask
        C_null = cs_null.order.astype(np.uint8)

        raw_null = C_null.tobytes()
        compressed_null = zlib.compress(raw_null, level=9)
        compress_null.append(len(compressed_null) / len(raw_null))

        C_null_f = C_null.astype(float)
        svd_null = np.linalg.svd(C_null_f, compute_uv=False)
        svd_null = svd_null[svd_null > 1e-10]
        p_sv_null = svd_null / np.sum(svd_null)
        erank_null.append(np.exp(-np.sum(p_sv_null * np.log(p_sv_null + 1e-15))))

        mi_null_samples = []
        for _ in range(20):
            i, j = rng.choice(N, 2, replace=False)
            row_i = C_null[i, :]
            row_j = C_null[j, :]
            joint = np.zeros((2, 2))
            for a in range(2):
                for b in range(2):
                    joint[a, b] = np.mean((row_i == a) & (row_j == b))
            p_i = np.sum(joint, axis=1)
            p_j = np.sum(joint, axis=0)
            mi = 0.0
            for a in range(2):
                for b in range(2):
                    if joint[a, b] > 0 and p_i[a] > 0 and p_j[b] > 0:
                        mi += joint[a, b] * np.log(joint[a, b] / (p_i[a] * p_j[b]))
            mi_null_samples.append(mi)
        row_mi_null.append(np.mean(mi_null_samples))

    print(f"N={N}:")
    print(f"  Compress ratio:  2-order={np.mean(compress_2o):.4f}({np.std(compress_2o):.4f})  "
          f"null={np.mean(compress_null):.4f}({np.std(compress_null):.4f})  "
          f"diff={np.mean(compress_2o)-np.mean(compress_null):.4f}")
    print(f"  Effective rank:  2-order={np.mean(erank_2o):.2f}({np.std(erank_2o):.2f})  "
          f"null={np.mean(erank_null):.2f}({np.std(erank_null):.2f})")
    print(f"  Row MI:          2-order={np.mean(row_mi_2o):.4f}({np.std(row_mi_2o):.4f})  "
          f"null={np.mean(row_mi_null):.4f}({np.std(row_mi_null):.4f})")
    print()


# ============================================================
# IDEA 116: PERCOLATION ON THE LINK GRAPH
# ============================================================
print("=" * 70)
print("IDEA 116: PERCOLATION ON THE LINK GRAPH")
print("=" * 70)
print()
print("Remove links with probability (1-p). Find the percolation threshold")
print("where a giant component appears.")
print()

def percolation_giant_comp(cs, p, rng):
    """Keep each link with probability p. Return fraction in largest component."""
    links = cs.link_matrix()
    N = cs.n
    # Keep each link with probability p
    keep = rng.random(links.shape) < p
    adj = ((links & keep) | (links & keep).T).astype(np.int32)

    # Find largest component
    visited = np.zeros(N, dtype=bool)
    largest = 0
    for start in range(N):
        if visited[start]:
            continue
        size = 0
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            size += 1
            for nbr in range(N):
                if adj[node, nbr] and not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        if size > largest:
            largest = size
    return largest / N

N = 50
n_trials = 40
p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"N={N}, {n_trials} trials")
print(f"{'p':>6s}  {'2-order GC':>12s}  {'null GC':>10s}  {'diff':>8s}")
print("-" * 42)

for p in p_values:
    gc_2o, gc_null = [], []
    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        gc_2o.append(percolation_giant_comp(cs, p, rng))

        of = cs.ordering_fraction()
        cs_null = random_dag_matched(N, of, rng)
        gc_null.append(percolation_giant_comp(cs_null, p, rng))

    m2 = np.mean(gc_2o)
    mn = np.mean(gc_null)
    print(f"{p:6.2f}  {m2:12.4f}  {mn:10.4f}  {m2-mn:8.4f}")

print()


# ============================================================
# IDEA 117: AUTOMORPHISM — CAUSAL NEIGHBORHOOD EQUIVALENCE
# ============================================================
print("=" * 70)
print("IDEA 117: APPROXIMATE SYMMETRIES — CAUSAL NEIGHBORHOOD CLASSES")
print("=" * 70)
print()
print("Count equivalence classes of elements with identical past/future sets.")
print("More classes = less symmetry = more 'structured'.")
print()

def causal_neighborhood_classes(cs):
    """Count elements with identical causal neighborhoods (past union future)."""
    N = cs.n
    # Represent each element by its row and column in the order matrix
    signatures = []
    for i in range(N):
        past = tuple(np.where(cs.order[:, i])[0])
        future = tuple(np.where(cs.order[i, :])[0])
        signatures.append((past, future))

    # Count unique signatures
    unique = len(set(signatures))
    return unique, N

for N in [30, 50, 70]:
    n_trials = 60 if N <= 50 else 30
    classes_2o, classes_null = [], []

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        n_classes, _ = causal_neighborhood_classes(cs)
        classes_2o.append(n_classes / N)  # Fraction of N that are unique

        of = cs.ordering_fraction()
        cs_null = random_dag_matched(N, of, rng)
        n_classes_null, _ = causal_neighborhood_classes(cs_null)
        classes_null.append(n_classes_null / N)

    print(f"N={N}: 2-order unique_frac={np.mean(classes_2o):.4f}({np.std(classes_2o):.4f})  "
          f"null={np.mean(classes_null):.4f}({np.std(classes_null):.4f})")

print()
print("If 2-order has FEWER unique classes => more symmetry from geometric embedding.")
print("If same as null => symmetry is just a density effect.")
print()


# ============================================================
# IDEA 118: CAUSAL MATRIX AS QUANTUM CHANNEL
# ============================================================
print("=" * 70)
print("IDEA 118: CAUSAL MATRIX AS QUANTUM CHANNEL")
print("=" * 70)
print()
print("Interpret C as a (classical) channel matrix. Compute:")
print("  - Classical capacity (mutual information maximized over input dist)")
print("  - Quantum capacity lower bound (coherent information)")
print()

def channel_capacity_lower_bound(C_float):
    """
    Treat the normalized causal matrix as a stochastic channel.
    C_normalized[i,:] = probability of j given i.
    Compute mutual information with uniform input.
    """
    N = C_float.shape[0]
    # Normalize rows to make a stochastic matrix
    row_sums = np.sum(C_float, axis=1)
    # Handle zero rows
    row_sums[row_sums == 0] = 1.0
    P = C_float / row_sums[:, None]  # P[i,j] = P(j|i)

    # Uniform input
    p_in = np.ones(N) / N
    p_out = p_in @ P  # marginal output distribution
    p_out = np.clip(p_out, 1e-15, None)

    # MI = sum_i p_in[i] sum_j P[i,j] log(P[i,j] / p_out[j])
    mi = 0.0
    for i in range(N):
        for j in range(N):
            if P[i, j] > 1e-15:
                mi += p_in[i] * P[i, j] * np.log(P[i, j] / p_out[j])
    return mi

for N in [20, 30, 50]:
    n_trials = 60 if N <= 30 else 30
    cap_2o, cap_null = [], []
    coh_2o, coh_null = [], []

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        C = cs.order.astype(float)
        cap = channel_capacity_lower_bound(C)
        cap_2o.append(cap / np.log(N))  # Normalize by log(N)

        # Coherent information: use von Neumann entropy
        # S(rho_out) - S(rho_env) where rho = C*C^T / Tr(C*C^T)
        rho = C @ C.T
        tr = np.trace(rho)
        if tr > 0:
            rho = rho / tr
            eigs_rho = np.linalg.eigvalsh(rho)
            eigs_rho = eigs_rho[eigs_rho > 1e-15]
            vn_entropy = -np.sum(eigs_rho * np.log(eigs_rho))
            coh_2o.append(vn_entropy / np.log(N))
        else:
            coh_2o.append(0.0)

        # Null
        of = cs.ordering_fraction()
        cs_null = FastCausalSet(N)
        n_pairs = N * (N - 1) // 2
        mask_null = rng.random(n_pairs) < of
        cs_null.order[np.triu_indices(N, k=1)] = mask_null
        C_null = cs_null.order.astype(float)

        cap_null.append(channel_capacity_lower_bound(C_null) / np.log(N))

        rho_null = C_null @ C_null.T
        tr_null = np.trace(rho_null)
        if tr_null > 0:
            rho_null = rho_null / tr_null
            eigs_rho_null = np.linalg.eigvalsh(rho_null)
            eigs_rho_null = eigs_rho_null[eigs_rho_null > 1e-15]
            coh_null.append(-np.sum(eigs_rho_null * np.log(eigs_rho_null)) / np.log(N))
        else:
            coh_null.append(0.0)

    print(f"N={N}:")
    print(f"  Channel cap/logN:  2-order={np.mean(cap_2o):.4f}({np.std(cap_2o):.4f})  "
          f"null={np.mean(cap_null):.4f}({np.std(cap_null):.4f})")
    print(f"  VN entropy/logN:   2-order={np.mean(coh_2o):.4f}({np.std(coh_2o):.4f})  "
          f"null={np.mean(coh_null):.4f}({np.std(coh_null):.4f})")
    # KS test
    ks_cap, p_cap = stats.ks_2samp(cap_2o, cap_null)
    print(f"  KS test (cap):     stat={ks_cap:.4f}, p={p_cap:.4e}")
    print()


# ============================================================
# IDEA 119: HASSE LAPLACIAN SPECTRAL PROPERTIES
# ============================================================
print("=" * 70)
print("IDEA 119: HASSE (LINK GRAPH) LAPLACIAN SPECTRAL PROPERTIES")
print("=" * 70)
print()
print("Graph Laplacian of the undirected link graph.")
print("Focus: spectral gap, algebraic connectivity (Fiedler value).")
print()

def hasse_laplacian_spectrum(cs):
    """Compute eigenvalues of the graph Laplacian of the link graph."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    evals = np.sort(np.linalg.eigvalsh(L))
    return evals

for N in [30, 50, 70]:
    n_trials = 60 if N <= 50 else 30

    fiedler_2o, fiedler_null = [], []
    gap_2o, gap_null = [], []
    max_eval_2o, max_eval_null = [], []

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        evals = hasse_laplacian_spectrum(cs)
        # Fiedler value = second smallest eigenvalue
        fiedler_2o.append(evals[1] if len(evals) > 1 else 0)
        # Spectral gap
        nz = evals[evals > 1e-10]
        gap_2o.append(nz[0] if len(nz) > 0 else 0)
        max_eval_2o.append(evals[-1])

        of = cs.ordering_fraction()
        cs_null = random_dag_matched(N, of, rng)
        evals_null = hasse_laplacian_spectrum(cs_null)
        fiedler_null.append(evals_null[1] if len(evals_null) > 1 else 0)
        nz_null = evals_null[evals_null > 1e-10]
        gap_null.append(nz_null[0] if len(nz_null) > 0 else 0)
        max_eval_null.append(evals_null[-1])

    print(f"N={N}:")
    print(f"  Fiedler (algeb conn):  2-order={np.mean(fiedler_2o):.4f}({np.std(fiedler_2o):.4f})  "
          f"null={np.mean(fiedler_null):.4f}({np.std(fiedler_null):.4f})")
    ks_f, p_f = stats.ks_2samp(fiedler_2o, fiedler_null)
    print(f"    KS test: stat={ks_f:.4f}, p={p_f:.4e}")
    print(f"  Spectral gap:          2-order={np.mean(gap_2o):.4f}({np.std(gap_2o):.4f})  "
          f"null={np.mean(gap_null):.4f}({np.std(gap_null):.4f})")
    print(f"  Max eigenvalue:        2-order={np.mean(max_eval_2o):.2f}({np.std(max_eval_2o):.2f})  "
          f"null={np.mean(max_eval_null):.2f}({np.std(max_eval_null):.2f})")
    print()


# ============================================================
# IDEA 120: SVD OF THE CAUSAL MATRIX C
# ============================================================
print("=" * 70)
print("IDEA 120: SINGULAR VALUE DECOMPOSITION OF CAUSAL MATRIX C")
print("=" * 70)
print()
print("SVD of C (upper-triangular boolean). Singular value distribution,")
print("condition number, and spectral entropy.")
print()

for N in [30, 50, 70]:
    n_trials = 60 if N <= 50 else 30

    cond_2o, cond_null = [], []
    spectral_ent_2o, spectral_ent_null = [], []
    top_sv_frac_2o, top_sv_frac_null = [], []
    marchenko_2o, marchenko_null = [], []

    for trial in range(n_trials):
        cs = make_2order_causet(N, rng)
        C = cs.order.astype(float)
        sv = np.linalg.svd(C, compute_uv=False)
        sv = sv[sv > 1e-10]

        if len(sv) > 1:
            cond_2o.append(sv[0] / sv[-1])
        else:
            cond_2o.append(float('inf'))

        # Spectral entropy of normalized singular values
        p_sv = sv / np.sum(sv)
        spectral_ent_2o.append(-np.sum(p_sv * np.log(p_sv + 1e-15)) / np.log(N))

        # Fraction of total in top singular value
        top_sv_frac_2o.append(sv[0]**2 / np.sum(sv**2))

        # Marchenko-Pastur-like: compare sv^2 distribution
        # For a random 0/1 matrix with density p, largest sv ~ N*p*(1+sqrt(1))
        of = cs.ordering_fraction()
        expected_largest = N * of  # rough scaling
        marchenko_2o.append(sv[0] / expected_largest if expected_largest > 0 else 0)

        # Null
        cs_null = FastCausalSet(N)
        n_pairs = N * (N - 1) // 2
        mask_null = rng.random(n_pairs) < of
        cs_null.order[np.triu_indices(N, k=1)] = mask_null
        C_null = cs_null.order.astype(float)
        sv_null = np.linalg.svd(C_null, compute_uv=False)
        sv_null = sv_null[sv_null > 1e-10]

        if len(sv_null) > 1:
            cond_null.append(sv_null[0] / sv_null[-1])
        else:
            cond_null.append(float('inf'))

        p_sv_null = sv_null / np.sum(sv_null)
        spectral_ent_null.append(-np.sum(p_sv_null * np.log(p_sv_null + 1e-15)) / np.log(N))
        top_sv_frac_null.append(sv_null[0]**2 / np.sum(sv_null**2))
        marchenko_null.append(sv_null[0] / expected_largest if expected_largest > 0 else 0)

    print(f"N={N}:")
    # Filter inf for condition number stats
    cond_2o_filt = [c for c in cond_2o if c < 1e10]
    cond_null_filt = [c for c in cond_null if c < 1e10]
    if cond_2o_filt and cond_null_filt:
        print(f"  Condition number:    2-order={np.mean(cond_2o_filt):.1f}({np.std(cond_2o_filt):.1f})  "
              f"null={np.mean(cond_null_filt):.1f}({np.std(cond_null_filt):.1f})")
    print(f"  Spectral entropy/logN: 2-order={np.mean(spectral_ent_2o):.4f}({np.std(spectral_ent_2o):.4f})  "
          f"null={np.mean(spectral_ent_null):.4f}({np.std(spectral_ent_null):.4f})")
    print(f"  Top SV fraction:     2-order={np.mean(top_sv_frac_2o):.4f}({np.std(top_sv_frac_2o):.4f})  "
          f"null={np.mean(top_sv_frac_null):.4f}({np.std(top_sv_frac_null):.4f})")
    print(f"  sigma_1/Np:          2-order={np.mean(marchenko_2o):.4f}({np.std(marchenko_2o):.4f})  "
          f"null={np.mean(marchenko_null):.4f}({np.std(marchenko_null):.4f})")
    ks_se, p_se = stats.ks_2samp(spectral_ent_2o, spectral_ent_null)
    ks_top, p_top = stats.ks_2samp(top_sv_frac_2o, top_sv_frac_null)
    print(f"  KS(spectral ent): stat={ks_se:.4f}, p={p_se:.4e}")
    print(f"  KS(top SV frac):  stat={ks_top:.4f}, p={p_top:.4e}")
    print()


# ============================================================
# SUMMARY & SCORING
# ============================================================
print("=" * 70)
print("SUMMARY & HONEST SCORING")
print("=" * 70)
print("""
Scoring each idea 1-10 on: novelty, rigor, audience size, null-resistance.

111. BD ACTION FLUCTUATIONS
     - Does the distribution shape or variance scaling differ from null?
     - Score based on whether the power-law exponent or distribution shape
       is genuinely novel and not explained by density matching.

112. MCMC AUTOCORRELATION
     - If tau_int diverges near beta_c => dynamical critical exponent z.
     - This would be genuinely new: no one has measured z for the BD transition.

113. HASSE DIAGRAM INVARIANTS
     - If any metric (diameter, clustering, girth) differs significantly
       from matched null => the link structure carries geometric info.

114. CROSS-DIMENSIONAL SCALING
     - Known theoretically that d-orders become sparser with d.
     - Novel if we find a non-trivial d-dependent phase structure.

115. INFORMATION-THEORETIC PROPERTIES
     - Compressibility and effective rank are novel angles on C.
     - Interesting if 2-orders are MORE compressible (= more structured) than null.

116. PERCOLATION ON LINK GRAPH
     - Novel combination: percolation theory + causal sets.
     - Interesting if threshold differs from random DAG.

117. SYMMETRIES / NEIGHBORHOOD CLASSES
     - Direct measurement of how geometric embedding constrains structure.

118. QUANTUM CHANNEL CAPACITY
     - Speculative but creative. Novel angle.

119. HASSE LAPLACIAN SPECTRUM
     - Fiedler value / algebraic connectivity is well-studied in graph theory.
     - Novel application to causal sets.

120. SVD OF C
     - Standard matrix analysis but novel application.
     - Interesting if singular value structure differs meaningfully from null.
""")
