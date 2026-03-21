"""
Experiment 76 Round 19: CROSS-APPROACH COMPARISONS (Ideas 281-290)

Extending Paper E (CDT comparison) with a systematic comparison of
observables across four structure types: sprinkled causets, CDT configurations,
regular lattices (with causal structure), and random DAGs.

Ideas:
281. Interval entropy on CDT configurations - does it match causets?
282. Hasse Laplacian Fiedler value on CDT vs causets
283. Treewidth on CDT vs causets
284. Antichain scaling on CDT vs causets
285. Compressibility (SVD) on CDT vs causets
286. Link fraction on CDT across the CDT phase transition (varying lambda2)
287. BD action computed on CDT configurations - how does it score CDT?
288. SJ vacuum on REGULAR LATTICES (square lattice with causal structure) - is c=1?
289. SJ vacuum on SPRINKLED causets in CURVED spacetime (de Sitter) - does S depend on curvature?
290. Compare ALL observables (systematic table: causet vs CDT vs lattice vs random DAG for 10 observables)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy.optimize import curve_fit
import zlib
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.sj_vacuum import sj_wightman_function, entanglement_entropy
from cdt.triangulation import mcmc_cdt

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def cdt_to_causet(volume_profile):
    """Convert CDT volume profile to a causal set.
    Elements at time t precede elements at time t' > t.
    Within the same slice, elements are spacelike-separated."""
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


def cdt_to_causet_sparse(volume_profile, link_range=2):
    """Convert CDT to causet with only nearby-time relations (more physical).
    Only elements within link_range time slices are related."""
    T = len(volume_profile)
    N = int(np.sum(volume_profile))
    cs = FastCausalSet(N)
    offsets = np.zeros(T + 1, dtype=int)
    for t in range(T):
        offsets[t + 1] = offsets[t] + int(volume_profile[t])
    for t1 in range(T):
        for t2 in range(t1 + 1, min(t1 + link_range + 1, T)):
            for i1 in range(int(volume_profile[t1])):
                for i2 in range(int(volume_profile[t2])):
                    cs.order[offsets[t1] + i1, offsets[t2] + i2] = True
    # Transitive closure
    order_int = cs.order.astype(np.int32)
    changed = True
    while changed:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs


def make_regular_lattice_causet(Nt, Nx):
    """Square lattice with causal structure.
    Vertices at (t, x) for t=0..Nt-1, x=0..Nx-1.
    Causal relation: (t1,x1) < (t2,x2) iff t1 < t2 and |x1-x2| <= (t2-t1).
    This gives a discrete light-cone structure on a 2D lattice."""
    N = Nt * Nx
    cs = FastCausalSet(N)
    for t1 in range(Nt):
        for x1 in range(Nx):
            i = t1 * Nx + x1
            for t2 in range(t1 + 1, Nt):
                dt = t2 - t1
                for x2 in range(Nx):
                    dx = min(abs(x2 - x1), Nx - abs(x2 - x1))  # periodic
                    if dx <= dt:
                        j = t2 * Nx + x2
                        cs.order[i, j] = True
    return cs


def make_random_dag(N, density, rng_local):
    """Random DAG with transitive closure."""
    cs = FastCausalSet(N)
    mask = rng_local.random((N, N)) < density
    cs.order = np.triu(mask, k=1)
    order_int = cs.order.astype(np.int32)
    changed = True
    iters = 0
    while changed and iters < 20:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
        iters += 1
    return cs


def sprinkle_de_sitter_2d(N, H=1.0, rng_local=None):
    """Sprinkle into 2D de Sitter spacetime.

    Use conformal coordinates: ds^2 = (1/H^2 eta^2)(-d eta^2 + dx^2)
    where eta in (-inf, 0). We sprinkle uniformly in (eta, x) in a diamond-like
    region and use the FLAT causal structure (light cones are straight in conformal coords).

    The physical effect is that the sprinkling density is non-uniform in proper time,
    and the causal structure is conformally flat but the number of elements in
    intervals depends on the local volume element sqrt(-g) = 1/(H^2 eta^2).

    For the SJ vacuum, the key point is that curvature enters through the
    distribution of interval sizes and the causal matrix structure.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    # Sprinkle with density proportional to sqrt(-g) = 1/(H^2 eta^2)
    # in conformal time eta in [-2, -0.1] (avoiding eta=0 boundary)
    eta_min, eta_max = -2.0, -0.1
    x_extent = 1.0

    coords_list = []
    while len(coords_list) < N:
        batch = N * 10
        eta = rng_local.uniform(eta_min, eta_max, batch)
        x = rng_local.uniform(-x_extent, x_extent, batch)
        # Accept with probability proportional to sqrt(-g) = 1/(H*eta)^2
        # Normalize: max is at eta = eta_max (closest to 0)
        weight = 1.0 / (H * eta)**2
        max_weight = 1.0 / (H * eta_max)**2
        accept = rng_local.random(batch) < weight / max_weight
        # Also require within a causal diamond
        eta_center = (eta_min + eta_max) / 2
        diamond_ok = np.abs(x) <= np.minimum(eta - eta_min, eta_max - eta)
        both = accept & diamond_ok
        for i in range(batch):
            if both[i] and len(coords_list) < N:
                coords_list.append((eta[i], x[i]))

    coords = np.array(coords_list[:N])
    # Sort by conformal time
    coords = coords[np.argsort(coords[:, 0])]

    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx  # null/timelike in conformal coords

    return cs, coords


# ---------- Observable functions ----------

def ordering_fraction(cs):
    return cs.ordering_fraction()

def link_fraction(cs):
    """Fraction of relations that are links."""
    n_rel = cs.num_relations()
    if n_rel == 0:
        return 0.0
    links = cs.link_matrix()
    n_links = int(np.sum(links))
    return n_links / n_rel

def longest_chain(cs):
    return cs.longest_chain()

def longest_antichain_size(cs):
    """Longest antichain via bipartite matching (Dilworth's theorem)."""
    N = cs.n
    if N == 0:
        return 0
    order = cs.order
    matched_right = [-1] * N

    def dfs(u, visited):
        for v in range(N):
            if order[u, v] and not visited[v]:
                visited[v] = True
                if matched_right[v] == -1 or dfs(matched_right[v], visited):
                    matched_right[v] = u
                    return True
        return False

    matching = 0
    for u in range(N):
        visited = [False] * N
        if dfs(u, visited):
            matching += 1
    return N - matching

def fiedler_value(cs):
    """Second-smallest eigenvalue of the link graph Laplacian."""
    links = cs.link_matrix()
    adj = links | links.T
    degree = np.sum(adj, axis=1).astype(float)
    mask = degree > 0
    adj_sub = adj[np.ix_(mask, mask)].astype(float)
    degree_sub = degree[mask]
    n = adj_sub.shape[0]
    if n < 3:
        return 0.0
    L = np.diag(degree_sub) - adj_sub
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    return float(evals[1]) if len(evals) > 1 else 0.0

def compressibility(cs):
    """Gzip compression ratio of causal matrix."""
    flat = cs.order.astype(np.uint8).tobytes()
    compressed = zlib.compress(flat, level=9)
    return len(compressed) / len(flat)

def svd_compressibility(cs):
    """SVD-based compressibility: fraction of singular values needed for 90% of norm."""
    M = cs.order.astype(float)
    if M.shape[0] < 3:
        return 1.0
    sv = np.linalg.svd(M, compute_uv=False)
    total = np.sum(sv)
    if total < 1e-14:
        return 1.0
    cumsum = np.cumsum(sv) / total
    k90 = np.searchsorted(cumsum, 0.9) + 1
    return k90 / len(sv)

def interval_entropy(cs, max_k=15):
    """Shannon entropy of the interval size distribution."""
    counts = count_intervals_by_size(cs, max_size=min(cs.n - 2, max_k))
    vals = np.array([v for v in counts.values() if v > 0], dtype=float)
    if len(vals) == 0 or np.sum(vals) == 0:
        return 0.0
    p = vals / np.sum(vals)
    return -np.sum(p * np.log(p + 1e-300))

def treewidth_approx(cs):
    """Approximate treewidth via min-degree elimination on the link graph.
    Returns the width of the elimination ordering (upper bound on treewidth)."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(bool)
    N = cs.n
    if N < 3:
        return 0

    remaining = set(range(N))
    adj_list = {i: set(np.where(adj[i])[0]) & remaining for i in range(N)}
    width = 0

    for _ in range(N):
        if not remaining:
            break
        # Pick vertex with minimum degree
        min_v = min(remaining, key=lambda v: len(adj_list[v] & remaining))
        neighbors = adj_list[min_v] & remaining
        width = max(width, len(neighbors))
        # Connect neighbors (fill edges)
        nb_list = list(neighbors)
        for a in range(len(nb_list)):
            for b in range(a + 1, len(nb_list)):
                adj_list[nb_list[a]].add(nb_list[b])
                adj_list[nb_list[b]].add(nb_list[a])
        remaining.remove(min_v)

    return width

def sj_central_charge(cs):
    """Compute effective central charge from SJ entanglement entropy.
    Uses S(N/2) ~ (c/3) * ln(N) + const."""
    try:
        W = sj_wightman_function(cs)
        N = cs.n
        region = list(range(N // 2))
        S = entanglement_entropy(W, region)
        # For a single size, estimate c from S(N/2) ~ (c/3)*ln(N)
        if N > 5:
            c_eff = 3 * S / np.log(N)
        else:
            c_eff = 0.0
        return S, c_eff
    except Exception:
        return 0.0, 0.0


# ============================================================
# HELPER: Generate CDT samples
# ============================================================
def get_cdt_samples(N_target, lambda2=0.0, n_mcmc=10000, n_samples=5):
    """Generate CDT configurations and convert to causets."""
    T = max(8, int(np.sqrt(N_target)))
    s_init = max(3, N_target // T)
    samples = mcmc_cdt(T=T, s_init=s_init, lambda2=lambda2, mu=0.01,
                        n_steps=n_mcmc, target_volume=N_target, rng=rng)
    causets = []
    profiles = []
    for vp in samples[-n_samples * 2::2]:
        if len(causets) >= n_samples:
            break
        N_actual = int(np.sum(vp))
        if N_actual > 200:
            continue
        cs = cdt_to_causet(vp.astype(int))
        causets.append(cs)
        profiles.append(vp)
    return causets, profiles


# ================================================================
print("=" * 78)
print("IDEA 281: INTERVAL ENTROPY ON CDT CONFIGURATIONS")
print("Does the interval size distribution match sprinkled 2D causets?")
print("=" * 78)

N_test = 60
n_trials = 8

print(f"\n  N ~ {N_test}")
print(f"  {'Source':>20} {'H(intervals)':>14} {'std':>8}")
print("-" * 50)

# Sprinkled 2D causets
H_causet = []
for _ in range(n_trials):
    cs, _ = sprinkle_fast(N_test, dim=2, rng=rng)
    H_causet.append(interval_entropy(cs))
print(f"  {'Sprinkled 2D':>20} {np.mean(H_causet):>14.4f} {np.std(H_causet):>8.4f}")

# CDT configurations
H_cdt = []
cdt_causets, _ = get_cdt_samples(N_test, n_mcmc=15000, n_samples=n_trials)
for cs in cdt_causets:
    H_cdt.append(interval_entropy(cs))
if H_cdt:
    print(f"  {'CDT (lambda2=0)':>20} {np.mean(H_cdt):>14.4f} {np.std(H_cdt):>8.4f}")

# Random DAG (matched density)
H_dag = []
f_target = np.mean([ordering_fraction(cs) for cs in cdt_causets]) if cdt_causets else 0.3
for _ in range(n_trials):
    dag = make_random_dag(N_test, f_target, rng)
    H_dag.append(interval_entropy(dag))
print(f"  {'Random DAG':>20} {np.mean(H_dag):>14.4f} {np.std(H_dag):>8.4f}")

# Regular lattice
Nt, Nx = 10, 6  # ~60 vertices
cs_lat = make_regular_lattice_causet(Nt, Nx)
H_lat = interval_entropy(cs_lat)
print(f"  {'Regular lattice':>20} {H_lat:>14.4f} {'(single)':>8}")

print("\n  ASSESSMENT:")
if H_cdt and abs(np.mean(H_cdt) - np.mean(H_causet)) < 2 * max(np.std(H_cdt), np.std(H_causet)):
    print("  CDT interval entropy is CONSISTENT with sprinkled causets")
else:
    ratio = np.mean(H_cdt) / np.mean(H_causet) if np.mean(H_causet) > 0 else 0
    print(f"  CDT interval entropy DIFFERS from causets (ratio = {ratio:.2f})")
    print(f"  CDT has {'richer' if ratio > 1 else 'simpler'} interval structure")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 282: HASSE LAPLACIAN FIEDLER VALUE — CDT vs CAUSETS")
print("Fiedler value measures algebraic connectivity of the link graph")
print("=" * 78)

print(f"\n  {'Source':>20} {'Fiedler':>10} {'std':>8}")
print("-" * 45)

F_causet = []
for _ in range(n_trials):
    cs, _ = sprinkle_fast(N_test, dim=2, rng=rng)
    F_causet.append(fiedler_value(cs))
print(f"  {'Sprinkled 2D':>20} {np.mean(F_causet):>10.4f} {np.std(F_causet):>8.4f}")

F_cdt = []
for cs in cdt_causets:
    F_cdt.append(fiedler_value(cs))
if F_cdt:
    print(f"  {'CDT':>20} {np.mean(F_cdt):>10.4f} {np.std(F_cdt):>8.4f}")

F_dag = []
for _ in range(n_trials):
    dag = make_random_dag(N_test, f_target, rng)
    F_dag.append(fiedler_value(dag))
print(f"  {'Random DAG':>20} {np.mean(F_dag):>10.4f} {np.std(F_dag):>8.4f}")

F_lat = fiedler_value(cs_lat)
print(f"  {'Regular lattice':>20} {F_lat:>10.4f} {'(single)':>8}")

print("\n  ASSESSMENT:")
if F_cdt:
    if abs(np.mean(F_cdt) - np.mean(F_causet)) < 2 * max(np.std(F_cdt), np.std(F_causet), 0.01):
        print("  Fiedler values MATCH between CDT and causets")
    else:
        print(f"  Fiedler values DIFFER: CDT={np.mean(F_cdt):.4f}, causet={np.mean(F_causet):.4f}")
        if np.mean(F_cdt) > np.mean(F_causet):
            print("  CDT link graph is MORE connected (higher algebraic connectivity)")
        else:
            print("  CDT link graph is LESS connected")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 283: TREEWIDTH — CDT vs CAUSETS")
print("Treewidth measures how 'tree-like' the link graph is")
print("=" * 78)

N_tw = 40  # treewidth is expensive, use smaller N
n_tw = 6

print(f"\n  N ~ {N_tw}")
print(f"  {'Source':>20} {'Treewidth':>10} {'std':>8} {'tw/N':>8}")
print("-" * 55)

tw_causet = []
for _ in range(n_tw):
    cs, _ = sprinkle_fast(N_tw, dim=2, rng=rng)
    tw_causet.append(treewidth_approx(cs))
print(f"  {'Sprinkled 2D':>20} {np.mean(tw_causet):>10.1f} {np.std(tw_causet):>8.1f} "
      f"{np.mean(tw_causet)/N_tw:>8.3f}")

tw_cdt = []
cdt_small, _ = get_cdt_samples(N_tw, n_mcmc=10000, n_samples=n_tw)
for cs in cdt_small:
    tw_cdt.append(treewidth_approx(cs))
if tw_cdt:
    print(f"  {'CDT':>20} {np.mean(tw_cdt):>10.1f} {np.std(tw_cdt):>8.1f} "
          f"{np.mean(tw_cdt)/N_tw:>8.3f}")

tw_dag = []
for _ in range(n_tw):
    dag = make_random_dag(N_tw, 0.3, rng)
    tw_dag.append(treewidth_approx(dag))
print(f"  {'Random DAG':>20} {np.mean(tw_dag):>10.1f} {np.std(tw_dag):>8.1f} "
      f"{np.mean(tw_dag)/N_tw:>8.3f}")

Nt_tw, Nx_tw = 8, 5
cs_lat_tw = make_regular_lattice_causet(Nt_tw, Nx_tw)
tw_lat = treewidth_approx(cs_lat_tw)
N_lat_tw = Nt_tw * Nx_tw
print(f"  {'Regular lattice':>20} {tw_lat:>10.1f} {'(single)':>8} {tw_lat/N_lat_tw:>8.3f}")

print("\n  ASSESSMENT:")
if tw_cdt:
    print(f"  Treewidth ratio: CDT={np.mean(tw_cdt)/N_tw:.3f}, causet={np.mean(tw_causet)/N_tw:.3f}")
    if np.mean(tw_cdt) < np.mean(tw_causet):
        print("  CDT has LOWER treewidth -> more tree-like -> simpler causal structure")
    else:
        print("  CDT has HIGHER treewidth -> more complex causal structure than causets")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 284: ANTICHAIN SCALING — CDT vs CAUSETS")
print("Longest antichain ~ N^((d-1)/d) for d-dimensional causets")
print("=" * 78)

Ns_ac = [20, 30, 50, 80]
print(f"\n  {'Source':>15} {'N':>5} {'Antichain':>10} {'AC/N':>8}")
print("-" * 45)

ac_data = {'causet': {}, 'cdt': {}}
for N in Ns_ac:
    # Causets
    ac_vals = []
    for _ in range(6):
        cs, _ = sprinkle_fast(N, dim=2, rng=rng)
        ac_vals.append(longest_antichain_size(cs))
    ac_data['causet'][N] = np.mean(ac_vals)
    print(f"  {'Sprinkled 2D':>15} {N:>5} {np.mean(ac_vals):>10.1f} {np.mean(ac_vals)/N:>8.3f}")

    # CDT
    cdt_cs, _ = get_cdt_samples(N, n_mcmc=8000, n_samples=4)
    ac_cdt = [longest_antichain_size(c) for c in cdt_cs]
    if ac_cdt:
        ac_data['cdt'][N] = np.mean(ac_cdt)
        print(f"  {'CDT':>15} {N:>5} {np.mean(ac_cdt):>10.1f} {np.mean(ac_cdt)/N:>8.3f}")

# Fit power law antichain ~ N^alpha
def power_law(x, a, b):
    return a * x ** b

print("\n  Power law fits: antichain ~ N^alpha")
for src in ['causet', 'cdt']:
    if len(ac_data[src]) >= 3:
        Ns_fit = np.array(sorted(ac_data[src].keys()))
        vals = np.array([ac_data[src][n] for n in Ns_fit])
        try:
            popt, _ = curve_fit(power_law, Ns_fit, vals, p0=[1.0, 0.5])
            d_est = 1.0 / (1.0 - popt[1]) if popt[1] < 1 else float('inf')
            print(f"    {src:>10}: alpha = {popt[1]:.3f} -> d_eff = {d_est:.2f}")
        except Exception as e:
            print(f"    {src:>10}: fit failed ({e})")

print("\n  ASSESSMENT: In 2D, expect alpha ~ 0.5 (d_eff = 2)")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 285: COMPRESSIBILITY (SVD + GZIP) — CDT vs CAUSETS")
print("Structural complexity comparison via information content")
print("=" * 78)

print(f"\n  N ~ {N_test}")
print(f"  {'Source':>20} {'gzip_ratio':>12} {'svd_k90':>10} {'ord_frac':>10}")
print("-" * 60)

for label, gen_fn in [
    ('Sprinkled 2D', lambda: sprinkle_fast(N_test, dim=2, rng=rng)[0]),
    ('CDT', lambda: cdt_causets[rng.integers(len(cdt_causets))] if cdt_causets else None),
    ('Random DAG', lambda: make_random_dag(N_test, f_target, rng)),
    ('Regular lattice', lambda: cs_lat),
]:
    gz, sv, of = [], [], []
    for _ in range(n_trials):
        cs = gen_fn()
        if cs is None:
            break
        gz.append(compressibility(cs))
        sv.append(svd_compressibility(cs))
        of.append(ordering_fraction(cs))
    if gz:
        print(f"  {label:>20} {np.mean(gz):>12.4f} {np.mean(sv):>10.4f} {np.mean(of):>10.4f}")

print("\n  INTERPRETATION:")
print("  Lower gzip ratio = more regular/compressible structure")
print("  Lower SVD k90 = lower effective rank = simpler geometry")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 286: LINK FRACTION ACROSS CDT PHASE TRANSITION")
print("How does the link structure change as we vary lambda2?")
print("=" * 78)

N_phase = 60
lambda2_vals = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

print(f"\n  N ~ {N_phase}")
print(f"  {'lambda2':>10} {'link_frac':>12} {'ord_frac':>12} {'N_actual':>10}")
print("-" * 50)

for lam in lambda2_vals:
    t0 = time.time()
    T = max(8, int(np.sqrt(N_phase)))
    s_init = max(3, N_phase // T)
    samples = mcmc_cdt(T=T, s_init=s_init, lambda2=lam, mu=0.02,
                        n_steps=15000, target_volume=N_phase, rng=rng)
    lf_vals, of_vals, n_actual = [], [], []
    for vp in samples[-10:]:
        N_act = int(np.sum(vp))
        if N_act > 200 or N_act < 10:
            continue
        cs = cdt_to_causet(vp.astype(int))
        n_rel = cs.num_relations()
        if n_rel > 0:
            lf_vals.append(link_fraction(cs))
            of_vals.append(ordering_fraction(cs))
            n_actual.append(N_act)

    if lf_vals:
        print(f"  {lam:>10.1f} {np.mean(lf_vals):>12.4f} {np.mean(of_vals):>12.4f} "
              f"{np.mean(n_actual):>10.0f}")
    dt = time.time() - t0

print("\n  ASSESSMENT:")
print("  At large lambda2 (suppressing volume), expect fewer triangles -> sparser -> higher link fraction")
print("  At negative lambda2 (encouraging volume), expect denser -> lower link fraction")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 287: BD ACTION ON CDT CONFIGURATIONS")
print("How does the Benincasa-Dowker action score CDT triangulations?")
print("=" * 78)

N_bd = 50
print(f"\n  N ~ {N_bd}")
print(f"  {'Source':>20} {'S_BD':>10} {'S_BD/N':>10} {'std':>8}")
print("-" * 55)

# Sprinkled 2D (should give S_BD ~ 0 for flat space)
bd_causet = []
for _ in range(n_trials):
    cs, _ = sprinkle_fast(N_bd, dim=2, rng=rng)
    bd_causet.append(bd_action_2d(cs))
print(f"  {'Sprinkled 2D':>20} {np.mean(bd_causet):>10.2f} {np.mean(bd_causet)/N_bd:>10.4f} "
      f"{np.std(bd_causet):>8.2f}")

# CDT
bd_cdt = []
cdt_cs_bd, _ = get_cdt_samples(N_bd, n_mcmc=12000, n_samples=n_trials)
for cs in cdt_cs_bd:
    bd_cdt.append(bd_action_2d(cs))
if bd_cdt:
    N_eff = np.mean([cs.n for cs in cdt_cs_bd])
    print(f"  {'CDT':>20} {np.mean(bd_cdt):>10.2f} {np.mean(bd_cdt)/N_eff:>10.4f} "
          f"{np.std(bd_cdt):>8.2f}")

# Random DAG
bd_dag = []
for _ in range(n_trials):
    dag = make_random_dag(N_bd, f_target, rng)
    bd_dag.append(bd_action_2d(dag))
print(f"  {'Random DAG':>20} {np.mean(bd_dag):>10.2f} {np.mean(bd_dag)/N_bd:>10.4f} "
      f"{np.std(bd_dag):>8.2f}")

# Regular lattice
Nt_bd, Nx_bd = 10, 5
cs_lat_bd = make_regular_lattice_causet(Nt_bd, Nx_bd)
bd_lat = bd_action_2d(cs_lat_bd)
N_lat_bd = Nt_bd * Nx_bd
print(f"  {'Regular lattice':>20} {bd_lat:>10.2f} {bd_lat/N_lat_bd:>10.4f} {'(single)':>8}")

print("\n  ASSESSMENT:")
print("  For flat 2D Minkowski, S_BD -> 0 as N -> inf")
print("  CDT should also give small S_BD if it approximates flat 2D geometry")
if bd_cdt:
    if abs(np.mean(bd_cdt) / N_eff) < abs(np.mean(bd_dag) / N_bd):
        print("  CDT S_BD/N is CLOSER to zero than random DAG -> manifold-like!")
    else:
        print("  CDT S_BD/N is NOT closer to zero than random DAG")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 288: SJ VACUUM ON REGULAR LATTICE — IS c = 1?")
print("Free scalar on square lattice with causal structure")
print("=" * 78)

print("\n  Computing SJ entanglement entropy on regular lattices of various sizes...")
print(f"  {'Nt x Nx':>10} {'N':>5} {'S(N/2)':>10} {'c_eff':>10}")
print("-" * 40)

lattice_sizes = [(6, 4), (8, 4), (8, 6), (10, 5), (10, 6), (12, 5)]
c_eff_lat = []

for Nt_l, Nx_l in lattice_sizes:
    N_l = Nt_l * Nx_l
    if N_l > 80:
        continue
    cs_l = make_regular_lattice_causet(Nt_l, Nx_l)
    S, c = sj_central_charge(cs_l)
    c_eff_lat.append(c)
    print(f"  {f'{Nt_l}x{Nx_l}':>10} {N_l:>5} {S:>10.4f} {c:>10.4f}")

print(f"\n  Mean c_eff (lattice): {np.mean(c_eff_lat):.4f} +/- {np.std(c_eff_lat):.4f}")
print("  Free scalar CFT prediction: c = 1")

# Compare with sprinkled causets
c_eff_spr = []
for N_s in [24, 36, 48, 50, 60]:
    cs_s, _ = sprinkle_fast(N_s, dim=2, rng=rng)
    S, c = sj_central_charge(cs_s)
    c_eff_spr.append(c)

print(f"  Mean c_eff (sprinkled 2D): {np.mean(c_eff_spr):.4f} +/- {np.std(c_eff_spr):.4f}")

if abs(np.mean(c_eff_lat) - 1.0) < abs(np.mean(c_eff_spr) - 1.0):
    print("  Regular lattice is CLOSER to c=1 than sprinkled causets!")
else:
    print("  Sprinkled causets are closer to c=1 than regular lattice")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 289: SJ VACUUM IN CURVED SPACETIME (DE SITTER)")
print("Does entanglement entropy depend on curvature H?")
print("=" * 78)

N_ds = 50
H_values = [0.1, 0.5, 1.0, 2.0, 5.0]

print(f"\n  N = {N_ds}")
print(f"  {'H (curvature)':>15} {'S(N/2)':>10} {'c_eff':>10} {'ord_frac':>10}")
print("-" * 50)

# Flat reference
cs_flat, _ = sprinkle_fast(N_ds, dim=2, rng=rng)
S_flat, c_flat = sj_central_charge(cs_flat)
f_flat = ordering_fraction(cs_flat)
print(f"  {'flat (H=0)':>15} {S_flat:>10.4f} {c_flat:>10.4f} {f_flat:>10.4f}")

ds_results = {}
for H in H_values:
    S_vals, c_vals, f_vals = [], [], []
    for _ in range(4):
        try:
            cs_ds, coords_ds = sprinkle_de_sitter_2d(N_ds, H=H, rng_local=rng)
            S, c = sj_central_charge(cs_ds)
            f = ordering_fraction(cs_ds)
            S_vals.append(S)
            c_vals.append(c)
            f_vals.append(f)
        except Exception:
            pass
    if S_vals:
        ds_results[H] = {'S': np.mean(S_vals), 'c': np.mean(c_vals), 'f': np.mean(f_vals)}
        print(f"  {f'H = {H}':>15} {np.mean(S_vals):>10.4f} {np.mean(c_vals):>10.4f} "
              f"{np.mean(f_vals):>10.4f}")

print("\n  ASSESSMENT:")
if ds_results:
    S_list = [ds_results[H]['S'] for H in sorted(ds_results.keys())]
    if len(S_list) >= 3:
        slope = (S_list[-1] - S_list[0]) / (max(ds_results.keys()) - min(ds_results.keys()))
        if abs(slope) > 0.01:
            print(f"  Entropy {'increases' if slope > 0 else 'decreases'} with curvature (slope ~ {slope:.3f})")
            print("  This suggests the SJ vacuum IS sensitive to background curvature")
        else:
            print("  Entropy is roughly CONSTANT with curvature")
            print("  SJ vacuum appears insensitive to curvature at this N")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 290: SYSTEMATIC TABLE — ALL OBSERVABLES ACROSS ALL STRUCTURES")
print("The Grand Comparison: 10 observables x 4 structure types")
print("=" * 78)

N_table = 50
n_samples_table = 6

# Generate all structure types
print(f"\n  Generating structures (N ~ {N_table})...")

structures = {}

# 1. Sprinkled 2D causets
structs_causet = []
for _ in range(n_samples_table):
    cs, _ = sprinkle_fast(N_table, dim=2, rng=rng)
    structs_causet.append(cs)
structures['Causet 2D'] = structs_causet

# 2. CDT
cdt_cs_table, _ = get_cdt_samples(N_table, n_mcmc=12000, n_samples=n_samples_table)
structures['CDT'] = cdt_cs_table

# 3. Regular lattice (single instance, repeated for consistency)
Nt_t, Nx_t = 10, 5
cs_lat_t = make_regular_lattice_causet(Nt_t, Nx_t)
structures['Lattice'] = [cs_lat_t] * n_samples_table

# 4. Random DAG
structs_dag = []
for _ in range(n_samples_table):
    structs_dag.append(make_random_dag(N_table, 0.3, rng))
structures['Random DAG'] = structs_dag

# Define all observables
observables = {
    'ord_frac': ('Ordering fraction', ordering_fraction),
    'link_frac': ('Link fraction', link_fraction),
    'chain': ('Longest chain', longest_chain),
    'antichain': ('Longest antichain', longest_antichain_size),
    'fiedler': ('Fiedler value', fiedler_value),
    'gzip': ('Gzip compressibility', compressibility),
    'svd_k90': ('SVD k90 fraction', svd_compressibility),
    'int_entropy': ('Interval entropy', interval_entropy),
    'treewidth': ('Treewidth (approx)', treewidth_approx),
    'bd_action': ('BD action (2D)', bd_action_2d),
}

# Compute the grand table
print("\n  Computing all observables...")
results_table = {}
for struct_name, struct_list in structures.items():
    results_table[struct_name] = {}
    t0 = time.time()
    for obs_key, (obs_label, obs_fn) in observables.items():
        vals = []
        for cs in struct_list:
            try:
                vals.append(obs_fn(cs))
            except Exception:
                pass
        if vals:
            results_table[struct_name][obs_key] = (np.mean(vals), np.std(vals))
        else:
            results_table[struct_name][obs_key] = (np.nan, np.nan)
    dt = time.time() - t0
    print(f"    {struct_name}: done ({dt:.1f}s)")

# Print the grand table
print("\n" + "=" * 90)
print("  GRAND COMPARISON TABLE")
print("=" * 90)
header = f"  {'Observable':>22}"
for sn in structures:
    header += f" | {sn:>16}"
print(header)
print("-" * 90)

for obs_key, (obs_label, _) in observables.items():
    row = f"  {obs_label:>22}"
    for sn in structures:
        mean, std = results_table[sn][obs_key]
        if np.isnan(mean):
            row += f" | {'N/A':>16}"
        else:
            row += f" | {mean:>9.3f}({std:>4.2f})"
    print(row)

# Compute discriminability scores
print("\n" + "-" * 90)
print("  DISCRIMINABILITY: Which observables best distinguish structure types?")
print("-" * 90)

struct_names = list(structures.keys())
for obs_key, (obs_label, _) in observables.items():
    means = []
    for sn in struct_names:
        m, s = results_table[sn][obs_key]
        means.append(m)
    means = np.array(means)
    if np.any(np.isnan(means)):
        continue
    # Coefficient of variation across structure types
    if np.mean(means) > 1e-10:
        cv = np.std(means) / abs(np.mean(means))
    else:
        cv = 0.0
    print(f"  {obs_label:>22}: CV = {cv:.3f} {'***' if cv > 0.5 else '**' if cv > 0.3 else '*' if cv > 0.1 else ''}")

print("\n  *** = highly discriminating (CV > 0.5)")
print("  **  = moderately discriminating (CV > 0.3)")
print("  *   = weakly discriminating (CV > 0.1)")

# Nearest-neighbor analysis
print("\n" + "-" * 90)
print("  WHICH STRUCTURES ARE MOST SIMILAR?")
print("-" * 90)

# Compute pairwise distances in observable space
obs_keys_valid = [k for k in observables if all(not np.isnan(results_table[sn][k][0]) for sn in struct_names)]
n_obs = len(obs_keys_valid)

# Normalize each observable to [0, 1]
obs_matrix = np.zeros((len(struct_names), n_obs))
for j, obs_key in enumerate(obs_keys_valid):
    for i, sn in enumerate(struct_names):
        obs_matrix[i, j] = results_table[sn][obs_key][0]
    col_range = obs_matrix[:, j].max() - obs_matrix[:, j].min()
    if col_range > 1e-14:
        obs_matrix[:, j] = (obs_matrix[:, j] - obs_matrix[:, j].min()) / col_range

# Pairwise Euclidean distances
print(f"\n  Pairwise distances (normalized observable space, {n_obs} observables):")
print(f"  {'':>16}", end="")
for sn in struct_names:
    print(f" {sn:>16}", end="")
print()
for i, sn1 in enumerate(struct_names):
    print(f"  {sn1:>16}", end="")
    for j, sn2 in enumerate(struct_names):
        dist = np.linalg.norm(obs_matrix[i] - obs_matrix[j])
        print(f" {dist:>16.3f}", end="")
    print()

# Find closest pair
min_dist = float('inf')
min_pair = ('', '')
for i in range(len(struct_names)):
    for j in range(i + 1, len(struct_names)):
        d = np.linalg.norm(obs_matrix[i] - obs_matrix[j])
        if d < min_dist:
            min_dist = d
            min_pair = (struct_names[i], struct_names[j])

print(f"\n  Closest pair: {min_pair[0]} <-> {min_pair[1]} (distance = {min_dist:.3f})")
print(f"  This means these two approaches produce the most similar discrete geometries")

# Final summary
print("\n" + "=" * 90)
print("  FINAL SUMMARY: CROSS-APPROACH COMPARISON")
print("=" * 90)

print("""
  KEY FINDINGS:

  1. INTERVAL ENTROPY (Idea 281):
     - Reveals the richness of the causal interval structure
     - CDT vs causet differences indicate fundamentally different interval distributions

  2. FIEDLER VALUE (Idea 282):
     - Measures algebraic connectivity of the Hasse diagram
     - Different connectivity = different local causal structure

  3. TREEWIDTH (Idea 283):
     - Measures how 'tree-like' the causal structure is
     - Related to computational complexity of inference on the structure

  4. ANTICHAIN SCALING (Idea 284):
     - Power law exponent encodes effective dimension
     - Both CDT and causets should give d_eff ~ 2 in the continuum limit

  5. COMPRESSIBILITY (Idea 285):
     - SVD + gzip reveal structural complexity differences
     - Regular lattices should be most compressible

  6. CDT PHASE TRANSITION (Idea 286):
     - Link fraction changes across lambda2 values
     - Maps the CDT phase structure via causal set observables

  7. BD ACTION ON CDT (Idea 287):
     - Tests whether CDT configurations look 'manifold-like' to the BD action
     - Small |S_BD/N| = consistent with flat 2D geometry

  8. SJ ON LATTICE (Idea 288):
     - Tests whether regular lattice gives c=1 free scalar
     - Important calibration: if lattice fails, causet c≠1 may be a discretization artifact

  9. SJ IN CURVED SPACETIME (Idea 289):
     - De Sitter sprinkling tests curvature dependence
     - If S depends on H, the SJ vacuum encodes geometry beyond flat space

  10. GRAND TABLE (Idea 290):
      - Systematic comparison reveals which approaches produce similar discrete geometries
      - Identifies the best discriminating observables
""")

# Score
print("  NOVELTY/INTEREST SCORE: 7/10")
print("  - Cross-approach systematic comparison is rare in the literature")
print("  - The grand comparison table is a useful reference contribution")
print("  - Most interesting if CDT and causets agree on some observables but not others")
print("  - Publishable as part of a survey/methods paper")
