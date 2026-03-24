"""
Experiment 111: LARGE-N VERIFICATION — Ideas 621-630

Pure combinatorial measurements at large N using sparse order matrices.
NO eigendecomposition needed. Verify or falsify key scaling claims.

621. Interval entropy H at N=1000-5000 (count intervals, no eigen)
622. Link fraction at N=5000-20000. Verify 4ln(N)/N with constant ~3.14 or ~4.0
623. Ordering fraction variance at N=5000-20000. Verify (2N+5)/[18N(N-1)]
624. Chain length at N=10000-50000. Verify 2sqrt(N) scaling. What's the constant?
625. Antichain width at N=10000-50000. Verify 2sqrt(N)
626. Hasse diameter at N=5000-20000. Does it really saturate at ~6?
627. Number of links at N=5000-20000. Verify (N+1)H_N - 2N
628. Number of maximal elements at N=5000-20000. Verify H_N
629. Hasse connectivity P(connected) at N=500-5000. When does it hit 100%?
630. Ordering fraction E[f] at N=10000. Verify exactly 0.5000
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time
from collections import deque

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)

OVERALL_START = time.time()

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Sprinkle cache to avoid recomputing for same (N, seed)
_sprinkle_cache = {}
def sprinkle_cached(n, seed=42, dim=2):
    """Cached sprinkle — reuse across ideas for same (n, seed)."""
    key = (n, seed, dim)
    if key not in _sprinkle_cache:
        _sprinkle_cache[key] = sprinkle_sparse(n, dim=dim, rng_local=np.random.default_rng(seed))
    return _sprinkle_cache[key]

def clear_cache():
    """Free memory from cache."""
    _sprinkle_cache.clear()


# ============================================================
# SPARSE UTILITIES (from exp102, extended)
# ============================================================

def sprinkle_sparse(n, dim=2, extent_t=1.0, rng_local=None):
    """
    Sprinkle into 2D Minkowski causal diamond, return coords + sparse order.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    coords_list = []
    total = 0
    while total < n:
        batch = max(n * 4, 10000)
        candidates = np.zeros((batch, dim))
        candidates[:, 0] = rng_local.uniform(-extent_t, extent_t, batch)
        for d in range(1, dim):
            candidates[:, d] = rng_local.uniform(-extent_t, extent_t, batch)
        r = np.sqrt(np.sum(candidates[:, 1:]**2, axis=1))
        mask = np.abs(candidates[:, 0]) + r <= extent_t
        coords_list.append(candidates[mask])
        total += int(np.sum(mask))
    coords = np.vstack(coords_list)[:n]
    coords = coords[np.argsort(coords[:, 0])]

    # Build sparse order relation
    rows, cols = [], []
    for i in range(n - 1):
        dt = coords[i + 1:, 0] - coords[i, 0]
        dx_sq = np.sum((coords[i + 1:, 1:] - coords[i, 1:])**2, axis=1)
        related = dt * dt >= dx_sq
        js = np.where(related)[0] + i + 1
        rows.extend([i] * len(js))
        cols.extend(js.tolist())

    data = np.ones(len(rows), dtype=np.int8)
    order_sparse = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)
    return coords, order_sparse


def compute_links_sparse(order_sparse, n):
    """Compute links from sparse order: link iff order[i,j] and no 2-step path."""
    order_float = order_sparse.astype(np.float32)
    path2 = order_float @ order_float
    path2_binary = (path2 > 0).astype(np.int8)
    links_sparse = order_sparse - order_sparse.multiply(path2_binary)
    links_sparse.eliminate_zeros()
    return links_sparse


def longest_chain_sparse(order_sparse, n):
    """Longest chain via DP on time-ordered elements using CSC format."""
    order_csc = order_sparse.tocsc()
    dp = np.ones(n, dtype=np.int32)
    for j in range(n):
        start, end = order_csc.indptr[j], order_csc.indptr[j + 1]
        predecessors = order_csc.indices[start:end]
        predecessors = predecessors[predecessors < j]
        if len(predecessors) > 0:
            dp[j] = np.max(dp[predecessors]) + 1
    return int(np.max(dp))


def greedy_antichain_sparse(order_sparse, n, rng_local=None):
    """Greedy antichain on sparse order matrix."""
    if rng_local is None:
        rng_local = np.random.default_rng()
    perm = rng_local.permutation(n)
    antichain_set = set()
    order_full = order_sparse + order_sparse.T

    for idx in perm:
        row = order_full.getrow(idx)
        _, cols_arr = row.nonzero()
        if len(antichain_set.intersection(cols_arr)) == 0:
            antichain_set.add(idx)
    return len(antichain_set)


def hasse_diameter_bfs(links_sparse, n, n_sources=20):
    """Estimate Hasse diameter via BFS from multiple sources."""
    adj = (links_sparse + links_sparse.T).tocsr()
    max_dist = 0
    # Use diverse sources: edges of time, middle, and random
    sources = list(set([0, n // 8, n // 4, 3 * n // 8, n // 2,
                        5 * n // 8, 3 * n // 4, 7 * n // 8, n - 1]))
    # Add some random sources
    extra = rng.integers(0, n, min(n_sources, 20)).tolist()
    sources = list(set(sources + extra))[:n_sources]

    for start in sources:
        dist = np.full(n, -1, dtype=np.int32)
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            s_, e_ = adj.indptr[u], adj.indptr[u + 1]
            for v in adj.indices[s_:e_]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        reachable = dist[dist >= 0]
        if len(reachable) > 0:
            max_dist = max(max_dist, int(np.max(reachable)))
    return max_dist


def count_maximal_elements_sparse(order_sparse, n):
    """Count elements with no successors (maximal elements)."""
    # An element i is maximal if row i of order_sparse is all zeros
    row_nnz = np.diff(order_sparse.indptr)
    return int(np.sum(row_nnz == 0))


def count_minimal_elements_sparse(order_sparse, n):
    """Count elements with no predecessors (minimal elements)."""
    order_csc = order_sparse.tocsc()
    col_nnz = np.diff(order_csc.indptr)
    return int(np.sum(col_nnz == 0))


def interval_counts_sparse(order_sparse, n, max_size=30):
    """
    Count intervals by interior size using sparse matrices.
    interval_matrix[i,j] = number of elements k with order[i,k] and order[k,j].
    This is (order @ order)[i,j] restricted to related pairs.
    """
    order_float = order_sparse.astype(np.float32)
    # path2[i,j] = number of 2-step paths from i to j
    path2 = order_float @ order_float
    # Only look at related pairs (entries in order_sparse)
    counts = {}
    # Extract entries of path2 at positions where order_sparse is nonzero
    rows, cols = order_sparse.nonzero()
    interior_sizes = np.array(path2[rows, cols]).flatten().astype(int)

    for k in range(max_size + 1):
        counts[k] = int(np.sum(interior_sizes == k))

    counts['larger'] = int(np.sum(interior_sizes > max_size))
    return counts


flush_print("=" * 80)
flush_print("EXPERIMENT 111: LARGE-N VERIFICATION (Ideas 621-630)")
flush_print("Pure combinatorial measurements — no eigendecomposition")
flush_print("=" * 80)
flush_print()


# ============================================================
# IDEA 621: INTERVAL ENTROPY H AT N=1000-5000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 621: INTERVAL ENTROPY H AT N=1000-5000")
flush_print("Count intervals by size, compute Shannon entropy of distribution.")
flush_print("Does H converge to a limit as N grows?")
flush_print("=" * 80)

int_ent_sizes = [100, 200, 500, 1000, 2000, 3000, 5000]
int_ent_vals = []
int_ent_max_k = []

for N in int_ent_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    t_spr = time.time() - t0

    max_k = min(N, 50)
    intervals = interval_counts_sparse(order_sparse, N, max_size=max_k)
    total = sum(v for k, v in intervals.items() if k != 'larger') + intervals.get('larger', 0)

    # Build probability distribution (exclude 'larger' key for entropy if small)
    probs = []
    for k in range(max_k + 1):
        if intervals.get(k, 0) > 0:
            probs.append(intervals[k] / total)
    if intervals.get('larger', 0) > 0:
        probs.append(intervals['larger'] / total)

    probs = np.array(probs)
    H = -np.sum(probs * np.log(probs))

    # Find the mode (most common interval size)
    mode_k = max(range(max_k + 1), key=lambda k: intervals.get(k, 0))

    elapsed = time.time() - t0
    int_ent_vals.append(H)
    int_ent_max_k.append(mode_k)
    flush_print(f"  N={N:5d}: H = {H:.6f}, n_intervals = {total:10d}, "
                f"mode_k = {mode_k}, links(k=0) = {intervals.get(0,0)}, "
                f"sprinkle={t_spr:.1f}s, total={elapsed:.1f}s")

    if elapsed > 55 and N < max(int_ent_sizes):
        flush_print(f"  (skipping larger sizes)")
        break

flush_print()
diffs = [abs(int_ent_vals[i+1] - int_ent_vals[i]) for i in range(len(int_ent_vals)-1)]
flush_print(f"  H values: {[f'{h:.4f}' for h in int_ent_vals]}")
flush_print(f"  Successive diffs: {[f'{d:.4f}' for d in diffs]}")
if len(diffs) >= 2 and diffs[-1] < diffs[0]:
    flush_print(f"  CONVERGING: last diff {diffs[-1]:.6f} < first diff {diffs[0]:.6f}")
    flush_print(f"  Estimated limit: {int_ent_vals[-1]:.4f}")
else:
    flush_print(f"  Convergence: UNCLEAR")
flush_print()


# ============================================================
# IDEA 622: LINK FRACTION AT N=5000-20000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 622: LINK FRACTION AT N=5000-20000")
flush_print("Verify link_frac = c * ln(N) / N. What is c?")
flush_print("=" * 80)

link_sizes = [500, 1000, 2000, 5000, 10000]
link_data = []

for N in link_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    t_spr = time.time() - t0

    links_sp = compute_links_sparse(order_sparse, N)
    n_links = links_sp.nnz
    n_rels = order_sparse.nnz

    link_frac = n_links / max(n_rels, 1)
    c_est = link_frac * N / np.log(N)

    elapsed = time.time() - t0
    link_data.append((N, n_links, n_rels, link_frac, c_est))
    flush_print(f"  N={N:6d}: links={n_links:8d}, rels={n_rels:12d}, "
                f"link_frac={link_frac:.6f}, c_est={c_est:.4f}, t={elapsed:.1f}s")

    if elapsed > 55 and N < max(link_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
c_vals = [d[4] for d in link_data]
flush_print(f"  c estimates (link_frac * N / ln(N)): {[f'{c:.4f}' for c in c_vals]}")
flush_print(f"  Mean c = {np.mean(c_vals):.4f} +/- {np.std(c_vals):.4f}")
flush_print(f"  Claimed: c ~ 4.0 (4ln(N)/N formula)")
flush_print(f"  Alternative: c ~ pi ~ 3.14")
flush_print()

# Also check links per element: links/N ~ a * ln(N)?
flush_print(f"  links/N analysis:")
for N, n_links, n_rels, lf, c in link_data:
    links_per_elem = n_links / N
    a_est = links_per_elem / np.log(N)
    flush_print(f"    N={N:6d}: links/N = {links_per_elem:.3f}, "
                f"a (links/N / lnN) = {a_est:.4f}")
flush_print()


# Free cache before multi-sample ideas
clear_cache()

# ============================================================
# IDEA 623: ORDERING FRACTION VARIANCE AT N=5000-20000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 623: ORDERING FRACTION VARIANCE AT N=5000-20000")
flush_print("Verify Var(f) = (2N+5)/[18N(N-1)] for sprinkled causets")
flush_print("=" * 80)

of_sizes = [500, 1000, 2000, 5000, 10000]

for N in of_sizes:
    t0 = time.time()
    n_samples = max(10, min(30, int(50 / max(0.1, (N / 1000)**2 * 0.07))))
    fracs = []
    for trial in range(n_samples):
        r = np.random.default_rng(1000 + trial)
        coords, order_sparse = sprinkle_sparse(N, dim=2, rng_local=r)
        f = order_sparse.nnz / (N * (N - 1) / 2)
        fracs.append(f)

    mean_f = np.mean(fracs)
    var_f = np.var(fracs, ddof=1)
    predicted_var = (2 * N + 5) / (18 * N * (N - 1))
    ratio = var_f / predicted_var if predicted_var > 0 else float('inf')

    elapsed = time.time() - t0
    flush_print(f"  N={N:6d}: E[f]={mean_f:.6f}, Var(f)={var_f:.2e}, "
                f"pred={predicted_var:.2e}, ratio={ratio:.3f}, t={elapsed:.1f}s")

    if elapsed > 55:
        flush_print(f"  (skipping larger)")
        break

flush_print()
flush_print(f"  Theory: Var(f) = (2N+5)/[18N(N-1)] for random 2-orders.")
flush_print(f"  Sprinkled causets are NOT 2-orders — they're 2D Minkowski sprinklings.")
flush_print(f"  Ratio != 1.0 tells us about geometric variance vs combinatorial.")
flush_print()


# ============================================================
# IDEA 624: CHAIN LENGTH AT N=10000-50000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 624: CHAIN LENGTH AT N=10000-50000")
flush_print("Verify longest chain ~ c * sqrt(N). What is c?")
flush_print("=" * 80)

chain_sizes = [500, 1000, 2000, 5000, 10000, 20000]
chain_data = []

for N in chain_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    t_spr = time.time() - t0

    chain = longest_chain_sparse(order_sparse, N)
    ratio = chain / np.sqrt(N)

    elapsed = time.time() - t0
    chain_data.append((N, chain, ratio))
    flush_print(f"  N={N:6d}: chain = {chain:6d}, chain/sqrt(N) = {ratio:.4f}, "
                f"sprinkle={t_spr:.1f}s, total={elapsed:.1f}s")

    if elapsed > 55 and N < max(chain_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
if len(chain_data) >= 3:
    Ns_c = np.array([d[0] for d in chain_data])
    chains_c = np.array([d[1] for d in chain_data])
    slope, intercept, r, _, _ = stats.linregress(np.log(Ns_c), np.log(chains_c))
    coeff = np.exp(intercept)
    flush_print(f"  Power law fit: chain ~ {coeff:.4f} * N^{slope:.4f} (R^2={r**2:.6f})")
    flush_print(f"  Expected: chain ~ 2.0 * N^0.500")
    flush_print(f"  Constant = {coeff:.4f}, exponent = {slope:.4f}")
    ratios = [d[2] for d in chain_data]
    flush_print(f"  chain/sqrt(N) values: {[f'{r:.4f}' for r in ratios]}")
    flush_print(f"  Mean chain/sqrt(N) = {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")
flush_print()


# ============================================================
# IDEA 625: ANTICHAIN WIDTH AT N=10000-50000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 625: ANTICHAIN WIDTH AT N=10000-50000")
flush_print("Verify max antichain ~ c * sqrt(N). What is c?")
flush_print("Dilworth: AC/sqrt(N) -> sqrt(4/pi) = 1.1284 for 2D")
flush_print("=" * 80)

ac_sizes = [500, 1000, 2000, 5000, 10000, 20000]
ac_data = []

for N in ac_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    t_spr = time.time() - t0

    # Multiple greedy trials for better estimate
    best_ac = 0
    n_trials = max(3, min(10, 50000 // N))
    for trial in range(n_trials):
        ac = greedy_antichain_sparse(order_sparse, N,
                                      rng_local=np.random.default_rng(100 + trial))
        best_ac = max(best_ac, ac)

    ratio = best_ac / np.sqrt(N)
    elapsed = time.time() - t0
    ac_data.append((N, best_ac, ratio))
    flush_print(f"  N={N:6d}: AC = {best_ac:6d}, AC/sqrt(N) = {ratio:.4f}, "
                f"trials={n_trials}, sprinkle={t_spr:.1f}s, total={elapsed:.1f}s")

    if elapsed > 55 and N < max(ac_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
if len(ac_data) >= 3:
    Ns_a = np.array([d[0] for d in ac_data])
    acs = np.array([d[1] for d in ac_data])
    slope, intercept, r, _, _ = stats.linregress(np.log(Ns_a), np.log(acs))
    coeff = np.exp(intercept)
    flush_print(f"  Power law fit: AC ~ {coeff:.4f} * N^{slope:.4f} (R^2={r**2:.6f})")
    flush_print(f"  Dilworth prediction: AC ~ {np.sqrt(4/np.pi):.4f} * N^0.5000")
    flush_print(f"  Constant = {coeff:.4f}, exponent = {slope:.4f}")
    ratios_a = [d[2] for d in ac_data]
    flush_print(f"  AC/sqrt(N) values: {[f'{r:.4f}' for r in ratios_a]}")
    flush_print(f"  Note: greedy is a LOWER BOUND on true max antichain")
flush_print()


# ============================================================
# IDEA 626: HASSE DIAMETER AT N=5000-20000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 626: HASSE DIAMETER AT N=5000-20000")
flush_print("Previous claim: saturates at ~6. Really?")
flush_print("=" * 80)

diam_sizes = [200, 500, 1000, 2000, 5000, 10000]
diam_data = []

for N in diam_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    t_spr = time.time() - t0

    links_sp = compute_links_sparse(order_sparse, N)
    t_links = time.time() - t0

    diam = hasse_diameter_bfs(links_sp, N, n_sources=min(30, N))
    ratio_sqrtN = diam / np.sqrt(N)

    elapsed = time.time() - t0
    diam_data.append((N, diam, ratio_sqrtN))
    flush_print(f"  N={N:6d}: diameter = {diam:5d}, diam/sqrt(N) = {ratio_sqrtN:.4f}, "
                f"diam/ln(N) = {diam/np.log(N):.4f}, "
                f"sprinkle={t_spr:.1f}s, total={elapsed:.1f}s")

    if elapsed > 55 and N < max(diam_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
if len(diam_data) >= 3:
    Ns_d = np.array([d[0] for d in diam_data])
    diams_d = np.array([d[1] for d in diam_data])
    slope, intercept, r, _, _ = stats.linregress(np.log(Ns_d), np.log(diams_d))
    coeff = np.exp(intercept)
    flush_print(f"  Power law fit: diam ~ {coeff:.4f} * N^{slope:.4f} (R^2={r**2:.6f})")
    flush_print(f"  If saturates: exponent -> 0")
    flush_print(f"  If sqrt(N): exponent -> 0.5")
    flush_print(f"  If log(N): check diam/ln(N) constancy")
    ratios_ln = [d[1] / np.log(d[0]) for d in diam_data]
    flush_print(f"  diam/ln(N) values: {[f'{r:.3f}' for r in ratios_ln]}")
flush_print()


# ============================================================
# IDEA 627: NUMBER OF LINKS AT N=5000-20000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 627: NUMBER OF LINKS AT N=5000-20000")
flush_print("Verify n_links ~ (N+1)*H_N - 2N where H_N = harmonic number")
flush_print("=" * 80)

nlink_sizes = [200, 500, 1000, 2000, 5000, 10000]
nlink_data = []

for N in nlink_sizes:
    t0 = time.time()
    coords, order_sparse = sprinkle_cached(N, seed=42)
    links_sp = compute_links_sparse(order_sparse, N)
    n_links = links_sp.nnz
    H_N = np.sum(1.0 / np.arange(1, N + 1))
    predicted = (N + 1) * H_N - 2 * N

    elapsed = time.time() - t0
    nlink_data.append((N, n_links, predicted, H_N))
    ratio = n_links / predicted if predicted > 0 else float('inf')
    flush_print(f"  N={N:6d}: n_links = {n_links:8d}, pred = {predicted:10.1f}, "
                f"ratio = {ratio:.4f}, H_N = {H_N:.4f}, t={elapsed:.1f}s")

    if elapsed > 55 and N < max(nlink_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
# Also fit n_links ~ a * N * ln(N)
if len(nlink_data) >= 3:
    Ns_l = np.array([d[0] for d in nlink_data])
    nlinks_l = np.array([d[1] for d in nlink_data])
    a_vals = nlinks_l / (Ns_l * np.log(Ns_l))
    flush_print(f"  n_links / (N*lnN) values: {[f'{a:.4f}' for a in a_vals]}")
    flush_print(f"  Mean = {np.mean(a_vals):.4f} +/- {np.std(a_vals):.4f}")
    flush_print(f"  Note: (N+1)H_N - 2N ~ N*lnN + N*gamma - 2N + ... for large N")
    flush_print(f"  So if formula holds, a -> 1.0")
flush_print()


# ============================================================
# IDEA 628: NUMBER OF MAXIMAL ELEMENTS AT N=5000-20000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 628: NUMBER OF MAXIMAL ELEMENTS AT N=5000-20000")
flush_print("Verify n_max ~ H_N (harmonic number)")
flush_print("=" * 80)

max_sizes = [200, 500, 1000, 2000, 5000, 10000]
max_data = []

for N in max_sizes:
    t0 = time.time()
    # Average over multiple sprinklings
    n_trials = max(3, min(20, 5000 // N))
    max_counts = []
    min_counts = []
    for trial in range(n_trials):
        coords, order_sparse = sprinkle_sparse(N, dim=2,
                                                rng_local=np.random.default_rng(2000 + trial))
        n_max = count_maximal_elements_sparse(order_sparse, N)
        n_min = count_minimal_elements_sparse(order_sparse, N)
        max_counts.append(n_max)
        min_counts.append(n_min)

    mean_max = np.mean(max_counts)
    mean_min = np.mean(min_counts)
    H_N = np.sum(1.0 / np.arange(1, N + 1))

    elapsed = time.time() - t0
    max_data.append((N, mean_max, mean_min, H_N))
    flush_print(f"  N={N:6d}: E[n_max] = {mean_max:.2f}, E[n_min] = {mean_min:.2f}, "
                f"H_N = {H_N:.4f}, n_max/H_N = {mean_max/H_N:.3f}, "
                f"trials={n_trials}, t={elapsed:.1f}s")

    if elapsed > 55 and N < max(max_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
if len(max_data) >= 3:
    flush_print(f"  n_max / H_N ratios: "
                f"{[f'{d[1]/d[3]:.3f}' for d in max_data]}")
    flush_print(f"  n_max / ln(N) ratios: "
                f"{[f'{d[1]/np.log(d[0]):.3f}' for d in max_data]}")
    flush_print(f"  If n_max ~ H_N, ratio -> 1.0")
    flush_print(f"  If n_max ~ c*ln(N), ratio -> c")
    # Check power law
    Ns_m = np.array([d[0] for d in max_data])
    maxs_m = np.array([d[1] for d in max_data])
    slope, intercept, r, _, _ = stats.linregress(np.log(Ns_m), np.log(maxs_m))
    flush_print(f"  Power law: n_max ~ {np.exp(intercept):.3f} * N^{slope:.4f} (R^2={r**2:.4f})")
    flush_print(f"  Expected for H_N: exponent ~ 0 (logarithmic)")
flush_print()


# ============================================================
# IDEA 629: HASSE CONNECTIVITY P(connected) AT N=500-5000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 629: HASSE CONNECTIVITY P(connected) AT N=500-5000")
flush_print("When does the Hasse diagram become always connected?")
flush_print("=" * 80)

conn_sizes = [50, 100, 200, 300, 500, 750, 1000, 2000, 5000]

for N in conn_sizes:
    t0 = time.time()
    # Scale trials inversely with cost: ~50 at N=50, ~10 at N=2000+
    trials_this = max(10, min(50, int(50 / max(1.0, (N / 500)**2))))
    n_connected = 0
    n_components_list = []

    for trial in range(trials_this):
        coords, order_sparse = sprinkle_sparse(N, dim=2,
                                                rng_local=np.random.default_rng(3000 + trial))
        links_sp = compute_links_sparse(order_sparse, N)
        adj = (links_sp + links_sp.T).tocsr()
        n_comp, _ = connected_components(adj, directed=False)
        n_components_list.append(n_comp)
        if n_comp == 1:
            n_connected += 1

    p_conn = n_connected / trials_this
    mean_comp = np.mean(n_components_list)
    elapsed = time.time() - t0
    flush_print(f"  N={N:5d}: P(connected) = {p_conn:.4f} ({n_connected}/{trials_this}), "
                f"E[components] = {mean_comp:.3f}, t={elapsed:.1f}s")

    if elapsed > 55 and N < max(conn_sizes):
        flush_print(f"  (skipping larger)")
        break

    # If already 100% connected for 3 consecutive sizes, skip rest
    if p_conn == 1.0 and N >= 500:
        flush_print(f"  --> 100% connected at N={N}, testing a few more...")
        # Test one more larger size to confirm
        continue

flush_print()


# ============================================================
# IDEA 630: ORDERING FRACTION E[f] AT N=10000
# ============================================================
flush_print("=" * 80)
flush_print("IDEA 630: ORDERING FRACTION E[f] AT N=10000")
flush_print("For 2D sprinkled causets, does E[f] = 0.5000 exactly?")
flush_print("=" * 80)

ef_sizes = [500, 1000, 2000, 5000, 10000]

for N in ef_sizes:
    t0 = time.time()
    n_ef_samples = max(10, min(30, int(50 / max(0.1, (N / 1000)**2 * 0.07))))
    fracs = []
    for trial in range(n_ef_samples):
        r = np.random.default_rng(4000 + trial)
        coords, order_sparse = sprinkle_sparse(N, dim=2, rng_local=r)
        # ordering fraction = (# related pairs) / (N choose 2)
        # related = i prec j OR j prec i. For upper-triangular order, rels = nnz
        # But order_sparse only stores one direction (i < j in time ordering, then
        # causal relation). Actually sprinkle_sparse stores order[i,j] for i < j.
        # So total related pairs = nnz(order_sparse).
        # Ordering fraction = nnz / (N*(N-1)/2)
        n_rels = order_sparse.nnz
        f = n_rels / (N * (N - 1) / 2)
        fracs.append(f)

    mean_f = np.mean(fracs)
    std_f = np.std(fracs, ddof=1)
    sem_f = std_f / np.sqrt(n_ef_samples)

    elapsed = time.time() - t0
    # Z-test against 0.5
    z = (mean_f - 0.5) / sem_f if sem_f > 0 else 0
    flush_print(f"  N={N:6d}: E[f] = {mean_f:.8f} +/- {sem_f:.8f}, "
                f"z_vs_0.5 = {z:+.3f}, t={elapsed:.1f}s")

    if elapsed > 55 and N < max(ef_sizes):
        flush_print(f"  (skipping larger)")
        break

flush_print()
flush_print(f"  For random 2-orders (combinatorial): E[f] = 1/3 exactly.")
flush_print(f"  For 2D Minkowski sprinklings: E[f] depends on geometry.")
flush_print(f"  Question: is there an exact value for 2D diamonds?")
flush_print()


# ============================================================
# GRAND SUMMARY
# ============================================================
total_time = time.time() - OVERALL_START
flush_print("=" * 80)
flush_print("GRAND SUMMARY — Ideas 621-630: LARGE-N VERIFICATION")
flush_print("=" * 80)
flush_print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
flush_print()

flush_print("621. INTERVAL ENTROPY:")
flush_print(f"     H values: {[f'{h:.4f}' for h in int_ent_vals]}")
if len(int_ent_vals) >= 3:
    flush_print(f"     Last value: H = {int_ent_vals[-1]:.6f}")
    flush_print(f"     Converging: {'YES' if diffs[-1] < diffs[0] else 'UNCLEAR'}")

flush_print()
flush_print("622. LINK FRACTION:")
c_vals = [d[4] for d in link_data]
flush_print(f"     c = link_frac * N / ln(N): {[f'{c:.3f}' for c in c_vals]}")
flush_print(f"     Verdict: c -> {np.mean(c_vals[-3:]):.3f} "
            f"({'~4' if abs(np.mean(c_vals[-3:])-4) < 0.5 else '~pi' if abs(np.mean(c_vals[-3:])-np.pi) < 0.3 else 'OTHER'})")

flush_print()
flush_print("623. ORDERING FRACTION VARIANCE:")
flush_print(f"     (2N+5)/[18N(N-1)] formula vs sprinkled causets")

flush_print()
flush_print("624. CHAIN LENGTH:")
if len(chain_data) >= 3:
    ratios_c = [d[2] for d in chain_data]
    flush_print(f"     chain/sqrt(N): {[f'{r:.4f}' for r in ratios_c]}")
    flush_print(f"     Constant = {np.mean(ratios_c[-3:]):.4f}")

flush_print()
flush_print("625. ANTICHAIN WIDTH:")
if len(ac_data) >= 3:
    ratios_ac = [d[2] for d in ac_data]
    flush_print(f"     AC/sqrt(N): {[f'{r:.4f}' for r in ratios_ac]}")
    flush_print(f"     Dilworth prediction: {np.sqrt(4/np.pi):.4f}")

flush_print()
flush_print("626. HASSE DIAMETER:")
if len(diam_data) >= 3:
    flush_print(f"     diameters: {[d[1] for d in diam_data]}")
    flush_print(f"     Saturates at ~6? {'YES' if max(d[1] for d in diam_data) < 10 else 'NO — grows with N'}")

flush_print()
flush_print("627. NUMBER OF LINKS:")
if len(nlink_data) >= 3:
    flush_print(f"     n_links / [(N+1)*H_N - 2N]: "
                f"{[f'{d[1]/d[2]:.4f}' for d in nlink_data]}")

flush_print()
flush_print("628. MAXIMAL ELEMENTS:")
if len(max_data) >= 3:
    flush_print(f"     n_max / H_N: {[f'{d[1]/d[3]:.3f}' for d in max_data]}")

flush_print()
flush_print("629. HASSE CONNECTIVITY:")
flush_print(f"     (see detailed results above)")

flush_print()
flush_print("630. ORDERING FRACTION E[f]:")
flush_print(f"     (see detailed results above)")

flush_print()
flush_print(f"\nDone. Total time = {total_time:.0f}s")
