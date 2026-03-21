"""
Experiment 77, Round 20: WILD CARD ROUND 2 — Ideas 291-300

10 ideas NOBODY in the field has considered. Pure conceptual exploration.

Ideas:
291. CAUSAL SET AS A GAME: Each element is a player, payoff = # descendants.
     Nash equilibrium = antichain. Price of anarchy = dimension?
292. KOLMOGOROV COMPLEXITY of the 2-order: bits needed to describe (u,v)
     given the causal set. Approximated via compression.
293. QUANTUM COMPLEXITY of the SJ state: how many gates to prepare it?
     Approximated via entanglement structure.
294. TOPOLOGICAL DATA ANALYSIS of 2-order space: persistent homology of
     the configuration space of permutation pairs.
295. FRACTAL DIMENSION of the Hasse diagram (box-counting on the link graph).
296. CAUSAL SET AS BOOLEAN NETWORK: synchronous dynamics on the DAG.
     Is it chaotic? (Derrida annealed approximation.)
297. RENORMALIZATION GROUP on the Hasse diagram: coarse-grain by merging
     clusters, track how observables flow.
298. CAUSAL MATRIX AS MARKOV CHAIN: C/sum(C) as transition matrix.
     Mixing time = geometry?
299. INFORMATION FLOW: max-flow min-cut between past and future boundaries.
     Connection to entanglement entropy?
300. EMERGENT SPATIAL TOPOLOGY: detect topology of embedding space
     (cylinder vs diamond vs torus) from causal structure alone.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, linalg
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.dimension import myrheim_meyer, _ordering_fraction_theory, _invert_ordering_fraction
import zlib
import time

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def make_2order_causet(N, rng):
    to = TwoOrder(N, rng=rng)
    return to, to.to_causet()

def random_dag(N, density, rng):
    cs = FastCausalSet(N)
    mask = rng.random((N, N)) < density
    cs.order = np.triu(mask, k=1)
    order_int = cs.order.astype(np.int32)
    changed = True
    while changed:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs

def sj_eigenvalues(cs):
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    return np.linalg.eigvalsh(H).real

def level_spacing_ratio(evals):
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    r_min = np.minimum(spacings[:-1], spacings[1:])
    r_max = np.maximum(spacings[:-1], spacings[1:])
    return np.mean(r_min / r_max)


# ================================================================
print("=" * 78)
print("IDEA 291: CAUSAL SET AS A GAME — PRICE OF ANARCHY")
print("Players = elements. Payoff(i) = # descendants. Nash eq = antichain.")
print("Price of Anarchy = (sum of payoffs at social opt) / (sum at worst NE)")
print("=" * 78)

# In a DAG, if each player picks "stay active", payoff = # descendants.
# A Nash equilibrium: no player can unilaterally increase descendants by
# "deactivating" (removal). An antichain IS a stable coalition.
#
# Social welfare = total descendants of active set.
# Best response: include yourself if you have descendants.
# Worst NE: maximal antichain (no one can improve by joining).
# Social optimum: include elements that maximize total descendant coverage.
#
# Price of Anarchy = social_opt / worst_NE_welfare.
# Question: does this ratio depend on dimension?

def descendant_count(cs):
    """For each element, count how many elements it precedes."""
    return cs.order.astype(int).sum(axis=1)

def ancestor_count(cs):
    """For each element, count how many elements precede it."""
    return cs.order.astype(int).sum(axis=0)

def social_welfare(cs, active_set):
    """Total distinct descendants of the active set."""
    N = cs.n
    covered = np.zeros(N, dtype=bool)
    for i in active_set:
        covered |= cs.order[i]
    return int(covered.sum())

def greedy_max_antichain(cs):
    """Find a maximal antichain greedily (not necessarily maximum)."""
    N = cs.n
    antichain = []
    available = set(range(N))
    # Sort by fewest relations (most "spacelike")
    total_relations = cs.order.astype(int).sum(axis=1) + cs.order.astype(int).sum(axis=0)
    order = np.argsort(total_relations)
    for i in order:
        if i not in available:
            continue
        # Check if i is spacelike to all current antichain members
        ok = True
        for j in antichain:
            if cs.order[i, j] or cs.order[j, i]:
                ok = False
                break
        if ok:
            antichain.append(i)
            # Remove all causally related elements
            for k in list(available):
                if cs.order[i, k] or cs.order[k, i]:
                    available.discard(k)
    return antichain

def price_of_anarchy_game(cs):
    """
    Social optimum: all elements active (every descendant covered).
    Worst NE (antichain): antichain members' descendants.
    PoA = social_opt / worst_NE.
    """
    N = cs.n
    # Social optimum: all active
    social_opt = social_welfare(cs, list(range(N)))
    if social_opt == 0:
        return float('nan')

    # Antichain NE
    antichain = greedy_max_antichain(cs)
    ne_welfare = social_welfare(cs, antichain)
    if ne_welfare == 0:
        return float('inf')

    return social_opt / ne_welfare

print("\n--- Price of Anarchy vs dimension (sprinkled causets) ---")
game_results = {}
for d in [2, 3, 4]:
    N = 60
    n_trials = 20
    poas = []
    for _ in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        poa = price_of_anarchy_game(cs)
        if np.isfinite(poa):
            poas.append(poa)
    mean_poa = np.mean(poas) if poas else float('nan')
    std_poa = np.std(poas) if poas else float('nan')
    game_results[d] = (mean_poa, std_poa)
    print(f"  d={d}: PoA = {mean_poa:.3f} ± {std_poa:.3f} (n={len(poas)})")

# Compare with random DAG
poas_rand = []
for _ in range(20):
    cs = random_dag(60, 0.3, rng)
    poa = price_of_anarchy_game(cs)
    if np.isfinite(poa):
        poas_rand.append(poa)
if poas_rand:
    print(f"  Random DAG: PoA = {np.mean(poas_rand):.3f} ± {np.std(poas_rand):.3f}")

print("\n  ASSESSMENT: Is Price of Anarchy dimension-dependent?")
if len(game_results) >= 2:
    vals = [game_results[d][0] for d in sorted(game_results)]
    spread = max(vals) - min(vals)
    trend = "increasing" if vals[-1] > vals[0] else "decreasing"
    print(f"    PoA trend with dimension: {trend}")
    print(f"    Spread across d=2,3,4: {spread:.3f}")
    if spread > 0.2:
        print(f"    >>> DIMENSION-DEPENDENT! Price of anarchy encodes geometry.")
    else:
        print(f"    >>> Weak or no dimension dependence.")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 292: KOLMOGOROV COMPLEXITY OF THE 2-ORDER")
print("How many bits to describe (u,v) given the causal set?")
print("Approximated via gzip compression of different representations.")
print("=" * 78)

def kolmogorov_approx(data_bytes):
    """Approximate Kolmogorov complexity via gzip compression."""
    compressed = zlib.compress(data_bytes, level=9)
    return len(compressed) * 8  # bits

def causet_complexity_matrix(cs):
    """Complexity of the causal matrix representation."""
    return kolmogorov_approx(cs.order.astype(np.uint8).tobytes())

def causet_complexity_2order(u, v):
    """Complexity of the 2-order representation."""
    data = np.concatenate([u, v]).astype(np.int32).tobytes()
    return kolmogorov_approx(data)

def causet_complexity_links(cs):
    """Complexity of just the link (Hasse) representation."""
    links = cs.link_matrix()
    return kolmogorov_approx(links.astype(np.uint8).tobytes())

print("\n--- Comparing representations: matrix vs 2-order vs links ---")
for N in [20, 40, 60, 80]:
    n_trials = 15
    bits_matrix, bits_2order, bits_links = [], [], []
    for _ in range(n_trials):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        bits_matrix.append(causet_complexity_matrix(cs))
        bits_2order.append(causet_complexity_2order(to.u, to.v))
        bits_links.append(causet_complexity_links(cs))

    # Theoretical minimum: 2-order needs 2*N*log2(N) bits (two permutations)
    theory_bits = 2 * N * np.log2(N)

    print(f"  N={N:3d}: matrix={np.mean(bits_matrix):7.0f} bits, "
          f"2-order={np.mean(bits_2order):7.0f} bits, "
          f"links={np.mean(bits_links):7.0f} bits, "
          f"theory_min={theory_bits:.0f} bits")

# Does complexity scale differently for manifold-like vs random?
print("\n--- Complexity scaling: sprinkled vs random 2-orders ---")
for N in [20, 40, 60, 80]:
    # Sprinkled (manifold-like)
    sprinkle_bits = []
    for _ in range(10):
        cs, _ = sprinkle_fast(N, dim=2, rng=rng)
        sprinkle_bits.append(causet_complexity_matrix(cs))

    # Random 2-order
    random_bits = []
    for _ in range(10):
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
        random_bits.append(causet_complexity_matrix(cs))

    ratio = np.mean(sprinkle_bits) / np.mean(random_bits) if np.mean(random_bits) > 0 else float('nan')
    print(f"  N={N:3d}: sprinkled={np.mean(sprinkle_bits):7.0f}, "
          f"random_2order={np.mean(random_bits):7.0f}, ratio={ratio:.3f}")

print("\n  ASSESSMENT: Is the 2-order more compressible than the matrix?")
print("  If ratio < 1: sprinkled causets are more structured (lower complexity).")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 293: QUANTUM COMPLEXITY OF THE SJ STATE")
print("How many quantum gates to prepare the SJ vacuum? Use entanglement")
print("structure as a proxy (circuit depth ~ entanglement entropy).")
print("=" * 78)

def sj_state_complexity(cs):
    """
    Approximate quantum complexity of preparing the SJ vacuum state.

    The SJ vacuum is the positive-frequency part of the Pauli-Jordan operator.
    Its complexity is related to the entanglement structure:
    - Compute the SJ 2-point function W
    - Bipartition the causet into past/future halves
    - Entanglement entropy of the bipartition ~ circuit depth
    """
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)  # Pauli-Jordan

    # Eigendecompose
    evals, evecs = np.linalg.eigh(1j * Delta)
    evals = evals.real

    # SJ prescription: keep positive eigenvalues
    pos_mask = evals > 1e-10
    n_pos = pos_mask.sum()

    if n_pos == 0:
        return {'n_modes': 0, 'entropy': 0.0, 'complexity_bound': 0.0}

    # SJ Wightman function
    W = evecs[:, pos_mask] @ np.diag(evals[pos_mask]) @ evecs[:, pos_mask].conj().T

    # Bipartition: first half vs second half (past vs future)
    mid = N // 2
    W_A = W[:mid, :mid]

    # Eigenvalues of reduced state
    w_evals = np.linalg.eigvalsh(W_A.real)
    w_evals = np.clip(w_evals, 1e-15, 1 - 1e-15)

    # von Neumann entropy
    S = -np.sum(w_evals * np.log2(np.abs(w_evals) + 1e-300) +
                (1 - w_evals) * np.log2(np.abs(1 - w_evals) + 1e-300))

    # Circuit complexity lower bound: Omega(S) gates needed
    # Upper bound: O(N * S) for generic state preparation
    complexity_lower = abs(S)
    complexity_upper = N * abs(S)

    return {
        'n_modes': int(n_pos),
        'entropy': abs(S),
        'complexity_lower': complexity_lower,
        'complexity_upper': complexity_upper,
    }

print("\n--- SJ state complexity vs N and dimension ---")
for d in [2, 3]:
    print(f"\n  d = {d}:")
    for N in [20, 30, 40, 50]:
        complexities = []
        entropies = []
        for _ in range(10):
            cs, _ = sprinkle_fast(N, dim=d, rng=rng)
            result = sj_state_complexity(cs)
            complexities.append(result['complexity_lower'])
            entropies.append(result['entropy'])
        print(f"    N={N:3d}: S_ent = {np.mean(entropies):.3f} ± {np.std(entropies):.3f}, "
              f"complexity_lower = {np.mean(complexities):.1f}")

# Fit: does complexity scale as N^alpha? What's alpha?
print("\n--- Complexity scaling exponent ---")
for d in [2, 3]:
    Ns = [20, 30, 40, 50, 60]
    mean_S = []
    for N in Ns:
        entropies = []
        for _ in range(8):
            cs, _ = sprinkle_fast(N, dim=d, rng=rng)
            result = sj_state_complexity(cs)
            entropies.append(result['entropy'])
        mean_S.append(np.mean(entropies))

    try:
        def power_law(x, a, b):
            return a * x ** b
        popt, _ = curve_fit(power_law, Ns, mean_S, p0=[1.0, 0.5])
        print(f"  d={d}: S ~ N^{popt[1]:.3f} (prefactor={popt[0]:.4f})")
        if 0.4 < popt[1] < 0.6:
            print(f"    >>> Sqrt(N) scaling — area law in 2d!")
        elif 0.9 < popt[1] < 1.1:
            print(f"    >>> Volume law!")
        elif popt[1] < 0.3:
            print(f"    >>> Sub-area law — highly structured state")
    except Exception as e:
        print(f"  d={d}: fit failed: {e}")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 294: TOPOLOGICAL DATA ANALYSIS — PERSISTENT HOMOLOGY OF 2-ORDERS")
print("Sample points in permutation-pair space, compute distance matrix,")
print("look at persistent homology of the configuration space.")
print("=" * 78)

# The space of 2-orders for N elements is (S_N)^2 / symmetry.
# We can sample random 2-orders, compute a distance between them
# (e.g., Kendall tau distance on permutations), and then do
# persistence analysis on the point cloud.

def kendall_tau_distance(p1, p2):
    """Normalized Kendall tau distance between two permutations."""
    n = len(p1)
    inversions = 0
    # Compose: sigma = p2 * p1^{-1}
    inv_p1 = np.argsort(p1)
    sigma = p2[inv_p1]
    for i in range(n):
        for j in range(i + 1, n):
            if sigma[i] > sigma[j]:
                inversions += 1
    return inversions / (n * (n - 1) / 2)

def two_order_distance(to1, to2):
    """Distance between two 2-orders: average Kendall tau of u and v."""
    return (kendall_tau_distance(to1.u, to2.u) + kendall_tau_distance(to1.v, to2.v)) / 2

# Sample 2-orders, compute distance matrix, analyze topology via Vietoris-Rips
# (simplified: use eigenvalues of distance matrix as a proxy for topology)
N_tda = 10  # Small N for tractability
n_samples = 50

print(f"\n--- Sampling {n_samples} random 2-orders on N={N_tda} ---")
t0 = time.time()
samples = [TwoOrder(N_tda, rng=rng) for _ in range(n_samples)]
dist_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(i + 1, n_samples):
        d = two_order_distance(samples[i], samples[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d
dt = time.time() - t0
print(f"  Distance matrix computed in {dt:.1f}s")

# Basic topological analysis via spectral properties
print(f"\n  Distance statistics:")
dists = dist_matrix[np.triu_indices(n_samples, k=1)]
print(f"    Mean distance: {np.mean(dists):.4f}")
print(f"    Std distance:  {np.std(dists):.4f}")
print(f"    Min distance:  {np.min(dists):.4f}")
print(f"    Max distance:  {np.max(dists):.4f}")

# Eigenvalues of the distance matrix (spectral geometry of config space)
evals_dist = np.linalg.eigvalsh(dist_matrix)
evals_dist = np.sort(evals_dist)[::-1]
print(f"\n  Top 10 eigenvalues of distance matrix:")
print(f"    {evals_dist[:10]}")

# Effective dimension of the point cloud (participation ratio)
evals_pos = evals_dist[evals_dist > 1e-10]
participation = (np.sum(evals_pos))**2 / np.sum(evals_pos**2)
print(f"\n  Effective dimension (participation ratio): {participation:.2f}")
print(f"  (cf. S_N has dimension N-1, so (S_N)^2 has dimension 2(N-1) = {2*(N_tda-1)})")

# Simplified Betti number detection: count connected components at various thresholds
print(f"\n  Simplified persistent homology (connected components):")
thresholds = np.linspace(0, np.max(dists), 20)
for thresh in thresholds[1::4]:
    adj = dist_matrix < thresh
    np.fill_diagonal(adj, True)
    # BFS to count components
    visited = set()
    components = 0
    for start in range(n_samples):
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nbr in range(n_samples):
                if adj[node, nbr] and nbr not in visited:
                    queue.append(nbr)
    print(f"    threshold={thresh:.4f}: {components} components (beta_0)")

print("\n  ASSESSMENT: What shape is the space of 2-orders?")
print(f"    Effective dimension {participation:.1f} vs theoretical {2*(N_tda-1)}")
if participation < N_tda:
    print("    >>> Configuration space is LOW-dimensional — strong correlations!")
else:
    print("    >>> High-dimensional, consistent with (S_N)^2 structure.")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 295: FRACTAL DIMENSION OF THE HASSE DIAGRAM")
print("Box-counting dimension of the link graph embedded via spectral coordinates.")
print("=" * 78)

def spectral_embedding(cs, n_dims=3):
    """Embed the Hasse diagram using eigenvectors of the graph Laplacian."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    evals, evecs = np.linalg.eigh(L)
    # Use eigenvectors 1..n_dims (skip the trivial zero eigenvector)
    coords = evecs[:, 1:n_dims+1]
    return coords

def box_counting_dimension(points, n_scales=10):
    """Estimate fractal dimension via box-counting."""
    if len(points) < 3:
        return float('nan')

    # Normalize to [0,1]^d
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0
    normalized = (points - mins) / ranges

    # Box sizes from 1/2 to 1/2^max_power
    max_power = min(6, int(np.log2(len(points))))
    if max_power < 2:
        return float('nan')

    log_eps = []
    log_N = []
    for power in range(1, max_power + 1):
        eps = 1.0 / (2 ** power)
        # Count occupied boxes
        box_indices = tuple((normalized / eps).astype(int).T)
        n_boxes = len(set(zip(*box_indices)))
        log_eps.append(np.log(eps))
        log_N.append(np.log(n_boxes))

    if len(log_eps) < 2:
        return float('nan')

    # Fit: log(N) = -D * log(eps) + const
    slope, _, _, _, _ = stats.linregress(log_eps, log_N)
    return -slope

print("\n--- Fractal dimension of Hasse diagram vs manifold dimension ---")
for d in [2, 3, 4]:
    N = 80
    n_trials = 12
    frac_dims = []
    for _ in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        try:
            coords = spectral_embedding(cs, n_dims=3)
            fd = box_counting_dimension(coords)
            if np.isfinite(fd):
                frac_dims.append(fd)
        except:
            pass
    if frac_dims:
        print(f"  d={d}: fractal dim = {np.mean(frac_dims):.3f} ± {np.std(frac_dims):.3f}")
    else:
        print(f"  d={d}: no valid measurements")

# Random DAG comparison
frac_dims_rand = []
for _ in range(12):
    cs = random_dag(80, 0.15, rng)
    try:
        coords = spectral_embedding(cs, n_dims=3)
        fd = box_counting_dimension(coords)
        if np.isfinite(fd):
            frac_dims_rand.append(fd)
    except:
        pass
if frac_dims_rand:
    print(f"  Random DAG: fractal dim = {np.mean(frac_dims_rand):.3f} ± {np.std(frac_dims_rand):.3f}")

print("\n  ASSESSMENT: Does fractal dimension encode manifold dimension?")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 296: CAUSAL SET AS BOOLEAN NETWORK")
print("Synchronous Boolean dynamics on the DAG. Is it chaotic?")
print("Derrida coefficient (sensitivity to perturbation) as order parameter.")
print("=" * 78)

def boolean_network_dynamics(cs, n_steps=50, n_trials=20):
    """
    Run synchronous Boolean dynamics on the causal order.
    State: sigma_i in {0,1}. Update: sigma_i(t+1) = MAJORITY of predecessors.
    If no predecessors, sigma stays fixed.

    Measure Derrida coefficient: average Hamming distance between
    trajectories that start 1 bit apart.
    """
    N = cs.n
    predecessors = [np.where(cs.order[:, i])[0] for i in range(N)]

    derrida_coeffs = []
    for trial in range(n_trials):
        # Random initial state
        state1 = rng.integers(0, 2, size=N)

        # Perturbed state (flip one random bit)
        state2 = state1.copy()
        flip_idx = rng.integers(0, N)
        state2[flip_idx] = 1 - state2[flip_idx]

        # Evolve both
        for step in range(n_steps):
            new1 = np.zeros(N, dtype=int)
            new2 = np.zeros(N, dtype=int)
            for i in range(N):
                preds = predecessors[i]
                if len(preds) == 0:
                    new1[i] = state1[i]
                    new2[i] = state2[i]
                else:
                    new1[i] = int(state1[preds].sum() > len(preds) / 2)
                    new2[i] = int(state2[preds].sum() > len(preds) / 2)
            state1 = new1
            state2 = new2

        hamming = np.sum(state1 != state2)
        derrida_coeffs.append(hamming / N)

    return np.mean(derrida_coeffs), np.std(derrida_coeffs)

print("\n--- Derrida coefficient (chaos measure) vs dimension ---")
for d in [2, 3, 4]:
    N = 50
    n_trials_outer = 10
    coeffs = []
    for _ in range(n_trials_outer):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        mu, _ = boolean_network_dynamics(cs, n_steps=30, n_trials=15)
        coeffs.append(mu)
    print(f"  d={d}: Derrida coeff = {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")
    if np.mean(coeffs) < 0.01:
        print(f"    >>> ORDERED phase (perturbations die)")
    elif np.mean(coeffs) > 0.1:
        print(f"    >>> CHAOTIC phase (perturbations spread)")
    else:
        print(f"    >>> Near EDGE OF CHAOS")

# Random DAG comparison
coeffs_rand = []
for _ in range(10):
    cs = random_dag(50, 0.2, rng)
    mu, _ = boolean_network_dynamics(cs, n_steps=30, n_trials=15)
    coeffs_rand.append(mu)
print(f"  Random DAG: Derrida coeff = {np.mean(coeffs_rand):.4f} ± {np.std(coeffs_rand):.4f}")

print("\n  ASSESSMENT: Are manifold-like causets at the edge of chaos?")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 297: RENORMALIZATION GROUP ON THE HASSE DIAGRAM")
print("Coarse-grain by merging link-connected clusters. Track observable flow.")
print("=" * 78)

def coarse_grain(cs, n_levels=3):
    """
    Coarse-grain the causal set by merging link-connected pairs.
    At each level, merge pairs of linked elements into single nodes.
    Track ordering fraction and density at each scale.
    """
    results = []
    current_order = cs.order.copy().astype(float)
    N_current = cs.n

    results.append({
        'N': N_current,
        'f': np.sum(current_order) / max(1, N_current * (N_current - 1) / 2),
        'density': np.sum(current_order > 0) / max(1, N_current * N_current),
    })

    for level in range(n_levels):
        if N_current < 4:
            break

        # Find links
        order_bool = current_order > 0.5
        path2 = order_bool.astype(np.int32) @ order_bool.astype(np.int32)
        links = order_bool & (path2 == 0)

        # Greedy pairing: match linked pairs
        merged = set()
        merge_map = {}  # old index -> new index
        new_idx = 0
        pairs = list(zip(*np.where(links)))

        for i, j in pairs:
            if i not in merged and j not in merged:
                merge_map[i] = new_idx
                merge_map[j] = new_idx
                merged.add(i)
                merged.add(j)
                new_idx += 1

        # Unmerged elements get their own new index
        for i in range(N_current):
            if i not in merged:
                merge_map[i] = new_idx
                new_idx += 1

        N_new = new_idx
        if N_new >= N_current or N_new < 3:
            break

        # Build coarse-grained order
        new_order = np.zeros((N_new, N_new))
        for i in range(N_current):
            for j in range(N_current):
                if current_order[i, j] > 0.5:
                    ni, nj = merge_map[i], merge_map[j]
                    if ni != nj:
                        new_order[ni, nj] = 1.0

        current_order = new_order
        N_current = N_new

        f = np.sum(current_order) / max(1, N_current * (N_current - 1) / 2)
        density = np.sum(current_order > 0) / max(1, N_current * N_current)
        results.append({'N': N_current, 'f': f, 'density': density})

    return results

print("\n--- RG flow of ordering fraction ---")
for d in [2, 3]:
    print(f"\n  d = {d}:")
    N = 80
    n_trials = 10
    all_flows = []
    for _ in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        flow = coarse_grain(cs, n_levels=5)
        all_flows.append(flow)

    # Average flow
    max_len = max(len(f) for f in all_flows)
    for level in range(min(max_len, 5)):
        fvals = [f[level]['f'] for f in all_flows if level < len(f)]
        Nvals = [f[level]['N'] for f in all_flows if level < len(f)]
        if fvals:
            print(f"    Level {level}: N={np.mean(Nvals):.0f}, f={np.mean(fvals):.4f} ± {np.std(fvals):.4f}")

# Check: does f flow to a fixed point?
print("\n  ASSESSMENT: Does the ordering fraction flow to a fixed point?")
print("  If f(level) → const, the RG has an IR fixed point (geometric phase).")
print("  If f → 0 or 1, it flows to a trivial fixed point.")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 298: CAUSAL MATRIX AS MARKOV CHAIN")
print("Transition matrix T = C / row_sum(C). Mixing time = geometry?")
print("=" * 78)

def markov_chain_analysis(cs):
    """
    Interpret the causal matrix as a Markov chain.
    T[i,j] = C[i,j] / sum_k C[i,k] (row-normalized).
    Elements with no successors are absorbing states.

    Measure: spectral gap, mixing time, stationary distribution entropy.
    """
    N = cs.n
    C = cs.order.astype(float)
    row_sums = C.sum(axis=1)

    # Handle absorbing states: elements with no successors
    absorbing = row_sums < 0.5
    n_absorbing = absorbing.sum()

    # For non-absorbing, normalize rows
    T = np.zeros((N, N))
    for i in range(N):
        if row_sums[i] > 0:
            T[i] = C[i] / row_sums[i]
        else:
            T[i, i] = 1.0  # absorbing state

    # Eigenvalues of T (for non-absorbing subspace)
    non_abs = ~absorbing
    n_non_abs = non_abs.sum()

    if n_non_abs < 3:
        return {'spectral_gap': float('nan'), 'mixing_time': float('nan'),
                'n_absorbing': int(n_absorbing), 'stationary_entropy': float('nan')}

    T_sub = T[np.ix_(non_abs, non_abs)]
    # Re-normalize after subsetting
    row_sums_sub = T_sub.sum(axis=1)
    mask_nonzero = row_sums_sub > 1e-10
    T_sub[mask_nonzero] = T_sub[mask_nonzero] / row_sums_sub[mask_nonzero, None]

    try:
        evals_T = np.linalg.eigvals(T_sub)
        evals_abs = np.abs(evals_T)
        evals_abs_sorted = np.sort(evals_abs)[::-1]

        if len(evals_abs_sorted) > 1:
            spectral_gap = 1.0 - evals_abs_sorted[1]
            mixing_time = 1.0 / max(spectral_gap, 1e-10) if spectral_gap > 0 else float('inf')
        else:
            spectral_gap = float('nan')
            mixing_time = float('nan')
    except:
        spectral_gap = float('nan')
        mixing_time = float('nan')

    # Stationary distribution entropy (from left eigenvector)
    try:
        evals_l, evecs_l = np.linalg.eig(T_sub.T)
        idx = np.argmin(np.abs(evals_l - 1.0))
        pi = np.abs(evecs_l[:, idx])
        pi = pi / pi.sum()
        stat_entropy = -np.sum(pi * np.log(pi + 1e-300))
    except:
        stat_entropy = float('nan')

    return {
        'spectral_gap': spectral_gap,
        'mixing_time': mixing_time,
        'n_absorbing': int(n_absorbing),
        'stationary_entropy': stat_entropy,
    }

print("\n--- Markov chain properties vs dimension ---")
for d in [2, 3, 4]:
    N = 60
    n_trials = 15
    gaps, mix_times, entropies = [], [], []
    for _ in range(n_trials):
        cs, _ = sprinkle_fast(N, dim=d, rng=rng)
        result = markov_chain_analysis(cs)
        if np.isfinite(result['spectral_gap']):
            gaps.append(result['spectral_gap'])
        if np.isfinite(result['mixing_time']) and result['mixing_time'] < 1e6:
            mix_times.append(result['mixing_time'])
        if np.isfinite(result['stationary_entropy']):
            entropies.append(result['stationary_entropy'])

    print(f"  d={d}: spectral_gap = {np.mean(gaps):.4f} ± {np.std(gaps):.4f}, "
          f"mixing_time = {np.mean(mix_times):.1f} ± {np.std(mix_times):.1f}, "
          f"stat_entropy = {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")

# Does mixing time scale with N?
print("\n--- Mixing time scaling with N (d=2) ---")
for N in [20, 30, 40, 60, 80]:
    mix_times = []
    for _ in range(10):
        cs, _ = sprinkle_fast(N, dim=2, rng=rng)
        result = markov_chain_analysis(cs)
        if np.isfinite(result['mixing_time']) and result['mixing_time'] < 1e6:
            mix_times.append(result['mixing_time'])
    if mix_times:
        print(f"  N={N:3d}: mixing_time = {np.mean(mix_times):.2f} ± {np.std(mix_times):.2f}")

print("\n  ASSESSMENT: Does mixing time encode geometric distance (proper time)?")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 299: INFORMATION FLOW — MAX-FLOW MIN-CUT")
print("Max flow from past boundary to future boundary. ~ Entanglement?")
print("=" * 78)

def maxflow_past_future(cs):
    """
    Compute max flow from past boundary (minimal elements) to
    future boundary (maximal elements) through the Hasse diagram.

    Each link has capacity 1. This gives the min-cut between
    past and future, which in holography ~ entanglement entropy.
    """
    N = cs.n
    links = cs.link_matrix()

    # Find minimal elements (no predecessors)
    has_predecessor = cs.order.astype(int).sum(axis=0) > 0
    minimal = np.where(~has_predecessor)[0]

    # Find maximal elements (no successors)
    has_successor = cs.order.astype(int).sum(axis=1) > 0
    maximal = np.where(~has_successor)[0]

    if len(minimal) == 0 or len(maximal) == 0:
        return {'max_flow': 0, 'n_minimal': 0, 'n_maximal': 0}

    # Build flow network: add super-source and super-sink
    # Node indices: 0..N-1 = causal set, N = source, N+1 = sink
    n_nodes = N + 2
    source = N
    sink = N + 1

    # Build capacity matrix as sparse
    row_indices = []
    col_indices = []
    capacities = []

    # Source to minimal elements
    for m in minimal:
        row_indices.append(source)
        col_indices.append(m)
        capacities.append(N)  # unlimited from source

    # Maximal to sink
    for m in maximal:
        row_indices.append(m)
        col_indices.append(sink)
        capacities.append(N)

    # Links with capacity 1
    link_pairs = np.where(links)
    for i, j in zip(link_pairs[0], link_pairs[1]):
        row_indices.append(i)
        col_indices.append(j)
        capacities.append(1)

    cap_matrix = csr_matrix((capacities, (row_indices, col_indices)),
                            shape=(n_nodes, n_nodes))

    try:
        result = maximum_flow(cap_matrix, source, sink)
        flow_value = result.flow_value
    except:
        flow_value = 0

    return {
        'max_flow': int(flow_value),
        'n_minimal': len(minimal),
        'n_maximal': len(maximal),
        'min_cut': int(flow_value),  # max-flow = min-cut
    }

print("\n--- Max-flow (min-cut) vs dimension and N ---")
for d in [2, 3, 4]:
    print(f"\n  d = {d}:")
    for N in [20, 40, 60, 80]:
        flows = []
        n_trials = 10
        for _ in range(n_trials):
            cs, _ = sprinkle_fast(N, dim=d, rng=rng)
            result = maxflow_past_future(cs)
            flows.append(result['max_flow'])
        print(f"    N={N:3d}: max_flow = {np.mean(flows):.1f} ± {np.std(flows):.1f}")

# Scaling: does min-cut scale as N^{(d-2)/d} (area law)?
print("\n--- Min-cut scaling (area law test) ---")
for d in [2, 3, 4]:
    Ns = [20, 30, 40, 60, 80]
    mean_flows = []
    for N in Ns:
        flows = []
        for _ in range(8):
            cs, _ = sprinkle_fast(N, dim=d, rng=rng)
            result = maxflow_past_future(cs)
            flows.append(result['max_flow'])
        mean_flows.append(np.mean(flows))

    try:
        def power_law(x, a, b):
            return a * x ** b
        popt, _ = curve_fit(power_law, Ns, mean_flows, p0=[1.0, 0.5])
        expected_exp = (d - 2) / d if d > 2 else 0.0  # area law: N^{(d-2)/d}
        print(f"  d={d}: min_cut ~ N^{popt[1]:.3f} (area law predicts {expected_exp:.3f})")
        if abs(popt[1] - expected_exp) < 0.15:
            print(f"    >>> CONSISTENT with area law!")
        else:
            print(f"    >>> Deviates from area law (diff = {abs(popt[1] - expected_exp):.3f})")
    except Exception as e:
        print(f"  d={d}: fit failed: {e}")

# Compare with SJ entanglement entropy
print("\n--- Min-cut vs SJ entanglement entropy ---")
N_compare = 40
for d in [2, 3]:
    flows_list = []
    entropies_list = []
    for _ in range(15):
        cs, _ = sprinkle_fast(N_compare, dim=d, rng=rng)
        flow_result = maxflow_past_future(cs)
        sj_result = sj_state_complexity(cs)
        flows_list.append(flow_result['max_flow'])
        entropies_list.append(sj_result['entropy'])

    r, p = stats.pearsonr(flows_list, entropies_list) if len(flows_list) > 2 else (float('nan'), float('nan'))
    print(f"  d={d}: correlation(min_cut, S_SJ) = {r:.3f} (p={p:.4f})")
    if abs(r) > 0.5 and p < 0.05:
        print(f"    >>> SIGNIFICANT correlation! Min-cut ~ entanglement!")
    else:
        print(f"    >>> Weak or no correlation.")


# ================================================================
print("\n" + "=" * 78)
print("IDEA 300: EMERGENT SPATIAL TOPOLOGY")
print("Can we distinguish cylinder from diamond from flat space")
print("using only the causal structure?")
print("=" * 78)

def sprinkle_cylinder(N, rng):
    """
    Sprinkle into a 2D causal cylinder: [0,1] x S^1 (periodic spatial direction).
    Metric: ds^2 = -dt^2 + dx^2, x ~ x + L.
    """
    L = 1.0  # spatial circumference
    t = rng.uniform(0, 1, N)
    x = rng.uniform(0, L, N)

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    cs = FastCausalSet(N)
    for i in range(N):
        dt = t[i+1:] - t[i]
        # Periodic distance
        dx = np.abs(x[i+1:] - x[i])
        dx = np.minimum(dx, L - dx)
        cs.order[i, i+1:] = dt**2 >= dx**2

    return cs

def sprinkle_torus(N, rng):
    """
    Sprinkle into a 2D causal torus: S^1_t x S^1_x (both periodic).
    Time also periodic — creates closed timelike curves.
    To avoid CTCs, use a "cut": unwrap time.
    Actually, for a proper causal set, use a spatial torus with flat time:
    t in [0,1], x in S^1.
    """
    # This is the same as cylinder but with larger circumference
    L = 2.0  # larger circumference → different topology
    t = rng.uniform(0, 1, N)
    x = rng.uniform(0, L, N)

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    cs = FastCausalSet(N)
    for i in range(N):
        dt = t[i+1:] - t[i]
        dx = np.abs(x[i+1:] - x[i])
        dx = np.minimum(dx, L - dx)
        cs.order[i, i+1:] = dt**2 >= dx**2

    return cs

def topology_observables(cs):
    """
    Observables that might distinguish spatial topology:
    1. Antichain size distribution (spatial slices)
    2. Path multiplicity (# of distinct geodesics between pairs)
    3. Cycle structure in the link graph (topology creates cycles)
    """
    N = cs.n
    links = cs.link_matrix()

    # 1. Ordering fraction
    f = cs.ordering_fraction()

    # 2. Link density
    n_links = links.sum()
    link_density = n_links / (N * (N - 1) / 2)

    # 3. "Spatial width" at midpoint: antichain size near t=N/2
    # Find elements with roughly half the causet in their past
    past_counts = cs.order.astype(int).sum(axis=0)
    future_counts = cs.order.astype(int).sum(axis=1)
    # Elements near the "equator"
    total = past_counts + future_counts
    midpoint_elements = np.where(np.abs(past_counts - future_counts) < N * 0.2)[0]

    # Among midpoint elements, count how many are spacelike to each other
    n_mid = len(midpoint_elements)
    if n_mid > 1:
        spacelike_count = 0
        for i in range(n_mid):
            for j in range(i + 1, n_mid):
                ei, ej = midpoint_elements[i], midpoint_elements[j]
                if not cs.order[ei, ej] and not cs.order[ej, ei]:
                    spacelike_count += 1
        spatial_width = spacelike_count / (n_mid * (n_mid - 1) / 2) if n_mid > 1 else 0
    else:
        spatial_width = 0

    # 4. Number of "independent paths" between minimal and maximal elements
    # (topological invariant: cylinder has more paths than diamond)
    has_predecessor = cs.order.astype(int).sum(axis=0) > 0
    has_successor = cs.order.astype(int).sum(axis=1) > 0
    n_minimal = (~has_predecessor).sum()
    n_maximal = (~has_successor).sum()

    # 5. Link graph clustering coefficient
    adj = (links | links.T).astype(float)
    degree = adj.sum(axis=1)
    clustering = 0.0
    n_counted = 0
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges among neighbors
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
        'n_minimal': int(n_minimal),
        'n_maximal': int(n_maximal),
        'clustering': clustering,
    }

print("\n--- Comparing topology: Diamond vs Cylinder vs Wide-Cylinder ---")
N_top = 60
n_trials = 15

for label, gen_func in [
    ("Diamond (d=2)", lambda: sprinkle_fast(N_top, dim=2, rng=rng)[0]),
    ("Cylinder (S^1, L=1)", lambda: sprinkle_cylinder(N_top, rng)),
    ("Wide cylinder (S^1, L=2)", lambda: sprinkle_torus(N_top, rng)),
]:
    obs_all = defaultdict(list)
    for _ in range(n_trials):
        cs = gen_func()
        obs = topology_observables(cs)
        for k, v in obs.items():
            obs_all[k].append(v)

    print(f"\n  {label}:")
    for k in ['f', 'link_density', 'spatial_width', 'n_minimal', 'n_maximal', 'clustering']:
        vals = obs_all[k]
        print(f"    {k:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Statistical test: can we distinguish topologies?
print("\n--- Statistical distinguishability (Mann-Whitney U test) ---")
diamond_obs = defaultdict(list)
cylinder_obs = defaultdict(list)
for _ in range(20):
    cs_d, _ = sprinkle_fast(N_top, dim=2, rng=rng)
    cs_c = sprinkle_cylinder(N_top, rng)
    obs_d = topology_observables(cs_d)
    obs_c = topology_observables(cs_c)
    for k in obs_d:
        diamond_obs[k].append(obs_d[k])
        cylinder_obs[k].append(obs_c[k])

for k in ['f', 'link_density', 'spatial_width', 'n_minimal', 'n_maximal', 'clustering']:
    stat, p = stats.mannwhitneyu(diamond_obs[k], cylinder_obs[k], alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {k:15s}: p = {p:.4f} {sig}")

print("\n  ASSESSMENT: Which observables distinguish spatial topology?")
print("  Significant differences suggest topology IS encoded in causal structure.")


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 78)
print("FINAL SUMMARY: WILD CARD ROUND 2 — IDEAS 291-300")
print("=" * 78)

print("""
IDEA 291 (Game Theory / Price of Anarchy):
  Tests whether the "inefficiency of selfish play" on a causal set
  depends on the embedding dimension. If PoA varies with d, it's a
  new dimension estimator from pure game theory.

IDEA 292 (Kolmogorov Complexity):
  Compares compression of different representations (matrix vs 2-order
  vs links). If 2-orders are significantly more compressible, the
  permutation representation carries less redundancy.

IDEA 293 (Quantum Complexity):
  SJ state preparation complexity via entanglement entropy proxy.
  The scaling exponent alpha in S ~ N^alpha determines whether the
  SJ vacuum has area-law or volume-law entanglement.

IDEA 294 (TDA of Configuration Space):
  The effective dimension of the space of 2-orders reveals the
  geometric structure of the configuration space itself.

IDEA 295 (Fractal Dimension):
  Box-counting dimension of spectrally-embedded Hasse diagram.
  If it differs from the manifold dimension, the causal structure
  has fractal sub-structure.

IDEA 296 (Boolean Network):
  Derrida coefficient measures chaos. If manifold-like causets sit
  at the edge of chaos, this connects to critical phenomena and
  the idea that spacetime geometry is a critical state.

IDEA 297 (Renormalization Group):
  Coarse-graining by merging linked pairs. If the ordering fraction
  flows to a fixed point, this is an IR fixed point of the causal
  set RG — connecting to the continuum limit.

IDEA 298 (Markov Chain):
  Spectral gap and mixing time of the causal transition matrix.
  If mixing time ~ proper time extent, this is a new operational
  definition of "distance" in a causal set.

IDEA 299 (Max-Flow Min-Cut):
  The min-cut between past and future boundaries. If it scales as
  an area law and correlates with SJ entanglement, this gives a
  purely classical (graph-theoretic) entanglement measure.

IDEA 300 (Emergent Topology):
  Whether causal structure alone can distinguish spatial topology
  (diamond vs cylinder). This is the ultimate test of "geometry
  from order."
""")

print("DONE. Ideas 291-300 complete.")
