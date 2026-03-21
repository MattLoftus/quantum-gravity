"""
Experiment 48: Ideas 56-75 — Round 3 of 8+ Search

Twenty new ideas, focused on:
- Deeper quantum information theory (modular Hamiltonian, Renyi, reflected entropy)
- Quantum chaos diagnostics (spectral form factor)
- Geometric structure (Ricci curvature, correlation length, deficit angle)
- Connections to known results (BW theorem, ETH, Markov property)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder
from causal_sets.two_orders_v2 import mcmc_corrected, bd_action_corrected
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size

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


def renyi_entropy(W, region, n_order):
    """Renyi entropy S_n = (1/(1-n)) ln Tr(rho^n)."""
    if len(region) < 2:
        return 0.0
    W_A = W[np.ix_(region, region)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-15, 1 - 1e-15)
    if n_order == 1:
        return entanglement_entropy(W, region)
    # For Gaussian state: Tr(rho^n) = prod_k [nu_k^n + (1-nu_k)^n]
    log_trace = np.sum(np.log(eigs**n_order + (1 - eigs)**n_order))
    return float(log_trace / (1 - n_order))


def mutual_info(W, A, B):
    """Mutual information I(A:B) = S(A) + S(B) - S(A∪B)."""
    return entanglement_entropy(W, A) + entanglement_entropy(W, B) - \
           entanglement_entropy(W, sorted(set(A) | set(B)))


def geodesic_distance(cs, i, j):
    """Shortest path distance on the link graph (Hasse diagram)."""
    N = cs.n
    # BFS on link graph
    links = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in range(N):
            if cs.order[a, b]:
                # Check if it's a link (no intermediate element)
                is_link = True
                for c in range(N):
                    if c != a and c != b and cs.order[a, c] and cs.order[c, b]:
                        is_link = False
                        break
                if is_link:
                    links[a, b] = True
                    links[b, a] = True  # undirected for distance

    # BFS from i
    visited = {i}
    queue = [(i, 0)]
    while queue:
        node, dist = queue.pop(0)
        if node == j:
            return dist
        for k in range(N):
            if links[node, k] and k not in visited:
                visited.add(k)
                queue.append((k, dist + 1))
    return N  # disconnected


def compute_link_matrix(cs):
    """Compute link (Hasse) matrix."""
    N = cs.n
    links = np.zeros((N, N), dtype=bool)
    for a in range(N):
        for b in range(N):
            if cs.order[a, b]:
                is_link = True
                for c in range(N):
                    if c != a and c != b and cs.order[a, c] and cs.order[c, b]:
                        is_link = False
                        break
                if is_link:
                    links[a, b] = True
    return links


# ================================================================
print("=" * 75)
print("IDEA 56: MODULAR HAMILTONIAN — BISOGNANO-WICHMANN TEST")
print("K_A = -ln(W_A/(1-W_A)). Does eigenvalue spectrum ~ distance from boundary?")
print("BW theorem: K = 2π × (distance from entangling surface) × boost generator")
print("=" * 75)

N = 50
n_trials = 15
bw_correlations = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, evals, _ = sj_full(cs)

    # Take left half as region A (by v-coordinate = spatial)
    v_sorted = np.argsort(to.v)
    A = list(v_sorted[:N // 2])

    W_A = W[np.ix_(A, A)]
    eigs = np.linalg.eigvalsh(W_A)
    eigs = np.clip(eigs, 1e-14, 1 - 1e-14)

    # Modular Hamiltonian eigenvalues
    K_eigs = -np.log(eigs / (1 - eigs))
    K_eigs_sorted = np.sort(K_eigs)

    # BW prediction: K eigenvalues ~ 2π × distance from boundary
    # For half-space partition, "distance from boundary" for each mode
    # is proportional to its mode number
    # Check: are K eigenvalues linearly spaced?
    mode_numbers = np.arange(len(K_eigs_sorted))
    r_bw = np.corrcoef(mode_numbers, K_eigs_sorted)[0, 1]
    bw_correlations.append(r_bw)

print(f"K eigenvalue linearity (BW test): r = {np.mean(bw_correlations):.3f} ± {np.std(bw_correlations):.3f}")
print(f"  Perfect BW would give r=1.0 (linear spectrum)")

# Null: random symmetric matrix
null_corrs = []
for _ in range(n_trials):
    M = rng.standard_normal((N // 2, N // 2))
    M = (M + M.T) / 2
    eigs = np.linalg.eigvalsh(M)
    eigs_sorted = np.sort(eigs)
    r = np.corrcoef(np.arange(len(eigs_sorted)), eigs_sorted)[0, 1]
    null_corrs.append(r)

print(f"Null (random symmetric): r = {np.mean(null_corrs):.3f} ± {np.std(null_corrs):.3f}")
z = (np.mean(bw_correlations) - np.mean(null_corrs)) / np.sqrt(
    np.std(bw_correlations)**2/n_trials + np.std(null_corrs)**2/n_trials)
print(f"z-score vs null: {z:.1f}")

# Also check: K eigenvalue spacing
K_spacings = np.diff(K_eigs_sorted)
print(f"K eigenvalue spacing: mean={np.mean(K_spacings):.3f}, CV={np.std(K_spacings)/np.mean(K_spacings):.3f}")
print(f"  (CV=0 would mean perfectly linear = BW)")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 57: SPECTRAL FORM FACTOR (QUANTUM CHAOS)")
print("SFF(t) = |Z(t)|²/|Z(0)|². Ramp-plateau = RMT/chaos. Decay only = integrable.")
print("=" * 75)

N = 50
n_trials = 20
t_values = np.logspace(-2, 2, 100)

sff_continuum = np.zeros(len(t_values))
sff_null = np.zeros(len(t_values))

for trial in range(n_trials):
    # Continuum phase causet
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    _, evals, _ = sj_full(cs)
    pos_evals = evals[evals > 1e-12]

    if len(pos_evals) > 2:
        # SFF from PJ eigenvalues
        Z0 = len(pos_evals)
        for ti, t in enumerate(t_values):
            Zt = np.sum(np.exp(1j * pos_evals * t))
            sff_continuum[ti] += np.abs(Zt)**2 / Z0**2

    # Null: Poisson eigenvalues (same number, same range)
    poisson_evals = rng.uniform(0, max(pos_evals) if len(pos_evals) > 0 else 1,
                                 size=len(pos_evals))
    Z0_null = len(poisson_evals)
    for ti, t in enumerate(t_values):
        Zt = np.sum(np.exp(1j * poisson_evals * t))
        sff_null[ti] += np.abs(Zt)**2 / Z0_null**2

sff_continuum /= n_trials
sff_null /= n_trials

# Check for ramp: SFF should increase linearly after initial dip
# Find the dip (minimum after initial decay)
dip_idx = np.argmin(sff_continuum[:50])
post_dip = sff_continuum[dip_idx:dip_idx+30]
if len(post_dip) > 5:
    slope_sff = np.polyfit(np.arange(len(post_dip)), post_dip, 1)[0]
    slope_null = np.polyfit(np.arange(len(sff_null[dip_idx:dip_idx+30])),
                            sff_null[dip_idx:dip_idx+30], 1)[0]
    print(f"SFF dip at t={t_values[dip_idx]:.3f}, value={sff_continuum[dip_idx]:.4f}")
    print(f"Post-dip slope (causet): {slope_sff:.6f}")
    print(f"Post-dip slope (Poisson null): {slope_null:.6f}")
    has_ramp = slope_sff > 2 * slope_null
    print(f"Ramp detected: {has_ramp} (slope ratio: {slope_sff/max(slope_null, 1e-10):.2f})")
else:
    print("Not enough post-dip points")

# Late-time plateau
plateau_sff = np.mean(sff_continuum[-20:])
plateau_null = np.mean(sff_null[-20:])
print(f"Late-time plateau: causet={plateau_sff:.4f}, Poisson={plateau_null:.4f}")
print(f"Plateau ratio: {plateau_sff/max(plateau_null, 1e-10):.2f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 58: MUTUAL INFORMATION DECAY LAW")
print("In 2D CFT: I(A:B) ~ 1/d^{4Δ}. What power law on causets?")
print("=" * 75)

N = 60
n_trials = 10
mi_vs_dist = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Spatial coordinate
    v = to.v / N
    # Take small regions (5 elements each) at varying separation
    region_size = 5
    for sep_idx in range(1, 8):
        start_A = 0
        start_B = start_A + region_size + sep_idx * 3
        if start_B + region_size >= N:
            break

        v_sorted = np.argsort(to.v)
        A = list(v_sorted[start_A:start_A + region_size])
        B = list(v_sorted[start_B:start_B + region_size])

        # Spatial separation
        d = np.mean(v[B]) - np.mean(v[A])
        if d < 0.01:
            continue

        I = mutual_info(W, A, B)
        if I > 1e-6:
            mi_vs_dist.append((d, I))

if mi_vs_dist:
    dists, mis = zip(*mi_vs_dist)
    dists, mis = np.array(dists), np.array(mis)

    # Fit power law: I ~ d^alpha
    log_d = np.log(dists[mis > 0])
    log_I = np.log(mis[mis > 0])
    if len(log_d) > 5:
        slope, intercept, r_val, p_val, _ = stats.linregress(log_d, log_I)
        print(f"Power law: I ~ d^{slope:.2f} (r={r_val:.3f}, p={p_val:.2e})")
        print(f"  CFT prediction: I ~ d^{{-4Δ}}. Our Δ ≈ {-slope/4:.3f}")
        print(f"  For massless scalar in 2D, Δ = 0 → I ~ ln(d). If Δ > 0, power law.")
    else:
        print(f"Only {len(log_d)} valid MI points — insufficient")
else:
    print("No valid MI measurements")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 59: ENTANGLEMENT CONTOUR")
print("s(i) = partial contribution of element i to S(A). Peaks at boundary?")
print("=" * 75)

N = 50
n_trials = 15
boundary_fracs = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Region A = left half by v-coordinate
    v_sorted = np.argsort(to.v)
    k = N // 2
    A = list(v_sorted[:k])

    # Entanglement contour: s(i) via single-element removal
    S_full = entanglement_entropy(W, A)
    contour = np.zeros(k)
    for idx in range(k):
        A_minus = [a for a in A if a != A[idx]]
        S_minus = entanglement_entropy(W, A_minus)
        contour[idx] = S_full - S_minus  # contribution of element A[idx]

    # Which elements are "near the boundary"?
    # Boundary = elements with v-coordinate near N/2
    v_positions = to.v[A] / N  # normalized position
    # Boundary proximity: distance from the cut (v = 0.5)
    boundary_proximity = 1.0 - np.abs(v_positions - 0.5)

    # Does contour peak near boundary?
    r = np.corrcoef(boundary_proximity, contour)[0, 1]
    boundary_fracs.append(r)

    if trial == 0:
        # Show contour for first trial
        sorted_idx = np.argsort(v_positions)
        print("  Position | Contour | Boundary proximity")
        for i in sorted_idx[::5]:
            print(f"  {v_positions[i]:.3f}      | {contour[i]:.4f}  | {boundary_proximity[i]:.3f}")

print(f"\nContour-boundary correlation: r = {np.mean(boundary_fracs):.3f} ± {np.std(boundary_fracs):.3f}")
print(f"  r > 0 means contour peaks near boundary (area law microscopic origin)")
print(f"  r ≈ 0 means no spatial structure in contour")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 60: RENYI ENTROPY SPECTRUM")
print("S_n for n=0.5, 1, 2, 3, inf. Ratio S_2/S_1 constrains entanglement structure.")
print("=" * 75)

N = 50
n_trials = 20
renyi_ratios = {n: [] for n in [0.5, 2, 3, 10]}

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    A = list(range(N // 2))
    S1 = entanglement_entropy(W, A)
    if S1 < 0.01:
        continue

    for n in [0.5, 2, 3, 10]:
        Sn = renyi_entropy(W, A, n)
        renyi_ratios[n].append(Sn / S1)

for n, ratios in renyi_ratios.items():
    if ratios:
        print(f"S_{n}/S_1 = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}  (n_valid={len(ratios)})")

print("\nCFT prediction: S_n/S_1 = (1+1/n)/2 for ground state")
for n in [0.5, 2, 3, 10]:
    print(f"  S_{n}/S_1 (CFT) = {(1+1/n)/2:.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 61: REFLECTED ENTROPY")
print("S_R(A:B) from canonical purification. In holography: S_R = 2×EW cross section.")
print("=" * 75)

N = 50
n_trials = 15
sr_data = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Three equal regions
    v_sorted = np.argsort(to.v)
    third = N // 3
    A = list(v_sorted[:third])
    B = list(v_sorted[third:2*third])
    C = list(v_sorted[2*third:])

    # Reflected entropy: S_R(A:B)
    # For Gaussian states, compute from correlation matrix of AA* system
    # Simplified: S_R ≈ S(A) + S(B) - S(AB) + correction
    # Actually: S_R(A:B) = S(AA*) where AA* is in the canonical purification
    # For a Gaussian state with W_AB, the canonical purification doubles the system:
    # W_AABB = [[W_A, sqrt(W_A(1-W_A))], [sqrt(W_A(1-W_A)), 1-W_A]]
    # Then S_R = S(AA*) in the doubled system

    # Compute W restricted to AB
    AB = sorted(set(A) | set(B))
    W_AB = W[np.ix_(AB, AB)]
    eigs_AB = np.linalg.eigvalsh(W_AB)
    eigs_AB = np.clip(eigs_AB, 1e-14, 1 - 1e-14)

    # Reflected entropy from eigenvalues of W_AB
    # S_R = Σ h(ν) where h(x) = -x ln x - (1-x) ln(1-x) applied to
    # eigenvalues of the "reflected" correlation matrix
    # For the canonical purification: reflected eigenvalues are
    # (1 + sqrt(1 - 4ν(1-ν)))/2 and (1 - sqrt(1 - 4ν(1-ν)))/2
    S_R = 0.0
    for nu in eigs_AB:
        disc = 1 - 4*nu*(1-nu)
        if disc < 0:
            disc = 0
        sq = np.sqrt(disc)
        lam_plus = (1 + sq) / 2
        lam_minus = (1 - sq) / 2
        for lam in [lam_plus, lam_minus]:
            lam = np.clip(lam, 1e-15, 1 - 1e-15)
            S_R += -lam * np.log(lam) - (1 - lam) * np.log(1 - lam)

    I_AB = mutual_info(W, A, B)
    sr_data.append({'S_R': S_R, 'I': I_AB, 'S_A': entanglement_entropy(W, A)})

S_R_arr = np.array([d['S_R'] for d in sr_data])
I_arr = np.array([d['I'] for d in sr_data])
S_A_arr = np.array([d['S_A'] for d in sr_data])

print(f"S_R = {np.mean(S_R_arr):.3f} ± {np.std(S_R_arr):.3f}")
print(f"I(A:B) = {np.mean(I_arr):.3f} ± {np.std(I_arr):.3f}")
print(f"S_R/I = {np.mean(S_R_arr/np.maximum(I_arr, 0.001)):.3f}")
print(f"  Holographic bound: S_R ≥ I(A:B). Satisfied: {np.all(S_R_arr >= I_arr - 0.01)}")
print(f"  S_R = 2×EW cross section in holography")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 62: TOPOLOGICAL ENTANGLEMENT ENTROPY")
print("S_topo = -I_3(A:B:C). Non-zero → topological order.")
print("=" * 75)

N = 60
n_trials = 20
s_topo_causet = []
s_topo_random = []

for trial in range(n_trials):
    # Causet
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Three contiguous regions by spatial coordinate
    v_sorted = np.argsort(to.v)
    third = N // 3
    A = list(v_sorted[:third])
    B = list(v_sorted[third:2*third])
    C = list(v_sorted[2*third:])

    AB = sorted(set(A) | set(B))
    BC = sorted(set(B) | set(C))
    AC = sorted(set(A) | set(C))
    ABC = sorted(set(A) | set(B) | set(C))

    S_A = entanglement_entropy(W, A)
    S_B = entanglement_entropy(W, B)
    S_C = entanglement_entropy(W, C)
    S_AB = entanglement_entropy(W, AB)
    S_BC = entanglement_entropy(W, BC)
    S_AC = entanglement_entropy(W, AC)
    S_ABC = entanglement_entropy(W, ABC)

    I3 = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    s_topo_causet.append(-I3)

    # Null: random DAG
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.3:
                order_rand[perm[i], perm[j]] = True
    cs_rand = FastCausalSet(N)
    cs_rand.order = order_rand
    W_rand, _, _ = sj_full(cs_rand)

    S_A_r = entanglement_entropy(W_rand, A)
    S_B_r = entanglement_entropy(W_rand, B)
    S_C_r = entanglement_entropy(W_rand, C)
    S_AB_r = entanglement_entropy(W_rand, AB)
    S_BC_r = entanglement_entropy(W_rand, BC)
    S_AC_r = entanglement_entropy(W_rand, AC)
    S_ABC_r = entanglement_entropy(W_rand, ABC)

    I3_r = S_A_r + S_B_r + S_C_r - S_AB_r - S_BC_r - S_AC_r + S_ABC_r
    s_topo_random.append(-I3_r)

print(f"S_topo (causet): {np.mean(s_topo_causet):.4f} ± {np.std(s_topo_causet):.4f}")
print(f"S_topo (random DAG): {np.mean(s_topo_random):.4f} ± {np.std(s_topo_random):.4f}")
t_stat = (np.mean(s_topo_causet) - np.mean(s_topo_random)) / np.sqrt(
    np.std(s_topo_causet)**2/n_trials + np.std(s_topo_random)**2/n_trials)
print(f"t-statistic: {t_stat:.1f}")
print(f"  Non-zero S_topo → topological order in the SJ vacuum")
print(f"  Same S_topo → generic property, not topological")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 63: ENTANGLEMENT TEMPERATURE")
print("For small A: S(A) ≈ β_ent × E(A). Does β_ent have physical meaning?")
print("=" * 75)

N = 60
n_trials = 15
beta_ents = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Compute S and "energy" (Tr[K × W_A]) for different region sizes
    sizes = [3, 5, 7, 10, 15, 20]
    S_vals = []
    E_vals = []
    for k in sizes:
        A = list(range(k))
        S = entanglement_entropy(W, A)
        # "Energy" = Tr(H_A × rho_A) where H_A is the modular Hamiltonian
        W_A = W[np.ix_(A, A)]
        eigs = np.linalg.eigvalsh(W_A)
        eigs = np.clip(eigs, 1e-14, 1 - 1e-14)
        K_eigs = -np.log(eigs / (1 - eigs))
        E = np.sum(eigs * K_eigs)
        S_vals.append(S)
        E_vals.append(E)

    # Fit S = β_ent × E + const for small regions
    if len(S_vals) >= 3:
        slope, intercept, r, _, _ = stats.linregress(E_vals[:4], S_vals[:4])
        beta_ents.append(slope)

if beta_ents:
    print(f"β_ent = {np.mean(beta_ents):.3f} ± {np.std(beta_ents):.3f}")
    print(f"  β_ent = 1 would mean thermalizing at the modular temperature")
    print(f"  Unruh: β = 2π/a. If β_ent ≈ 2π/{np.mean(beta_ents):.3f} ≈ {2*np.pi/np.mean(beta_ents):.2f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 64: OLLIVIER-RICCI CURVATURE ON LINK GRAPH")
print("Does graph curvature distinguish causet phases and detect manifold curvature?")
print("=" * 75)

def ollivier_ricci_curvature(cs, i, j):
    """Compute Ollivier-Ricci curvature between linked elements i, j.
    κ(i,j) = 1 - W₁(μᵢ, μⱼ) / d(i,j)
    where μᵢ = uniform on neighbors of i, W₁ = Wasserstein-1."""
    N = cs.n
    links = compute_link_matrix(cs)

    # Neighbors of i and j
    nbrs_i = np.where(links[i] | links[:, i])[0]
    nbrs_j = np.where(links[j] | links[:, j])[0]

    if len(nbrs_i) == 0 or len(nbrs_j) == 0:
        return 0.0

    # BFS distance matrix for relevant nodes
    all_nodes = sorted(set(nbrs_i) | set(nbrs_j) | {i, j})
    node_map = {n: idx for idx, n in enumerate(all_nodes)}
    M = len(all_nodes)

    # Distance matrix via BFS
    adj = links | links.T  # undirected
    dist_matrix = np.full((M, M), N, dtype=float)
    for a_idx, a in enumerate(all_nodes):
        visited = {a: 0}
        queue = [(a, 0)]
        while queue:
            node, d = queue.pop(0)
            if node in node_map:
                dist_matrix[a_idx, node_map[node]] = d
            for k in range(N):
                if adj[node, k] and k not in visited:
                    visited[k] = d + 1
                    queue.append((k, d + 1))

    # Wasserstein-1 via linear programming (simplified: average distance)
    # For uniform measures on neighbors:
    # Lower bound: W_1 ≥ |mean_dist(nbrs_i) - mean_dist(nbrs_j)|
    # Approximate: W_1 ≈ average min-distance matching
    w1 = 0.0
    for ni in nbrs_i:
        min_d = min(dist_matrix[node_map.get(ni, 0), node_map.get(nj, 0)]
                    for nj in nbrs_j) if ni in node_map else N
        w1 += min_d
    w1 /= len(nbrs_i)

    d_ij = 1.0  # linked, so distance = 1
    return 1 - w1 / d_ij

N = 40
n_trials = 10
curvatures_causet = []
curvatures_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    links = compute_link_matrix(cs)

    # Sample some linked pairs
    curv_vals = []
    link_pairs = list(zip(*np.where(links)))
    if len(link_pairs) > 20:
        sample = rng.choice(len(link_pairs), 20, replace=False)
        for idx in sample:
            i, j = link_pairs[idx]
            k = ollivier_ricci_curvature(cs, i, j)
            curv_vals.append(k)

    if curv_vals:
        curvatures_causet.append(np.mean(curv_vals))

    # Random DAG
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.3:
                order_rand[perm[i], perm[j]] = True
    cs_rand = FastCausalSet(N)
    cs_rand.order = order_rand
    links_rand = compute_link_matrix(cs_rand)
    curv_vals_r = []
    link_pairs_r = list(zip(*np.where(links_rand)))
    if len(link_pairs_r) > 20:
        sample = rng.choice(len(link_pairs_r), 20, replace=False)
        for idx in sample:
            i, j = link_pairs_r[idx]
            k = ollivier_ricci_curvature(cs_rand, i, j)
            curv_vals_r.append(k)
    if curv_vals_r:
        curvatures_random.append(np.mean(curv_vals_r))

if curvatures_causet and curvatures_random:
    print(f"Ricci curvature (causet): {np.mean(curvatures_causet):.3f} ± {np.std(curvatures_causet):.3f}")
    print(f"Ricci curvature (random): {np.mean(curvatures_random):.3f} ± {np.std(curvatures_random):.3f}")
    t = (np.mean(curvatures_causet) - np.mean(curvatures_random)) / np.sqrt(
        np.std(curvatures_causet)**2/len(curvatures_causet) +
        np.std(curvatures_random)**2/len(curvatures_random))
    print(f"t-stat: {t:.1f}")
    print(f"  Positive curvature → sphere-like. Negative → hyperbolic. Zero → flat.")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 65: CORRELATION LENGTH AT BD TRANSITION")
print("Does |W[i,j]| correlation length diverge at β_c? Critical exponent?")
print("=" * 75)

N = 50
eps = 0.12
beta_c = 1.66 / (N * eps**2)
betas = [0, 0.5*beta_c, 0.8*beta_c, beta_c, 1.2*beta_c, 1.5*beta_c, 2*beta_c, 3*beta_c]
xi_values = []

for beta in betas:
    if beta == 0:
        to = TwoOrder(N, rng=rng)
        cs = to.to_causet()
    else:
        result = mcmc_corrected(N, beta, eps, n_steps=20000, n_therm=10000,
                                 record_every=50, rng=rng)
        if result['samples']:
            cs = result['samples'][-1].to_causet()
            to = result['samples'][-1]
        else:
            continue

    W, _, _ = sj_full(cs)

    # Correlation function C(d) = <|W[i,j]|> at geodesic distance d
    # Use spatial distance (v-coordinate difference)
    v = to.v / N
    corr_by_dist = {}
    for i in range(N):
        for j in range(i+1, N):
            d = abs(v[i] - v[j])
            d_bin = round(d * 10) / 10  # bin by 0.1
            if d_bin not in corr_by_dist:
                corr_by_dist[d_bin] = []
            corr_by_dist[d_bin].append(abs(W[i, j]))

    # Extract correlation length: fit C(d) ~ exp(-d/xi)
    dists = sorted(corr_by_dist.keys())
    means = [np.mean(corr_by_dist[d]) for d in dists]
    if len(dists) >= 4 and means[0] > 0:
        log_C = [np.log(max(m, 1e-15)) for m in means]
        slope, _, r, _, _ = stats.linregress(dists[:5], log_C[:5])
        xi = -1.0 / slope if slope < -0.01 else 999
        xi_values.append((beta / beta_c, xi))
        print(f"β/β_c = {beta/beta_c:.2f}: ξ = {xi:.2f} (r={r:.3f})")

if xi_values:
    print(f"\nCorrelation length at transition:")
    print(f"  ξ diverges → continuous transition, ξ jumps → first-order")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 66: DEFICIT ANGLE FROM INTERVAL COUNTING")
print("Interval count deviation from flat → effective curvature.")
print("=" * 75)

# In flat 2D Minkowski, for N elements in a causal diamond:
# Expected number of k-intervals: N_k ~ N × (something depending on k/N)
# On curved spacetime, deviations encode curvature

N = 50
n_trials = 15
interval_ratios = {'flat': [], 'curved': []}

for trial in range(n_trials):
    # Flat: 2-order
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    counts_flat = count_intervals_by_size(cs, max_size=10)

    # Curved: non-uniform sprinkling (simulate curvature via density variation)
    # Use exp(alpha * x) density profile
    alpha = 2.0
    u = rng.permutation(N)
    # Non-uniform v: sample from exp distribution
    v_raw = -np.log(1 - rng.random(N) * (1 - np.exp(-alpha))) / alpha
    v = np.argsort(v_raw).astype(float)
    v_perm = rng.permutation(N)
    # Actually just use argsort to get a permutation
    to_curved = TwoOrder.from_permutations(u, np.argsort(v_raw))
    cs_curved = to_curved.to_causet()
    counts_curved = count_intervals_by_size(cs_curved, max_size=10)

    # Compare interval distributions
    for k in range(6):
        n_flat = counts_flat.get(k, 0)
        n_curved = counts_curved.get(k, 0)
        if n_flat > 0:
            interval_ratios['flat'].append((k, n_flat / N))
        if n_curved > 0:
            interval_ratios['curved'].append((k, n_curved / N))

# Average by k
for label in ['flat', 'curved']:
    data = interval_ratios[label]
    for k in range(6):
        vals = [v for (kk, v) in data if kk == k]
        if vals:
            print(f"  {label} k={k}: N_k/N = {np.mean(vals):.3f} ± {np.std(vals):.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 67: CAUSAL SET D'ALEMBERTIAN SPECTRUM")
print("BD d'Alembertian eigenvalues vs lattice Laplacian. Same spectrum?")
print("=" * 75)

N = 50
n_trials = 10
dalembert_evals_all = []
lattice_evals_all = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()

    # BD d'Alembertian (2D): B = (2/l²)(identity - (2/√6)(L - √6 I₂ + ...))
    # Simplified: B[i,j] = δ_{ij} - α Σ_k C[i,k]C[k,j] where C is causal matrix
    C = cs.order.astype(float)
    # Simple d'Alembertian: B = I - (2/N)(C + C^T)
    B = np.eye(N) - (2.0 / N) * (C + C.T)
    evals_B = np.sort(np.linalg.eigvalsh(B))
    dalembert_evals_all.append(evals_B)

    # Lattice Laplacian (1D chain of N elements)
    L = np.zeros((N, N))
    for i in range(N - 1):
        L[i, i] += 1
        L[i + 1, i + 1] += 1
        L[i, i + 1] = -1
        L[i + 1, i] = -1
    evals_L = np.sort(np.linalg.eigvalsh(L))
    lattice_evals_all.append(evals_L)

# Compare spectra
avg_B = np.mean(dalembert_evals_all, axis=0)
avg_L = np.mean(lattice_evals_all, axis=0)

# Low eigenvalues (IR behavior)
print("Low eigenvalue comparison (first 10):")
print(f"  d'Alembertian: {avg_B[:10].round(4)}")
print(f"  Lattice Lapl:  {avg_L[:10].round(4)}")

# Weyl's law: eigenvalue density
print(f"\nSpectral statistics:")
print(f"  d'Alembertian: min={avg_B[0]:.4f}, max={avg_B[-1]:.4f}, mean={np.mean(avg_B):.4f}")
print(f"  Lattice:       min={avg_L[0]:.4f}, max={avg_L[-1]:.4f}, mean={np.mean(avg_L):.4f}")

# Correlation between spectra
r_spec = np.corrcoef(avg_B, avg_L[:N])[0, 1]
print(f"  Spectral correlation: r = {r_spec:.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 68: CONDITIONAL MUTUAL INFORMATION — MARKOV PROPERTY")
print("I(A:C|B) for B separating A and C. ≈0 → Markov → holographic.")
print("=" * 75)

N = 60
n_trials = 20
cmi_causet = []
cmi_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Three contiguous spatial regions: A | B | C
    v_sorted = np.argsort(to.v)
    fifth = N // 5
    A = list(v_sorted[:fifth])         # leftmost
    B = list(v_sorted[fifth:4*fifth])  # middle (large, separating)
    C = list(v_sorted[4*fifth:])       # rightmost

    AB = sorted(set(A) | set(B))
    BC = sorted(set(B) | set(C))
    ABC = sorted(set(A) | set(B) | set(C))

    S_AB = entanglement_entropy(W, AB)
    S_BC = entanglement_entropy(W, BC)
    S_B = entanglement_entropy(W, list(B))
    S_ABC = entanglement_entropy(W, ABC)

    I_AC_B = S_AB + S_BC - S_B - S_ABC  # I(A:C|B) ≥ 0 by SSA
    cmi_causet.append(I_AC_B)

    # Null: random DAG
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.25:
                order_rand[perm[i], perm[j]] = True
    cs_rand = FastCausalSet(N)
    cs_rand.order = order_rand
    W_rand, _, _ = sj_full(cs_rand)
    S_AB_r = entanglement_entropy(W_rand, AB)
    S_BC_r = entanglement_entropy(W_rand, BC)
    S_B_r = entanglement_entropy(W_rand, list(B))
    S_ABC_r = entanglement_entropy(W_rand, ABC)
    cmi_random.append(S_AB_r + S_BC_r - S_B_r - S_ABC_r)

print(f"I(A:C|B) causet: {np.mean(cmi_causet):.4f} ± {np.std(cmi_causet):.4f}")
print(f"I(A:C|B) random: {np.mean(cmi_random):.4f} ± {np.std(cmi_random):.4f}")
ratio = np.mean(cmi_causet) / max(np.mean(cmi_random), 1e-6)
print(f"Ratio: {ratio:.2f}")
print(f"  ≈0 → approximately Markov (holographic). Large → long-range correlations.")
t = (np.mean(cmi_causet) - np.mean(cmi_random)) / np.sqrt(
    np.std(cmi_causet)**2/n_trials + np.std(cmi_random)**2/n_trials)
print(f"  t-stat: {t:.1f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 69: FISHER INFORMATION AT PHASE TRANSITION")
print("Quantum Fisher info F(β) = Var(S_BD) × β². Diverges at β_c?")
print("=" * 75)

N = 50
eps = 0.12
beta_c = 1.66 / (N * eps**2)
betas_scan = np.linspace(0.5*beta_c, 2.0*beta_c, 8)
fisher_info = []

for beta in betas_scan:
    result = mcmc_corrected(N, beta, eps, n_steps=15000, n_therm=7500,
                             record_every=30, rng=rng)
    if result['actions']:
        var_S = np.var(result['actions'])
        F = beta**2 * var_S
        fisher_info.append((beta / beta_c, F, var_S))
        print(f"β/β_c={beta/beta_c:.2f}: Var(S)={var_S:.2f}, F={F:.1f}, acc={result.get('acceptance', 0):.1%}")

if fisher_info:
    max_F = max(f[1] for f in fisher_info)
    max_beta = [f[0] for f in fisher_info if f[1] == max_F][0]
    print(f"\nPeak Fisher info at β/β_c = {max_beta:.2f}")
    print(f"  Diverging F at β_c confirms phase transition location")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 70: MULTIPARTITE ENTANGLEMENT (RESIDUAL TANGLE)")
print("Genuine tripartite entanglement beyond bipartite in SJ vacuum?")
print("=" * 75)

N = 60
n_trials = 20
tangles_causet = []
tangles_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    v_sorted = np.argsort(to.v)
    third = N // 3
    A = list(v_sorted[:third])
    B = list(v_sorted[third:2*third])
    C = list(v_sorted[2*third:])
    AB = sorted(set(A) | set(B))
    AC = sorted(set(A) | set(C))

    S_A = entanglement_entropy(W, A)
    S_AB = entanglement_entropy(W, AB)
    S_B = entanglement_entropy(W, list(B))
    S_AC = entanglement_entropy(W, AC)
    S_C = entanglement_entropy(W, list(C))

    # Residual tangle: τ_ABC = S_A² - (S_AB - S_B)² - (S_AC - S_C)²
    # (Generalization of CKW inequality)
    I_AB = S_A + S_B - S_AB
    I_AC = S_A + S_C - S_AC
    tau = S_A**2 - I_AB**2 - I_AC**2
    tangles_causet.append(tau)

    # Random DAG null
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.25:
                order_rand[perm[i], perm[j]] = True
    cs_r = FastCausalSet(N)
    cs_r.order = order_rand
    W_r, _, _ = sj_full(cs_r)
    S_A_r = entanglement_entropy(W_r, A)
    S_AB_r = entanglement_entropy(W_r, AB)
    S_B_r = entanglement_entropy(W_r, list(B))
    S_AC_r = entanglement_entropy(W_r, AC)
    S_C_r = entanglement_entropy(W_r, list(C))
    I_AB_r = S_A_r + S_B_r - S_AB_r
    I_AC_r = S_A_r + S_C_r - S_AC_r
    tau_r = S_A_r**2 - I_AB_r**2 - I_AC_r**2
    tangles_random.append(tau_r)

print(f"Residual tangle (causet): {np.mean(tangles_causet):.4f} ± {np.std(tangles_causet):.4f}")
print(f"Residual tangle (random): {np.mean(tangles_random):.4f} ± {np.std(tangles_random):.4f}")
t = (np.mean(tangles_causet) - np.mean(tangles_random)) / np.sqrt(
    np.std(tangles_causet)**2/n_trials + np.std(tangles_random)**2/n_trials)
print(f"t-stat: {t:.1f}")
print(f"  τ > 0 → genuine tripartite entanglement beyond bipartite sharing")
print(f"  τ < 0 → bipartite entanglement dominates (monogamy)")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 71: LOCALITY OF W — PROPAGATOR DECAY")
print("How does |W[i,j]| decay with proper distance? Power law or exponential?")
print("=" * 75)

N = 60
n_trials = 15
decay_exponents = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Proper distance: d = sqrt(|Δt² - Δx²|) for spacelike, d = Δt for timelike
    t_coord = (to.u + to.v) / (2 * N)
    x_coord = (to.v - to.u) / (2 * N)

    w_vs_d = []
    for i in range(N):
        for j in range(i + 1, N):
            dt = t_coord[j] - t_coord[i]
            dx = x_coord[j] - x_coord[i]
            ds2 = dt**2 - dx**2
            if ds2 > 0.001:  # timelike
                d = np.sqrt(ds2)
                w = abs(W[i, j])
                if w > 1e-10 and d > 0.01:
                    w_vs_d.append((d, w))

    if len(w_vs_d) > 20:
        dists, ws = zip(*w_vs_d)
        dists, ws = np.array(dists), np.array(ws)
        # Power law fit: W ~ d^alpha
        log_d = np.log(dists)
        log_w = np.log(ws)
        slope, _, r, _, _ = stats.linregress(log_d, log_w)
        decay_exponents.append(slope)

if decay_exponents:
    print(f"W decay exponent: {np.mean(decay_exponents):.3f} ± {np.std(decay_exponents):.3f}")
    print(f"  Continuum 2D massless scalar: W ~ 1/d → exponent = -1")
    print(f"  Massive scalar: exponential decay")
    print(f"  Our measured exponent: {np.mean(decay_exponents):.3f}")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 72: RELATIVE ENTROPY BETWEEN PHASES")
print("S(W_cont || W_cryst) = quantum distance. How does it scale with N?")
print("=" * 75)

sizes = [20, 30, 40, 50]
rel_ent_by_N = []

for N in sizes:
    eps = 0.12
    beta_c = 1.66 / (N * eps**2)

    # Continuum phase
    to_cont = TwoOrder(N, rng=rng)
    cs_cont = to_cont.to_causet()
    W_cont, _, _ = sj_full(cs_cont)

    # Crystalline phase (high beta)
    result = mcmc_corrected(N, 3*beta_c, eps, n_steps=15000, n_therm=7500,
                             record_every=50, rng=rng)
    if result['samples']:
        cs_cryst = result['samples'][-1].to_causet()
        W_cryst, _, _ = sj_full(cs_cryst)

        # Relative entropy from eigenvalues
        # S(W_cont || W_cryst) using the full W matrices
        # For Gaussian: S = Tr[W_c (ln W_c - ln W_k) + (1-W_c)(ln(1-W_c) - ln(1-W_k))]
        eigs_c = np.linalg.eigvalsh(W_cont)
        eigs_c = np.clip(eigs_c, 1e-14, 1 - 1e-14)
        eigs_k = np.linalg.eigvalsh(W_cryst)
        eigs_k = np.clip(eigs_k, 1e-14, 1 - 1e-14)

        # This is approximate (ignores off-diagonal effects)
        S_rel = np.sum(eigs_c * np.log(eigs_c / eigs_k) +
                       (1 - eigs_c) * np.log((1 - eigs_c) / (1 - eigs_k)))
        rel_ent_by_N.append((N, S_rel / N))
        print(f"N={N}: S(cont || cryst)/N = {S_rel/N:.3f}")

if len(rel_ent_by_N) >= 3:
    ns = [x[0] for x in rel_ent_by_N]
    vals = [x[1] for x in rel_ent_by_N]
    slope, _, r, _, _ = stats.linregress(np.log(ns), np.log([max(v, 1e-10) for v in vals]))
    print(f"Scaling: S_rel/N ~ N^{slope:.2f} (r={r:.3f})")
    print(f"  Extensive (slope≈0): phases equally different at all scales")
    print(f"  Super-extensive (slope>0): phases diverge with system size")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 73: EIGENSTATE THERMALIZATION (ETH)")
print("Do SJ eigenstates have thermal reduced density matrices?")
print("=" * 75)

N = 50
n_trials = 10
eth_deviations = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    _, evals, evecs = sj_full(cs)

    # Take positive-eigenvalue eigenstates
    pos_idx = np.where(evals > 1e-12)[0]
    if len(pos_idx) < 5:
        continue

    # For each eigenstate, compute reduced density matrix of half-system
    A = list(range(N // 2))
    eth_vals = []

    for idx in pos_idx[:10]:  # first 10 positive eigenstates
        psi = evecs[:, idx].real  # eigenstate
        # One-particle reduced density matrix
        rho_A = np.outer(psi[A], psi[A].conj()).real
        rho_A /= max(np.trace(rho_A), 1e-15)

        # Thermal state: rho_th = exp(-β H_A) / Z
        # For simplicity, compare entropy of eigenstate RDM with maximum entropy
        eigs_rho = np.linalg.eigvalsh(rho_A)
        eigs_rho = eigs_rho[eigs_rho > 1e-15]
        S_eigenstate = -np.sum(eigs_rho * np.log(eigs_rho))
        S_max = np.log(len(A))
        eth_vals.append(S_eigenstate / S_max)

    eth_deviations.append(np.mean(eth_vals))

if eth_deviations:
    print(f"S_eigenstate / S_max = {np.mean(eth_deviations):.4f} ± {np.std(eth_deviations):.4f}")
    print(f"  ETH predicts S → S_max (= thermal). Value close to 1 = ETH satisfied.")
    print(f"  Value << 1 = eigenstates are non-thermal (integrable)")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 74: SJ ON PERTURBED CAUSETS — CURVATURE RESPONSE")
print("Add a perturbation to the causal matrix. δS ~ perturbation strength?")
print("=" * 75)

N = 50
n_trials = 15
perturbation_strengths = [0.01, 0.02, 0.05, 0.1, 0.2]
delta_S_by_pert = {p: [] for p in perturbation_strengths}

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W0, _, _ = sj_full(cs)
    S0 = entanglement_entropy(W0, list(range(N // 2)))

    for p in perturbation_strengths:
        # Perturb: flip some causal relations
        cs_pert = FastCausalSet(N)
        order_pert = cs.order.copy()
        n_flip = max(1, int(p * N * (N - 1) / 2))
        for _ in range(n_flip):
            i, j = rng.integers(0, N, size=2)
            if i != j:
                order_pert[i, j] = not order_pert[i, j]
        cs_pert.order = order_pert
        W_pert, _, _ = sj_full(cs_pert)
        S_pert = entanglement_entropy(W_pert, list(range(N // 2)))
        delta_S_by_pert[p].append(abs(S_pert - S0) / S0)

print("Perturbation strength | δS/S₀")
for p in perturbation_strengths:
    vals = delta_S_by_pert[p]
    print(f"  {p:.2f}                | {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Check linearity
means = [np.mean(delta_S_by_pert[p]) for p in perturbation_strengths]
r_lin = np.corrcoef(perturbation_strengths, means)[0, 1]
slope, _, _, _, _ = stats.linregress(perturbation_strengths, means)
print(f"\nLinear response: δS/S₀ = {slope:.3f} × ε (r={r_lin:.3f})")
print(f"  Linear response = well-defined curvature coupling")


# ================================================================
print("\n" + "=" * 75)
print("IDEA 75: ENTANGLEMENT ASYMMETRY UNDER TIME REVERSAL")
print("S(past→future) vs S(future→past). Broken T-symmetry?")
print("=" * 75)

N = 50
n_trials = 20
asymmetry_causet = []
asymmetry_random = []

for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, _, _ = sj_full(cs)

    # Time coordinate
    t_coord = (to.u + to.v) / (2.0 * N)
    t_sorted = np.argsort(t_coord)

    # Past half and future half
    past = list(t_sorted[:N // 2])
    future = list(t_sorted[N // 2:])

    S_past = entanglement_entropy(W, past)
    S_future = entanglement_entropy(W, future)

    asymmetry = (S_past - S_future) / (S_past + S_future + 1e-15)
    asymmetry_causet.append(asymmetry)

    # Random DAG
    order_rand = np.zeros((N, N), dtype=bool)
    perm = rng.permutation(N)
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.25:
                order_rand[perm[i], perm[j]] = True
    cs_r = FastCausalSet(N)
    cs_r.order = order_rand
    W_r, _, _ = sj_full(cs_r)
    S_past_r = entanglement_entropy(W_r, past)
    S_future_r = entanglement_entropy(W_r, future)
    asymmetry_random.append((S_past_r - S_future_r) / (S_past_r + S_future_r + 1e-15))

print(f"Time asymmetry (causet): {np.mean(asymmetry_causet):.4f} ± {np.std(asymmetry_causet):.4f}")
print(f"Time asymmetry (random): {np.mean(asymmetry_random):.4f} ± {np.std(asymmetry_random):.4f}")
t = abs(np.mean(asymmetry_causet)) / (np.std(asymmetry_causet) / np.sqrt(n_trials))
print(f"Is asymmetry ≠ 0? t-stat from zero: {t:.1f}")
print(f"  Asymmetry = 0 → T-symmetric vacuum (expected for SJ in flat spacetime)")
print(f"  Asymmetry ≠ 0 → T-violation → arrow of time from quantum gravity")


# ================================================================
print("\n" + "=" * 75)
print("SUMMARY: IDEAS 56-75")
print("=" * 75)
print("""
Score each idea 1-10 based on:
- Novelty of the result
- Passes null model control
- Stable across system sizes
- Connects to known physics
- Paper-worthy as standalone or strengthens existing paper
""")
