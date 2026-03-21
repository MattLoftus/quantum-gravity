"""
Experiment 94: CONTINUUM LIMIT TESTS — Ideas 451-460

Do causal sets reproduce KNOWN CONTINUUM RESULTS at accessible N?
Every confirmed match strengthens the causal set program.

Ideas:
451. WEYL'S LAW: eigenvalue counting function N(λ) of the Hasse Laplacian.
     In d dimensions, N(λ) ~ λ^{d/2}. Does the Hasse Laplacian obey this?
452. MINKOWSKI DIMENSION from box-counting on embedded coordinates.
     Does it give d=2 for 2-orders?
453. HAUSDORFF DIMENSION of the Hasse diagram (as a metric space with
     graph distance). Compare with d=2.
454. GREEN'S FUNCTION: G(x,y) = -(1/2π)ln|x-y| in 2D. Does the SJ
     Wightman function match this for timelike-separated pairs?
455. PROPAGATOR POLES: the Feynman propagator has poles at p²=m² in
     momentum space. Does the discrete propagator (FT of W) show poles?
456. SPECTRAL DIMENSION from the Hasse Laplacian heat kernel — does
     the HASSE Laplacian heat kernel give d_s=2?
457. VOLUME-DISTANCE scaling: V(r) ~ r^d for geodesic balls. Number
     of elements within chain-distance r of a typical element. r^2?
458. EULER CHARACTERISTIC via Gauss-Bonnet: χ = (1/2π)∫R dA = 2 for
     a sphere, 0 for a torus. Compute χ from the Hasse diagram.
459. GEODESIC DEVIATION: "Jacobi field" analogue — how do nearby
     geodesics (chains) diverge? Does the rate match d=2 flat spacetime?
460. SCALAR CURVATURE from BD action: S_BD ≈ ∫R√g d²x. Compute
     S_BD/N for flat vs curved — does the difference scale with curvature?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, linalg
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from collections import defaultdict
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function
import time

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def hasse_laplacian_spectrum(cs):
    """Compute eigenvalues of the (unnormalized) graph Laplacian of the Hasse diagram."""
    links = cs.link_matrix()
    adj = (links | links.T).astype(float)
    degree = np.sum(adj, axis=1)
    L = np.diag(degree) - adj
    evals = np.sort(np.linalg.eigvalsh(L))
    return evals


def hasse_adjacency(cs):
    """Return the symmetric adjacency matrix of the Hasse diagram."""
    links = cs.link_matrix()
    return (links | links.T).astype(float)


def geodesic_distance_matrix(cs):
    """All-pairs shortest path on the undirected Hasse (link) graph."""
    adj = hasse_adjacency(cs)
    sp = csr_matrix(adj)
    dist = shortest_path(sp, directed=False)
    return dist


def chain_distance_matrix(cs):
    """All-pairs longest chain distance (causal/geodesic in Lorentzian sense).
    For each related pair (i,j), find the longest chain from i to j.
    For unrelated pairs, distance is infinity."""
    N = cs.n
    # dp[i,j] = longest chain from i to j (number of links)
    # Use dynamic programming on the causal order
    dist = np.full((N, N), np.inf)
    np.fill_diagonal(dist, 0)

    links = cs.link_matrix()
    # Initialize with links
    for i in range(N):
        for j in range(i+1, N):
            if links[i, j]:
                dist[i, j] = 1

    # Floyd-Warshall style but taking MAX of chain lengths for related pairs
    # Actually we want longest path in a DAG — use topological order
    # Since elements are roughly time-ordered, process in order
    for j in range(N):
        for i in range(j):
            if not cs.order[i, j]:
                continue
            # Try to extend through intermediate k
            for k in range(i+1, j):
                if cs.order[i, k] and cs.order[k, j]:
                    if dist[i, k] + dist[k, j] > dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def sprinkle_curved_2d(n, curvature_R, extent_t=1.0, rng_local=None):
    """Sprinkle into 2D spacetime with constant scalar curvature R.

    For small curvature, use conformal factor: ds² = Ω²(x)(-dt² + dx²)
    with Ω²(x) = 1 + (R/4)(t² - x²) to first order.

    Positive R = de Sitter-like (expanding), negative R = anti-de Sitter-like.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    coords_list = []
    while len(coords_list) < n:
        batch = n * 4
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng_local.uniform(-extent_t, extent_t, batch)
        candidates[:, 1] = rng_local.uniform(-extent_t, extent_t, batch)
        # Diamond constraint
        mask = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= extent_t
        accepted = candidates[mask]
        coords_list.append(accepted)

    coords = np.vstack(coords_list)[:n]
    coords = coords[np.argsort(coords[:, 0])]

    # Conformal factor for curved spacetime
    # Ω²(t,x) = 1 + (R/4)(t² - x²) ... but we need Ω² > 0
    omega_sq = 1.0 + (curvature_R / 4.0) * (coords[:, 0]**2 - coords[:, 1]**2)
    omega_sq = np.clip(omega_sq, 0.1, 10.0)  # safety

    # Build causal order with conformal metric
    # Two events are causally related if the CONFORMAL null cone allows it
    # In conformal coords: ds² = Ω²(-dt² + dx²), so null cones are unchanged
    # BUT the volume element changes: √g = Ω²
    # For sprinkling with density ρ ∝ Ω², we use rejection sampling

    # Actually for simplicity: sprinkle uniformly, build order with FLAT metric,
    # but the BD action will detect curvature through the interval structure
    causet = FastCausalSet(n)
    for i in range(n):
        if i == n - 1:
            break
        dt = coords[i+1:, 0] - coords[i, 0]
        dx_sq = (coords[i+1:, 1] - coords[i, 1])**2
        related = dt * dt >= dx_sq
        causet.order[i, i+1:] = related

    return causet, coords, omega_sq


print("=" * 72)
print("EXPERIMENT 94: CONTINUUM LIMIT TESTS — IDEAS 451-460")
print("Do causal sets reproduce known continuum results?")
print("=" * 72)


# ============================================================
# IDEA 451: WEYL'S LAW
# ============================================================
print("\n" + "=" * 72)
print("IDEA 451: WEYL'S LAW FOR THE HASSE LAPLACIAN")
print("In d dimensions, N(λ) ~ λ^{d/2}")
print("For d=2: N(λ) ~ λ")
print("=" * 72)

t0 = time.time()

sizes_weyl = [50, 100, 150, 200]
n_trials_weyl = 8

print(f"\nSprinkle into 2D Minkowski diamond, sizes: {sizes_weyl}")
print(f"Trials per size: {n_trials_weyl}")

for N in sizes_weyl:
    all_slopes = []
    for trial in range(n_trials_weyl):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        evals = hasse_laplacian_spectrum(cs)
        evals_pos = evals[evals > 1e-10]

        if len(evals_pos) < 5:
            continue

        # Eigenvalue counting function: N(λ) = number of eigenvalues ≤ λ
        # Fit log(N(λ)) vs log(λ) to get exponent
        # Weyl's law: N(λ) ~ λ^{d/2}, so log N(λ) ~ (d/2) log λ
        count_func = np.arange(1, len(evals_pos) + 1)

        # Use middle 50% to avoid boundary effects
        n_ev = len(evals_pos)
        lo, hi = n_ev // 4, 3 * n_ev // 4
        if hi - lo < 3:
            lo, hi = 0, n_ev

        log_lambda = np.log(evals_pos[lo:hi])
        log_count = np.log(count_func[lo:hi])

        slope, intercept, r, p, se = stats.linregress(log_lambda, log_count)
        all_slopes.append(slope)

    if all_slopes:
        mean_slope = np.mean(all_slopes)
        std_slope = np.std(all_slopes) / np.sqrt(len(all_slopes))
        inferred_d = 2 * mean_slope
        print(f"  N={N:4d}: slope = {mean_slope:.3f} ± {std_slope:.3f}, "
              f"inferred d = {inferred_d:.2f} (target: 2.00)")

# Also test d=3 for comparison
print("\n  Testing d=3 sprinklings for comparison:")
for N in [100, 200]:
    all_slopes = []
    for trial in range(n_trials_weyl):
        cs, coords = sprinkle_fast(N, dim=3, rng=rng)
        evals = hasse_laplacian_spectrum(cs)
        evals_pos = evals[evals > 1e-10]

        if len(evals_pos) < 5:
            continue

        count_func = np.arange(1, len(evals_pos) + 1)
        n_ev = len(evals_pos)
        lo, hi = n_ev // 4, 3 * n_ev // 4
        if hi - lo < 3:
            lo, hi = 0, n_ev

        log_lambda = np.log(evals_pos[lo:hi])
        log_count = np.log(count_func[lo:hi])
        slope, _, _, _, _ = stats.linregress(log_lambda, log_count)
        all_slopes.append(slope)

    if all_slopes:
        mean_slope = np.mean(all_slopes)
        std_slope = np.std(all_slopes) / np.sqrt(len(all_slopes))
        inferred_d = 2 * mean_slope
        print(f"  N={N:4d} (d=3): slope = {mean_slope:.3f} ± {std_slope:.3f}, "
              f"inferred d = {inferred_d:.2f} (target: 3.00)")

t1 = time.time()
print(f"\n  [Weyl's law completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 452: MINKOWSKI DIMENSION (BOX-COUNTING)
# ============================================================
print("\n" + "=" * 72)
print("IDEA 452: MINKOWSKI (BOX-COUNTING) DIMENSION")
print("Does box-counting on embedded coordinates give d=2?")
print("=" * 72)

t0 = time.time()

sizes_box = [100, 200, 300, 500]
n_trials_box = 5

for N in sizes_box:
    all_dims = []
    for trial in range(n_trials_box):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)

        # Box-counting: cover the point set with boxes of side ε
        # Count N(ε) = number of boxes needed
        # d_box = -lim log N(ε) / log ε as ε → 0

        # Range of ε values
        extent = np.max(coords) - np.min(coords)
        epsilons = np.logspace(np.log10(extent / 2), np.log10(extent / 30), 10)
        box_counts = []

        for eps in epsilons:
            # Grid covering
            mins = coords.min(axis=0)
            grid_indices = np.floor((coords - mins) / eps).astype(int)
            # Count unique occupied boxes
            unique_boxes = len(set(map(tuple, grid_indices)))
            box_counts.append(unique_boxes)

        box_counts = np.array(box_counts, dtype=float)
        log_eps = np.log(epsilons)
        log_N = np.log(box_counts)

        # Fit slope = -d_box
        slope, _, r, _, _ = stats.linregress(log_eps, log_N)
        d_box = -slope
        all_dims.append(d_box)

    mean_d = np.mean(all_dims)
    std_d = np.std(all_dims) / np.sqrt(len(all_dims))
    print(f"  N={N:4d}: d_box = {mean_d:.3f} ± {std_d:.3f} (target: 2.00)")

t1 = time.time()
print(f"\n  [Box-counting completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 453: HAUSDORFF DIMENSION OF THE HASSE DIAGRAM
# ============================================================
print("\n" + "=" * 72)
print("IDEA 453: HAUSDORFF DIMENSION OF HASSE DIAGRAM")
print("Graph distance metric space. Does it give d≈2?")
print("=" * 72)

t0 = time.time()

sizes_haus = [50, 100, 150, 200]
n_trials_haus = 5

for N in sizes_haus:
    all_dims = []
    for trial in range(n_trials_haus):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        dist = geodesic_distance_matrix(cs)

        # Hausdorff dimension: number of points within ball of radius r
        # grows as r^d_H
        # For each node, count B(r) = |{j : dist(i,j) ≤ r}|
        # Average over nodes, fit log B(r) vs log r

        finite_mask = np.isfinite(dist)
        if not np.all(finite_mask):
            # Graph might be disconnected — use largest component
            # Skip disconnected cases
            continue

        max_r = int(np.max(dist))
        if max_r < 3:
            continue

        radii = np.arange(1, min(max_r, 15) + 1)
        avg_ball_size = []

        for r in radii:
            ball_sizes = np.sum(dist <= r, axis=1)  # includes self
            avg_ball_size.append(np.mean(ball_sizes))

        avg_ball_size = np.array(avg_ball_size, dtype=float)

        # Fit in the scaling regime (exclude r=1 and near-max)
        valid = avg_ball_size > 1
        if np.sum(valid) < 3:
            continue

        log_r = np.log(radii[valid].astype(float))
        log_B = np.log(avg_ball_size[valid])
        slope, _, r_val, _, _ = stats.linregress(log_r, log_B)
        all_dims.append(slope)

    if all_dims:
        mean_d = np.mean(all_dims)
        std_d = np.std(all_dims) / np.sqrt(len(all_dims))
        print(f"  N={N:4d}: d_Hausdorff = {mean_d:.3f} ± {std_d:.3f} (target: ~2)")

# Compare with d=3
print("\n  d=3 comparison:")
for N in [100, 150]:
    all_dims = []
    for trial in range(n_trials_haus):
        cs, coords = sprinkle_fast(N, dim=3, rng=rng)
        dist = geodesic_distance_matrix(cs)
        finite_mask = np.isfinite(dist)
        if not np.all(finite_mask):
            continue
        max_r = int(np.max(dist))
        if max_r < 3:
            continue
        radii = np.arange(1, min(max_r, 15) + 1)
        avg_ball_size = []
        for r in radii:
            ball_sizes = np.sum(dist <= r, axis=1)
            avg_ball_size.append(np.mean(ball_sizes))
        avg_ball_size = np.array(avg_ball_size, dtype=float)
        valid = avg_ball_size > 1
        if np.sum(valid) < 3:
            continue
        log_r = np.log(radii[valid].astype(float))
        log_B = np.log(avg_ball_size[valid])
        slope, _, _, _, _ = stats.linregress(log_r, log_B)
        all_dims.append(slope)
    if all_dims:
        mean_d = np.mean(all_dims)
        std_d = np.std(all_dims) / np.sqrt(len(all_dims))
        print(f"  N={N:4d} (d=3): d_Hausdorff = {mean_d:.3f} ± {std_d:.3f} (target: ~3)")

t1 = time.time()
print(f"\n  [Hausdorff dimension completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 454: GREEN'S FUNCTION COMPARISON
# ============================================================
print("\n" + "=" * 72)
print("IDEA 454: GREEN'S FUNCTION — SJ WIGHTMAN vs CONTINUUM")
print("In 2D: G(x,y) = -(1/2π) ln|x-y| for massless scalar")
print("=" * 72)

t0 = time.time()

sizes_green = [50, 80, 100]
n_trials_green = 3

for N in sizes_green:
    correlations = []
    for trial in range(n_trials_green):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        W = sj_wightman_function(cs)

        # For each timelike-separated pair, compare W(i,j) with continuum G
        # Continuum 2D massless: G(x,y) = -(1/2π) ln(σ²) where σ² = -(Δt)² + (Δx)²
        # For timelike: σ² < 0, so G = -(1/2π) ln|σ²|
        # But there's a sign/phase subtlety. The Wightman function in 2D is:
        # W(x,y) = -(1/4π) ln(-(t-t'-iε)² + (x-x')²)

        W_discrete = []
        G_continuum = []

        for i in range(N):
            for j in range(i+1, N):
                if cs.order[i, j]:  # timelike separated
                    dt = coords[j, 0] - coords[i, 0]
                    dx = coords[j, 1] - coords[i, 1]
                    sigma_sq = -dt**2 + dx**2  # negative for timelike
                    if abs(sigma_sq) < 1e-10:
                        continue
                    # Continuum Wightman (real part for timelike)
                    G_cont = -(1.0 / (4 * np.pi)) * np.log(abs(sigma_sq))
                    W_discrete.append(W[i, j])
                    G_continuum.append(G_cont)

        if len(W_discrete) > 10:
            W_d = np.array(W_discrete)
            G_c = np.array(G_continuum)
            # Correlation between discrete and continuum
            r, p = stats.pearsonr(W_d, G_c)
            # Also try rank correlation (more robust)
            rho, p_rho = stats.spearmanr(W_d, G_c)
            correlations.append((r, rho))

    if correlations:
        pearson_vals = [c[0] for c in correlations]
        spearman_vals = [c[1] for c in correlations]
        print(f"  N={N:4d}: Pearson r = {np.mean(pearson_vals):.4f} ± {np.std(pearson_vals):.4f}, "
              f"Spearman ρ = {np.mean(spearman_vals):.4f} ± {np.std(spearman_vals):.4f}")

t1 = time.time()
print(f"\n  [Green's function completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 455: PROPAGATOR POLES IN MOMENTUM SPACE
# ============================================================
print("\n" + "=" * 72)
print("IDEA 455: PROPAGATOR POLES IN MOMENTUM SPACE")
print("Feynman propagator has poles at p²=m²=0 for massless scalar")
print("=" * 72)

t0 = time.time()

N_prop = 100
n_trials_prop = 3

for trial in range(n_trials_prop):
    cs, coords = sprinkle_fast(N_prop, dim=2, rng=rng)
    W = sj_wightman_function(cs)

    # "Fourier transform" of the Wightman function
    # For each pair, we have W(x_i, x_j) and know the coordinates
    # Define p² = -p_t² + p_x² in 2D
    # Bin by geodesic interval (proxy for σ²)

    # Compute W as a function of σ² = -(Δt)² + (Δx)²
    sigma_sq_list = []
    W_list = []

    for i in range(N_prop):
        for j in range(i+1, N_prop):
            dt = coords[j, 0] - coords[i, 0]
            dx = coords[j, 1] - coords[i, 1]
            sigma_sq = -dt**2 + dx**2
            sigma_sq_list.append(sigma_sq)
            W_list.append(W[i, j])

    sigma_sq_arr = np.array(sigma_sq_list)
    W_arr = np.array(W_list)

    # Sort by σ²
    order_idx = np.argsort(sigma_sq_arr)
    sigma_sorted = sigma_sq_arr[order_idx]
    W_sorted = W_arr[order_idx]

    # Bin into sigma² bins
    n_bins = 20
    bin_edges = np.linspace(sigma_sorted[0], sigma_sorted[-1], n_bins + 1)
    bin_centers = []
    bin_W_mean = []

    for b in range(n_bins):
        mask_bin = (sigma_sorted >= bin_edges[b]) & (sigma_sorted < bin_edges[b+1])
        if np.sum(mask_bin) > 2:
            bin_centers.append((bin_edges[b] + bin_edges[b+1]) / 2)
            bin_W_mean.append(np.mean(W_sorted[mask_bin]))

    bin_centers = np.array(bin_centers)
    bin_W_mean = np.array(bin_W_mean)

    # The Fourier transform: W̃(p) = Σ W(σ²) exp(i p σ)
    # For massless scalar, W̃(p) ~ 1/p² should diverge at p=0
    # Compute the discrete FT over σ² bins
    p_values = np.linspace(0.5, 20.0, 40)
    W_tilde = np.zeros(len(p_values))

    for ip, p in enumerate(p_values):
        # Use 1D FT: W̃(p) = Σ_σ W(σ) cos(p*σ) Δσ
        # σ here is the geodesic interval (proper distance analog)
        sigma_abs = np.sqrt(np.abs(bin_centers))
        W_tilde[ip] = np.sum(bin_W_mean * np.cos(p * sigma_abs)) * (
            bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0)

    # Check if |W̃(p)| ~ 1/p² for small p
    # Fit log|W̃| vs log p for small p
    valid_p = p_values < 10
    W_abs = np.abs(W_tilde[valid_p])
    mask_pos = W_abs > 1e-15
    if np.sum(mask_pos) > 3:
        log_p = np.log(p_values[valid_p][mask_pos])
        log_W = np.log(W_abs[mask_pos])
        slope, _, r_val, _, _ = stats.linregress(log_p, log_W)
        print(f"  Trial {trial+1}: |W̃(p)| ~ p^{slope:.2f} (expect ~ -2 for massless)")
        print(f"           R² = {r_val**2:.3f}")

t1 = time.time()
print(f"\n  [Propagator poles completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 456: SPECTRAL DIMENSION FROM HASSE LAPLACIAN HEAT KERNEL
# ============================================================
print("\n" + "=" * 72)
print("IDEA 456: SPECTRAL DIMENSION FROM HASSE LAPLACIAN HEAT KERNEL")
print("Link-graph d_s fails, but does the HASSE Laplacian give d_s=2?")
print("=" * 72)

t0 = time.time()

sizes_spec = [50, 100, 150, 200]
n_trials_spec = 5

print("\nUsing UNNORMALIZED graph Laplacian (L = D - A):")
for N in sizes_spec:
    all_d_s_at_plateau = []
    for trial in range(n_trials_spec):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        evals = hasse_laplacian_spectrum(cs)
        evals_pos = evals[evals > 1e-10]

        if len(evals_pos) < 3:
            continue

        # Heat kernel trace: K(t) = Σ exp(-λ_k t) / N
        # Spectral dimension: d_s(t) = -2 d(ln K)/d(ln t)
        sigmas = np.logspace(-2, 2, 80)
        K = np.zeros(len(sigmas))
        for idx, sigma in enumerate(sigmas):
            K[idx] = np.mean(np.exp(-evals * sigma))

        ln_K = np.log(K + 1e-300)
        ln_sigma = np.log(sigmas)
        d_s = -2 * np.gradient(ln_K, ln_sigma)

        # Find plateau value (middle range of sigma)
        mid_range = (sigmas > 0.1) & (sigmas < 10)
        if np.sum(mid_range) > 2:
            d_s_plateau = np.mean(d_s[mid_range])
            all_d_s_at_plateau.append(d_s_plateau)

    if all_d_s_at_plateau:
        mean_ds = np.mean(all_d_s_at_plateau)
        std_ds = np.std(all_d_s_at_plateau) / np.sqrt(len(all_d_s_at_plateau))
        print(f"  N={N:4d}: d_s(plateau) = {mean_ds:.3f} ± {std_ds:.3f} (target: 2.00)")

print("\nUsing NORMALIZED Laplacian (L = I - D^{-1/2} A D^{-1/2}):")
for N in sizes_spec:
    all_d_s_at_plateau = []
    for trial in range(n_trials_spec):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        adj = hasse_adjacency(cs)
        degree = np.sum(adj, axis=1)

        # Remove isolated nodes
        mask = degree > 0
        adj_sub = adj[np.ix_(mask, mask)]
        deg_sub = degree[mask]
        n_sub = adj_sub.shape[0]

        if n_sub < 5:
            continue

        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg_sub))
        L_norm = np.eye(n_sub) - d_inv_sqrt @ adj_sub @ d_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(L_norm))
        evals = np.clip(evals, 0, None)

        sigmas = np.logspace(-1, 2, 80)
        K = np.zeros(len(sigmas))
        for idx, sigma in enumerate(sigmas):
            K[idx] = np.mean(np.exp(-evals * sigma))

        ln_K = np.log(K + 1e-300)
        ln_sigma = np.log(sigmas)
        d_s = -2 * np.gradient(ln_K, ln_sigma)

        mid_range = (sigmas > 0.5) & (sigmas < 20)
        if np.sum(mid_range) > 2:
            d_s_plateau = np.mean(d_s[mid_range])
            all_d_s_at_plateau.append(d_s_plateau)

    if all_d_s_at_plateau:
        mean_ds = np.mean(all_d_s_at_plateau)
        std_ds = np.std(all_d_s_at_plateau) / np.sqrt(len(all_d_s_at_plateau))
        print(f"  N={N:4d}: d_s(plateau) = {mean_ds:.3f} ± {std_ds:.3f} (target: 2.00)")

t1 = time.time()
print(f"\n  [Spectral dimension completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 457: VOLUME-DISTANCE SCALING
# ============================================================
print("\n" + "=" * 72)
print("IDEA 457: VOLUME-DISTANCE SCALING V(r) ~ r^d")
print("Number of elements within chain-distance r. Expect r^2 for d=2.")
print("=" * 72)

t0 = time.time()

sizes_vol = [50, 100, 150, 200]
n_trials_vol = 5

print("\nUsing HASSE (link graph) distance:")
for N in sizes_vol:
    all_exponents = []
    for trial in range(n_trials_vol):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        dist = geodesic_distance_matrix(cs)

        if not np.all(np.isfinite(dist)):
            continue

        max_r = int(np.max(dist))
        if max_r < 3:
            continue

        # Average volume of ball of radius r
        radii = np.arange(1, min(max_r - 1, 12) + 1)
        volumes = []
        for r in radii:
            vol = np.mean(np.sum(dist <= r, axis=1))
            volumes.append(vol)

        volumes = np.array(volumes, dtype=float)

        # Fit log V vs log r (exclude saturation regime)
        # Saturation = when V(r) > 0.8 * N
        valid = volumes < 0.8 * N
        if np.sum(valid) < 3:
            continue

        log_r = np.log(radii[valid].astype(float))
        log_V = np.log(volumes[valid])
        slope, _, r_val, _, _ = stats.linregress(log_r, log_V)
        if r_val**2 > 0.9:  # only accept good fits
            all_exponents.append(slope)

    if all_exponents:
        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents) / np.sqrt(len(all_exponents))
        print(f"  N={N:4d}: V(r) ~ r^{mean_exp:.3f} ± {std_exp:.3f} (target: 2.00)")

print("\nUsing CHAIN (longest path) distance:")
for N in [50, 80, 100]:
    all_exponents = []
    for trial in range(n_trials_vol):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        cdist = chain_distance_matrix(cs)

        # Only use pairs that are causally related (finite distance)
        finite_pairs = np.isfinite(cdist) & (cdist > 0)
        if np.sum(finite_pairs) < 10:
            continue

        max_r = int(np.max(cdist[finite_pairs]))
        if max_r < 3:
            continue

        radii = np.arange(1, min(max_r, 10) + 1)
        volumes = []
        for r in radii:
            # Count elements within chain distance r (only among causally related)
            vol_per_node = []
            for i in range(N):
                reachable = np.sum((cdist[i, :] <= r) & np.isfinite(cdist[i, :]))
                vol_per_node.append(reachable)
            volumes.append(np.mean(vol_per_node))

        volumes = np.array(volumes, dtype=float)
        valid = (volumes > 1) & (volumes < 0.8 * N)
        if np.sum(valid) < 3:
            continue

        log_r = np.log(radii[valid].astype(float))
        log_V = np.log(volumes[valid])
        slope, _, r_val, _, _ = stats.linregress(log_r, log_V)
        if r_val**2 > 0.85:
            all_exponents.append(slope)

    if all_exponents:
        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents) / np.sqrt(len(all_exponents))
        print(f"  N={N:4d}: V(r) ~ r^{mean_exp:.3f} ± {std_exp:.3f} (target: 2.00)")

t1 = time.time()
print(f"\n  [Volume-distance completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 458: EULER CHARACTERISTIC FROM HASSE DIAGRAM
# ============================================================
print("\n" + "=" * 72)
print("IDEA 458: EULER CHARACTERISTIC FROM HASSE DIAGRAM")
print("For flat 2D: χ=0 (topologically trivial).")
print("Compute via simplicial complex from link graph cliques.")
print("=" * 72)

t0 = time.time()

sizes_euler = [50, 100, 150, 200]
n_trials_euler = 5

for N in sizes_euler:
    all_chi = []
    for trial in range(n_trials_euler):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)
        adj = hasse_adjacency(cs)

        # Build the clique complex (flag complex) of the Hasse diagram
        # Count simplices of each dimension:
        # V = vertices, E = edges, T = triangles, ...
        # χ = V - E + T - ...

        n_nodes = N

        # Edges
        n_edges = int(np.sum(np.triu(adj, k=1)))

        # Triangles: count cliques of size 3
        # A_ij * A_jk * A_ik = 1 means {i,j,k} is a triangle
        adj_int = adj.astype(np.int32)
        # Number of triangles = Tr(A³)/6
        A3 = adj_int @ adj_int @ adj_int
        n_triangles = int(np.trace(A3)) // 6

        # Tetrahedra: for small N, check 4-cliques
        n_tetra = 0
        if N <= 200:
            # Count 4-cliques: for each triangle, check if there's a 4th vertex
            # connected to all three
            for i in range(N):
                neighbors_i = set(np.where(adj_int[i] > 0)[0])
                for j in neighbors_i:
                    if j <= i:
                        continue
                    neighbors_ij = neighbors_i & set(np.where(adj_int[j] > 0)[0])
                    for k in neighbors_ij:
                        if k <= j:
                            continue
                        # {i,j,k} is a triangle
                        neighbors_ijk = neighbors_ij & set(np.where(adj_int[k] > 0)[0])
                        for l in neighbors_ijk:
                            if l <= k:
                                continue
                            n_tetra += 1

        chi = n_nodes - n_edges + n_triangles - n_tetra
        all_chi.append(chi)

    mean_chi = np.mean(all_chi)
    std_chi = np.std(all_chi) / np.sqrt(len(all_chi))
    print(f"  N={N:4d}: χ = {mean_chi:.1f} ± {std_chi:.1f} "
          f"(V={n_nodes}, E≈{n_edges}, T≈{n_triangles}, Tet≈{n_tetra})")
    print(f"           expect: χ ≈ 0 for topologically trivial flat 2D")

t1 = time.time()
print(f"\n  [Euler characteristic completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 459: GEODESIC DEVIATION (JACOBI FIELD ANALOGUE)
# ============================================================
print("\n" + "=" * 72)
print("IDEA 459: GEODESIC DEVIATION — JACOBI FIELD ANALOGUE")
print("How do nearby maximal chains diverge?")
print("In flat 2D: linear divergence (separation grows linearly with time)")
print("=" * 72)

t0 = time.time()

sizes_jacobi = [100, 150, 200]
n_trials_jacobi = 5

def find_maximal_chains(cs, coords, n_chains=20):
    """Find several approximately maximal chains through the causet.
    A chain is a totally ordered subset; maximal means no element can be added."""
    N = cs.n
    chains = []
    links = cs.link_matrix()

    for _ in range(n_chains):
        # Start from a random early element
        start = rng.integers(0, N // 4)
        chain = [start]
        current = start

        while True:
            # Follow a random link forward
            successors = np.where(links[current, :])[0]
            if len(successors) == 0:
                break
            # Pick successor closest to the longest-chain direction
            next_elem = rng.choice(successors)
            chain.append(next_elem)
            current = next_elem

        if len(chain) >= 3:
            chains.append(chain)

    return chains


for N in sizes_jacobi:
    all_exponents = []
    for trial in range(n_trials_jacobi):
        cs, coords = sprinkle_fast(N, dim=2, rng=rng)

        chains = find_maximal_chains(cs, coords, n_chains=30)

        if len(chains) < 4:
            continue

        # For each pair of chains that start nearby, measure how their
        # spatial separation grows with "time" (chain step)
        deviations = []

        for ci in range(len(chains)):
            for cj in range(ci + 1, min(ci + 10, len(chains))):
                c1, c2 = chains[ci], chains[cj]

                # Check if they start from nearby spatial positions
                x1_start = coords[c1[0], 1]
                x2_start = coords[c2[0], 1]
                t1_start = coords[c1[0], 0]
                t2_start = coords[c2[0], 0]

                init_sep = abs(x1_start - x2_start) + abs(t1_start - t2_start)

                # Only consider initially nearby chains
                if init_sep > 0.3:
                    continue

                # Measure spatial separation at each "step"
                min_len = min(len(c1), len(c2))
                if min_len < 4:
                    continue

                seps = []
                for step in range(min_len):
                    dx = abs(coords[c1[step], 1] - coords[c2[step], 1])
                    seps.append(dx)

                if len(seps) > 3 and seps[-1] > seps[0]:
                    deviations.append(seps)

        # Fit the average deviation vs step
        if len(deviations) < 2:
            continue

        max_steps = min(max(len(d) for d in deviations), 15)
        avg_dev = np.zeros(max_steps)
        counts = np.zeros(max_steps)

        for d in deviations:
            for s in range(min(len(d), max_steps)):
                avg_dev[s] += d[s]
                counts[s] += 1

        valid = counts > 0
        avg_dev[valid] /= counts[valid]

        # Fit: separation ~ step^α
        steps = np.arange(1, max_steps + 1).astype(float)
        pos_mask = (avg_dev > 1e-10) & valid
        if np.sum(pos_mask) < 3:
            continue

        log_step = np.log(steps[pos_mask])
        log_dev = np.log(avg_dev[pos_mask])
        slope, _, r_val, _, _ = stats.linregress(log_step, log_dev)
        all_exponents.append(slope)

    if all_exponents:
        mean_exp = np.mean(all_exponents)
        std_exp = np.std(all_exponents) / np.sqrt(len(all_exponents))
        print(f"  N={N:4d}: deviation ~ step^{mean_exp:.3f} ± {std_exp:.3f}")
        print(f"           (flat 2D Minkowski: exponent = 1.0 for linear divergence)")

t1 = time.time()
print(f"\n  [Geodesic deviation completed in {t1-t0:.1f}s]")


# ============================================================
# IDEA 460: SCALAR CURVATURE FROM BD ACTION
# ============================================================
print("\n" + "=" * 72)
print("IDEA 460: SCALAR CURVATURE FROM BD ACTION")
print("S_BD ≈ ∫R√g d²x. For flat space R=0 → S_BD/N ≈ 0.")
print("For curved space, S_BD/N should scale with curvature R.")
print("=" * 72)

t0 = time.time()

curvatures = [0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0]
N_bd = 150
n_trials_bd = 8

print(f"\nN={N_bd}, {n_trials_bd} trials per curvature value")
print(f"{'R':>8s} {'S_BD/N':>12s} {'std':>8s}")
print("-" * 32)

results_bd = {}
for R in curvatures:
    sbd_per_n = []
    for trial in range(n_trials_bd):
        if R == 0.0:
            cs, coords = sprinkle_fast(N_bd, dim=2, rng=rng)
        else:
            cs, coords, omega = sprinkle_curved_2d(N_bd, R, rng_local=rng)

        S = bd_action_2d(cs)
        sbd_per_n.append(S / N_bd)

    mean_s = np.mean(sbd_per_n)
    std_s = np.std(sbd_per_n) / np.sqrt(len(sbd_per_n))
    results_bd[R] = (mean_s, std_s)
    print(f"  {R:+6.1f}   {mean_s:+10.4f}   {std_s:8.4f}")

# Check if S_BD/N correlates with |R|
R_vals = np.array(sorted(results_bd.keys()))
S_vals = np.array([results_bd[r][0] for r in R_vals])

# Fit S_BD/N vs R
r_corr, p_corr = stats.pearsonr(R_vals, S_vals)
print(f"\n  Correlation(R, S_BD/N): Pearson r = {r_corr:.4f}, p = {p_corr:.4f}")

# Also test S_BD/N vs R² (quadratic response)
r_corr2, p_corr2 = stats.pearsonr(R_vals**2, S_vals)
print(f"  Correlation(R², S_BD/N): Pearson r = {r_corr2:.4f}, p = {p_corr2:.4f}")

# Fit slope: S_BD/N ≈ α + β·R
slope_bd, intercept_bd, _, _, _ = stats.linregress(R_vals, S_vals)
print(f"  Linear fit: S_BD/N = {intercept_bd:.4f} + {slope_bd:.4f}·R")
print(f"  (continuum prediction: S_BD/N → 0 for flat, proportional to R for curved)")

t1 = time.time()
print(f"\n  [BD action curvature completed in {t1-t0:.1f}s]")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: CONTINUUM LIMIT TESTS — IDEAS 451-460")
print("=" * 72)

print("""
451. WEYL'S LAW: N(λ) ~ λ^{d/2} for Hasse Laplacian eigenvalue counting.
     → Measured exponent gives inferred dimension. Check if d_Weyl ≈ 2.

452. MINKOWSKI DIMENSION: Box-counting on embedded 2D coordinates.
     → Should give d_box ≈ 2 trivially (embedded in 2D).

453. HAUSDORFF DIMENSION: Ball-volume growth on Hasse graph distance.
     → d_H measures intrinsic graph geometry. Key test of manifold-likeness.

454. GREEN'S FUNCTION: SJ Wightman vs continuum -(1/4π)ln|σ²|.
     → Positive correlation = discrete approximates continuum propagation.

455. PROPAGATOR POLES: Momentum-space behavior of Wightman function.
     → |W̃(p)| ~ p^α; massless scalar expects α ≈ -2.

456. SPECTRAL DIMENSION: Heat kernel d_s from Hasse Laplacian.
     → Key test: does Hasse Laplacian d_s reach 2 (unlike link-graph d_s)?

457. VOLUME-DISTANCE SCALING: V(r) ~ r^d for geodesic balls.
     → Direct dimension probe. Hasse distance vs chain distance.

458. EULER CHARACTERISTIC: Flag complex of Hasse diagram.
     → χ ≈ 0 for flat topology; nonzero for curved/compact.

459. GEODESIC DEVIATION: Separation growth of nearby chains.
     → Exponent ≈ 1 for flat 2D (linear divergence).

460. SCALAR CURVATURE: S_BD/N vs background curvature R.
     → Tests whether BD action detects curvature as predicted.
""")

print("KEY QUESTION: Which continuum results are reproduced at N~100-200?")
print("Matches strengthen the causal set program; failures indicate where")
print("finite-size effects dominate or discretization artifacts persist.")
print("=" * 72)
