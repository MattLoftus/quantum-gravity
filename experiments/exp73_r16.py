"""
Experiment 73: Round 16 — Ideas 251-260
FRESH APPROACHES to the SJ Vacuum

Key lessons from prior rounds:
- c_eff diverges (not a useful observable alone)
- ER=EPR correlation is generic (appears in null models)
- GUE level spacing is generic (random matrix universality)
- BUT SJ vacuum HAS specific properties: heavy-tailed eigenvalue density
  (kurtosis=53), exact ER=EPR proof for 2-orders, different c_eff than CDT

NEW IDEAS:
251. SJ vacuum with Sorkin-Yazdi TRUNCATION — remove near-zero modes
252. SJ vacuum with 1/2 normalization instead of 2/N
253. Massive scalar at multiple masses — mass gap emergence
254. SJ vacuum on REGULAR causets (lattice-like 2-orders, not random)
255. Two-point function in MOMENTUM space
256. Entanglement entropy of CAUSAL DIAMONDS of varying size
257. SJ vacuum FIDELITY between nearby causets
258. SJ state as MIXED state (partial trace) — purity
259. SJ vacuum on CRYSTALLINE phase (deep KR regime)
260. Compare SJ Wightman with RETARDED Green's function directly
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, linalg
from scipy.optimize import curve_fit
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function, entanglement_entropy
from causal_sets.bd_action import count_intervals_by_size
import time

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def make_2order_causet(N, rng):
    to = TwoOrder(N, rng=rng)
    return to, to.to_causet()


def sj_wightman_detailed(cs):
    """Compute W with full spectral details."""
    N = cs.n
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real

    pos = evals > 1e-12
    W = np.zeros((N, N), dtype=complex)
    for k in range(N):
        if pos[k]:
            v = evecs[:, k]
            W += evals[k] * np.outer(v, v.conj())
    W = W.real
    return W, evals, evecs


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


def random_dag(N, density, rng):
    """Random DAG with given density (transitively closed)."""
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


def retarded_green(cs):
    """Retarded Green's function: G_R[i,j] = C[i,j] (upper triangular causal matrix).
    For massless scalar, G_R is proportional to the causal matrix.
    With normalization: G_R = (1/N) * C."""
    N = cs.n
    C = cs.order.astype(float)
    return C / N


print("=" * 78)
print("EXPERIMENT 73: ROUND 16 — IDEAS 251-260")
print("FRESH APPROACHES TO THE SJ VACUUM")
print("=" * 78)


# ================================================================
# IDEA 251: SJ VACUUM WITH SORKIN-YAZDI TRUNCATION
# ================================================================
print("\n" + "=" * 78)
print("IDEA 251: SJ vacuum with Sorkin-Yazdi TRUNCATION")
print("Remove near-zero eigenvalue modes from iDelta. Does c_eff improve?")
print("Yazdi (2017) showed truncation removes IR artifacts.")
print("=" * 78)

N = 40
n_trials = 20
truncation_fracs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  # fraction of modes to remove

print(f"N={N}, {n_trials} trials, truncation fractions: {truncation_fracs}")

results_251 = {f: {'entropies': [], 'max_evals': []} for f in truncation_fracs}

for trial in range(n_trials):
    to, cs = make_2order_causet(N, rng)
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)
    H = 1j * Delta
    evals, evecs = np.linalg.eigh(H)
    evals = evals.real

    for frac in truncation_fracs:
        # Truncation: zero out the smallest |eigenvalue| modes
        n_remove = int(frac * N)
        abs_evals = np.abs(evals)
        sorted_idx = np.argsort(abs_evals)

        trunc_evals = evals.copy()
        if n_remove > 0:
            trunc_evals[sorted_idx[:n_remove]] = 0.0

        # Build truncated W from positive truncated eigenvalues
        pos = trunc_evals > 1e-12
        W_trunc = np.zeros((N, N))
        for k in range(N):
            if pos[k]:
                v = evecs[:, k].real  # eigenvectors of Hermitian matrix
                W_trunc += trunc_evals[k] * np.outer(v, v)

        # Entanglement entropy for half-partition
        half = list(range(N // 2))
        S = entanglement_entropy(W_trunc, half)
        results_251[frac]['entropies'].append(S)
        results_251[frac]['max_evals'].append(np.max(np.abs(trunc_evals)))

print("\n  Truncation | Mean S(N/2) | Std S(N/2) | c_eff proxy")
print("  " + "-" * 60)
for frac in truncation_fracs:
    ents = results_251[frac]['entropies']
    mean_S = np.mean(ents)
    std_S = np.std(ents)
    # c_eff ~ 3*S / ln(N) for 1+1D CFT
    c_eff = 3 * mean_S / np.log(N)
    print(f"  {frac:10.0%} | {mean_S:11.4f} | {std_S:10.4f} | {c_eff:.4f}")

# Test: does truncation STABILIZE c_eff across different N?
print("\n  Truncation effect on c_eff scaling:")
Ns_251 = [20, 30, 40, 50]
for frac in [0.0, 0.1, 0.3]:
    c_effs = []
    for Ni in Ns_251:
        ents_Ni = []
        for _ in range(10):
            to_i, cs_i = make_2order_causet(Ni, rng)
            Ci = cs_i.order.astype(float)
            Di = (2.0 / Ni) * (Ci.T - Ci)
            Hi = 1j * Di
            ei, vi = np.linalg.eigh(Hi)
            ei = ei.real

            n_rem = int(frac * Ni)
            abs_ei = np.abs(ei)
            si = np.argsort(abs_ei)
            te = ei.copy()
            if n_rem > 0:
                te[si[:n_rem]] = 0.0

            pos_i = te > 1e-12
            Wi = np.zeros((Ni, Ni))
            for k in range(Ni):
                if pos_i[k]:
                    vk = vi[:, k].real
                    Wi += te[k] * np.outer(vk, vk)

            S_i = entanglement_entropy(Wi, list(range(Ni // 2)))
            ents_Ni.append(S_i)
        c_effs.append(3 * np.mean(ents_Ni) / np.log(Ni))

    print(f"  frac={frac:.0%}: c_eff(N) = {['%.3f' % c for c in c_effs]}")

# Verdict
c0 = [3 * np.mean(results_251[0.0]['entropies']) / np.log(N)]
c3 = [3 * np.mean(results_251[0.3]['entropies']) / np.log(N)]
print(f"\n  VERDICT: Truncation at 30% {'REDUCES' if c3[0] < c0[0] else 'INCREASES'} c_eff from {c0[0]:.3f} to {c3[0]:.3f}")
# Check if N-scaling improves
print("  (Check N-scaling above — does truncated c_eff converge to a constant?)")


# ================================================================
# IDEA 252: SJ VACUUM WITH 1/2 NORMALIZATION
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 252: SJ vacuum with 1/2 normalization (not 2/N)")
print("Standard: iDelta = (2/N)(C^T - C). Alternative: iDelta = (1/2)(C^T - C)")
print("The 1/2 comes from the continuum; 2/N is the discrete version.")
print("Different normalization = different physics regime?")
print("=" * 78)

N = 40
n_trials = 15
norms = {'2/N': 2.0 / N, '1/2': 0.5, '1/N': 1.0 / N, '1/sqrt(N)': 1.0 / np.sqrt(N)}

print(f"N={N}, {n_trials} trials")

results_252 = {}
for name, norm_factor in norms.items():
    entropies = []
    max_evals_list = []
    kurtoses = []
    r_vals = []

    for trial in range(n_trials):
        to, cs = make_2order_causet(N, rng)
        C = cs.order.astype(float)
        Delta = norm_factor * (C.T - C)
        H = 1j * Delta
        evals = np.linalg.eigvalsh(H).real

        max_evals_list.append(np.max(np.abs(evals)))
        kurtoses.append(stats.kurtosis(evals))
        r_vals.append(level_spacing_ratio(evals))

        # Build W from positive eigenvalues
        evals_full, evecs = np.linalg.eigh(H)
        evals_full = evals_full.real
        pos = evals_full > 1e-12
        W = np.zeros((N, N))
        for k in range(N):
            if pos[k]:
                v = evecs[:, k].real
                W += evals_full[k] * np.outer(v, v)

        S = entanglement_entropy(W, list(range(N // 2)))
        entropies.append(S)

    results_252[name] = {
        'S_mean': np.mean(entropies), 'S_std': np.std(entropies),
        'max_eval': np.mean(max_evals_list),
        'kurtosis': np.mean(kurtoses), 'r': np.mean(r_vals)
    }

print("\n  Normalization | S(N/2) mean | max|lambda| | kurtosis | <r>")
print("  " + "-" * 70)
for name in norms:
    r = results_252[name]
    print(f"  {name:13s} | {r['S_mean']:11.4f} | {r['max_eval']:11.4f} | {r['kurtosis']:8.2f} | {r['r']:.4f}")

# Key: does 1/sqrt(N) give bounded eigenvalues AND finite entropy?
print(f"\n  VERDICT: 1/sqrt(N) normalization gives max|lambda|={results_252['1/sqrt(N)']['max_eval']:.3f}")
print(f"  (Should be O(1) if this is the 'right' scaling for a continuum limit)")
print(f"  Kurtosis comparison: 2/N → {results_252['2/N']['kurtosis']:.1f}, "
      f"1/sqrt(N) → {results_252['1/sqrt(N)']['kurtosis']:.1f}")


# ================================================================
# IDEA 253: MASSIVE SCALAR — MASS GAP EMERGENCE
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 253: Massive scalar at multiple masses — mass gap emergence")
print("Modify iDelta → iDelta + m^2 * I. How does the SJ spectrum change?")
print("At what mass does the spectrum develop a gap?")
print("=" * 78)

N = 40
masses = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
n_trials = 15

print(f"N={N}, masses={masses}, {n_trials} trials")

results_253 = {m: {'gaps': [], 'entropies': [], 'n_pos': []} for m in masses}

for trial in range(n_trials):
    to, cs = make_2order_causet(N, rng)
    C = cs.order.astype(float)
    Delta = (2.0 / N) * (C.T - C)

    for m in masses:
        # Massive Pauli-Jordan: the mass enters through the retarded propagator
        # For massive scalar: G_R satisfies (Box + m^2)G_R = delta
        # On a causet, Sorkin's d'Alembertian B gives B*G_R = delta
        # The simplest approach: modify the eigenvalues of iDelta
        # lambda_k -> sign(lambda_k) * sqrt(lambda_k^2 + m^2) would be wrong
        # Correct approach: add mass to the operator
        # H_m = i*Delta + m^2 * identity shift? No — mass modifies the propagator.
        #
        # Johnston (2009) eq 3.14: for massive scalar, the retarded propagator
        # is modified, and iDelta_m = G_R_m - G_A_m.
        # For simplicity, we use the approximation:
        # iDelta_m = iDelta * exp(-m^2 * tau^2) where tau is a "proper time" proxy
        # Actually, simplest correct approach for 2-orders:
        # Just add m^2 to the Hamiltonian: H_m = H + m^2 * (identity diagonal part)
        # This shifts all eigenvalues by m^2.
        # But more physical: suppress long-range correlations.
        # Use Yukawa-like damping: Delta_m[i,j] = Delta[i,j] * exp(-m * d(i,j))
        # where d(i,j) is the geodesic distance.

        if m == 0:
            H_m = 1j * Delta
        else:
            # Geodesic distance on the causet (longest chain between i and j)
            # Approximation: use the order matrix to define a distance
            # d(i,j) = 1 if directly related, else path length through causal structure
            # For efficiency, use "causal interval size" as distance proxy
            order_int = C.astype(np.int32)
            # Simple approach: multiply Delta by exp(-m * |i-j|/N) as a crude mass
            # (i,j in natural ordering ~ time separation)
            idx = np.arange(N)
            dist = np.abs(idx[:, None] - idx[None, :]).astype(float) / N
            damping = np.exp(-m * dist)
            Delta_m = Delta * damping
            H_m = 1j * Delta_m

        evals, evecs = np.linalg.eigh(H_m)
        evals = evals.real

        # Spectral gap: smallest positive eigenvalue
        pos_evals = evals[evals > 1e-12]
        gap = np.min(pos_evals) if len(pos_evals) > 0 else 0.0
        results_253[m]['gaps'].append(gap)
        results_253[m]['n_pos'].append(len(pos_evals))

        # Entropy
        pos = evals > 1e-12
        W = np.zeros((N, N))
        for k in range(N):
            if pos[k]:
                v = evecs[:, k].real
                W += evals[k] * np.outer(v, v)
        S = entanglement_entropy(W, list(range(N // 2)))
        results_253[m]['entropies'].append(S)

print("\n  Mass | Spectral gap | S(N/2) | N_positive modes")
print("  " + "-" * 60)
for m in masses:
    gap = np.mean(results_253[m]['gaps'])
    S = np.mean(results_253[m]['entropies'])
    np_modes = np.mean(results_253[m]['n_pos'])
    print(f"  {m:5.1f} | {gap:12.6f} | {S:6.3f} | {np_modes:.1f}")

# Does gap scale linearly with m?
gaps_arr = [np.mean(results_253[m]['gaps']) for m in masses if m > 0]
masses_pos = [m for m in masses if m > 0]
if len(gaps_arr) > 2:
    slope, intercept, r, p, se = stats.linregress(np.log(masses_pos), np.log(gaps_arr))
    print(f"\n  Gap scaling: gap ~ m^{slope:.2f} (r^2={r**2:.3f})")
    print(f"  VERDICT: Mass {'OPENS' if slope > 0.3 else 'does NOT clearly open'} a spectral gap")


# ================================================================
# IDEA 254: SJ VACUUM ON REGULAR (LATTICE-LIKE) CAUSETS
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 254: SJ vacuum on REGULAR causets (lattice-like 2-orders)")
print("Instead of random permutations, use structured permutations that")
print("approximate a regular lattice. Compare spectral properties.")
print("=" * 78)

N = 40
n_trials = 15

# Regular 2-order: u = identity, v = identity (total order = 1D chain)
# Slightly perturbed regular: u = identity, v = small perturbation of identity
# "Lattice" 2-order: u, v arranged as a 2D grid

def make_regular_causet(N, perturbation=0.0, rng=None):
    """Make a 2-order that approximates a regular lattice.
    perturbation=0 gives total order (chain).
    perturbation=1 gives fully random."""
    u = np.arange(N, dtype=float)
    v = np.arange(N, dtype=float)
    if perturbation > 0 and rng is not None:
        # Add Gaussian noise and re-rank
        noise_u = rng.normal(0, perturbation * N, N)
        noise_v = rng.normal(0, perturbation * N, N)
        u = np.argsort(np.argsort(u + noise_u))  # rank after perturbation
        v = np.argsort(np.argsort(v + noise_v))
    else:
        u = np.arange(N)
        v = np.arange(N)
    cs = FastCausalSet(N)
    cs.order = (u[:, None] < u[None, :]) & (v[:, None] < v[None, :])
    return cs

def make_grid_causet(side):
    """Make a 2D grid causet. N = side^2. Element (a,b) precedes (c,d) iff a<c AND b<d."""
    N = side * side
    cs = FastCausalSet(N)
    for i in range(N):
        ai, bi = i // side, i % side
        for j in range(N):
            aj, bj = j // side, j % side
            if ai < aj and bi < bj:
                cs.order[i, j] = True
    return cs

perturbations = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]

print(f"N={N}, perturbation levels: {perturbations}")
print("\n  Part A: Perturbed chain → random 2-order")

results_254 = {}
for p in perturbations:
    evals_list = []
    kurtoses = []
    r_vals = []
    entropies = []
    for trial in range(n_trials):
        cs = make_regular_causet(N, perturbation=p, rng=rng)
        ev = sj_eigenvalues(cs)
        evals_list.append(ev)
        kurtoses.append(stats.kurtosis(ev))
        r_vals.append(level_spacing_ratio(ev))

        W, _, _ = sj_wightman_detailed(cs)
        S = entanglement_entropy(W, list(range(N // 2)))
        entropies.append(S)

    results_254[p] = {
        'kurtosis': np.mean(kurtoses), 'r': np.mean(r_vals),
        'S': np.mean(entropies), 'S_std': np.std(entropies)
    }

print("\n  Perturbation | kurtosis | <r>   | S(N/2)")
print("  " + "-" * 50)
for p in perturbations:
    r = results_254[p]
    print(f"  {p:12.2f} | {r['kurtosis']:8.2f} | {r['r']:.4f} | {r['S']:.4f}")

# Part B: Grid causet
print("\n  Part B: Grid causets (2D lattice)")
for side in [4, 5, 6]:
    cs_grid = make_grid_causet(side)
    Ng = side * side
    ev_grid = sj_eigenvalues(cs_grid)
    kurt_grid = stats.kurtosis(ev_grid)
    r_grid = level_spacing_ratio(ev_grid)
    W_grid, _, _ = sj_wightman_detailed(cs_grid)
    S_grid = entanglement_entropy(W_grid, list(range(Ng // 2)))
    of_grid = cs_grid.ordering_fraction()
    print(f"  {side}x{side} (N={Ng:2d}): kurtosis={kurt_grid:.2f}, <r>={r_grid:.4f}, "
          f"S={S_grid:.4f}, f={of_grid:.3f}")

print("\n  VERDICT: Compare kurtosis of chain (p=0) vs random (p=1) — "
      f"{results_254[0.0]['kurtosis']:.1f} vs {results_254[1.0]['kurtosis']:.1f}")


# ================================================================
# IDEA 255: TWO-POINT FUNCTION IN MOMENTUM SPACE
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 255: SJ two-point function in MOMENTUM space")
print("For 2-orders, elements have 'coordinates' (u_i, v_i).")
print("Define lightcone coords: t=(u+v)/2, x=(u-v)/2. Then Fourier transform W.")
print("=" * 78)

N = 50
n_trials = 10

print(f"N={N}, {n_trials} trials")

momentum_spectra = []
for trial in range(n_trials):
    to = TwoOrder(N, rng=rng)
    cs = to.to_causet()
    W, evals, evecs = sj_wightman_detailed(cs)

    # "Coordinates" from the two permutations
    u = to.u.astype(float) / N
    v = to.v.astype(float) / N
    t = (u + v) / 2
    x = (u - v) / 2

    # Fourier transform of W(x_i, x_j) to get W(k)
    # W(k) = sum_{i,j} W[i,j] * exp(-i*k*(x_i - x_j))
    n_k = 30
    k_vals = np.linspace(-N * np.pi, N * np.pi, n_k)
    W_k = np.zeros(n_k)
    for ik, k in enumerate(k_vals):
        phases = np.exp(-1j * k * x)
        W_k[ik] = np.abs(np.dot(phases, W @ phases.conj())) / N

    momentum_spectra.append(W_k)

avg_Wk = np.mean(momentum_spectra, 0)
# Fit power law in |k|
k_pos = k_vals[k_vals > 0.5]
Wk_pos = avg_Wk[k_vals > 0.5]
if len(k_pos) > 3 and np.all(Wk_pos > 0):
    try:
        slope, intercept, r, p, se = stats.linregress(np.log(k_pos), np.log(Wk_pos))
        print(f"\n  W(k) ~ k^{slope:.2f} (r^2={r**2:.3f})")
        print(f"  Continuum expectation for massless 1+1D: W(k) ~ 1/|k| (slope=-1)")
        print(f"  VERDICT: {'MATCHES' if -1.5 < slope < -0.5 else 'DOES NOT MATCH'} "
              f"continuum (slope={slope:.2f})")
    except:
        print("  Could not fit power law to W(k)")
else:
    print("  W(k) data insufficient for power law fit")

# Print spectrum
print("\n  k (sample) | W(k)")
print("  " + "-" * 30)
for i in range(0, n_k, 5):
    print(f"  {k_vals[i]:10.2f}  | {avg_Wk[i]:.6f}")


# ================================================================
# IDEA 256: ENTANGLEMENT ENTROPY OF CAUSAL DIAMONDS
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 256: Entanglement entropy of CAUSAL DIAMONDS of varying size")
print("Pick element pairs (i,j) with i<j, define diamond = {k: i<k<j}.")
print("Compute S(diamond) vs diamond_size. Area law or volume law?")
print("=" * 78)

N = 50
n_trials = 10

print(f"N={N}, {n_trials} trials")

# Collect (diamond_size, entropy) pairs
diamond_data = []
for trial in range(n_trials):
    to, cs = make_2order_causet(N, rng)
    W = sj_wightman_function(cs)
    order = cs.order

    # Find all diamonds of various sizes
    for i in range(N):
        for j in range(i + 2, min(i + N, N)):
            if order[i, j]:
                # Diamond = elements causally between i and j
                diamond = [k for k in range(N) if order[i, k] and order[k, j]]
                if 2 <= len(diamond) <= N // 2:
                    S = entanglement_entropy(W, diamond)
                    diamond_data.append((len(diamond), S))

if diamond_data:
    sizes, entropies = zip(*diamond_data)
    sizes = np.array(sizes)
    entropies = np.array(entropies)

    # Bin by size
    unique_sizes = np.unique(sizes)
    print(f"\n  Found {len(diamond_data)} diamonds with sizes {int(sizes.min())}-{int(sizes.max())}")
    print("\n  Diamond size | Mean S | Std S | Count")
    print("  " + "-" * 50)
    binned_sizes = []
    binned_S = []
    for s in sorted(unique_sizes):
        mask = sizes == s
        if np.sum(mask) >= 3:
            binned_sizes.append(s)
            binned_S.append(np.mean(entropies[mask]))
            print(f"  {s:12d} | {np.mean(entropies[mask]):6.3f} | "
                  f"{np.std(entropies[mask]):5.3f} | {int(np.sum(mask))}")

    # Fit: S ~ a * L^alpha (area law: alpha ~ (d-2)/(d-1), volume law: alpha ~ 1)
    if len(binned_sizes) > 3:
        bs = np.array(binned_sizes, dtype=float)
        be = np.array(binned_S)
        mask_fit = be > 0
        if np.sum(mask_fit) > 2:
            slope, intercept, r, p, se = stats.linregress(
                np.log(bs[mask_fit]), np.log(be[mask_fit]))
            print(f"\n  S ~ L^{slope:.2f} (r^2={r**2:.3f})")
            print(f"  Volume law: alpha=1, Area law in 1+1D: alpha=0 (log correction)")
            print(f"  VERDICT: {'VOLUME LAW' if slope > 0.7 else 'SUB-VOLUME' if slope > 0.3 else 'AREA-LIKE'}")
else:
    print("  No diamonds found (unexpected)")


# ================================================================
# IDEA 257: SJ VACUUM FIDELITY BETWEEN NEARBY CAUSETS
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 257: SJ vacuum FIDELITY between nearby causets")
print("Start with a 2-order, make ONE swap move. How much does W change?")
print("Fidelity F = Tr(sqrt(sqrt(W1) W2 sqrt(W1))) for density matrices.")
print("For Wightman functions, use overlap: F = Tr(W1 W2) / sqrt(Tr(W1^2) Tr(W2^2))")
print("=" * 78)

N = 30
n_swaps_list = [1, 2, 5, 10, 20, 50]
n_trials = 15

print(f"N={N}, swap counts: {n_swaps_list}, {n_trials} trials per swap count")

results_257 = {}
for n_swaps in n_swaps_list:
    fidelities = []
    for trial in range(n_trials):
        to1 = TwoOrder(N, rng=rng)
        cs1 = to1.to_causet()
        W1 = sj_wightman_function(cs1)

        # Apply n_swaps moves
        to2 = to1.copy()
        for _ in range(n_swaps):
            to2 = swap_move(to2, rng)
        cs2 = to2.to_causet()
        W2 = sj_wightman_function(cs2)

        # Hilbert-Schmidt fidelity (overlap)
        num = np.trace(W1 @ W2)
        denom = np.sqrt(np.trace(W1 @ W1) * np.trace(W2 @ W2))
        F = num / denom if denom > 1e-15 else 0.0
        fidelities.append(F)

    results_257[n_swaps] = {'mean': np.mean(fidelities), 'std': np.std(fidelities)}

print("\n  N_swaps | Mean Fidelity | Std")
print("  " + "-" * 40)
for ns in n_swaps_list:
    r = results_257[ns]
    print(f"  {ns:7d} | {r['mean']:13.6f} | {r['std']:.6f}")

# Fit decay: F ~ exp(-alpha * n_swaps)?
ns_arr = np.array(n_swaps_list, dtype=float)
f_arr = np.array([results_257[ns]['mean'] for ns in n_swaps_list])
if np.all(f_arr > 0):
    try:
        log_f = np.log(f_arr)
        slope, intercept, r, p, se = stats.linregress(ns_arr, log_f)
        print(f"\n  Fidelity decay: F ~ exp({slope:.4f} * n_swaps)")
        print(f"  Decay rate: {-slope:.4f} per swap (r^2={r**2:.3f})")
        print(f"  VERDICT: SJ vacuum {'IS' if -slope > 0.01 else 'is NOT'} sensitive to single-swap perturbations")
    except:
        print("  Could not fit exponential decay")


# ================================================================
# IDEA 258: SJ STATE AS MIXED STATE — PURITY
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 258: SJ state as MIXED state (partial trace) — purity")
print("Partition causet into region A (first half) and complement B.")
print("Reduced state rho_A from W. Purity = Tr(rho_A^2).")
print("How does purity scale with N? Compare causets vs random DAGs.")
print("=" * 78)

Ns = [15, 20, 25, 30, 40, 50]
n_trials = 12

print(f"N values: {Ns}, {n_trials} trials each")

results_258_causet = {}
results_258_dag = {}

for Ni in Ns:
    purities_cs = []
    purities_dag = []

    for trial in range(n_trials):
        # Causet
        to, cs = make_2order_causet(Ni, rng)
        W_cs = sj_wightman_function(cs)
        A = list(range(Ni // 2))
        W_A = W_cs[np.ix_(A, A)]
        evals_A = np.linalg.eigvalsh(W_A)
        evals_A = np.clip(evals_A, 0, 1)
        # Purity of Gaussian state from Wightman eigenvalues
        # For a Gaussian state, purity = product_k 1/sqrt(nu_k(1-nu_k)) ... but this diverges
        # Better: use the von Neumann entropy S = -Tr(rho ln rho) and
        # purity = exp(-S_2) where S_2 = -ln(Tr(rho^2)) is Renyi-2 entropy
        # For Gaussian state: S_2 = -sum_k ln(nu_k^2 + (1-nu_k)^2)
        nu = evals_A
        nu = np.clip(nu, 1e-10, 1 - 1e-10)
        S2 = -np.sum(np.log(nu**2 + (1 - nu)**2))
        purity = np.exp(-S2)
        purities_cs.append(purity)

        # Random DAG null
        of_cs = cs.ordering_fraction()
        dag = random_dag(Ni, of_cs, rng)
        W_dag = sj_wightman_function(dag)
        W_A_dag = W_dag[np.ix_(A, A)]
        evals_A_dag = np.linalg.eigvalsh(W_A_dag)
        evals_A_dag = np.clip(evals_A_dag, 0, 1)
        nu_dag = np.clip(evals_A_dag, 1e-10, 1 - 1e-10)
        S2_dag = -np.sum(np.log(nu_dag**2 + (1 - nu_dag)**2))
        purity_dag = np.exp(-S2_dag)
        purities_dag.append(purity_dag)

    results_258_causet[Ni] = {'mean': np.mean(purities_cs), 'std': np.std(purities_cs)}
    results_258_dag[Ni] = {'mean': np.mean(purities_dag), 'std': np.std(purities_dag)}

print("\n     N | Purity (causet) | Purity (DAG) | Ratio")
print("  " + "-" * 55)
for Ni in Ns:
    pc = results_258_causet[Ni]['mean']
    pd = results_258_dag[Ni]['mean']
    ratio = pc / pd if pd > 1e-15 else float('inf')
    print(f"  {Ni:4d} | {pc:15.6e} | {pd:12.6e} | {ratio:.3f}")

# Fit scaling: purity ~ exp(-alpha * N)
Ns_arr = np.array(Ns, dtype=float)
log_p_cs = np.log([results_258_causet[Ni]['mean'] for Ni in Ns])
log_p_dag = np.log([results_258_dag[Ni]['mean'] for Ni in Ns])
sl_cs, _, r_cs, _, _ = stats.linregress(Ns_arr, log_p_cs)
sl_dag, _, r_dag, _, _ = stats.linregress(Ns_arr, log_p_dag)
print(f"\n  Purity scaling: causet ~ exp({sl_cs:.4f}*N) (r^2={r_cs**2:.3f})")
print(f"                  DAG    ~ exp({sl_dag:.4f}*N) (r^2={r_dag**2:.3f})")
print(f"  VERDICT: Causets {'MORE' if sl_cs > sl_dag else 'LESS'} mixed than random DAGs")


# ================================================================
# IDEA 259: SJ VACUUM ON CRYSTALLINE PHASE (DEEP KR REGIME)
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 259: SJ vacuum on CRYSTALLINE phase (deep Kleitman-Rothschild)")
print("At high beta (ordered phase), the 2-order approximates the total order.")
print("What happens to the SJ vacuum in this crystalline limit?")
print("Compare: beta=0 (random) vs beta >> beta_c (crystalline)")
print("=" * 78)

N = 30
eps = 0.12
beta_c = 1.66 / (N * eps**2)
betas = [0.0, 0.5 * beta_c, beta_c, 2 * beta_c, 5 * beta_c, 10 * beta_c]
n_mcmc = 8000
n_therm = 4000
n_trials = 5

print(f"N={N}, eps={eps}, beta_c={beta_c:.2f}")
print(f"Betas: {['%.1f (%.1fx bc)' % (b, b/beta_c) for b in betas]}")

results_259 = {}
for beta in betas:
    kurtoses = []
    r_vals = []
    entropies = []
    ordering_fracs = []

    for trial in range(n_trials):
        # Run short MCMC to get a sample at this beta
        current = TwoOrder(N, rng=rng)
        current_cs = current.to_causet()
        current_S = bd_action_corrected(current_cs, eps)

        for step in range(n_mcmc):
            proposed = swap_move(current, rng)
            proposed_cs = proposed.to_causet()
            proposed_S = bd_action_corrected(proposed_cs, eps)
            dS = beta * (proposed_S - current_S)
            if dS <= 0 or rng.random() < np.exp(-dS):
                current = proposed
                current_cs = proposed_cs
                current_S = proposed_S

        # Measure SJ properties on the thermalized sample
        if step >= n_therm:
            ev = sj_eigenvalues(current_cs)
            kurtoses.append(stats.kurtosis(ev))
            r_vals.append(level_spacing_ratio(ev))
            ordering_fracs.append(current_cs.ordering_fraction())

            W = sj_wightman_function(current_cs)
            S = entanglement_entropy(W, list(range(N // 2)))
            entropies.append(S)

    results_259[beta] = {
        'kurtosis': np.mean(kurtoses) if kurtoses else np.nan,
        'r': np.mean(r_vals) if r_vals else np.nan,
        'S': np.mean(entropies) if entropies else np.nan,
        'f': np.mean(ordering_fracs) if ordering_fracs else np.nan,
    }

print("\n  beta/beta_c | Ord.frac | kurtosis | <r>   | S(N/2)")
print("  " + "-" * 60)
for beta in betas:
    r = results_259[beta]
    ratio = beta / beta_c
    print(f"  {ratio:11.1f} | {r['f']:8.3f} | {r['kurtosis']:8.2f} | {r['r']:.4f} | {r['S']:.4f}")

# Look for sharp change at beta_c
if results_259[0.0]['kurtosis'] is not np.nan and results_259[10 * beta_c]['kurtosis'] is not np.nan:
    k_random = results_259[0.0]['kurtosis']
    k_crystal = results_259[10 * beta_c]['kurtosis']
    print(f"\n  VERDICT: Kurtosis changes from {k_random:.1f} (random) to {k_crystal:.1f} (crystalline)")
    print(f"  The SJ vacuum {'DETECTS' if abs(k_random - k_crystal) > 5 else 'does NOT clearly detect'} the phase transition")


# ================================================================
# IDEA 260: COMPARE SJ WIGHTMAN WITH RETARDED GREEN'S FUNCTION
# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 260: Compare SJ Wightman with RETARDED Green's function")
print("W_SJ comes from the positive-frequency projection of iDelta.")
print("G_R is just the causal matrix C (or C/N).")
print("How much of W is 'explained' by G_R? Correlation, ratio, residual structure?")
print("=" * 78)

N = 50
n_trials = 15

print(f"N={N}, {n_trials} trials")

correlations = []
ratio_means = []
residual_norms = []
residual_kurtoses = []

for trial in range(n_trials):
    to, cs = make_2order_causet(N, rng)
    W = sj_wightman_function(cs)
    G_R = retarded_green(cs)

    # Only compare on related pairs (where G_R is nonzero)
    mask = cs.order
    W_vals = W[mask]
    GR_vals = G_R[mask]

    if len(W_vals) > 5:
        corr = np.corrcoef(W_vals, GR_vals)[0, 1]
        correlations.append(corr)

        ratio = W_vals / (GR_vals + 1e-15)
        ratio_means.append(np.mean(ratio))

        # Residual: W - alpha * G_R (best fit)
        alpha_fit = np.dot(W_vals, GR_vals) / (np.dot(GR_vals, GR_vals) + 1e-15)
        residual = W_vals - alpha_fit * GR_vals
        residual_norms.append(np.linalg.norm(residual) / np.linalg.norm(W_vals))
        if len(residual) > 4:
            residual_kurtoses.append(stats.kurtosis(residual))

    # Also compare on SPACELIKE pairs (where G_R = 0 but W may be nonzero)
    spacelike = ~cs.order & ~cs.order.T & ~np.eye(N, dtype=bool)

print(f"\n  Correlation(W, G_R) on causal pairs:")
print(f"    Mean: {np.mean(correlations):.4f} +/- {np.std(correlations):.4f}")
print(f"  Mean ratio W/G_R: {np.mean(ratio_means):.4f}")
print(f"  Relative residual ||W - alpha*G_R|| / ||W||: {np.mean(residual_norms):.4f}")
print(f"  Residual kurtosis: {np.mean(residual_kurtoses):.2f}")

# Key question: does W have significant SPACELIKE correlations?
spacelike_norms = []
for trial in range(min(10, n_trials)):
    to, cs = make_2order_causet(N, rng)
    W = sj_wightman_function(cs)
    spacelike = ~cs.order & ~cs.order.T & ~np.eye(N, dtype=bool)
    W_spacelike = np.abs(W[spacelike])
    W_causal = np.abs(W[cs.order])
    if len(W_causal) > 0:
        spacelike_norms.append(np.mean(W_spacelike) / np.mean(W_causal))

print(f"\n  Spacelike W / Causal W ratio: {np.mean(spacelike_norms):.4f}")
print(f"  VERDICT: W has {'SIGNIFICANT' if np.mean(spacelike_norms) > 0.1 else 'NEGLIGIBLE'} "
      f"spacelike correlations vs G_R")
if np.mean(correlations) > 0.8:
    print(f"  W and G_R are HIGHLY correlated ({np.mean(correlations):.3f}) — "
          f"most of W is 'explained' by causal structure")
else:
    print(f"  W and G_R have moderate correlation ({np.mean(correlations):.3f}) — "
          f"quantum corrections are significant")


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n\n" + "=" * 78)
print("FINAL SUMMARY — IDEAS 251-260")
print("=" * 78)

print("""
  251. TRUNCATION: Removing near-zero modes from iDelta. Check above for
       whether this stabilizes c_eff across N.

  252. NORMALIZATION: 1/sqrt(N) normalization gives O(1) eigenvalues.
       This may be the correct continuum scaling.

  253. MASSIVE SCALAR: Mass term via Yukawa damping of correlations.
       Check if spectral gap scales linearly with mass.

  254. REGULAR CAUSETS: Lattice-like 2-orders vs random. The chain (total order)
       is a degenerate case. Check kurtosis interpolation.

  255. MOMENTUM SPACE: Fourier transform of W using 2-order coordinates.
       Check if W(k) ~ 1/|k| (massless scalar propagator).

  256. CAUSAL DIAMONDS: Entanglement of sub-diamonds of varying size.
       Check if S ~ area or S ~ volume.

  257. FIDELITY: How sensitive is the SJ vacuum to single-swap perturbations?
       Exponential decay rate characterizes the vacuum's rigidity.

  258. PURITY: Reduced state purity scaling with N. Compare causets vs DAGs.
       Faster purity decay = more entangled.

  259. CRYSTALLINE PHASE: SJ vacuum across the BD phase transition.
       Does kurtosis, <r>, or entropy detect the transition?

  260. W vs G_R: How much of the Wightman function is 'explained' by the
       retarded Green's function? Spacelike correlations are the quantum part.
""")

# Score each idea
print("  SCORING (based on results above):")
print("  " + "-" * 50)
scores = {
    251: "Truncation c_eff",
    252: "Normalization physics",
    253: "Mass gap",
    254: "Regular causets",
    255: "Momentum space W(k)",
    256: "Diamond entanglement",
    257: "Vacuum fidelity",
    258: "Purity scaling",
    259: "Crystalline SJ",
    260: "W vs G_R",
}
for idea, name in scores.items():
    print(f"  Idea {idea}: {name} — see results above")

print("\n" + "=" * 78)
print("EXPERIMENT 73 COMPLETE")
print("=" * 78)
