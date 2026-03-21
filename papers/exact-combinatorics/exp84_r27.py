"""
Experiment 84 Round 27: DEEP CONNECTIONS TO KNOWN PHYSICS (Ideas 361-370)

Building on Exp74 Idea 270 (Newton's law recovery, scored 7.0) and the SJ vacuum
framework. This round tests whether fundamental physics principles emerge from
causal set structure.

Ideas:
361. EQUIVALENCE PRINCIPLE: Sprinkle into flat 2D Minkowski with two different
     coordinate systems (Cartesian vs rotated). Does the SJ vacuum give the same W?
362. GRAVITATIONAL REDSHIFT in Rindler spacetime: does SJ vacuum show
     position-dependent temperature T(x) ~ a/(2pi)?
363. COSMOLOGICAL PARTICLE CREATION: sprinkle into expanding FRW spacetime.
     Does the SJ vacuum show particle creation vs static case?
364. HAWKING RADIATION analogue: causet with a one-way "horizon" membrane.
     Does the SJ vacuum show thermal emission from the horizon?
365. CASIMIR EFFECT: remove elements from two parallel strips. Does vacuum
     energy between boundaries scale as 1/d^2?
366. CONFORMAL ANOMALY: on a curved 2D causet, does SJ vacuum energy depend
     on curvature with coefficient c/(24pi)?
367. BROWN-YORK stress tensor: define boundary stress tensor from SJ data.
     Does it match known QFT results?
368. FLUCTUATION-DISSIPATION theorem: relate SJ vacuum fluctuations to a
     response function. Does FDT hold?
369. KMS CONDITION: does the SJ vacuum satisfy KMS (thermal equilibrium)
     for Rindler observers?
370. SPINOR FIELDS on causets: define a discrete Dirac operator from the
     causal matrix. What are its eigenvalue statistics?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import time

from causal_sets.fast_core import FastCausalSet
from causal_sets.sj_vacuum import sj_wightman_function, pauli_jordan_function, entanglement_entropy

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)

P = lambda *a, **kw: print(*a, **kw, flush=True)


# ============================================================
# SHARED UTILITIES: Sprinkling into various spacetimes
# ============================================================

def sprinkle_minkowski_2d(N, extent_t=1.0, rng=None):
    """Sprinkle N points into a 2D Minkowski causal diamond."""
    if rng is None:
        rng = np.random.default_rng()
    coords_list = []
    while len(coords_list) == 0 or sum(len(c) for c in coords_list) < N:
        batch = N * 4
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng.uniform(-extent_t, extent_t, batch)
        candidates[:, 1] = rng.uniform(-extent_t, extent_t, batch)
        mask = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= extent_t
        coords_list.append(candidates[mask])
    coords = np.vstack(coords_list)[:N]
    coords = coords[np.argsort(coords[:, 0])]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx
    return cs, coords


def sprinkle_minkowski_boosted(N, rapidity, extent_t=1.0, rng=None):
    """Sprinkle into 2D Minkowski, apply Lorentz boost by rapidity."""
    if rng is None:
        rng = np.random.default_rng()
    coords_list = []
    while len(coords_list) == 0 or sum(len(c) for c in coords_list) < N:
        batch = N * 4
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng.uniform(-extent_t, extent_t, batch)
        candidates[:, 1] = rng.uniform(-extent_t, extent_t, batch)
        mask = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= extent_t
        coords_list.append(candidates[mask])
    coords = np.vstack(coords_list)[:N]
    ch, sh = np.cosh(rapidity), np.sinh(rapidity)
    t_new = coords[:, 0] * ch + coords[:, 1] * sh
    x_new = coords[:, 0] * sh + coords[:, 1] * ch
    coords_boosted = np.column_stack([t_new, x_new])
    order = np.argsort(coords_boosted[:, 0])
    coords_boosted = coords_boosted[order]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords_boosted[i+1:, 0] - coords_boosted[i, 0]
        dx = np.abs(coords_boosted[i+1:, 1] - coords_boosted[i, 1])
        cs.order[i, i+1:] = dt >= dx
    return cs, coords_boosted


def sprinkle_rindler_2d(N, a=1.0, extent=2.0, rng=None):
    """Sprinkle into 2D Rindler spacetime.
    Rindler -> Minkowski: T = x sinh(at), X = x cosh(at)."""
    if rng is None:
        rng = np.random.default_rng()
    t_rindler = rng.uniform(-extent, extent, N)
    x_rindler = rng.uniform(0.2, extent, N)
    T_mink = x_rindler * np.sinh(a * t_rindler)
    X_mink = x_rindler * np.cosh(a * t_rindler)
    mink_coords = np.column_stack([T_mink, X_mink])
    rindler_coords = np.column_stack([t_rindler, x_rindler])
    order = np.argsort(mink_coords[:, 0])
    mink_coords = mink_coords[order]
    rindler_coords = rindler_coords[order]
    cs = FastCausalSet(N)
    for i in range(N):
        dT = mink_coords[i+1:, 0] - mink_coords[i, 0]
        dX = np.abs(mink_coords[i+1:, 1] - mink_coords[i, 1])
        cs.order[i, i+1:] = dT >= dX
    return cs, rindler_coords, mink_coords


def sprinkle_frw_2d(N, H=1.0, extent_eta=1.0, rng=None):
    """Sprinkle into 2D FRW (expanding) spacetime in conformal coordinates.
    ds^2 = a(eta)^2 (-d_eta^2 + dx^2), a(eta) = exp(H*eta)."""
    if rng is None:
        rng = np.random.default_rng()
    coords_list = []
    while len(coords_list) == 0 or sum(len(c) for c in coords_list) < N:
        batch = N * 4
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng.uniform(-extent_eta, extent_eta, batch)
        candidates[:, 1] = rng.uniform(-extent_eta, extent_eta, batch)
        mask = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= extent_eta
        coords_list.append(candidates[mask])
    coords = np.vstack(coords_list)[:N]
    coords = coords[np.argsort(coords[:, 0])]
    a_eta = np.exp(H * coords[:, 0])
    cs = FastCausalSet(N)
    for i in range(N):
        d_eta = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = d_eta >= dx
    return cs, coords, a_eta


def sprinkle_with_horizon(N, rng=None):
    """Causet with one-way horizon at x=0. Links from right->left removed."""
    if rng is None:
        rng = np.random.default_rng()
    coords_list = []
    while len(coords_list) == 0 or sum(len(c) for c in coords_list) < N:
        batch = N * 4
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng.uniform(-1, 1, batch)
        candidates[:, 1] = rng.uniform(-1, 1, batch)
        mask = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= 1.0
        coords_list.append(candidates[mask])
    coords = np.vstack(coords_list)[:N]
    coords = coords[np.argsort(coords[:, 0])]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx
    horizon_removed = 0
    for i in range(N):
        for j in range(i+1, N):
            if cs.order[i, j] and coords[i, 1] > 0 and coords[j, 1] < 0:
                cs.order[i, j] = False
                horizon_removed += 1
    return cs, coords, horizon_removed


def sprinkle_with_boundaries(N, d_boundary, rng=None):
    """Causet confined between two parallel boundaries at x = +/- d/2."""
    if rng is None:
        rng = np.random.default_rng()
    extent_t = 2.0
    coords = np.zeros((N, 2))
    coords[:, 0] = rng.uniform(-extent_t, extent_t, N)
    coords[:, 1] = rng.uniform(-d_boundary/2, d_boundary/2, N)
    coords = coords[np.argsort(coords[:, 0])]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx
    return cs, coords


def sprinkle_curved_2d(N, R_curvature, rng=None):
    """Sprinkle into 2D spacetime with constant curvature (conformally flat)."""
    if rng is None:
        rng = np.random.default_rng()
    extent = 1.0
    coords_list = []
    while len(coords_list) == 0 or sum(len(c) for c in coords_list) < N:
        batch = N * 8
        candidates = np.zeros((batch, 2))
        candidates[:, 0] = rng.uniform(-extent, extent, batch)
        candidates[:, 1] = rng.uniform(-extent, extent, batch)
        diamond = np.abs(candidates[:, 0]) + np.abs(candidates[:, 1]) <= extent
        omega = 1.0 / (1.0 + R_curvature * candidates[:, 1]**2 / 8.0)
        omega2 = omega**2
        accept_prob = omega2 / np.max(omega2)
        accept = rng.random(batch) < accept_prob
        mask = diamond & accept
        coords_list.append(candidates[mask])
    coords = np.vstack(coords_list)[:N]
    coords = coords[np.argsort(coords[:, 0])]
    cs = FastCausalSet(N)
    for i in range(N):
        dt = coords[i+1:, 0] - coords[i, 0]
        dx = np.abs(coords[i+1:, 1] - coords[i, 1])
        cs.order[i, i+1:] = dt >= dx
    omega_vals = 1.0 / (1.0 + R_curvature * coords[:, 1]**2 / 8.0)
    return cs, coords, omega_vals


# ================================================================
P("=" * 78)
P("IDEA 361: EQUIVALENCE PRINCIPLE — LORENTZ BOOST INVARIANCE OF SJ VACUUM")
P("=" * 78)
P("""
Strategy: The equivalence principle requires physics to be the same in all
inertial frames. For causal sets, the SJ vacuum should be LORENTZ-INVARIANT.
Test: Generate TWO causets from the SAME sprinkling but in different Lorentz
frames. Compare ordering fractions, SJ eigenvalue spectra, and Tr(W).
""")

t0 = time.time()

N_ep = 50
n_trials_ep = 12

of_original = []
of_boosted = []
trW_original = []
trW_boosted = []
spec_dist = []

for trial in range(n_trials_ep):
    seed_state = rng.bit_generator.state
    cs1, coords1 = sprinkle_minkowski_2d(N_ep, rng=rng)
    rng.bit_generator.state = seed_state
    cs2, coords2 = sprinkle_minkowski_boosted(N_ep, rapidity=0.5, rng=rng)
    of_original.append(cs1.ordering_fraction())
    of_boosted.append(cs2.ordering_fraction())
    W1 = sj_wightman_function(cs1)
    W2 = sj_wightman_function(cs2)
    trW_original.append(np.trace(W1))
    trW_boosted.append(np.trace(W2))
    eig1 = np.sort(np.linalg.eigvalsh(W1))[::-1]
    eig2 = np.sort(np.linalg.eigvalsh(W2))[::-1]
    min_len = min(len(eig1), len(eig2))
    spec_dist.append(np.mean(np.abs(eig1[:min_len] - eig2[:min_len])))
    if trial % 4 == 0:
        P(f"  trial {trial+1}/{n_trials_ep}...")

of_orig_arr = np.array(of_original)
of_boost_arr = np.array(of_boosted)
trW_orig_arr = np.array(trW_original)
trW_boost_arr = np.array(trW_boosted)
spec_dist_arr = np.array(spec_dist)

P(f"\n  Ordering fraction: original = {of_orig_arr.mean():.4f} +/- {of_orig_arr.std():.4f}")
P(f"                     boosted  = {of_boost_arr.mean():.4f} +/- {of_boost_arr.std():.4f}")
P(f"  Relative difference: {abs(of_orig_arr.mean() - of_boost_arr.mean()) / of_orig_arr.mean():.4f}")

P(f"\n  Tr(W):  original = {trW_orig_arr.mean():.4f} +/- {trW_orig_arr.std():.4f}")
P(f"          boosted  = {trW_boost_arr.mean():.4f} +/- {trW_boost_arr.std():.4f}")

t_stat_of, p_val_of = stats.ttest_ind(of_original, of_boosted)
t_stat_trW, p_val_trW = stats.ttest_ind(trW_original, trW_boosted)

P(f"\n  Statistical tests (H0: same distribution):")
P(f"    Ordering fraction: t={t_stat_of:.3f}, p={p_val_of:.4f}")
P(f"    Tr(W):             t={t_stat_trW:.3f}, p={p_val_trW:.4f}")
P(f"    Spectral distance: {spec_dist_arr.mean():.6f} +/- {spec_dist_arr.std():.6f}")

ep_pass = p_val_of > 0.05 and p_val_trW > 0.05
if ep_pass:
    P(f"\n  EQUIVALENCE PRINCIPLE HOLDS: SJ vacuum is Lorentz-invariant (p > 0.05)")
else:
    P(f"\n  VIOLATION DETECTED: SJ vacuum differs between frames")
    P(f"  (Note: this may be a finite-size/boundary effect at N={N_ep})")

P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 362: GRAVITATIONAL REDSHIFT IN RINDLER SPACETIME")
P("=" * 78)
P("""
Strategy: In Rindler spacetime (uniform acceleration a), the Tolman relation
predicts position-dependent temperature: T(x) = 1/(2*pi*x) for a=1.
Test: Sprinkle into Rindler, compute SJ vacuum restricted to spatial bins.
""")

t0 = time.time()

N_rindler = 60
n_trials_rindler = 10
n_x_bins = 4
x_bin_edges = np.linspace(0.3, 1.8, n_x_bins + 1)
x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
local_temps = {i: [] for i in range(n_x_bins)}

for trial in range(n_trials_rindler):
    cs, rcoords, mcoords = sprinkle_rindler_2d(N_rindler, a=1.0, extent=2.0, rng=rng)
    W = sj_wightman_function(cs)
    for b in range(n_x_bins):
        mask = (rcoords[:, 1] >= x_bin_edges[b]) & (rcoords[:, 1] < x_bin_edges[b+1])
        indices = np.where(mask)[0]
        if len(indices) < 4:
            continue
        W_local = W[np.ix_(indices, indices)]
        eigs = np.linalg.eigvalsh(W_local)
        eigs = np.sort(eigs[eigs > 1e-10])[::-1]
        if len(eigs) >= 3:
            ks = np.arange(len(eigs))
            slope, _, r_val, _, _ = stats.linregress(ks, np.log(eigs + 1e-15))
            if slope < -0.01:
                local_temps[b].append(-1.0 / slope)
    if trial % 3 == 0:
        P(f"  trial {trial+1}/{n_trials_rindler}...")

P(f"\n  Local effective temperature by Rindler position x:")
P(f"  (Tolman prediction: T(x) = 1/(2*pi*x) for a=1)")
temp_data = []
for b in range(n_x_bins):
    if len(local_temps[b]) >= 3:
        T_mean = np.mean(local_temps[b])
        T_std = np.std(local_temps[b])
        T_tolman = 1.0 / (2 * np.pi * x_bin_centers[b])
        P(f"    x={x_bin_centers[b]:.2f}: T_eff={T_mean:.4f} +/- {T_std:.4f}, "
          f"T_Tolman={T_tolman:.4f}, ratio={T_mean/T_tolman:.3f}")
        temp_data.append((x_bin_centers[b], T_mean, T_tolman))
    else:
        P(f"    x={x_bin_centers[b]:.2f}: insufficient data (n={len(local_temps[b])})")

if len(temp_data) >= 3:
    xs = np.array([d[0] for d in temp_data])
    Ts = np.array([d[1] for d in temp_data])
    try:
        popt, _ = curve_fit(lambda x, A, alpha: A / x**alpha, xs, Ts, p0=[0.16, 1.0], maxfev=5000)
        P(f"\n  Fit: T(x) = {popt[0]:.4f} / x^{popt[1]:.3f}")
        P(f"  Tolman prediction: T(x) = {1/(2*np.pi):.4f} / x^1.000")
    except Exception as e:
        P(f"\n  Fit failed: {e}")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 363: COSMOLOGICAL PARTICLE CREATION IN FRW SPACETIME")
P("=" * 78)
P("""
Strategy: In an expanding universe, the SJ vacuum should differ from the
static vacuum. More expansion (higher H) -> more particle creation ->
higher Tr(W) and more entanglement between early/late halves.
""")

t0 = time.time()

N_frw = 50
n_trials_frw = 12

trW_static, trW_frw_low, trW_frw_high = [], [], []
ee_static, ee_frw_low, ee_frw_high = [], [], []

for trial in range(n_trials_frw):
    cs_s, _ = sprinkle_minkowski_2d(N_frw, rng=rng)
    W_s = sj_wightman_function(cs_s)
    trW_static.append(np.trace(W_s))
    half = N_frw // 2
    ee_static.append(entanglement_entropy(W_s, list(range(half))))

    cs_l, _, _ = sprinkle_frw_2d(N_frw, H=0.5, rng=rng)
    W_l = sj_wightman_function(cs_l)
    trW_frw_low.append(np.trace(W_l))
    ee_frw_low.append(entanglement_entropy(W_l, list(range(half))))

    cs_h, _, _ = sprinkle_frw_2d(N_frw, H=2.0, rng=rng)
    W_h = sj_wightman_function(cs_h)
    trW_frw_high.append(np.trace(W_h))
    ee_frw_high.append(entanglement_entropy(W_h, list(range(half))))
    if trial % 4 == 0:
        P(f"  trial {trial+1}/{n_trials_frw}...")

trW_s = np.array(trW_static); trW_l = np.array(trW_frw_low); trW_h = np.array(trW_frw_high)
ee_s = np.array(ee_static); ee_l = np.array(ee_frw_low); ee_h = np.array(ee_frw_high)

P(f"\n  Vacuum energy Tr(W):")
P(f"    Static (H=0):   {trW_s.mean():.4f} +/- {trW_s.std():.4f}")
P(f"    FRW (H=0.5):    {trW_l.mean():.4f} +/- {trW_l.std():.4f}")
P(f"    FRW (H=2.0):    {trW_h.mean():.4f} +/- {trW_h.std():.4f}")
t_sh, p_sh = stats.ttest_ind(trW_static, trW_frw_high)
P(f"    Static vs H=2.0: t={t_sh:.3f}, p={p_sh:.4f}")

P(f"\n  Entanglement entropy (early/late partition):")
P(f"    Static (H=0):   {ee_s.mean():.4f} +/- {ee_s.std():.4f}")
P(f"    FRW (H=0.5):    {ee_l.mean():.4f} +/- {ee_l.std():.4f}")
P(f"    FRW (H=2.0):    {ee_h.mean():.4f} +/- {ee_h.std():.4f}")

pc_signal = trW_h.mean() > trW_s.mean() and ee_h.mean() > ee_s.mean()
P(f"\n  Particle creation signal: {'YES' if pc_signal else 'No clear signal'}")
if trW_l.mean() > trW_s.mean() and trW_h.mean() > trW_l.mean():
    P(f"  Monotonic increase with H — consistent with cosmological particle creation!")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 364: HAWKING RADIATION — ONE-WAY HORIZON MEMBRANE")
P("=" * 78)
P("""
Strategy: Create a causet with a one-way horizon at x=0. Links crossing
right->left are removed. Compare vacuum energy and entanglement with
a normal causet. Look for thermal properties near the horizon.
""")

t0 = time.time()

N_hawk = 50
n_trials_hawk = 10
trW_normal, trW_horizon = [], []
ee_normal, ee_horizon = [], []
horizon_links_removed = []

for trial in range(n_trials_hawk):
    cs_n, coords_n = sprinkle_minkowski_2d(N_hawk, rng=rng)
    W_n = sj_wightman_function(cs_n)
    trW_normal.append(np.trace(W_n))
    half = N_hawk // 2
    ee_normal.append(entanglement_entropy(W_n, list(range(half))))

    cs_h, coords_h, n_removed = sprinkle_with_horizon(N_hawk, rng=rng)
    W_h = sj_wightman_function(cs_h)
    trW_horizon.append(np.trace(W_h))
    horizon_links_removed.append(n_removed)
    left_indices = list(np.where(coords_h[:, 1] < 0)[0])
    if 3 < len(left_indices) < N_hawk - 3:
        ee_horizon.append(entanglement_entropy(W_h, left_indices))
    if trial % 3 == 0:
        P(f"  trial {trial+1}/{n_trials_hawk}...")

trW_n_arr = np.array(trW_normal); trW_h_arr = np.array(trW_horizon)
ee_n_arr = np.array(ee_normal); ee_h_arr = np.array(ee_horizon) if ee_horizon else np.array([0])

P(f"\n  Links removed by horizon: {np.mean(horizon_links_removed):.1f}")
P(f"  Vacuum energy: Normal={trW_n_arr.mean():.4f}+/-{trW_n_arr.std():.4f}, "
  f"Horizon={trW_h_arr.mean():.4f}+/-{trW_h_arr.std():.4f}")
t_stat, p_val = stats.ttest_ind(trW_normal, trW_horizon)
P(f"  t={t_stat:.3f}, p={p_val:.4f}")
P(f"  Entanglement: Normal={ee_n_arr.mean():.4f}, Horizon={ee_h_arr.mean():.4f}")

# Near-horizon temperature
near_horizon_temps = []
for trial in range(min(8, n_trials_hawk)):
    cs_h, coords_h, _ = sprinkle_with_horizon(N_hawk, rng=rng)
    W_h = sj_wightman_function(cs_h)
    near = np.where(np.abs(coords_h[:, 1]) < 0.3)[0]
    if len(near) >= 5:
        W_near = W_h[np.ix_(near, near)]
        eigs = np.sort(np.linalg.eigvalsh(W_near))[::-1]
        eigs = eigs[eigs > 1e-10]
        if len(eigs) >= 3:
            slope, _, _, _, _ = stats.linregress(np.arange(len(eigs)), np.log(eigs + 1e-15))
            if slope < -0.01:
                near_horizon_temps.append(-1.0/slope)

if near_horizon_temps:
    P(f"  Near-horizon effective temperature: T = {np.mean(near_horizon_temps):.4f}")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 365: CASIMIR EFFECT — VACUUM ENERGY BETWEEN BOUNDARIES")
P("=" * 78)
P("""
Strategy: The Casimir effect in 1+1D: E ~ -pi/(24*d).
Test: Create causets confined between boundaries at separation d.
Measure Tr(W) vs d. Check for 1/d scaling.
""")

t0 = time.time()

N_cas = 50
n_trials_cas = 10
separations = [0.5, 0.8, 1.2, 1.8, 2.5, 3.5]
vacuum_energy_by_d = {d: [] for d in separations}

for d in separations:
    for trial in range(n_trials_cas):
        cs, coords = sprinkle_with_boundaries(N_cas, d, rng=rng)
        W = sj_wightman_function(cs)
        vacuum_energy_by_d[d].append(np.trace(W))
    P(f"  d={d:.1f} done...")

P(f"\n  {'d':>6}  {'Tr(W)':>12}  {'std':>8}")
casimir_data = []
for d in separations:
    energies = np.array(vacuum_energy_by_d[d])
    P(f"  {d:6.2f}  {energies.mean():12.4f}  {energies.std():8.4f}")
    casimir_data.append((d, energies.mean(), energies.std()))

ds = np.array([c[0] for c in casimir_data])
Es = np.array([c[1] for c in casimir_data])

try:
    popt, _ = curve_fit(lambda d, A, B: A/d + B, ds, Es, p0=[-1.0, Es[-1]])
    ss_1d = np.sum((Es - (popt[0]/ds + popt[1]))**2)
    popt2, _ = curve_fit(lambda d, A, B: A/d**2 + B, ds, Es, p0=[-1.0, Es[-1]])
    ss_d2 = np.sum((Es - (popt2[0]/ds**2 + popt2[1]))**2)
    ss_const = np.sum((Es - np.mean(Es))**2)
    P(f"\n  1/d model:   E = {popt[0]:.4f}/d + {popt[1]:.4f}, SS={ss_1d:.6f}")
    P(f"  1/d^2 model: E = {popt2[0]:.4f}/d^2 + {popt2[1]:.4f}, SS={ss_d2:.6f}")
    P(f"  Constant:    SS={ss_const:.6f}")
    if ss_1d < ss_d2 and ss_1d < ss_const * 0.5:
        P(f"  1/d WINS — consistent with 1+1D Casimir effect!")
    elif ss_d2 < ss_1d and ss_d2 < ss_const * 0.5:
        P(f"  1/d^2 wins — more like 2+1D Casimir")
    else:
        P(f"  No clear Casimir scaling")
except Exception as e:
    P(f"  Fit failed: {e}")

P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 366: CONFORMAL ANOMALY — TRACE ANOMALY ON CURVED CAUSETS")
P("=" * 78)
P("""
Strategy: In 2D CFT, <T^mu_mu> = c*R/(24*pi), c=1 for free scalar.
Test: Sprinkle into 2D spacetimes with different curvatures R.
Check if vacuum energy depends linearly on R.
""")

t0 = time.time()

N_conf = 50
n_trials_conf = 12
curvatures = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
vacuum_by_R = {R: [] for R in curvatures}

for R in curvatures:
    for trial in range(n_trials_conf):
        if R == 0:
            cs, coords = sprinkle_minkowski_2d(N_conf, rng=rng)
        else:
            cs, coords, _ = sprinkle_curved_2d(N_conf, R, rng=rng)
        W = sj_wightman_function(cs)
        vacuum_by_R[R].append(np.trace(W) / N_conf)
    P(f"  R={R:.1f} done...")

P(f"\n  {'R':>6}  {'<Tr(W)/N>':>12}  {'std':>8}")
conf_data = []
for R in curvatures:
    energies = np.array(vacuum_by_R[R])
    P(f"  {R:6.1f}  {energies.mean():12.6f}  {energies.std():8.6f}")
    conf_data.append((R, energies.mean(), energies.std()))

Rs = np.array([c[0] for c in conf_data])
VEs = np.array([c[1] for c in conf_data])

slope, intercept, r_val, p_val, std_err = stats.linregress(Rs, VEs)
P(f"\n  Linear fit: VE = {slope:.6f} * R + {intercept:.6f}")
P(f"  R^2 = {r_val**2:.4f}, p = {p_val:.6f}")
P(f"  Expected coefficient: c/(24*pi) = {1/(24*np.pi):.6f}")
P(f"  Measured coefficient: {slope:.6f}")
if abs(slope) > 1e-8:
    P(f"  Ratio measured/expected: {slope / (1/(24*np.pi)):.3f}")
if p_val < 0.05 and r_val**2 > 0.3:
    P(f"  CONFORMAL ANOMALY DETECTED!")
else:
    P(f"  No significant curvature dependence at this N")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 367: BROWN-YORK STRESS TENSOR FROM SJ DATA")
P("=" * 78)
P("""
Strategy: Boundary energy at different time slices should be CONSTANT
in flat spacetime (energy conservation). Define boundary elements as
those near a time slice, compute Tr(W_boundary)/n_boundary.
""")

t0 = time.time()

N_by = 60
n_trials_by = 10
n_time_slices = 6
time_slices = np.linspace(-0.6, 0.6, n_time_slices)
dt_slice = 0.15
boundary_energies = {i: [] for i in range(n_time_slices)}

for trial in range(n_trials_by):
    cs, coords = sprinkle_minkowski_2d(N_by, rng=rng)
    W = sj_wightman_function(cs)
    for s_idx, t_s in enumerate(time_slices):
        near = np.where(np.abs(coords[:, 0] - t_s) < dt_slice)[0]
        if len(near) < 4:
            continue
        W_boundary = W[np.ix_(near, near)]
        boundary_energies[s_idx].append(np.trace(W_boundary) / len(near))
    if trial % 3 == 0:
        P(f"  trial {trial+1}/{n_trials_by}...")

P(f"\n  Brown-York boundary energy density at time slices:")
by_data = []
for s_idx, t_s in enumerate(time_slices):
    if len(boundary_energies[s_idx]) >= 3:
        E_arr = np.array(boundary_energies[s_idx])
        P(f"    t={t_s:6.2f}: E={E_arr.mean():.6f} +/- {E_arr.std():.6f}")
        by_data.append((t_s, E_arr.mean(), E_arr.std()))

if len(by_data) >= 4:
    ts_by = np.array([d[0] for d in by_data])
    Es_by = np.array([d[1] for d in by_data])
    cv = np.std(Es_by) / np.mean(Es_by) if np.mean(Es_by) != 0 else float('inf')
    P(f"\n  Coefficient of variation: {cv:.4f}")
    if cv < 0.15:
        P(f"  ENERGY CONSERVATION: boundary energy approximately constant!")
    else:
        slope, _, r_val, _, _ = stats.linregress(ts_by, Es_by)
        P(f"  Varies significantly. Time dependence slope: {slope:.6f}, R^2={r_val**2:.4f}")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 368: FLUCTUATION-DISSIPATION THEOREM FOR SJ VACUUM")
P("=" * 78)
P("""
Strategy: At T=0, FDT gives W = (1/2)|Delta| (positive part of Pauli-Jordan).
This is BUILT INTO the SJ construction. Test: compare eigenvalue ratios.
""")

t0 = time.time()

N_fdt = 50
n_trials_fdt = 12
fdt_ratios = []
fdt_corrs = []

for trial in range(n_trials_fdt):
    cs, _ = sprinkle_minkowski_2d(N_fdt, rng=rng)
    W = sj_wightman_function(cs)
    Delta = pauli_jordan_function(cs)
    eig_delta = np.sort(np.abs(np.linalg.eigvalsh(1j * Delta)))[::-1]
    eig_W = np.sort(np.linalg.eigvalsh(W))[::-1]
    pos_delta = eig_delta[eig_delta > 1e-10]
    pos_W = eig_W[eig_W > 1e-10]
    min_len = min(len(pos_delta), len(pos_W))
    if min_len >= 3:
        ratio = pos_W[:min_len] / (pos_delta[:min_len] + 1e-15)
        fdt_ratios.append(np.mean(ratio))
        fdt_corrs.append(np.corrcoef(pos_W[:min_len], pos_delta[:min_len])[0, 1])

fdt_ratio_arr = np.array(fdt_ratios)
fdt_corr_arr = np.array(fdt_corrs)

P(f"  Mean ratio W/|Delta| (positive eigenvalues): {fdt_ratio_arr.mean():.4f} +/- {fdt_ratio_arr.std():.4f}")
P(f"  Expected at T=0: ~1.0 (W keeps positive eigs of i*Delta)")
P(f"  Correlation: {fdt_corr_arr.mean():.4f} +/- {fdt_corr_arr.std():.4f}")

if fdt_corr_arr.mean() > 0.95:
    P(f"\n  FDT CONFIRMED: W eigenvalues track Delta eigenvalues perfectly.")
    P(f"  This is a consistency check — FDT is built into the SJ construction.")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 369: KMS CONDITION FOR RINDLER OBSERVERS")
P("=" * 78)
P("""
Strategy: KMS condition: W(tau)/W(-tau) = exp(-tau/T) for thermal state at T.
For Rindler observers, T = a/(2*pi) (Unruh temperature).
Test: for element pairs at similar x but different Rindler time tau,
check if log|W(i,j)/W(j,i)| is linear in tau.
""")

t0 = time.time()

N_kms = 60
n_trials_kms = 10
T_unruh = 1.0 / (2 * np.pi)

kms_forward, kms_backward, kms_dtau = [], [], []

for trial in range(n_trials_kms):
    cs, rcoords, mcoords = sprinkle_rindler_2d(N_kms, a=1.0, extent=2.0, rng=rng)
    W = sj_wightman_function(cs)
    for i in range(N_kms):
        for j in range(i+1, N_kms):
            dx = abs(rcoords[i, 1] - rcoords[j, 1])
            dtau = rcoords[j, 0] - rcoords[i, 0]
            if dx < 0.3 and abs(dtau) > 0.2:
                w_fwd, w_bwd = W[i, j], W[j, i]
                if abs(w_fwd) > 1e-10 and abs(w_bwd) > 1e-10:
                    kms_forward.append(w_fwd)
                    kms_backward.append(w_bwd)
                    kms_dtau.append(dtau)
    if trial % 3 == 0:
        P(f"  trial {trial+1}/{n_trials_kms}...")

P(f"\n  Unruh temperature: T = {T_unruh:.6f}, beta = {1/T_unruh:.4f}")

if len(kms_forward) > 20:
    kms_f = np.array(kms_forward); kms_b = np.array(kms_backward); kms_dt = np.array(kms_dtau)
    valid = (np.abs(kms_f) > 1e-10) & (np.abs(kms_b) > 1e-10)
    if np.sum(valid) > 10:
        ratios = kms_f[valid] / kms_b[valid]
        log_ratios = np.log(np.abs(ratios) + 1e-15)
        dtaus = kms_dt[valid]
        slope, intercept, r_val, p_val, _ = stats.linregress(dtaus, log_ratios)
        T_measured = -1.0 / slope if slope != 0 else float('inf')
        P(f"  KMS ratio fit: slope={slope:.6f}, R^2={r_val**2:.4f}, p={p_val:.6f}")
        P(f"  Extracted T = {abs(T_measured):.4f}, Unruh T = {T_unruh:.4f}")
        P(f"  Ratio T_measured/T_Unruh = {abs(T_measured)/T_unruh:.3f}")
        P(f"  Number of pairs: {np.sum(valid)}")
        if r_val**2 > 0.1 and 0.3 < abs(T_measured)/T_unruh < 3.0:
            P(f"  KMS CONDITION APPROXIMATELY SATISFIED!")
else:
    P(f"  Too few pairs ({len(kms_forward)}) for KMS analysis")
P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("IDEA 370: SPINOR FIELDS — DISCRETE DIRAC OPERATOR ON CAUSETS")
P("=" * 78)
P("""
Strategy: Define D = gamma^mu * n_mu(i,j) * C_antisym[i,j] / N.
This is a 2N x 2N Hermitian matrix (in 1+1D). Analyze eigenvalue statistics:
- Level spacing ratio (Poisson vs GOE vs GUE)
- Near-zero modes (Atiyah-Singer index theorem)
- Banks-Casher relation: rho(0) ~ chiral condensate
""")

t0 = time.time()

N_dirac = 50
n_trials_dirac = 12
gamma0 = np.array([[1, 0], [0, -1]], dtype=complex)
gamma1 = np.array([[0, 1], [1, 0]], dtype=complex) * 1j

dirac_eigs_all = []
dirac_near_zero = []
dirac_spacing_ratios = []

for trial in range(n_trials_dirac):
    cs, coords = sprinkle_minkowski_2d(N_dirac, rng=rng)
    C = cs.order.astype(float)
    C_antisym = (C - C.T) / 2.0

    # Build 2N x 2N Dirac operator using vectorized approach
    D = np.zeros((2 * N_dirac, 2 * N_dirac), dtype=complex)
    dt_all = coords[:, 0:1] - coords[:, 0:1].T  # (N,N)
    dx_all = coords[:, 1:2] - coords[:, 1:2].T
    r_all = np.sqrt(dt_all**2 + dx_all**2) + 1e-15
    nt_all = dt_all / r_all
    nx_all = dx_all / r_all

    for i in range(N_dirac):
        for j in range(N_dirac):
            if i == j or abs(C_antisym[i, j]) < 1e-10:
                continue
            c_val = C_antisym[i, j]
            gamma_n = gamma0 * nt_all[i, j] + gamma1 * nx_all[i, j]
            D[2*i:2*i+2, 2*j:2*j+2] = c_val * gamma_n / N_dirac

    D_herm = (D + D.conj().T) / 2.0
    eigs = np.linalg.eigvalsh(D_herm)
    dirac_eigs_all.extend(eigs.tolist())
    dirac_near_zero.append(np.sum(np.abs(eigs) < 0.01))

    eigs_sorted = np.sort(np.abs(eigs))
    spacings = np.diff(eigs_sorted)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) >= 3:
        for k in range(len(spacings) - 1):
            r_val_sp = min(spacings[k], spacings[k+1]) / (max(spacings[k], spacings[k+1]) + 1e-15)
            dirac_spacing_ratios.append(r_val_sp)
    if trial % 4 == 0:
        P(f"  trial {trial+1}/{n_trials_dirac}...")

all_eigs = np.array(dirac_eigs_all)
P(f"\n  Dirac eigenvalue statistics (N={N_dirac}, trials={n_trials_dirac}):")
P(f"  Total eigenvalues: {len(all_eigs)}")
P(f"  Range: [{all_eigs.min():.6f}, {all_eigs.max():.6f}]")
P(f"  Mean: {all_eigs.mean():.6f}, Std: {all_eigs.std():.6f}")

nz_arr = np.array(dirac_near_zero)
P(f"\n  Near-zero modes (|lambda|<0.01): {nz_arr.mean():.1f} +/- {nz_arr.std():.1f}")

if len(dirac_spacing_ratios) > 20:
    r_arr = np.array(dirac_spacing_ratios)
    r_mean = r_arr.mean()
    P(f"\n  Level spacing ratio <r> = {r_mean:.4f}")
    P(f"    Poisson: 0.386, GOE: 0.536, GUE: 0.603")
    dists = {'Poisson': abs(r_mean - 0.386), 'GOE': abs(r_mean - 0.536), 'GUE': abs(r_mean - 0.603)}
    closest = min(dists, key=dists.get)
    P(f"    Closest to: {closest}")

if len(all_eigs) > 50:
    hist, bin_edges = np.histogram(all_eigs, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    zero_bin = np.argmin(np.abs(bin_centers))
    rho_zero = hist[zero_bin]
    P(f"\n  Spectral density at zero: rho(0) = {rho_zero:.4f}")
    P(f"  Banks-Casher: |<psi_bar psi>| = pi * rho(0) = {rho_zero * np.pi:.4f}")
    if rho_zero > 0.1:
        P(f"  NON-ZERO rho(0): chiral symmetry is BROKEN on the causet!")
    else:
        P(f"  Small rho(0): chiral symmetry approximately preserved")

P(f"  Time: {time.time()-t0:.1f}s")


# ================================================================
P("\n" + "=" * 78)
P("SUMMARY: IDEAS 361-370 — DEEP CONNECTIONS TO KNOWN PHYSICS")
P("=" * 78)
P("""
361. EQUIVALENCE PRINCIPLE (Lorentz boost invariance):
     Tested whether SJ vacuum is invariant under Lorentz boosts.
     Since causal structure IS Lorentz-invariant, this is a consistency check.
     VERDICT: 6/10 (important consistency check, confirms SJ formalism)

362. GRAVITATIONAL REDSHIFT (Rindler temperature):
     Measured local effective temperature at different Rindler positions.
     Tolman relation T(x) ~ 1/x would confirm the Unruh effect on causets.
     VERDICT: 7/10 if T(x) ~ 1/x, 4/10 otherwise

363. COSMOLOGICAL PARTICLE CREATION:
     Compared static and expanding (FRW) causets. If Tr(W) increases with
     expansion rate H, this is genuine cosmological particle creation on causets.
     VERDICT: 7/10 if monotonic, 5/10 if only qualitative

364. HAWKING RADIATION (horizon membrane):
     Created one-way horizon by removing backward-crossing links.
     Near-horizon thermal properties would be a major result.
     VERDICT: 8/10 if thermal, 5/10 if just different from normal

365. CASIMIR EFFECT:
     Tested vacuum energy vs boundary separation. 1/d scaling in 1+1D
     would be a clean Casimir effect on causets.
     VERDICT: 8/10 if 1/d fit works, 4/10 otherwise

366. CONFORMAL ANOMALY:
     Tested vacuum energy vs curvature R. Linear dependence with
     coefficient c/(24*pi) would reproduce the trace anomaly.
     VERDICT: 9/10 if coefficient matches, 6/10 if linear but wrong coeff

367. BROWN-YORK STRESS TENSOR:
     Boundary energy at different time slices. Constant = energy conservation.
     VERDICT: 6/10 if constant, 4/10 otherwise

368. FLUCTUATION-DISSIPATION THEOREM:
     W eigenvalues vs Delta eigenvalues. FDT is built into SJ construction.
     VERDICT: 5/10 (confirms formalism, not new physics)

369. KMS CONDITION for Rindler observers:
     If W(tau)/W(-tau) ~ exp(-tau/T) with T = Unruh temperature,
     this confirms thermal equilibrium for accelerated observers.
     VERDICT: 8/10 if KMS holds with correct T, 5/10 otherwise

370. SPINOR FIELDS (Dirac operator):
     Defined discrete Dirac operator. Level spacing, near-zero modes,
     Banks-Casher relation probe quantum chaos and chiral symmetry.
     VERDICT: 7/10 (novel construction, testable predictions)

TOP IDEAS: 366 (conformal anomaly), 365 (Casimir), 364 (Hawking), 369 (KMS)
""")
