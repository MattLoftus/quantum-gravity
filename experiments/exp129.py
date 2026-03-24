"""
Experiment 129: SYNESTHESIA — Causal Set Properties in Different Sensory Modalities
(Ideas 801-810)

Translate causal set observables into different "sensory" channels and analyze
whether the translated representations separate manifold-like from non-manifold-like
causets, reveal dimension-dependence, or expose geometric features.

801. TASTE: flavor profile (sweetness=ordering, sourness=links, bitterness=entropy, umami=Fiedler)
802. FEEL: SJ Wightman as pressure field — locate pressure points
803. SEE: PJ eigenvalue density as color gradient — what "color" is a causet?
804. HEAR: poset as musical sequence — autocorrelation, spectral flatness, rhythmic regularity
805. SMELL: MCMC action volatility — does it peak at beta_c?
806. TOUCH: Hasse Laplacian as stiffness matrix — Young's modulus vs dimension
807. PROPRIOCEPTION: MM estimator accuracy vs position in causet
808. BALANCE: left-right asymmetry of width profile
809. PAIN: elements whose removal maximally changes SJ entropy
810. RHYTHM: autocorrelation of antichain width profile — dominant frequency vs d
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.linalg import eigh
from scipy.signal import correlate
import time

from causal_sets.fast_core import FastCausalSet, sprinkle_fast
from causal_sets.two_orders import TwoOrder
from causal_sets.d_orders import DOrder
from causal_sets.bd_action import count_intervals_by_size, bd_action_2d
from causal_sets.sj_vacuum import pauli_jordan_function, sj_wightman_function

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# SHARED UTILITIES
# ============================================================

def random_2order(N, rng_local=None):
    """Generate a random 2-order and convert to FastCausalSet."""
    if rng_local is None:
        rng_local = rng
    to = TwoOrder(N, rng=rng_local)
    return to.to_causet(), to


def random_dorder(d, N, rng_local=None):
    """Generate a random d-order and convert to FastCausalSet."""
    if rng_local is None:
        rng_local = rng
    do = DOrder(d, N, rng=rng_local)
    return do.to_causet_fast(), do


def random_dag(N, density, rng_local):
    """Generate a random DAG with given relation density (null model)."""
    cs = FastCausalSet(N)
    mask = rng_local.random((N, N)) < density
    cs.order = np.triu(mask, k=1)
    changed = True
    order_int = cs.order.astype(np.int32)
    while changed:
        new_order = (order_int @ order_int > 0) | cs.order
        changed = np.any(new_order != cs.order)
        cs.order = new_order
        order_int = cs.order.astype(np.int32)
    return cs


def hasse_adjacency(cs):
    """Symmetric adjacency matrix of the Hasse diagram."""
    links = cs.link_matrix()
    return (links | links.T).astype(np.float64)


def hasse_laplacian(cs):
    """Laplacian of the Hasse diagram."""
    adj = hasse_adjacency(cs)
    degree = np.sum(adj, axis=1)
    return np.diag(degree) - adj


def ordering_fraction(cs):
    """Fraction of causally related pairs."""
    N = cs.n
    return cs.num_relations() / (N * (N - 1) / 2)


def link_fraction(cs):
    """Fraction of pairs that are links."""
    N = cs.n
    links = cs.link_matrix()
    return int(np.sum(links)) / (N * (N - 1) / 2)


def interval_entropy(cs):
    """Shannon entropy of the interval size distribution."""
    intervals = count_intervals_by_size(cs, max_size=10)
    counts = np.array([intervals.get(k, 0) for k in range(11)], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def fiedler_value(cs):
    """Second-smallest eigenvalue of the Hasse Laplacian (algebraic connectivity)."""
    L = hasse_laplacian(cs)
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


def width_profile(cs):
    """Compute the width (antichain size) at each height level.
    Height of element i = length of longest chain ending at i."""
    N = cs.n
    if N == 0:
        return np.array([])
    # Compute heights
    heights = np.zeros(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            heights[j] = np.max(heights[preds]) + 1
    max_h = int(np.max(heights))
    widths = np.array([np.sum(heights == h) for h in range(max_h + 1)])
    return widths


def sj_entropy(cs):
    """Von Neumann entropy of the SJ Wightman function (full causet)."""
    W = sj_wightman_function(cs)
    evals = np.linalg.eigvalsh(W)
    evals = evals[evals > 1e-12]
    evals = np.clip(evals, 1e-30, 1 - 1e-30)
    return -np.sum(evals * np.log(evals) + (1 - evals) * np.log(1 - evals))


def mm_dimension_from_frac(f_ord):
    """Myrheim-Meyer dimension from ordering fraction."""
    from math import lgamma
    if f_ord <= 0 or f_ord >= 1:
        return float('nan')
    f = f_ord / 2.0  # convert to directional fraction
    d_low, d_high = 0.5, 20.0
    for _ in range(100):
        d_mid = (d_low + d_high) / 2
        try:
            log_f = lgamma(d_mid + 1) + lgamma(d_mid / 2) - np.log(4) - lgamma(3 * d_mid / 2)
            f_mid = np.exp(log_f)
        except (ValueError, OverflowError):
            f_mid = 0.0
        if f_mid > f:
            d_low = d_mid
        else:
            d_high = d_mid
    return (d_low + d_high) / 2


print("=" * 80)
print("EXPERIMENT 129: SYNESTHESIA — Causal Sets in Different Sensory Modalities")
print("Ideas 801-810")
print("=" * 80)
sys.stdout.flush()


# ============================================================
# IDEA 801: TASTE THE CAUSAL MATRIX
# ============================================================
print("\n" + "=" * 80)
print("IDEA 801: TASTE the Causal Matrix — Flavor Profiles")
print("=" * 80)
print("""
Map causal set properties to taste dimensions:
  Sweetness = ordering fraction (more order = sweeter)
  Sourness  = link fraction (more links = more sour)
  Bitterness = interval entropy (more complex intervals = more bitter)
  Umami     = Fiedler value (algebraic connectivity = savory depth)

Question: Do 2-orders, d-orders, sprinkled causets, and random DAGs have
distinguishable flavor profiles? Is there a dimension-dependent taste?
""")
sys.stdout.flush()

t0 = time.time()

N_taste = 60
n_trials = 8

categories = {
    '2-order': lambda: random_2order(N_taste)[0],
    '3-order': lambda: random_dorder(3, N_taste)[0],
    '4-order': lambda: random_dorder(4, N_taste)[0],
    'sprinkle_2d': lambda: sprinkle_fast(N_taste, dim=2, rng=rng)[0],
    'sprinkle_3d': lambda: sprinkle_fast(N_taste, dim=3, rng=rng)[0],
    'sprinkle_4d': lambda: sprinkle_fast(N_taste, dim=4, rng=rng)[0],
    'random_DAG': lambda: random_dag(N_taste, 0.15, rng),
}

taste_data = {cat: {'sweet': [], 'sour': [], 'bitter': [], 'umami': []}
              for cat in categories}

for cat, gen in categories.items():
    for trial in range(n_trials):
        cs = gen()
        taste_data[cat]['sweet'].append(ordering_fraction(cs))
        taste_data[cat]['sour'].append(link_fraction(cs))
        taste_data[cat]['bitter'].append(interval_entropy(cs))
        taste_data[cat]['umami'].append(fiedler_value(cs))

print(f"{'Category':<14} {'Sweet':>8} {'Sour':>8} {'Bitter':>8} {'Umami':>8}")
print("-" * 50)
for cat in categories:
    s = np.mean(taste_data[cat]['sweet'])
    so = np.mean(taste_data[cat]['sour'])
    b = np.mean(taste_data[cat]['bitter'])
    u = np.mean(taste_data[cat]['umami'])
    print(f"{cat:<14} {s:8.4f} {so:8.4f} {b:8.4f} {u:8.4f}")

# Null test: can we distinguish 2-order from random DAG by flavor alone?
sweet_2o = taste_data['2-order']['sweet']
sweet_dag = taste_data['random_DAG']['sweet']
t_sweet, p_sweet = stats.ttest_ind(sweet_2o, sweet_dag)
umami_2o = taste_data['2-order']['umami']
umami_dag = taste_data['random_DAG']['umami']
t_umami, p_umami = stats.ttest_ind(umami_2o, umami_dag)

print(f"\nNull test (2-order vs random DAG):")
print(f"  Sweetness: t={t_sweet:.3f}, p={p_sweet:.4f} {'*DISTINCT*' if p_sweet < 0.05 else 'not distinct'}")
print(f"  Umami:     t={t_umami:.3f}, p={p_umami:.4f} {'*DISTINCT*' if p_umami < 0.05 else 'not distinct'}")

# Dimension dependence: does taste change monotonically with d?
dims_taste = [2, 3, 4]
dim_cats = ['sprinkle_2d', 'sprinkle_3d', 'sprinkle_4d']
print(f"\nDimension dependence (sprinkled causets):")
for flavor in ['sweet', 'sour', 'bitter', 'umami']:
    vals = [np.mean(taste_data[dc][flavor]) for dc in dim_cats]
    # Spearman correlation with dimension
    r, p = stats.spearmanr(dims_taste, vals)
    print(f"  {flavor:>7}: d=2->{vals[0]:.4f}, d=3->{vals[1]:.4f}, d=4->{vals[2]:.4f}  "
          f"Spearman r={r:.3f}, p={p:.3f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 802: FEEL THE SJ VACUUM — PRESSURE FIELD
# ============================================================
print("\n" + "=" * 80)
print("IDEA 802: FEEL the SJ Vacuum — Pressure Field")
print("=" * 80)
print("""
Map W[i,j] (SJ Wightman function) to a pressure field:
  Pressure(i) = sum_j |W[i,j]|  (total "force" on element i)

Questions: Where are the pressure points (local maxima)?
Do they correspond to geometric features (center, boundary)?
""")
sys.stdout.flush()

t0 = time.time()

N_press = 50
n_press_trials = 6

for label, gen_func in [('2-order', lambda: random_2order(N_press)),
                         ('sprinkle_2d', lambda: sprinkle_fast(N_press, dim=2, rng=rng))]:
    pressures_boundary = []
    pressures_center = []
    for trial in range(n_press_trials):
        if label == '2-order':
            cs, to = gen_func()
            # Use rank as proxy for position (fractional height in poset)
            heights = np.zeros(N_press)
            for j in range(N_press):
                preds = np.where(cs.order[:j, j])[0]
                heights[j] = (np.max(heights[preds]) + 1) if len(preds) > 0 else 0
            max_h = max(heights.max(), 1)
            frac_height = heights / max_h
        else:
            cs, coords = gen_func()
            # Fractional height = normalized time coordinate
            t_coords = coords[:, 0]
            frac_height = (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-15)

        W = sj_wightman_function(cs)
        pressure = np.sum(np.abs(W), axis=1)

        # Split into boundary (bottom/top 20%) and center (middle 60%)
        boundary_mask = (frac_height < 0.2) | (frac_height > 0.8)
        center_mask = (frac_height >= 0.2) & (frac_height <= 0.8)
        if np.sum(center_mask) > 0 and np.sum(boundary_mask) > 0:
            pressures_boundary.append(np.mean(pressure[boundary_mask]))
            pressures_center.append(np.mean(pressure[center_mask]))

    if pressures_boundary and pressures_center:
        t_val, p_val = stats.ttest_ind(pressures_center, pressures_boundary)
        print(f"  {label}: center_pressure={np.mean(pressures_center):.4f} +/- {np.std(pressures_center):.4f}, "
              f"boundary_pressure={np.mean(pressures_boundary):.4f} +/- {np.std(pressures_boundary):.4f}")
        print(f"    Center vs Boundary: t={t_val:.3f}, p={p_val:.4f} "
              f"{'*DIFFERENT*' if p_val < 0.05 else 'not different'}")

# Null test: random DAG pressure distribution
pressures_dag_cv = []
for trial in range(n_press_trials):
    cs_dag = random_dag(N_press, 0.15, rng)
    W_dag = sj_wightman_function(cs_dag)
    p_dag = np.sum(np.abs(W_dag), axis=1)
    pressures_dag_cv.append(np.std(p_dag) / (np.mean(p_dag) + 1e-15))

pressures_2o_cv = []
for trial in range(n_press_trials):
    cs, _ = random_2order(N_press)
    W = sj_wightman_function(cs)
    p = np.sum(np.abs(W), axis=1)
    pressures_2o_cv.append(np.std(p) / (np.mean(p) + 1e-15))

t_cv, p_cv = stats.ttest_ind(pressures_2o_cv, pressures_dag_cv)
print(f"\n  Null test — pressure CV (coeff of variation):")
print(f"    2-order: {np.mean(pressures_2o_cv):.4f} +/- {np.std(pressures_2o_cv):.4f}")
print(f"    random DAG: {np.mean(pressures_dag_cv):.4f} +/- {np.std(pressures_dag_cv):.4f}")
print(f"    t={t_cv:.3f}, p={p_cv:.4f} {'*DISTINCT*' if p_cv < 0.05 else 'not distinct'}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 803: SEE THE EIGENVALUE DENSITY — COLOR GRADIENT
# ============================================================
print("\n" + "=" * 80)
print("IDEA 803: SEE the Eigenvalue Density — Color of a Causal Set")
print("=" * 80)
print("""
Map PJ eigenvalue density to a color: wavelength proportional to eigenvalue.
Compute the "dominant wavelength" = mean positive eigenvalue of i*iDelta.
Also compute the spectral "hue spread" = std of positive eigenvalues.

Does color change across the BD transition (ordering fraction)?
Does it depend on dimension?
""")
sys.stdout.flush()

t0 = time.time()

N_color = 50
n_color_trials = 8

def spectral_color(cs):
    """Compute mean and std of positive PJ eigenvalues (proxy for 'color')."""
    iDelta = pauli_jordan_function(cs)
    evals = np.linalg.eigvalsh(1j * iDelta).real
    pos_evals = evals[evals > 1e-12]
    if len(pos_evals) == 0:
        return 0.0, 0.0, 0
    return float(np.mean(pos_evals)), float(np.std(pos_evals)), len(pos_evals)

color_results = {}
for label, gen in [('2-order', lambda: random_2order(N_color)[0]),
                    ('3-order', lambda: random_dorder(3, N_color)[0]),
                    ('4-order', lambda: random_dorder(4, N_color)[0]),
                    ('sprinkle_2d', lambda: sprinkle_fast(N_color, dim=2, rng=rng)[0]),
                    ('sprinkle_4d', lambda: sprinkle_fast(N_color, dim=4, rng=rng)[0]),
                    ('random_DAG', lambda: random_dag(N_color, 0.15, rng))]:
    means, stds, counts = [], [], []
    for trial in range(n_color_trials):
        m, s, c = spectral_color(gen())
        means.append(m)
        stds.append(s)
        counts.append(c)
    color_results[label] = (means, stds, counts)
    print(f"  {label:<14}: dominant_lam = {np.mean(means):.4f} +/- {np.std(means):.4f}, "
          f"hue_spread = {np.mean(stds):.4f}, n_modes = {np.mean(counts):.1f}")

# Null test
t_col, p_col = stats.ttest_ind(color_results['2-order'][0], color_results['random_DAG'][0])
print(f"\n  Null test (dominant lam, 2-order vs DAG): t={t_col:.3f}, p={p_col:.4f}")

# Dimension dependence
print(f"\n  Dimension trend (dominant lam):")
for lab in ['sprinkle_2d', 'sprinkle_4d']:
    print(f"    {lab}: {np.mean(color_results[lab][0]):.4f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 804: HEAR THE CAUSAL SET — MUSICAL SEQUENCE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 804: HEAR the Causal Set — Musical Analysis")
print("=" * 80)
print("""
Map elements to notes:
  Pitch    = height in poset (layer index)
  Duration = number of descendants
  Volume   = degree (in-degree + out-degree in Hasse diagram)

Generate a sequence (sorted by height then by degree within layer).
Compute: autocorrelation of pitch, spectral flatness, rhythmic regularity.
Are manifold-like causets more "musical" than random DAGs?
""")
sys.stdout.flush()

t0 = time.time()

N_music = 80

def musical_properties(cs):
    """Extract musical properties from a causal set."""
    N = cs.n
    # Heights
    heights = np.zeros(N, dtype=int)
    for j in range(N):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            heights[j] = int(np.max(heights[preds])) + 1

    # Descendants count
    descendants = np.sum(cs.order, axis=1)  # number of elements above i

    # Degree in Hasse diagram
    links = cs.link_matrix()
    hasse_degree = np.sum(links, axis=0) + np.sum(links, axis=1)

    # Sort elements by height, then by degree within height
    order_idx = np.lexsort((hasse_degree, heights))
    pitch_seq = heights[order_idx].astype(float)
    volume_seq = hasse_degree[order_idx].astype(float)

    if len(pitch_seq) < 5:
        return {'autocorr_1': 0, 'spectral_flatness': 0, 'rhythmic_reg': 0}

    # Normalize
    pitch_seq = (pitch_seq - pitch_seq.mean()) / (pitch_seq.std() + 1e-15)
    volume_seq = (volume_seq - volume_seq.mean()) / (volume_seq.std() + 1e-15)

    # 1. Autocorrelation of pitch at lag 1
    autocorr = np.correlate(pitch_seq, pitch_seq, mode='full')
    mid = len(autocorr) // 2
    autocorr = autocorr / (autocorr[mid] + 1e-15)
    autocorr_1 = autocorr[mid + 1] if mid + 1 < len(autocorr) else 0

    # 2. Spectral flatness of pitch sequence (geometric mean / arithmetic mean of power spectrum)
    fft_vals = np.abs(np.fft.rfft(pitch_seq))**2
    fft_vals = fft_vals[1:]  # remove DC
    if len(fft_vals) > 0 and np.mean(fft_vals) > 1e-15:
        log_mean = np.mean(np.log(fft_vals + 1e-30))
        spectral_flatness = np.exp(log_mean) / (np.mean(fft_vals) + 1e-15)
    else:
        spectral_flatness = 0

    # 3. Rhythmic regularity: autocorrelation of volume sequence — peak-to-mean ratio
    vol_autocorr = np.correlate(volume_seq, volume_seq, mode='full')
    vol_mid = len(vol_autocorr) // 2
    vol_autocorr = vol_autocorr / (vol_autocorr[vol_mid] + 1e-15)
    # Look for peaks in the second half (lags 1..N/2)
    second_half = vol_autocorr[vol_mid+1:vol_mid+N//2]
    if len(second_half) > 0:
        rhythmic_reg = float(np.max(second_half) / (np.mean(np.abs(second_half)) + 1e-15))
    else:
        rhythmic_reg = 0

    return {'autocorr_1': autocorr_1, 'spectral_flatness': spectral_flatness,
            'rhythmic_reg': rhythmic_reg}

n_music_trials = 10
music_results = {}
for label, gen in [('2-order', lambda: random_2order(N_music)[0]),
                    ('sprinkle_2d', lambda: sprinkle_fast(N_music, dim=2, rng=rng)[0]),
                    ('sprinkle_4d', lambda: sprinkle_fast(N_music, dim=4, rng=rng)[0]),
                    ('random_DAG', lambda: random_dag(N_music, 0.15, rng))]:
    ac1s, sfs, rrs = [], [], []
    for trial in range(n_music_trials):
        props = musical_properties(gen())
        ac1s.append(props['autocorr_1'])
        sfs.append(props['spectral_flatness'])
        rrs.append(props['rhythmic_reg'])
    music_results[label] = (ac1s, sfs, rrs)
    print(f"  {label:<14}: autocorr_1={np.mean(ac1s):.4f}+/-{np.std(ac1s):.4f}, "
          f"spec_flat={np.mean(sfs):.4f}+/-{np.std(sfs):.4f}, "
          f"rhythm_reg={np.mean(rrs):.2f}+/-{np.std(rrs):.2f}")

# Null test: 2-order vs random DAG
for metric_name, idx in [('autocorr_1', 0), ('spectral_flatness', 1), ('rhythmic_regularity', 2)]:
    t_m, p_m = stats.ttest_ind(music_results['2-order'][idx], music_results['random_DAG'][idx])
    print(f"  Null test {metric_name}: t={t_m:.3f}, p={p_m:.4f} "
          f"{'*DISTINCT*' if p_m < 0.05 else 'not distinct'}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 805: SMELL THE PHASE TRANSITION — MCMC VOLATILITY
# ============================================================
print("\n" + "=" * 80)
print("IDEA 805: SMELL the Phase Transition — Action Volatility")
print("=" * 80)
print("""
Map the MCMC action trajectory to "volatility" (std of action over a rolling window).
Like perfume evaporation: volatility should peak at beta_c (the BD transition).

Method: run MCMC at several beta values, compute rolling-window std of action.
""")
sys.stdout.flush()

t0 = time.time()

from causal_sets.mcmc import mcmc_bd_action

N_smell = 25  # small for speed
betas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
n_mcmc_steps = 600
window = 10

print(f"  Running MCMC at {len(betas)} beta values, N={N_smell}, {n_mcmc_steps} steps each...")
sys.stdout.flush()

volatilities = {}
for beta in betas:
    cs_init, _ = sprinkle_fast(N_smell, dim=2, rng=rng)
    result = mcmc_bd_action(cs_init, beta=beta, n_steps=n_mcmc_steps,
                            target_n=N_smell, n_size_penalty=1.0,
                            rng=rng, record_every=1)
    actions = np.array(result['actions'])
    # Rolling volatility
    if len(actions) > window:
        rolling_std = np.array([np.std(actions[max(0,i-window):i+1])
                                for i in range(window, len(actions))])
        vol = float(np.mean(rolling_std))
    else:
        vol = float(np.std(actions))
    volatilities[beta] = vol
    print(f"    beta={beta:.1f}: mean_action={np.mean(actions):.2f}, volatility={vol:.4f}, "
          f"accept_rate={result['accept_rate']:.3f}")

# Find peak volatility
peak_beta = max(volatilities, key=volatilities.get)
print(f"\n  Peak volatility at beta = {peak_beta:.1f}")
print(f"  (Expected: peak near beta_c ~ 1-3 for small N)")

# Null test: volatility should be LOWER at beta=0 (random) and beta=8 (frozen)
# vs near the transition
v_low = volatilities[0.0]
v_high = volatilities[8.0]
v_peak = volatilities[peak_beta]
print(f"\n  Null test: vol(beta=0)={v_low:.4f}, vol(peak)={v_peak:.4f}, vol(beta=8)={v_high:.4f}")
print(f"  Peak > endpoints: {v_peak > v_low and v_peak > v_high}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 806: TOUCH THE HASSE DIAGRAM — STIFFNESS / YOUNG'S MODULUS
# ============================================================
print("\n" + "=" * 80)
print("IDEA 806: TOUCH the Hasse Diagram — Stiffness Matrix & Young's Modulus")
print("=" * 80)
print("""
Interpret the graph Laplacian as a stiffness matrix (like a spring network).
Young's modulus analog: E = (force per unit area) / (strain)
Here: E ~ lambda_2 * N^{2/d} (Fiedler value scaled by expected area).

Does E change systematically with dimension d?
""")
sys.stdout.flush()

t0 = time.time()

N_touch = 80
n_touch_trials = 8

touch_results = {}
for d, label in [(2, 'sprinkle_2d'), (3, 'sprinkle_3d'), (4, 'sprinkle_4d')]:
    fiedler_vals = []
    youngs_vals = []
    for trial in range(n_touch_trials):
        cs, _ = sprinkle_fast(N_touch, dim=d, rng=rng)
        L = hasse_laplacian(cs)
        evals = np.sort(np.linalg.eigvalsh(L))
        lam2 = float(evals[1]) if len(evals) > 1 else 0
        fiedler_vals.append(lam2)
        # Young's modulus proxy: lambda_2 * N^(2/d)
        E = lam2 * N_touch**(2.0/d)
        youngs_vals.append(E)
    touch_results[label] = (fiedler_vals, youngs_vals)
    print(f"  {label}: Fiedler={np.mean(fiedler_vals):.4f}+/-{np.std(fiedler_vals):.4f}, "
          f"Young's={np.mean(youngs_vals):.2f}+/-{np.std(youngs_vals):.2f}")

# Add null: random DAG
fiedler_dag, youngs_dag = [], []
for trial in range(n_touch_trials):
    cs = random_dag(N_touch, 0.15, rng)
    L = hasse_laplacian(cs)
    evals = np.sort(np.linalg.eigvalsh(L))
    lam2 = float(evals[1]) if len(evals) > 1 else 0
    fiedler_dag.append(lam2)
    youngs_dag.append(lam2 * N_touch**(2.0/2))
touch_results['random_DAG'] = (fiedler_dag, youngs_dag)
print(f"  random_DAG:   Fiedler={np.mean(fiedler_dag):.4f}+/-{np.std(fiedler_dag):.4f}, "
      f"Young's={np.mean(youngs_dag):.2f}+/-{np.std(youngs_dag):.2f}")

# Dimension dependence
dims_touch = [2, 3, 4]
youngs_means = [np.mean(touch_results[f'sprinkle_{d}d'][1]) for d in dims_touch]
r_touch, p_touch = stats.spearmanr(dims_touch, youngs_means)
print(f"\n  Young's modulus vs d: Spearman r={r_touch:.3f}, p={p_touch:.3f}")

# Null test vs DAG
t_touch, p_t_touch = stats.ttest_ind(touch_results['sprinkle_2d'][0], fiedler_dag)
print(f"  Fiedler (2d sprinkle vs DAG): t={t_touch:.3f}, p={p_t_touch:.4f} "
      f"{'*DISTINCT*' if p_t_touch < 0.05 else 'not distinct'}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 807: PROPRIOCEPTION — MM ACCURACY VS POSITION
# ============================================================
print("\n" + "=" * 80)
print("IDEA 807: PROPRIOCEPTION — MM Dimension Accuracy vs Position")
print("=" * 80)
print("""
Compute the MM dimension estimator using only the causal diamond (interval)
around each element. Do central elements estimate d more accurately than
boundary elements?

Method: For each element i, take the interval I(min, i) u I(i, max) as a
sub-causet and compute MM dimension from its ordering fraction.
""")
sys.stdout.flush()

t0 = time.time()

N_prop = 80
n_prop_trials = 5

for d in [2, 3, 4]:
    center_errors = []
    boundary_errors = []
    for trial in range(n_prop_trials):
        cs, coords = sprinkle_fast(N_prop, dim=d, rng=rng)
        t_coords = coords[:, 0]
        frac_t = (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-15)

        # For each element, compute local ordering fraction
        # using past interval (all elements below i) and future interval (all above i)
        for i in range(N_prop):
            # Elements in the past of i (those that precede i)
            past = np.where(cs.order[:, i])[0]
            # Elements in the future of i (those that i precedes)
            future = np.where(cs.order[i, :])[0]
            # Combined interval around i
            interval = np.concatenate([past, [i], future])
            n_int = len(interval)
            if n_int < 8:
                continue
            # Sub-causet ordering fraction
            sub_order = cs.order[np.ix_(interval, interval)]
            n_rel = int(np.sum(sub_order))
            f_local = n_rel / (n_int * (n_int - 1) / 2)
            d_est = mm_dimension_from_frac(f_local)
            if np.isnan(d_est):
                continue
            error = abs(d_est - d)
            if 0.2 < frac_t[i] < 0.8:
                center_errors.append(error)
            else:
                boundary_errors.append(error)

    if center_errors and boundary_errors:
        t_prop, p_prop = stats.ttest_ind(center_errors, boundary_errors)
        print(f"  d={d}: center_error={np.mean(center_errors):.3f}+/-{np.std(center_errors):.3f} "
              f"(n={len(center_errors)}), "
              f"boundary_error={np.mean(boundary_errors):.3f}+/-{np.std(boundary_errors):.3f} "
              f"(n={len(boundary_errors)})")
        print(f"    Center more accurate: {np.mean(center_errors) < np.mean(boundary_errors)}, "
              f"t={t_prop:.3f}, p={p_prop:.4f}")
    else:
        print(f"  d={d}: insufficient data (center={len(center_errors)}, boundary={len(boundary_errors)})")

# Null test: random DAG should have no center/boundary distinction
center_err_dag, boundary_err_dag = [], []
for trial in range(n_prop_trials):
    cs = random_dag(N_prop, 0.15, rng)
    heights = np.zeros(N_prop, dtype=int)
    for j in range(N_prop):
        preds = np.where(cs.order[:j, j])[0]
        if len(preds) > 0:
            heights[j] = int(np.max(heights[preds])) + 1
    max_h = max(heights.max(), 1)
    frac_h = heights / max_h
    for i in range(N_prop):
        past = np.where(cs.order[:, i])[0]
        future = np.where(cs.order[i, :])[0]
        interval = np.concatenate([past, [i], future])
        n_int = len(interval)
        if n_int < 8:
            continue
        sub_order = cs.order[np.ix_(interval, interval)]
        n_rel = int(np.sum(sub_order))
        f_local = n_rel / (n_int * (n_int - 1) / 2)
        d_est = mm_dimension_from_frac(f_local)
        if np.isnan(d_est):
            continue
        error = abs(d_est - 2)
        if 0.2 < frac_h[i] < 0.8:
            center_err_dag.append(error)
        else:
            boundary_err_dag.append(error)

if center_err_dag and boundary_err_dag:
    t_dag, p_dag = stats.ttest_ind(center_err_dag, boundary_err_dag)
    print(f"\n  Null (random DAG): center_error={np.mean(center_err_dag):.3f}, "
          f"boundary_error={np.mean(boundary_err_dag):.3f}, t={t_dag:.3f}, p={p_dag:.4f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 808: BALANCE — LEFT-RIGHT ASYMMETRY OF WIDTH PROFILE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 808: BALANCE — Width Profile Asymmetry")
print("=" * 80)
print("""
Compute the width profile W(h) = number of elements at height h.
Define asymmetry = |sum_{h}(W(h) - W(H-h))| / sum(W(h))  where H = max height.
A "balanced" causet has low asymmetry. Does balance correlate with manifold-likeness?
""")
sys.stdout.flush()

t0 = time.time()

N_bal = 100
n_bal_trials = 10

def width_asymmetry(cs):
    """Compute asymmetry of the width profile."""
    wp = width_profile(cs)
    if len(wp) < 3:
        return 0.0
    H = len(wp)
    # Mirror comparison
    mirrored = wp[::-1]
    # Pad to same length
    diff = np.sum(np.abs(wp - mirrored))
    return diff / (np.sum(wp) + 1e-15)

bal_results = {}
for label, gen in [('2-order', lambda: random_2order(N_bal)[0]),
                    ('sprinkle_2d', lambda: sprinkle_fast(N_bal, dim=2, rng=rng)[0]),
                    ('sprinkle_3d', lambda: sprinkle_fast(N_bal, dim=3, rng=rng)[0]),
                    ('sprinkle_4d', lambda: sprinkle_fast(N_bal, dim=4, rng=rng)[0]),
                    ('random_DAG', lambda: random_dag(N_bal, 0.15, rng))]:
    asym_vals = []
    for trial in range(n_bal_trials):
        cs = gen()
        asym_vals.append(width_asymmetry(cs))
    bal_results[label] = asym_vals
    print(f"  {label:<14}: asymmetry = {np.mean(asym_vals):.4f} +/- {np.std(asym_vals):.4f}")

# Null test
t_bal, p_bal = stats.ttest_ind(bal_results['2-order'], bal_results['random_DAG'])
print(f"\n  Null test (2-order vs DAG): t={t_bal:.3f}, p={p_bal:.4f} "
      f"{'*DISTINCT*' if p_bal < 0.05 else 'not distinct'}")

# Dimension dependence
dims_bal = [2, 3, 4]
asym_means_bal = [np.mean(bal_results[f'sprinkle_{d}d']) for d in dims_bal]
r_bal, p_r_bal = stats.spearmanr(dims_bal, asym_means_bal)
print(f"  Asymmetry vs d: Spearman r={r_bal:.3f}, p={p_r_bal:.3f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 809: PAIN — ELEMENTS WHOSE REMOVAL MAXIMALLY CHANGES SJ ENTROPY
# ============================================================
print("\n" + "=" * 80)
print("IDEA 809: PAIN — Critical Elements for SJ Entropy")
print("=" * 80)
print("""
For each element i, compute delta_S(i) = |S_SJ(C) - S_SJ(C \ {i})|.
"Painful" elements are those with large delta_S. Questions:
  1. How many painful elements are there (top 10%)?
  2. Do they cluster (are they near each other in the poset)?
  3. Are they geometric (center vs boundary)?
""")
sys.stdout.flush()

t0 = time.time()

N_pain = 35  # small for computational tractability
n_pain_trials = 4

for label, gen in [('2-order', lambda: random_2order(N_pain)),
                    ('sprinkle_2d', lambda: sprinkle_fast(N_pain, dim=2, rng=rng))]:
    pain_center_frac = []
    pain_clustering = []
    n_painful_list = []

    for trial in range(n_pain_trials):
        if label == '2-order':
            cs, _ = gen()
            heights = np.zeros(N_pain, dtype=int)
            for j in range(N_pain):
                preds = np.where(cs.order[:j, j])[0]
                if len(preds) > 0:
                    heights[j] = int(np.max(heights[preds])) + 1
            max_h = max(heights.max(), 1)
            frac_pos = heights / max_h
        else:
            cs, coords = gen()
            t_coords = coords[:, 0]
            frac_pos = (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-15)

        # Full SJ entropy
        S_full = sj_entropy(cs)

        # Remove each element and compute entropy
        delta_S = np.zeros(N_pain)
        for i in range(N_pain):
            keep = np.ones(N_pain, dtype=bool)
            keep[i] = False
            sub_cs = FastCausalSet(N_pain - 1)
            sub_cs.order = cs.order[np.ix_(keep, keep)]
            S_sub = sj_entropy(sub_cs)
            delta_S[i] = abs(S_full - S_sub)

        # Top 10% = "painful"
        threshold = np.percentile(delta_S, 90)
        painful = delta_S >= threshold
        n_painful = int(np.sum(painful))
        n_painful_list.append(n_painful)

        # Are painful elements more central or boundary?
        if n_painful > 0:
            pain_positions = frac_pos[painful]
            center_frac = np.mean((pain_positions > 0.2) & (pain_positions < 0.8))
            pain_center_frac.append(center_frac)

            # Clustering: fraction of painful-painful pairs that are causally related
            painful_idx = np.where(painful)[0]
            if len(painful_idx) > 1:
                n_pairs = 0
                n_related = 0
                for a in range(len(painful_idx)):
                    for b in range(a+1, len(painful_idx)):
                        n_pairs += 1
                        if cs.order[painful_idx[a], painful_idx[b]] or cs.order[painful_idx[b], painful_idx[a]]:
                            n_related += 1
                pain_clustering.append(n_related / n_pairs if n_pairs > 0 else 0)

    print(f"  {label}:")
    print(f"    Mean painful elements (top 10%): {np.mean(n_painful_list):.1f}")
    if pain_center_frac:
        print(f"    Fraction of painful in center (0.2-0.8): {np.mean(pain_center_frac):.3f}")
        # If random, expect ~60% in center (60% of range)
        expected_center = 0.6
        print(f"    (Expected if uniform: {expected_center:.3f})")
    if pain_clustering:
        overall_of = ordering_fraction(cs)
        print(f"    Clustering (related fraction): {np.mean(pain_clustering):.3f} "
              f"(baseline ordering fraction: ~{overall_of:.3f})")

# Null test: do random DAGs have the same pain distribution?
dag_pain_std = []
for trial in range(n_pain_trials):
    cs = random_dag(N_pain, 0.15, rng)
    S_full = sj_entropy(cs)
    delta_S = np.zeros(N_pain)
    for i in range(N_pain):
        keep = np.ones(N_pain, dtype=bool)
        keep[i] = False
        sub_cs = FastCausalSet(N_pain - 1)
        sub_cs.order = cs.order[np.ix_(keep, keep)]
        delta_S[i] = abs(S_full - sj_entropy(sub_cs))
    dag_pain_std.append(np.std(delta_S))

causet_pain_std = []
for trial in range(n_pain_trials):
    cs, _ = random_2order(N_pain)
    S_full = sj_entropy(cs)
    delta_S = np.zeros(N_pain)
    for i in range(N_pain):
        keep = np.ones(N_pain, dtype=bool)
        keep[i] = False
        sub_cs = FastCausalSet(N_pain - 1)
        sub_cs.order = cs.order[np.ix_(keep, keep)]
        delta_S[i] = abs(S_full - sj_entropy(sub_cs))
    causet_pain_std.append(np.std(delta_S))

t_pain, p_pain = stats.ttest_ind(causet_pain_std, dag_pain_std)
print(f"\n  Null test — pain variability (std of delta_S):")
print(f"    2-order: {np.mean(causet_pain_std):.4f} +/- {np.std(causet_pain_std):.4f}")
print(f"    random DAG: {np.mean(dag_pain_std):.4f} +/- {np.std(dag_pain_std):.4f}")
print(f"    t={t_pain:.3f}, p={p_pain:.4f} {'*DISTINCT*' if p_pain < 0.05 else 'not distinct'}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# IDEA 810: RHYTHM — AUTOCORRELATION OF WIDTH PROFILE
# ============================================================
print("\n" + "=" * 80)
print("IDEA 810: RHYTHM — Width Profile Autocorrelation")
print("=" * 80)
print("""
Compute the autocorrelation function of the width profile W(h).
Is there a dominant frequency (periodic structure in layer widths)?
Does it depend on dimension d?

Method: FFT of centered width profile -> power spectrum -> dominant frequency.
""")
sys.stdout.flush()

t0 = time.time()

N_rhythm = 120
n_rhythm_trials = 10

def rhythm_analysis(cs):
    """Analyze rhythmic properties of the width profile."""
    wp = width_profile(cs).astype(float)
    if len(wp) < 5:
        return {'dominant_freq': 0, 'spectral_entropy': 0, 'peak_ratio': 0}

    # Center the profile
    wp_centered = wp - wp.mean()

    # Autocorrelation
    acf = correlate(wp_centered, wp_centered, mode='full')
    mid = len(acf) // 2
    acf = acf[mid:] / (acf[mid] + 1e-15)  # normalize

    # Power spectrum
    ps = np.abs(np.fft.rfft(wp_centered))**2
    ps = ps[1:]  # remove DC
    freqs = np.fft.rfftfreq(len(wp_centered))[1:]

    if len(ps) == 0 or np.sum(ps) < 1e-15:
        return {'dominant_freq': 0, 'spectral_entropy': 0, 'peak_ratio': 0}

    # Dominant frequency
    dominant_idx = np.argmax(ps)
    dominant_freq = freqs[dominant_idx] if len(freqs) > dominant_idx else 0

    # Spectral entropy (how spread the power is)
    ps_norm = ps / (np.sum(ps) + 1e-15)
    ps_norm = ps_norm[ps_norm > 0]
    spectral_entropy = -np.sum(ps_norm * np.log(ps_norm))

    # Peak ratio: max power / mean power (how peaked the spectrum is)
    peak_ratio = np.max(ps) / (np.mean(ps) + 1e-15)

    return {'dominant_freq': float(dominant_freq),
            'spectral_entropy': float(spectral_entropy),
            'peak_ratio': float(peak_ratio)}

rhythm_results = {}
for label, gen in [('2-order', lambda: random_2order(N_rhythm)[0]),
                    ('sprinkle_2d', lambda: sprinkle_fast(N_rhythm, dim=2, rng=rng)[0]),
                    ('sprinkle_3d', lambda: sprinkle_fast(N_rhythm, dim=3, rng=rng)[0]),
                    ('sprinkle_4d', lambda: sprinkle_fast(N_rhythm, dim=4, rng=rng)[0]),
                    ('random_DAG', lambda: random_dag(N_rhythm, 0.15, rng))]:
    dfs, ses, prs = [], [], []
    for trial in range(n_rhythm_trials):
        r = rhythm_analysis(gen())
        dfs.append(r['dominant_freq'])
        ses.append(r['spectral_entropy'])
        prs.append(r['peak_ratio'])
    rhythm_results[label] = (dfs, ses, prs)
    print(f"  {label:<14}: dom_freq={np.mean(dfs):.4f}+/-{np.std(dfs):.4f}, "
          f"spec_entropy={np.mean(ses):.3f}+/-{np.std(ses):.3f}, "
          f"peak_ratio={np.mean(prs):.2f}+/-{np.std(prs):.2f}")

# Null test
for metric_name, idx in [('dominant_freq', 0), ('spectral_entropy', 1), ('peak_ratio', 2)]:
    t_r, p_r = stats.ttest_ind(rhythm_results['2-order'][idx], rhythm_results['random_DAG'][idx])
    print(f"  Null test {metric_name}: t={t_r:.3f}, p={p_r:.4f} "
          f"{'*DISTINCT*' if p_r < 0.05 else 'not distinct'}")

# Dimension dependence
dims_rhythm = [2, 3, 4]
for metric_name, idx in [('dominant_freq', 0), ('spectral_entropy', 1), ('peak_ratio', 2)]:
    vals = [np.mean(rhythm_results[f'sprinkle_{d}d'][idx]) for d in dims_rhythm]
    r_rhy, p_rhy = stats.spearmanr(dims_rhythm, vals)
    print(f"  {metric_name} vs d: {vals[0]:.4f} -> {vals[1]:.4f} -> {vals[2]:.4f}, "
          f"Spearman r={r_rhy:.3f}, p={p_rhy:.3f}")

print(f"\n  Time: {time.time()-t0:.1f}s")
sys.stdout.flush()


# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("GRAND SUMMARY: SYNESTHESIA — Causal Sets Across Sensory Modalities")
print("=" * 80)

print("""
801 TASTE: Flavor profiles (sweetness, sourness, bitterness, umami) computed.
     Key question: do d-orders and sprinkled causets have distinct tastes from DAGs?

802 FEEL: SJ Wightman pressure field analyzed. Center vs boundary pressure tested.
     Key question: do manifold-like causets have structured pressure distributions?

803 SEE: Causal sets assigned a "color" via PJ eigenvalue density.
     Key question: does the color change with dimension?

804 HEAR: Musical properties (autocorrelation, spectral flatness, rhythm) computed.
     Key question: are causets more "musical" than random DAGs?

805 SMELL: MCMC action volatility measured across beta values.
     Key question: does volatility peak at the phase transition?

806 TOUCH: Young's modulus (Fiedler x N^{2/d}) as stiffness measure.
     Key question: does stiffness depend on dimension?

807 PROPRIOCEPTION: MM dimension accuracy vs position in causet.
     Key question: do central elements "know" their dimension better?

808 BALANCE: Width profile asymmetry measured.
     Key question: does balance correlate with manifold-likeness?

809 PAIN: Critical elements (whose removal changes SJ entropy most) identified.
     Key question: do they cluster? Are they geometric?

810 RHYTHM: Width profile autocorrelation and dominant frequency.
     Key question: is there a d-dependent rhythmic structure?
""")

# Count significant results
print("Significance summary (p < 0.05 in null tests):")
print("  Check the individual idea outputs above for detailed p-values.")
print("  The synesthesia framework translates geometric/causal properties into")
print("  interpretable 'sensory' channels -- each providing an independent view")
print("  of what makes manifold-like causets special.")

print("\n" + "=" * 80)
print("EXPERIMENT 129 COMPLETE")
print("=" * 80)
