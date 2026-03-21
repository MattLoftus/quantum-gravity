"""
Experiment 69: BD Phase Transition — Deep Physics (Ideas 211-220)

Focus: The UNKNOWN physics of the BD first-order transition.
Known: interval entropy jumps 87%, link fraction jumps 60%, Fiedler jumps 74%,
       ordering fraction barely changes (1%). First-order transition at eps=0.12.
Unknown: critical exponents, latent heat scaling, metastability, nucleation
         dynamics, KR (crystalline) phase structure, tricritical point.

Parameters: N=30-70, eps=0.12, beta_c = 1.66 / (N * eps^2)

Ideas:
211. LATENT HEAT SCALING: Measure action discontinuity Delta_S at beta_c
     for N=30,40,50,60,70. For first-order: Delta_S ~ N (extensive).
     For second-order: Delta_S -> 0. This determines the thermodynamic order.
212. HYSTERESIS LOOP: Heat-then-cool cycle. Start ordered (high beta),
     slowly decrease beta. Then start disordered, slowly increase beta.
     Hysteresis width = signature of first-order metastability.
213. ACTION HISTOGRAM BIMODALITY: At beta=beta_c, the action histogram
     should show TWO peaks (coexisting phases) for first-order transitions.
     Measure the peak separation and valley depth vs N.
214. BINDER CUMULANT: U_4 = 1 - <S^4>/(3<S^2>^2). For first-order
     transitions, U_4 has a minimum that deepens as 2/3 with N. For
     second-order, U_4 -> 2/3 monotonically.
215. METASTABLE LIFETIME: Start in ordered (KR) phase at beta slightly
     below beta_c. Measure how many MCMC steps until the system "melts"
     to disordered phase. Lifetime should grow exponentially with N for
     first-order.
216. NUCLEATION DYNAMICS: During a transition event (ordered->disordered),
     track link fraction step-by-step. Does it change gradually (second-order)
     or in a single sharp jump (nucleation event)?
217. KR PHASE STRUCTURE: Deep in ordered phase (beta=5*beta_c), what IS
     the causal set? Measure: chain length/N, antichain/sqrt(N), link
     fraction, layer structure (fraction of elements at each "time slice").
     Is it a total order? A layered structure? A lattice?
218. SPECIFIC HEAT PEAK: C_V = beta^2 * (<S^2> - <S>^2) / N.
     For first-order: C_V peak ~ N. For second-order: C_V peak ~ N^{alpha/nu}.
     Measure peak height vs N and extract the scaling exponent.
219. SUSCEPTIBILITY OF LINK FRACTION: chi_L = N * (<L^2> - <L>^2).
     Link fraction is the "order parameter" (60% jump). Its fluctuations
     should diverge at beta_c. Measure peak height scaling with N.
220. TRICRITICAL SEARCH: Scan eps from 0.05 to 0.25 at fixed N=50.
     The transition may weaken as eps changes. If the Binder dip vanishes
     at some eps_tc, that's a tricritical point where first-order becomes
     second-order.
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.two_orders_v2 import bd_action_corrected
from causal_sets.bd_action import count_intervals_by_size
from causal_sets.fast_core import FastCausalSet
import time

rng = np.random.default_rng(42)

# ============================================================
# SHARED MCMC UTILITIES
# ============================================================

def beta_c(N, eps):
    """Critical coupling from Glaser et al."""
    return 1.66 / (N * eps**2)


def run_mcmc(N, beta, eps, n_steps, n_therm, record_every, rng,
             return_trajectory=False):
    """MCMC returning post-thermalization samples.
    If return_trajectory=True, also return action at every step."""
    current = TwoOrder(N, rng=rng)
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0

    samples = []
    trajectory = [] if return_trajectory else None

    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if return_trajectory:
            trajectory.append(current_S)

        if step >= n_therm and (step - n_therm) % record_every == 0:
            samples.append((current.copy(), current_cs, current_S))

    result = {
        'samples': samples,
        'accept_rate': n_acc / n_steps,
    }
    if return_trajectory:
        result['trajectory'] = np.array(trajectory)
    return result


def run_mcmc_from_config(initial_two_order, beta, eps, n_steps, rng,
                         record_every=100):
    """MCMC starting from a specific configuration. No thermalization."""
    current = initial_two_order.copy()
    current_cs = current.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0
    samples = []

    for step in range(n_steps):
        proposed = swap_move(current, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step % record_every == 0:
            samples.append((current.copy(), current_cs, current_S))

    return {
        'samples': samples,
        'accept_rate': n_acc / n_steps,
    }


def link_fraction(cs):
    """links / total relations."""
    links = cs.link_matrix()
    n_links = int(np.sum(links))
    n_rels = cs.num_relations()
    if n_rels == 0:
        return 1.0
    return n_links / n_rels


def get_ordered_config(N, eps, rng):
    """Get an ordered-phase configuration by running MCMC at high beta."""
    bc = beta_c(N, eps)
    result = run_mcmc(N, 5.0 * bc, eps, n_steps=8000, n_therm=6000,
                      record_every=500, rng=rng)
    if result['samples']:
        return result['samples'][-1][0]  # return last TwoOrder
    # fallback: identity permutations (maximally ordered)
    to = TwoOrder.__new__(TwoOrder)
    to.N = N
    to.u = np.arange(N)
    to.v = np.arange(N)
    return to


print("=" * 78)
print("EXPERIMENT 69: BD PHASE TRANSITION — DEEP PHYSICS (IDEAS 211-220)")
print("=" * 78)
print(f"eps=0.12, beta_c(N=50) = {beta_c(50, 0.12):.3f}")
print()


# ============================================================
# IDEA 211: LATENT HEAT SCALING
# ============================================================

print("=" * 78)
print("IDEA 211: LATENT HEAT SCALING — Delta_S(beta_c) vs N")
print("=" * 78)
print("If first-order: Delta_S ~ N (extensive latent heat)")
print("If second-order: Delta_S -> 0 as N -> infinity")
print()

t0 = time.time()
Ns_211 = [30, 40, 50, 60, 70]
eps = 0.12
n_steps_211 = 20000
n_therm_211 = 10000

latent_heats = []
for N in Ns_211:
    bc = beta_c(N, eps)
    # Measure action just below and just above beta_c
    beta_below = 0.5 * bc
    beta_above = 2.0 * bc

    result_below = run_mcmc(N, beta_below, eps, n_steps_211, n_therm_211,
                            200, rng)
    result_above = run_mcmc(N, beta_above, eps, n_steps_211, n_therm_211,
                            200, rng)

    actions_below = [s[2] for s in result_below['samples']]
    actions_above = [s[2] for s in result_above['samples']]

    S_dis = np.mean(actions_below)
    S_ord = np.mean(actions_above)
    delta_S = abs(S_dis - S_ord)
    delta_S_per_N = delta_S / N

    latent_heats.append({
        'N': N, 'delta_S': delta_S, 'delta_S_per_N': delta_S_per_N,
        'S_dis': S_dis, 'S_ord': S_ord,
        'S_dis_std': np.std(actions_below),
        'S_ord_std': np.std(actions_above),
    })

    print(f"  N={N:3d}: S_dis={S_dis:.3f}+/-{np.std(actions_below):.3f}, "
          f"S_ord={S_ord:.3f}+/-{np.std(actions_above):.3f}, "
          f"Delta_S={delta_S:.3f}, Delta_S/N={delta_S_per_N:.4f}")

# Fit Delta_S = a * N^alpha
Ns_arr = np.array([d['N'] for d in latent_heats])
dS_arr = np.array([d['delta_S'] for d in latent_heats])

if np.all(dS_arr > 0):
    log_N = np.log(Ns_arr)
    log_dS = np.log(dS_arr)
    slope, intercept, r, p, se = stats.linregress(log_N, log_dS)
    print(f"\n  Scaling fit: Delta_S ~ N^{slope:.3f} (R^2={r**2:.4f}, SE={se:.3f})")
    print(f"  First-order prediction: exponent = 1.0")
    print(f"  Measured exponent: {slope:.3f}")
    if slope > 0.7:
        print(f"  --> CONSISTENT with first-order (extensive latent heat)")
    elif slope < 0.3:
        print(f"  --> CONSISTENT with second-order or crossover")
    else:
        print(f"  --> AMBIGUOUS — could be weak first-order or finite-size effects")

    # Also check Delta_S/N constancy (first-order prediction)
    dS_per_N = dS_arr / Ns_arr
    cv = np.std(dS_per_N) / np.mean(dS_per_N)
    print(f"  Delta_S/N values: {dS_per_N}")
    print(f"  Coefficient of variation of Delta_S/N: {cv:.3f}")
    print(f"  (CV < 0.2 means roughly constant -> first-order)")
else:
    print("  WARNING: some Delta_S values are zero or negative")

dt_211 = time.time() - t0
print(f"\n  [Idea 211 completed in {dt_211:.1f}s]")


# ============================================================
# IDEA 212: HYSTERESIS LOOP
# ============================================================

print("\n" + "=" * 78)
print("IDEA 212: HYSTERESIS LOOP — metastability signature")
print("=" * 78)
print("Heating (high beta -> low) vs Cooling (low beta -> high)")
print()

t0 = time.time()
N_212 = 50
bc_212 = beta_c(N_212, eps)
# Scan beta from 0 to 4*beta_c in 12 steps
beta_scan = np.linspace(0, 4 * bc_212, 12)
n_steps_per_beta = 5000  # steps at each beta value

# COOLING: start disordered (low beta), increase beta
print("  COOLING branch (disordered -> ordered):")
current_cool = TwoOrder(N_212, rng=rng)  # random start
cooling_actions = []
cooling_links = []

for beta_val in beta_scan:
    current_cs = current_cool.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0

    action_samples = []
    link_samples = []

    for step in range(n_steps_per_beta):
        proposed = swap_move(current_cool, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta_val * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current_cool = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= 2000 and step % 200 == 0:
            action_samples.append(current_S)
            link_samples.append(link_fraction(current_cs))

    mean_S = np.mean(action_samples) if action_samples else current_S
    mean_L = np.mean(link_samples) if link_samples else 0
    cooling_actions.append(mean_S)
    cooling_links.append(mean_L)
    print(f"    beta/bc={beta_val/bc_212:.2f}: <S>/N={mean_S/N_212:.4f}, "
          f"<L>={mean_L:.4f}, acc={n_acc/n_steps_per_beta:.3f}")

# HEATING: start ordered (high beta), decrease beta
print("\n  HEATING branch (ordered -> disordered):")
current_heat = get_ordered_config(N_212, eps, rng)
heating_actions = []
heating_links = []

for beta_val in beta_scan[::-1]:
    current_cs = current_heat.to_causet()
    current_S = bd_action_corrected(current_cs, eps)
    n_acc = 0

    action_samples = []
    link_samples = []

    for step in range(n_steps_per_beta):
        proposed = swap_move(current_heat, rng)
        proposed_cs = proposed.to_causet()
        proposed_S = bd_action_corrected(proposed_cs, eps)

        dS = beta_val * (proposed_S - current_S)
        if dS <= 0 or rng.random() < np.exp(-dS):
            current_heat = proposed
            current_cs = proposed_cs
            current_S = proposed_S
            n_acc += 1

        if step >= 2000 and step % 200 == 0:
            action_samples.append(current_S)
            link_samples.append(link_fraction(current_cs))

    mean_S = np.mean(action_samples) if action_samples else current_S
    mean_L = np.mean(link_samples) if link_samples else 0
    heating_actions.append(mean_S)
    heating_links.append(mean_L)
    print(f"    beta/bc={beta_val/bc_212:.2f}: <S>/N={mean_S/N_212:.4f}, "
          f"<L>={mean_L:.4f}, acc={n_acc/n_steps_per_beta:.3f}")

# Reverse heating to match beta ordering
heating_actions = heating_actions[::-1]
heating_links = heating_links[::-1]

# Compute hysteresis width
action_diff = np.array(cooling_actions) - np.array(heating_actions)
link_diff = np.array(cooling_links) - np.array(heating_links)
max_action_hyst = np.max(np.abs(action_diff))
max_link_hyst = np.max(np.abs(link_diff))

print(f"\n  HYSTERESIS ANALYSIS:")
print(f"  Max action hysteresis: {max_action_hyst:.4f} (at beta/bc ~ "
      f"{beta_scan[np.argmax(np.abs(action_diff))]/bc_212:.2f})")
print(f"  Max link hysteresis:   {max_link_hyst:.4f} (at beta/bc ~ "
      f"{beta_scan[np.argmax(np.abs(link_diff))]/bc_212:.2f})")

# Compare with random noise level
print(f"\n  Cooling <S>/N at each beta/bc:")
for i, b in enumerate(beta_scan):
    arrow = " <-- transition region" if 0.8 < b/bc_212 < 1.5 else ""
    hyst = cooling_actions[i] - heating_actions[i]
    print(f"    {b/bc_212:.2f}: cool={cooling_actions[i]/N_212:.4f}, "
          f"heat={heating_actions[i]/N_212:.4f}, diff={hyst/N_212:.5f}{arrow}")

dt_212 = time.time() - t0
print(f"\n  [Idea 212 completed in {dt_212:.1f}s]")


# ============================================================
# IDEA 213: ACTION HISTOGRAM BIMODALITY
# ============================================================

print("\n" + "=" * 78)
print("IDEA 213: ACTION HISTOGRAM BIMODALITY AT beta_c")
print("=" * 78)
print("First-order => double-peaked histogram. Second-order => single peak.")
print()

t0 = time.time()
Ns_213 = [30, 50, 70]
n_steps_213 = 30000
n_therm_213 = 15000

for N in Ns_213:
    bc = beta_c(N, eps)
    # Run long chain at exactly beta_c to collect many action samples
    result = run_mcmc(N, bc, eps, n_steps_213, n_therm_213,
                      record_every=50, rng=rng)
    actions = np.array([s[2] for s in result['samples']])

    print(f"  N={N}, beta_c={bc:.3f}, {len(actions)} samples:")
    print(f"    <S> = {np.mean(actions):.4f}, std = {np.std(actions):.4f}")
    print(f"    min = {np.min(actions):.4f}, max = {np.max(actions):.4f}")
    print(f"    skewness = {stats.skew(actions):.3f}, "
          f"kurtosis = {stats.kurtosis(actions):.3f}")

    # Test for bimodality using Hartigan's dip test approximation
    # Simple approach: check if there are two modes in a histogram
    n_bins = max(15, len(actions) // 10)
    hist, bin_edges = np.histogram(actions, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Find peaks (local maxima in histogram)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append((i, hist[i], bin_centers[i]))

    if len(peaks) >= 2:
        # Sort by height, take top two
        peaks.sort(key=lambda x: x[1], reverse=True)
        p1, p2 = peaks[0], peaks[1]
        # Valley between them
        i_lo = min(p1[0], p2[0])
        i_hi = max(p1[0], p2[0])
        if i_hi > i_lo:
            valley = np.min(hist[i_lo:i_hi+1])
            peak_avg = (p1[1] + p2[1]) / 2
            valley_ratio = valley / peak_avg if peak_avg > 0 else 1.0
            separation = abs(p1[2] - p2[2])
            print(f"    BIMODAL: 2 peaks at S={p1[2]:.3f} and S={p2[2]:.3f}")
            print(f"    Peak separation: {separation:.4f}")
            print(f"    Valley/peak ratio: {valley_ratio:.3f} "
                  f"(0=deep valley, 1=no valley)")
        else:
            print(f"    Two peaks found but overlapping")
    else:
        print(f"    UNIMODAL: {len(peaks)} peak(s) detected")
        if len(peaks) == 1:
            print(f"    Peak at S={peaks[0][2]:.3f}")

    # Also report excess kurtosis as bimodality indicator
    # Bimodal distributions often have negative excess kurtosis
    kurt = stats.kurtosis(actions)
    print(f"    Excess kurtosis = {kurt:.3f} "
          f"({'suggestive of bimodality' if kurt < -0.5 else 'unimodal-like'})")
    print()

dt_213 = time.time() - t0
print(f"  [Idea 213 completed in {dt_213:.1f}s]")


# ============================================================
# IDEA 214: BINDER CUMULANT
# ============================================================

print("\n" + "=" * 78)
print("IDEA 214: BINDER CUMULANT U_4")
print("=" * 78)
print("U_4 = 1 - <S^4>/(3<S^2>^2)")
print("First-order: minimum dips to U_4 < 2/3, depth grows with N")
print("Second-order: U_4 -> 2/3 at criticality")
print()

t0 = time.time()
Ns_214 = [30, 40, 50, 60, 70]
n_steps_214 = 20000
n_therm_214 = 10000
beta_multiples = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0]

print(f"{'N':>4s}", end="")
for m in beta_multiples:
    print(f" {'%.1fbc'%m:>8s}", end="")
print()
print("-" * (4 + 9 * len(beta_multiples)))

binder_mins = {}

for N in Ns_214:
    bc = beta_c(N, eps)
    binder_vals = []

    for mult in beta_multiples:
        beta_val = mult * bc
        result = run_mcmc(N, beta_val, eps, n_steps_214, n_therm_214,
                          200, rng)
        actions = np.array([s[2] for s in result['samples']])

        # Center the action around its mean for the cumulant
        S_mean = np.mean(actions)
        S_centered = actions - S_mean
        S2 = np.mean(S_centered**2)
        S4 = np.mean(S_centered**4)

        if S2 > 1e-20:
            U4 = 1.0 - S4 / (3.0 * S2**2)
        else:
            U4 = 2.0/3.0  # trivial case

        binder_vals.append(U4)

    binder_mins[N] = np.min(binder_vals)
    print(f"{N:>4d}", end="")
    for u in binder_vals:
        print(f" {u:>8.4f}", end="")
    print(f"  min={np.min(binder_vals):.4f}")

# Check if Binder minimum deepens with N
print(f"\n  Binder minimum vs N:")
for N in Ns_214:
    print(f"    N={N}: U4_min = {binder_mins[N]:.4f}")

Ns_b = np.array(list(binder_mins.keys()))
U4_mins = np.array(list(binder_mins.values()))

# For first-order: minimum should deepen (become more negative) with N
if len(Ns_b) >= 3:
    slope_b, _, r_b, _, _ = stats.linregress(Ns_b, U4_mins)
    print(f"    Linear fit slope: {slope_b:.6f} per element")
    print(f"    R^2 = {r_b**2:.4f}")
    if slope_b < -0.001 and r_b**2 > 0.5:
        print(f"    --> Binder dip DEEPENS with N: FIRST-ORDER signature")
    elif abs(slope_b) < 0.001:
        print(f"    --> Binder dip roughly constant: SECOND-ORDER or crossover")
    else:
        print(f"    --> Binder dip becomes shallower: crossover or finite-size")

dt_214 = time.time() - t0
print(f"\n  [Idea 214 completed in {dt_214:.1f}s]")


# ============================================================
# IDEA 215: METASTABLE LIFETIME
# ============================================================

print("\n" + "=" * 78)
print("IDEA 215: METASTABLE LIFETIME — ordered phase decay below beta_c")
print("=" * 78)
print("Start in KR phase, quench to beta < beta_c. Measure steps to melt.")
print("First-order: lifetime ~ exp(c*N). Second-order: power-law.")
print()

t0 = time.time()
Ns_215 = [30, 40, 50]  # smaller N to keep runtime reasonable
n_trials = 3
quench_beta_mult = 0.7  # quench to 70% of beta_c (just below transition)
max_steps = 15000

lifetimes = {}

for N in Ns_215:
    bc = beta_c(N, eps)
    beta_quench = quench_beta_mult * bc
    trial_lifetimes = []

    for trial in range(n_trials):
        # Get ordered config
        seed_rng = np.random.default_rng(42 + trial * 1000 + N)
        ordered = get_ordered_config(N, eps, seed_rng)
        ordered_cs = ordered.to_causet()
        ordered_lf = link_fraction(ordered_cs)
        ordered_S = bd_action_corrected(ordered_cs, eps)

        # Get reference disordered link fraction
        dis_result = run_mcmc(N, 0.0, eps, 3000, 1500, 500, seed_rng)
        dis_lf = np.mean([link_fraction(s[1]) for s in dis_result['samples']])

        # Threshold: halfway between ordered and disordered link fraction
        threshold = (ordered_lf + dis_lf) / 2.0

        # Quench: run at beta_quench starting from ordered config
        current = ordered.copy()
        current_cs = ordered_cs
        current_S = ordered_S
        melted_step = max_steps  # default if doesn't melt

        for step in range(max_steps):
            proposed = swap_move(current, seed_rng)
            proposed_cs = proposed.to_causet()
            proposed_S = bd_action_corrected(proposed_cs, eps)

            dS = beta_quench * (proposed_S - current_S)
            if dS <= 0 or seed_rng.random() < np.exp(-dS):
                current = proposed
                current_cs = proposed_cs
                current_S = proposed_S

            if step % 200 == 0:
                lf = link_fraction(current_cs)
                if lf < threshold:
                    melted_step = step
                    break

        trial_lifetimes.append(melted_step)

    lifetimes[N] = trial_lifetimes
    mean_lt = np.mean(trial_lifetimes)
    print(f"  N={N}: lifetimes = {trial_lifetimes}, "
          f"mean = {mean_lt:.0f} steps, "
          f"survived = {sum(1 for t in trial_lifetimes if t >= max_steps)}/{n_trials}")

# Check scaling
print(f"\n  Lifetime scaling with N:")
Ns_lt = np.array(list(lifetimes.keys()))
mean_lts = np.array([np.mean(lifetimes[N]) for N in Ns_lt])

if np.all(mean_lts > 0):
    # Exponential fit: log(lifetime) ~ a*N + b
    log_lts = np.log(mean_lts)
    slope_lt, intercept_lt, r_lt, _, _ = stats.linregress(Ns_lt, log_lts)
    print(f"  Exponential fit: lifetime ~ exp({slope_lt:.4f} * N)")
    print(f"  R^2 = {r_lt**2:.4f}")

    # Power-law fit: log(lifetime) ~ a*log(N) + b
    log_Ns = np.log(Ns_lt)
    slope_pl, _, r_pl, _, _ = stats.linregress(log_Ns, log_lts)
    print(f"  Power-law fit:   lifetime ~ N^{slope_pl:.2f}")
    print(f"  R^2 = {r_pl**2:.4f}")

    if r_lt**2 > r_pl**2 + 0.05:
        print(f"  --> Exponential fits better: FIRST-ORDER metastability")
    elif r_pl**2 > r_lt**2 + 0.05:
        print(f"  --> Power-law fits better: SECOND-ORDER critical slowing")
    else:
        print(f"  --> Both fits similar: inconclusive (need larger N range)")

dt_215 = time.time() - t0
print(f"\n  [Idea 215 completed in {dt_215:.1f}s]")


# ============================================================
# IDEA 216: NUCLEATION DYNAMICS
# ============================================================

print("\n" + "=" * 78)
print("IDEA 216: NUCLEATION DYNAMICS — step-by-step transition tracking")
print("=" * 78)
print("During ordered->disordered melting, does link fraction change")
print("gradually (second-order) or in a sharp jump (nucleation)?")
print()

t0 = time.time()
N_216 = 50
bc_216 = beta_c(N_216, eps)

# Start ordered, quench to 0.6*beta_c (stronger quench for faster transition)
ordered_216 = get_ordered_config(N_216, eps, rng)
beta_quench_216 = 0.6 * bc_216

current = ordered_216.copy()
current_cs = current.to_causet()
current_S = bd_action_corrected(current_cs, eps)

# Track link fraction at every recorded step
track_every = 50
lf_trajectory = []
action_trajectory = []

for step in range(12000):
    proposed = swap_move(current, rng)
    proposed_cs = proposed.to_causet()
    proposed_S = bd_action_corrected(proposed_cs, eps)

    dS = beta_quench_216 * (proposed_S - current_S)
    if dS <= 0 or rng.random() < np.exp(-dS):
        current = proposed
        current_cs = proposed_cs
        current_S = proposed_S

    if step % track_every == 0:
        lf_trajectory.append(link_fraction(current_cs))
        action_trajectory.append(current_S)

lf_traj = np.array(lf_trajectory)
act_traj = np.array(action_trajectory)

# Find the biggest single-step jump in link fraction
diffs = np.diff(lf_traj)
max_jump_idx = np.argmax(np.abs(diffs))
max_jump = diffs[max_jump_idx]

# Characterize the transition
# Total change
total_change = lf_traj[-1] - lf_traj[0]
# Fraction of total change in the biggest single step
if abs(total_change) > 1e-6:
    jump_fraction = abs(max_jump) / abs(total_change)
else:
    jump_fraction = 0.0

print(f"  N={N_216}, quench to {0.6:.1f}*beta_c")
print(f"  Link fraction: {lf_traj[0]:.4f} (start) -> {lf_traj[-1]:.4f} (end)")
print(f"  Total change: {total_change:+.4f}")
print(f"  Biggest single-step jump: {max_jump:+.4f} at step {max_jump_idx * track_every}")
print(f"  Jump/total: {jump_fraction:.3f} "
      f"({'SHARP nucleation' if jump_fraction > 0.3 else 'gradual transition'})")

# Compute running variance (fluctuations should peak at transition)
window = 10
if len(lf_traj) > 2 * window:
    running_var = []
    for i in range(window, len(lf_traj) - window):
        running_var.append(np.var(lf_traj[i-window:i+window]))
    max_var_idx = np.argmax(running_var) + window
    print(f"  Peak fluctuation at step {max_var_idx * track_every} "
          f"(var={np.max(running_var):.6f})")

# Show trajectory in 4 segments
n_seg = 4
seg_len = len(lf_traj) // n_seg
print(f"\n  Trajectory (link fraction, {n_seg} segments):")
for seg in range(n_seg):
    seg_data = lf_traj[seg*seg_len:(seg+1)*seg_len]
    step_start = seg * seg_len * track_every
    step_end = (seg+1) * seg_len * track_every
    print(f"    Steps {step_start:5d}-{step_end:5d}: "
          f"<L>={np.mean(seg_data):.4f} +/- {np.std(seg_data):.4f}")

dt_216 = time.time() - t0
print(f"\n  [Idea 216 completed in {dt_216:.1f}s]")


# ============================================================
# IDEA 217: KR PHASE STRUCTURE
# ============================================================

print("\n" + "=" * 78)
print("IDEA 217: KR (CRYSTALLINE) PHASE STRUCTURE")
print("=" * 78)
print("Deep in ordered phase: what IS the causal set?")
print()

t0 = time.time()
N_217 = 50
bc_217 = beta_c(N_217, eps)

# Run deep in ordered phase
beta_deep = 5.0 * bc_217
result_kr = run_mcmc(N_217, beta_deep, eps, 20000, 12000, 500, rng)
n_kr_samples = len(result_kr['samples'])

print(f"  N={N_217}, beta={beta_deep:.2f} = 5*beta_c, {n_kr_samples} samples")
print()

# Measure structural properties on each KR sample
chain_lengths = []
antichain_sizes = []
link_fracs = []
ordering_fracs = []
n_layers_list = []
max_width_list = []
n_links_list = []

for two_order, cs, action in result_kr['samples']:
    N = cs.n
    chain_lengths.append(cs.longest_chain())

    # Antichain via pi = v * u^{-1}
    u_inv = np.argsort(two_order.u)
    pi = two_order.v[u_inv]
    # LDS = LIS of reversed
    rev_pi = pi[::-1]
    tails = []
    for x in rev_pi:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    antichain_sizes.append(len(tails))

    link_fracs.append(link_fraction(cs))
    ordering_fracs.append(cs.ordering_fraction())

    # Layer structure: assign each element to a "layer" based on longest
    # chain from minimal elements
    order = cs.order.astype(np.int32)
    depth = np.zeros(N, dtype=int)
    # Topological sort by finding elements with no predecessors first
    for j in range(N):
        preds = np.where(cs.order[:, j])[0]
        if len(preds) > 0:
            depth[j] = np.max(depth[preds]) + 1
    n_layers = int(np.max(depth)) + 1
    n_layers_list.append(n_layers)

    # Width of widest layer
    layer_counts = np.bincount(depth, minlength=n_layers)
    max_width_list.append(int(np.max(layer_counts)))

    n_links = int(np.sum(cs.link_matrix()))
    n_links_list.append(n_links)

print(f"  KR Phase Properties (averaged over {n_kr_samples} samples):")
print(f"    Chain length:      {np.mean(chain_lengths):.1f} +/- {np.std(chain_lengths):.1f} "
      f"(chain/N = {np.mean(chain_lengths)/N_217:.3f})")
print(f"    Antichain size:    {np.mean(antichain_sizes):.1f} +/- {np.std(antichain_sizes):.1f} "
      f"(ac/sqrt(N) = {np.mean(antichain_sizes)/np.sqrt(N_217):.3f})")
print(f"    Link fraction:     {np.mean(link_fracs):.4f} +/- {np.std(link_fracs):.4f}")
print(f"    Ordering fraction: {np.mean(ordering_fracs):.4f} +/- {np.std(ordering_fracs):.4f}")
print(f"    Number of layers:  {np.mean(n_layers_list):.1f} +/- {np.std(n_layers_list):.1f}")
print(f"    Max layer width:   {np.mean(max_width_list):.1f} +/- {np.std(max_width_list):.1f}")
print(f"    Number of links:   {np.mean(n_links_list):.1f} +/- {np.std(n_links_list):.1f}")

# Characterize the structure
avg_chain_ratio = np.mean(chain_lengths) / N_217
avg_ac_ratio = np.mean(antichain_sizes) / np.sqrt(N_217)

print(f"\n  Structural diagnosis:")
if avg_chain_ratio > 0.9:
    print(f"    Chain/N = {avg_chain_ratio:.3f} -> NEAR-TOTAL ORDER (single chain)")
elif avg_chain_ratio > 0.5:
    print(f"    Chain/N = {avg_chain_ratio:.3f} -> ELONGATED (chain-dominated)")
else:
    print(f"    Chain/N = {avg_chain_ratio:.3f} -> WIDE (antichain-dominated)")

avg_of = np.mean(ordering_fracs)
print(f"    Ordering fraction = {avg_of:.3f}")
if avg_of > 0.9:
    print(f"    -> Nearly total order: most pairs are comparable")
elif avg_of > 0.6:
    print(f"    -> Highly ordered but with some incomparable pairs")
else:
    print(f"    -> Moderate ordering: significant causal set structure")

avg_max_w = np.mean(max_width_list)
avg_layers = np.mean(n_layers_list)
print(f"    Layer structure: {avg_layers:.0f} layers, widest has {avg_max_w:.0f} elements")
if avg_max_w < 3:
    print(f"    -> LATTICE-LIKE: narrow layers suggest a crystalline lattice")
elif avg_max_w < N_217 / avg_layers:
    print(f"    -> LAYERED: moderate-width layers")
else:
    print(f"    -> AMORPHOUS: wide layers, little regularity")

# Compare with disordered phase
result_dis = run_mcmc(N_217, 0.0, eps, 10000, 5000, 500, rng)

dis_chains = []
dis_acs = []
dis_lfs = []
dis_ofs = []

for two_order, cs, action in result_dis['samples']:
    N = cs.n
    dis_chains.append(cs.longest_chain())
    u_inv = np.argsort(two_order.u)
    pi = two_order.v[u_inv]
    rev_pi = pi[::-1]
    tails = []
    for x in rev_pi:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    dis_acs.append(len(tails))
    dis_lfs.append(link_fraction(cs))
    dis_ofs.append(cs.ordering_fraction())

print(f"\n  Comparison with disordered phase (beta=0):")
print(f"    {'Property':25s} {'Disordered':>12s} {'KR (ordered)':>12s} {'Ratio':>8s}")
print(f"    {'-'*60}")
props = [
    ("Chain/N", np.mean(dis_chains)/N_217, np.mean(chain_lengths)/N_217),
    ("Antichain/sqrt(N)", np.mean(dis_acs)/np.sqrt(N_217),
     np.mean(antichain_sizes)/np.sqrt(N_217)),
    ("Link fraction", np.mean(dis_lfs), np.mean(link_fracs)),
    ("Ordering fraction", np.mean(dis_ofs), np.mean(ordering_fracs)),
]
for name, v_dis, v_kr in props:
    ratio = v_kr / v_dis if abs(v_dis) > 1e-10 else float('inf')
    print(f"    {name:25s} {v_dis:>12.4f} {v_kr:>12.4f} {ratio:>8.2f}x")

dt_217 = time.time() - t0
print(f"\n  [Idea 217 completed in {dt_217:.1f}s]")


# ============================================================
# IDEA 218: SPECIFIC HEAT PEAK SCALING
# ============================================================

print("\n" + "=" * 78)
print("IDEA 218: SPECIFIC HEAT PEAK SCALING")
print("=" * 78)
print("C_V = beta^2 * Var(S) / N")
print("First-order: C_V_peak ~ N. Second-order: C_V_peak ~ N^{alpha/nu}")
print()

t0 = time.time()
Ns_218 = [30, 40, 50, 60, 70]
# Fine scan near beta_c
beta_mults_cv = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 2.0]
n_steps_218 = 20000
n_therm_218 = 10000

cv_peaks = {}

for N in Ns_218:
    bc = beta_c(N, eps)
    cv_vals = []

    for mult in beta_mults_cv:
        beta_val = mult * bc
        result = run_mcmc(N, beta_val, eps, n_steps_218, n_therm_218,
                          200, rng)
        actions = np.array([s[2] for s in result['samples']])

        var_S = np.var(actions)
        cv = beta_val**2 * var_S / N if beta_val > 0 else 0.0
        cv_vals.append(cv)

    cv_peak = np.max(cv_vals)
    cv_peak_mult = beta_mults_cv[np.argmax(cv_vals)]
    cv_peaks[N] = cv_peak

    print(f"  N={N}: C_V peak = {cv_peak:.4f} at beta/bc = {cv_peak_mult:.2f}")
    cv_line = "    C_V: " + " ".join([f"{v:.3f}" for v in cv_vals])
    print(cv_line)

# Fit C_V_peak vs N
print(f"\n  Specific heat peak scaling:")
Ns_cv = np.array(list(cv_peaks.keys()))
peaks_cv = np.array(list(cv_peaks.values()))

if np.all(peaks_cv > 0):
    log_N = np.log(Ns_cv)
    log_cv = np.log(peaks_cv)
    slope_cv, intercept_cv, r_cv, _, se_cv = stats.linregress(log_N, log_cv)
    print(f"  C_V_peak ~ N^{slope_cv:.3f} (R^2={r_cv**2:.4f}, SE={se_cv:.3f})")
    print(f"  First-order prediction: exponent = 1.0")
    print(f"  Measured exponent: {slope_cv:.3f}")

    if slope_cv > 0.7:
        print(f"  --> CONSISTENT with first-order (C_V ~ N)")
    elif 0.0 < slope_cv < 0.5:
        print(f"  --> CONSISTENT with second-order (alpha/nu < 1)")
    else:
        print(f"  --> AMBIGUOUS")

dt_218 = time.time() - t0
print(f"\n  [Idea 218 completed in {dt_218:.1f}s]")


# ============================================================
# IDEA 219: LINK FRACTION SUSCEPTIBILITY
# ============================================================

print("\n" + "=" * 78)
print("IDEA 219: LINK FRACTION SUSCEPTIBILITY")
print("=" * 78)
print("chi_L = N * Var(link_fraction)")
print("Peak should diverge at beta_c; scaling determines universality class")
print()

t0 = time.time()
Ns_219 = [30, 40, 50, 60, 70]
beta_mults_chi = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 2.0]
n_steps_219 = 20000
n_therm_219 = 10000

chi_peaks = {}

for N in Ns_219:
    bc = beta_c(N, eps)
    chi_vals = []

    for mult in beta_mults_chi:
        beta_val = mult * bc
        result = run_mcmc(N, beta_val, eps, n_steps_219, n_therm_219,
                          200, rng)
        lf_samples = [link_fraction(s[1]) for s in result['samples']]

        var_lf = np.var(lf_samples)
        chi = N * var_lf
        chi_vals.append(chi)

    chi_peak = np.max(chi_vals)
    chi_peak_mult = beta_mults_chi[np.argmax(chi_vals)]
    chi_peaks[N] = chi_peak

    print(f"  N={N}: chi_L peak = {chi_peak:.4f} at beta/bc = {chi_peak_mult:.2f}")

# Fit chi_peak vs N
print(f"\n  Susceptibility peak scaling:")
Ns_chi = np.array(list(chi_peaks.keys()))
peaks_chi = np.array(list(chi_peaks.values()))

if np.all(peaks_chi > 0):
    log_N = np.log(Ns_chi)
    log_chi = np.log(peaks_chi)
    slope_chi, intercept_chi, r_chi, _, se_chi = stats.linregress(log_N, log_chi)
    print(f"  chi_L_peak ~ N^{slope_chi:.3f} (R^2={r_chi**2:.4f}, SE={se_chi:.3f})")
    print(f"  First-order prediction: exponent ~ 1.0 (volume scaling)")
    print(f"  Second-order (mean-field): exponent = gamma/nu")
    print(f"  Measured exponent: {slope_chi:.3f}")

dt_219 = time.time() - t0
print(f"\n  [Idea 219 completed in {dt_219:.1f}s]")


# ============================================================
# IDEA 220: TRICRITICAL SEARCH — eps scan
# ============================================================

print("\n" + "=" * 78)
print("IDEA 220: TRICRITICAL SEARCH — scan eps at fixed N=50")
print("=" * 78)
print("Does the first-order transition weaken at some eps?")
print("Measure action variance peak (proxy for latent heat) vs eps.")
print()

t0 = time.time()
N_220 = 50
eps_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
n_steps_220 = 15000
n_therm_220 = 8000

print(f"  {'eps':>6s} {'beta_c':>8s} {'Var(S)_peak':>12s} {'<S>_dis':>10s} "
      f"{'<S>_ord':>10s} {'Delta_S':>10s} {'C_V_peak':>10s}")
print(f"  {'-'*70}")

eps_results = []

for eps_val in eps_values:
    bc = beta_c(N_220, eps_val)

    # Measure at beta=0.5*bc (disordered) and beta=2*bc (ordered)
    res_dis = run_mcmc(N_220, 0.5 * bc, eps_val, n_steps_220, n_therm_220,
                       200, rng)
    res_ord = run_mcmc(N_220, 2.0 * bc, eps_val, n_steps_220, n_therm_220,
                       200, rng)

    # Also measure at beta_c for variance
    res_crit = run_mcmc(N_220, bc, eps_val, n_steps_220, n_therm_220,
                        200, rng)

    S_dis = np.mean([s[2] for s in res_dis['samples']])
    S_ord = np.mean([s[2] for s in res_ord['samples']])
    delta_S = abs(S_dis - S_ord)

    actions_crit = np.array([s[2] for s in res_crit['samples']])
    var_S = np.var(actions_crit)
    cv_peak = bc**2 * var_S / N_220

    eps_results.append({
        'eps': eps_val, 'bc': bc, 'var_S': var_S, 'S_dis': S_dis,
        'S_ord': S_ord, 'delta_S': delta_S, 'cv_peak': cv_peak,
    })

    print(f"  {eps_val:>6.3f} {bc:>8.3f} {var_S:>12.4f} {S_dis:>10.4f} "
          f"{S_ord:>10.4f} {delta_S:>10.4f} {cv_peak:>10.4f}")

# Check if transition strength varies with eps
print(f"\n  Transition strength vs eps:")
eps_arr = np.array([r['eps'] for r in eps_results])
dS_arr = np.array([r['delta_S'] for r in eps_results])
cv_arr = np.array([r['cv_peak'] for r in eps_results])

# Normalize delta_S by N to get latent heat per element
dS_per_N = dS_arr / N_220

for r in eps_results:
    print(f"    eps={r['eps']:.3f}: Delta_S/N = {r['delta_S']/N_220:.5f}, "
          f"C_V_peak = {r['cv_peak']:.4f}")

# Check for minimum (tricritical point would show vanishing Delta_S)
min_idx = np.argmin(dS_per_N)
max_idx = np.argmax(dS_per_N)
print(f"\n  Weakest transition at eps = {eps_arr[min_idx]:.3f} "
      f"(Delta_S/N = {dS_per_N[min_idx]:.5f})")
print(f"  Strongest transition at eps = {eps_arr[max_idx]:.3f} "
      f"(Delta_S/N = {dS_per_N[max_idx]:.5f})")

if dS_per_N[min_idx] < 0.3 * dS_per_N[max_idx]:
    print(f"  --> Transition weakens significantly: possible tricritical region")
else:
    print(f"  --> Transition strength varies but no clear vanishing point")

dt_220 = time.time() - t0
print(f"\n  [Idea 220 completed in {dt_220:.1f}s]")


# ============================================================
# FINAL SUMMARY AND SCORING
# ============================================================

print("\n" + "=" * 78)
print("FINAL SUMMARY: IDEAS 211-220")
print("=" * 78)

print("""
BD Phase Transition Deep Physics — N=30-70, eps=0.12

The 10 ideas probe the UNKNOWN physics of the first-order BD transition:
thermodynamic characterization, metastability, microscopic dynamics,
and phase diagram topology.
""")

# Collect all timing
print("Results and Scores:")
print("-" * 78)

# Score each idea based on what we found
ideas = [
    (211, "Latent heat scaling", "Delta_S vs N"),
    (212, "Hysteresis loop", "Heating/cooling asymmetry"),
    (213, "Action histogram bimodality", "Double-peaked at beta_c?"),
    (214, "Binder cumulant", "U_4 minimum deepening?"),
    (215, "Metastable lifetime", "Decay time vs N"),
    (216, "Nucleation dynamics", "Sharp vs gradual transition"),
    (217, "KR phase structure", "What is the ordered phase?"),
    (218, "Specific heat peak", "C_V_peak scaling with N"),
    (219, "Link fraction susceptibility", "chi_L peak scaling with N"),
    (220, "Tricritical search", "Transition strength vs eps"),
]

for num, title, what in ideas:
    print(f"\n  Idea {num}: {title}")
    print(f"    Measured: {what}")

print(f"""

KEY QUESTION ANSWERED:
  Is the BD transition genuinely first-order at eps=0.12?
  Evidence from: latent heat scaling, Binder cumulant, specific heat,
  histogram bimodality, hysteresis, and metastable lifetimes.

  A coherent picture should emerge from the 10 measurements above.
""")
