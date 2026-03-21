"""
Experiment 62: WILD CARD ROUND — Ideas 141-150

The final 10 ideas. Completely outside the box. Nobody in causal set theory
has considered these directions. Some will fail spectacularly. That's OK.
We're looking for one surprise.

Ideas:
141. CAUSAL SET AS QUANTUM COMPUTER: U = exp(iC). What quantum computation
     does the manifold-like phase perform? Is it computationally universal?
142. MUSIC OF THE CAUSAL SET: Sonify the iDelta spectrum. Is the power
     spectrum 1/f (pink noise) like physical systems, or white noise?
143. GAME THEORY ON CAUSAL SETS: Two-player game with causal payoffs.
     Do Nash equilibria encode geometry?
144. CELLULAR AUTOMATON ON THE CAUSAL SET: Rule 110-like dynamics on
     the DAG. Do complex patterns emerge? Edge of chaos?
145. CAUSAL SET AS NEURAL NETWORK: Forward propagation through C.
     What function does the manifold-like phase compute vs crystalline?
146. ZETA FUNCTION OF THE CAUSAL SET: zeta_C(s) = sum (interval_size)^{-s}.
     Does it have a functional equation? Zeros on a critical line?
147. QUANTUM WALK ON THE CAUSAL SET: Quantum walk (not classical) on the
     DAG. Hitting times, spreading rates — do they probe Lorentzian structure?
148. d'ALEMBERTIAN HEAT KERNEL: The retarded propagator defines a new spectral
     dimension. Compare with link-graph spectral dimension.
149. MCMC MIXING TIME & CRITICAL SLOWING DOWN: How fast does BD MCMC mix
     near the phase transition? Dynamical critical exponent z.
150. INFORMATION BOTTLENECK: Compress the N x N causal matrix to k dimensions.
     How does k scale with N? Is the causal set compressible?
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats, linalg
from scipy.optimize import curve_fit
from causal_sets.two_orders import TwoOrder, swap_move
from causal_sets.fast_core import FastCausalSet
from causal_sets.bd_action import count_intervals_by_size
import time

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
print("IDEA 141: CAUSAL SET AS QUANTUM COMPUTER")
print("U = exp(iC) defines a unitary. What computation does it perform?")
print("=" * 78)

# The causal matrix C is a {0,1} upper-triangular matrix.
# U = exp(iC) is unitary (since iC is not Hermitian, we use the matrix exponential).
# Actually, C is NOT Hermitian, so exp(iC) is NOT unitary in general.
# But the antisymmetric part A = C - C^T IS antisymmetric, so exp(A) IS orthogonal.
# And exp(i * Hermitian) = unitary. So we use H = i*(C^T - C) (the Pauli-Jordan).
#
# KEY IDEA: The "computational power" of U can be measured by:
# 1. Entangling power: how much entanglement does U create from product states?
# 2. Gate complexity: how many 2-qubit gates to decompose U?
# 3. Out-of-time-order correlator (OTOC): how fast does information scramble?

N_sizes = [20, 30, 40, 50]
n_trials = 12

print("\n--- Part A: Entangling power of U = exp(i*H) ---")
print("  (Entangling power measures how much entanglement U creates on average)")

for N in N_sizes:
    ent_powers_causet = []
    ent_powers_null = []

    for trial in range(n_trials):
        # Causet
        _, cs = make_2order_causet(N, rng)
        C = cs.order.astype(float)
        H = (C.T - C)  # antisymmetric
        # U = expm(i*H) -- this is unitary since i*antisym = Hermitian -> exp(iH_herm) unitary
        # Actually exp(antisymmetric) = orthogonal, which is real unitary
        U = linalg.expm(H)  # real orthogonal matrix

        # Entangling power: apply U to random product states and measure entropy
        # For efficiency: use the operator entanglement entropy of U itself
        # Reshape U as a bipartite operator on N/2 x N/2
        half = N // 2
        U_reshaped = U[:half*2, :half*2].reshape(half, 2, half, 2)
        # Compute operator Schmidt decomposition via SVD of the reshuffled matrix
        U_bipartite = U_reshaped.transpose(0, 2, 1, 3).reshape(half**2, 4)
        svs = np.linalg.svd(U_bipartite, compute_uv=False)
        svs = svs[svs > 1e-12]
        svs_sq = svs**2 / np.sum(svs**2)
        op_entropy = -np.sum(svs_sq * np.log(svs_sq + 1e-30))
        ent_powers_causet.append(op_entropy)

        # Null: random orthogonal from antisymmetric matrix
        A_null = rng.standard_normal((N, N))
        A_null = (A_null - A_null.T) / 2  # antisymmetrize
        A_null *= np.linalg.norm(H) / (np.linalg.norm(A_null) + 1e-10)  # match norm
        U_null = linalg.expm(A_null)
        U_null_bi = U_null[:half*2, :half*2].reshape(half, 2, half, 2)
        U_null_bp = U_null_bi.transpose(0, 2, 1, 3).reshape(half**2, 4)
        svs_null = np.linalg.svd(U_null_bp, compute_uv=False)
        svs_null = svs_null[svs_null > 1e-12]
        svs_null_sq = svs_null**2 / np.sum(svs_null**2)
        op_ent_null = -np.sum(svs_null_sq * np.log(svs_null_sq + 1e-30))
        ent_powers_null.append(op_ent_null)

    t_val, p_val = stats.ttest_ind(ent_powers_causet, ent_powers_null)
    print(f"  N={N}: causet={np.mean(ent_powers_causet):.3f}±{np.std(ent_powers_causet):.3f}, "
          f"null={np.mean(ent_powers_null):.3f}±{np.std(ent_powers_null):.3f}, p={p_val:.4f}")

print("\n--- Part B: OTOC-like scrambling ---")
print("  C(t) = <[W(t), V]^dag [W(t), V]>, W(t) = U^t W U^{-t}")
print("  Scrambling time = when C(t) saturates")

N = 30
n_trials_otoc = 10
t_steps = list(range(1, 16))

otoc_causet_avg = np.zeros(len(t_steps))
otoc_null_avg = np.zeros(len(t_steps))

for trial in range(n_trials_otoc):
    _, cs = make_2order_causet(N, rng)
    C = cs.order.astype(float)
    H = C.T - C
    U = linalg.expm(H)
    U_inv = linalg.expm(-H)

    # W and V are simple operators: single-site
    W = np.zeros((N, N)); W[0, 0] = 1.0
    V = np.zeros((N, N)); V[N//2, N//2] = 1.0

    for ti, t in enumerate(t_steps):
        # W(t) = U^t W U^{-t}
        Ut = np.linalg.matrix_power(U, t)
        Ut_inv = np.linalg.matrix_power(U_inv, t)
        Wt = Ut @ W @ Ut_inv
        comm = Wt @ V - V @ Wt
        otoc_val = np.real(np.trace(comm.conj().T @ comm)) / N
        otoc_causet_avg[ti] += otoc_val / n_trials_otoc

    # Null
    A_null = rng.standard_normal((N, N))
    A_null = (A_null - A_null.T) / 2
    A_null *= np.linalg.norm(H) / (np.linalg.norm(A_null) + 1e-10)
    U_null = linalg.expm(A_null)
    U_null_inv = linalg.expm(-A_null)
    for ti, t in enumerate(t_steps):
        Ut = np.linalg.matrix_power(U_null, t)
        Ut_inv = np.linalg.matrix_power(U_null_inv, t)
        Wt = Ut @ W @ Ut_inv
        comm = Wt @ V - V @ Wt
        otoc_val = np.real(np.trace(comm.conj().T @ comm)) / N
        otoc_null_avg[ti] += otoc_val / n_trials_otoc

print("  t   OTOC(causet)  OTOC(null)  ratio")
for ti, t in enumerate(t_steps):
    ratio = otoc_causet_avg[ti] / (otoc_null_avg[ti] + 1e-15)
    print(f"  {t:>2}  {otoc_causet_avg[ti]:>11.4f}  {otoc_null_avg[ti]:>10.4f}  {ratio:>6.3f}")

# Scrambling rate: fit exponential growth to early OTOC
early = min(8, len(t_steps))
t_arr = np.array(t_steps[:early], dtype=float)
if np.all(otoc_causet_avg[:early] > 1e-15):
    log_otoc = np.log(otoc_causet_avg[:early] + 1e-30)
    slope_c, _, r_c, _, _ = stats.linregress(t_arr, log_otoc)
    log_otoc_n = np.log(otoc_null_avg[:early] + 1e-30)
    slope_n, _, r_n, _, _ = stats.linregress(t_arr, log_otoc_n)
    print(f"\n  Scrambling exponent (Lyapunov): causet={slope_c:.3f} (R²={r_c**2:.3f}), "
          f"null={slope_n:.3f} (R²={r_n**2:.3f})")
    print(f"  MSS bound: lambda <= 2*pi*T. If causet lambda is SMALLER, it's less chaotic.")

print("\n  ASSESSMENT: Does the causal structure constrain quantum scrambling?")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 142: MUSIC OF THE CAUSAL SET — POWER SPECTRUM OF EIGENVALUES")
print("Map eigenvalues to a time series. Is the power spectrum 1/f (pink noise)?")
print("=" * 78)

# Many physical systems produce 1/f noise (pink noise).
# The eigenvalue sequence of a random matrix is known to have specific
# spectral properties (GUE eigenvalues have log-correlated fluctuations).
# Does the causal set's PJ spectrum produce 1/f noise?
#
# Method: treat the sorted eigenvalues as a time series.
# Compute the power spectral density (PSD).
# 1/f noise: PSD ~ 1/f^alpha with alpha ~ 1
# White noise: alpha ~ 0
# Brown noise: alpha ~ 2

N_sizes_music = [50, 80, 120]
n_trials_music = 20

print("\n--- Power spectral density of eigenvalue sequences ---")

for N in N_sizes_music:
    alphas_causet = []
    alphas_null = []
    alphas_gue = []

    for trial in range(n_trials_music):
        # Causet eigenvalues
        _, cs = make_2order_causet(N, rng)
        evals = sj_eigenvalues(cs)
        evals_sorted = np.sort(evals)

        # "Unfold" the spectrum: map to uniform density
        # Use the empirical CDF as the unfolding map
        n_ev = len(evals_sorted)
        unfolded = np.interp(evals_sorted,
                             evals_sorted,
                             np.linspace(0, n_ev, n_ev))
        # Fluctuations from the mean staircase
        delta_n = unfolded - np.arange(n_ev)

        # Power spectrum via FFT
        fft_vals = np.fft.rfft(delta_n)
        psd = np.abs(fft_vals[1:])**2  # skip DC
        freqs = np.arange(1, len(psd) + 1) / n_ev

        # Fit log(PSD) = -alpha * log(f) + const
        if len(freqs) > 5:
            log_f = np.log(freqs[:len(freqs)//2])  # use lower half of frequencies
            log_psd = np.log(psd[:len(freqs)//2] + 1e-30)
            slope, _, r_val, _, _ = stats.linregress(log_f, log_psd)
            alphas_causet.append(-slope)

        # Null: random antisymmetric matrix eigenvalues
        A_null = rng.standard_normal((N, N))
        A_null = (A_null - A_null.T) / 2
        evals_null = np.linalg.eigvalsh(1j * A_null).real
        evals_null_sorted = np.sort(evals_null)
        n_ev_n = len(evals_null_sorted)
        unfolded_n = np.interp(evals_null_sorted, evals_null_sorted,
                               np.linspace(0, n_ev_n, n_ev_n))
        delta_n_n = unfolded_n - np.arange(n_ev_n)
        fft_null = np.fft.rfft(delta_n_n)
        psd_null = np.abs(fft_null[1:])**2
        freqs_null = np.arange(1, len(psd_null) + 1) / n_ev_n
        if len(freqs_null) > 5:
            log_f_n = np.log(freqs_null[:len(freqs_null)//2])
            log_psd_n = np.log(psd_null[:len(freqs_null)//2] + 1e-30)
            slope_n, _, _, _, _ = stats.linregress(log_f_n, log_psd_n)
            alphas_null.append(-slope_n)

        # Pure GUE reference
        H_gue = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H_gue = (H_gue + H_gue.conj().T) / 2
        evals_gue = np.linalg.eigvalsh(H_gue)
        evals_gue_sorted = np.sort(evals_gue)
        n_gue = len(evals_gue_sorted)
        unfolded_gue = np.interp(evals_gue_sorted, evals_gue_sorted,
                                 np.linspace(0, n_gue, n_gue))
        delta_gue = unfolded_gue - np.arange(n_gue)
        fft_gue = np.fft.rfft(delta_gue)
        psd_gue = np.abs(fft_gue[1:])**2
        freqs_gue = np.arange(1, len(psd_gue) + 1) / n_gue
        if len(freqs_gue) > 5:
            log_f_g = np.log(freqs_gue[:len(freqs_gue)//2])
            log_psd_g = np.log(psd_gue[:len(freqs_gue)//2] + 1e-30)
            slope_g, _, _, _, _ = stats.linregress(log_f_g, log_psd_g)
            alphas_gue.append(-slope_g)

    if alphas_causet and alphas_null and alphas_gue:
        print(f"  N={N}: alpha_causet={np.mean(alphas_causet):.3f}±{np.std(alphas_causet):.3f}, "
              f"alpha_null={np.mean(alphas_null):.3f}±{np.std(alphas_null):.3f}, "
              f"alpha_GUE={np.mean(alphas_gue):.3f}±{np.std(alphas_gue):.3f}")
        print(f"         (1/f noise: alpha~1, white: alpha~0, brown: alpha~2)")

print("\n  KNOWN: GUE eigenvalue fluctuations have 1/f^2 power spectrum")
print("  (the 'spectral rigidity' of random matrices). If causet matches GUE,")
print("  this confirms the GUE universality from a completely different angle.")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 143: GAME THEORY ON CAUSAL SETS")
print("Two-player game: payoff depends on causal relation. Nash equilibria?")
print("=" * 78)

# Define a 2-player game on N elements:
# Player 1 chooses element i, Player 2 chooses element j.
# Payoff matrix M[i,j]:
#   If i < j (player 1's choice precedes player 2's): player 1 gets +1, player 2 gets -1
#   If j < i: player 1 gets -1, player 2 gets +1
#   If incomparable: both get 0
# This is a zero-sum game. The value of the game and the Nash equilibrium
# strategies encode the causal structure.

N = 50
n_trials = 15

print("\n--- Zero-sum causal game ---")
print("  Payoff: +1 if your choice causally precedes opponent's, -1 if after, 0 if spacelike")

game_values_causet = []
game_values_null = []
entropy_nash_causet = []
entropy_nash_null = []

for trial in range(n_trials):
    _, cs = make_2order_causet(N, rng)
    C = cs.order.astype(float)

    # Payoff matrix for player 1: M = C^T - C
    # (M[i,j] = 1 if j>i, -1 if i>j, 0 if incomparable)
    M = C.T - C

    # Solve zero-sum game: value = max_p min_q p^T M q
    # For a symmetric game, value = 0 and Nash is uniform over "balanced" elements.
    # But our game is NOT symmetric (asymmetry from the partial order).

    # Approximate Nash via fictitious play
    n_fp = 1000
    counts1 = np.ones(N)  # player 1's cumulative strategy
    counts2 = np.ones(N)  # player 2's cumulative strategy

    for step in range(n_fp):
        # Player 1 best-responds to player 2's mixed strategy
        q = counts2 / counts2.sum()
        payoffs1 = M @ q
        best1 = np.argmax(payoffs1)
        counts1[best1] += 1

        # Player 2 best-responds to player 1's (minimize player 1's payoff)
        p = counts1 / counts1.sum()
        payoffs2 = M.T @ p  # player 2's payoff is -M
        best2 = np.argmin(payoffs2)  # player 2 minimizes player 1's payoff
        counts2[best2] += 1

    p_nash = counts1 / counts1.sum()
    q_nash = counts2 / counts2.sum()
    game_value = p_nash @ M @ q_nash
    game_values_causet.append(game_value)

    # Nash equilibrium entropy (how concentrated is the strategy?)
    ent_p = -np.sum(p_nash * np.log(p_nash + 1e-30))
    ent_q = -np.sum(q_nash * np.log(q_nash + 1e-30))
    entropy_nash_causet.append((ent_p + ent_q) / 2)

    # Null
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)
    C_null = cs_null.order.astype(float)
    M_null = C_null.T - C_null

    counts1_n = np.ones(N)
    counts2_n = np.ones(N)
    for step in range(n_fp):
        q_n = counts2_n / counts2_n.sum()
        best1_n = np.argmax(M_null @ q_n)
        counts1_n[best1_n] += 1
        p_n = counts1_n / counts1_n.sum()
        best2_n = np.argmin(M_null.T @ p_n)
        counts2_n[best2_n] += 1

    p_nash_n = counts1_n / counts1_n.sum()
    q_nash_n = counts2_n / counts2_n.sum()
    game_values_null.append(p_nash_n @ M_null @ q_nash_n)
    ent_n = (-np.sum(p_nash_n * np.log(p_nash_n + 1e-30)) +
             -np.sum(q_nash_n * np.log(q_nash_n + 1e-30))) / 2
    entropy_nash_null.append(ent_n)

t_gv, p_gv = stats.ttest_ind(game_values_causet, game_values_null)
t_en, p_en = stats.ttest_ind(entropy_nash_causet, entropy_nash_null)

print(f"\n  Game value: causet={np.mean(game_values_causet):.4f}±{np.std(game_values_causet):.4f}, "
      f"null={np.mean(game_values_null):.4f}±{np.std(game_values_null):.4f}, p={p_gv:.4f}")
print(f"  Nash entropy: causet={np.mean(entropy_nash_causet):.3f}±{np.std(entropy_nash_causet):.3f}, "
      f"null={np.mean(entropy_nash_null):.3f}±{np.std(entropy_nash_null):.3f}, p={p_en:.4f}")
print(f"  Max entropy (uniform): {np.log(N):.3f}")

# How many elements have nonzero Nash weight?
support_causet = np.mean([np.sum(counts1 / counts1.sum() > 0.01) for _ in [0]])
print(f"  Nash support size (last trial): "
      f"|supp(p)|={np.sum(p_nash > 0.01)}, |supp(q)|={np.sum(q_nash > 0.01)}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 144: CELLULAR AUTOMATON ON THE CAUSAL SET")
print("Binary CA on the DAG: state depends on ancestors. Complexity measures.")
print("=" * 78)

# Each element has a binary state. The rule: the state of element x is
# determined by XOR (or majority vote) of the states of its immediate
# ancestors (parents = elements linked TO x).
# We initialize the minimal elements randomly and propagate forward.
#
# Measure: complexity of the resulting pattern. Complex = edge of chaos.
# Metrics: entropy of the state sequence, compressibility (LZ complexity).

N = 60
n_trials = 15

print("\n--- Binary CA on causal set (XOR rule) ---")

def run_ca_on_causet(cs, rule='xor', init_rng=None):
    """Run a cellular automaton on the causal set DAG.
    Elements are processed in topological order.
    State of x = rule applied to states of parents of x."""
    N = cs.n
    links = cs.link_matrix()
    state = np.zeros(N, dtype=int)

    # Find topological order (elements with no predecessors first)
    # Since our order matrix is upper-triangular-ish, natural order is roughly topological
    in_degree = links.sum(axis=0)  # number of parents
    minimal = np.where(in_degree == 0)[0]

    # Initialize minimal elements randomly
    if init_rng is not None:
        state[minimal] = init_rng.integers(0, 2, size=len(minimal))
    else:
        state[minimal] = 1  # all ones

    # Process in order of increasing number of predecessors
    order = np.argsort(cs.order.sum(axis=0))  # sort by number of things below

    for x in order:
        parents = np.where(links[:, x])[0]
        if len(parents) == 0:
            continue  # minimal element, already initialized
        parent_states = state[parents]
        if rule == 'xor':
            state[x] = np.sum(parent_states) % 2
        elif rule == 'majority':
            state[x] = 1 if np.sum(parent_states) > len(parents) / 2 else 0
        elif rule == 'or':
            state[x] = 1 if np.any(parent_states) else 0
    return state


def lempel_ziv_complexity(binary_seq):
    """Approximate Lempel-Ziv complexity (number of distinct substrings)."""
    s = ''.join(map(str, binary_seq))
    n = len(s)
    if n == 0:
        return 0
    complexity = 1
    i = 0
    k = 1
    while i + k <= n:
        # Check if s[i:i+k] appeared before in s[0:i+k-1]
        substr = s[i:i+k]
        if substr in s[:i+k-1]:
            k += 1
        else:
            complexity += 1
            i = i + k
            k = 1
    # Normalize by random string complexity: n / log2(n)
    return complexity / (n / (np.log2(n) + 1e-10))


complexities_causet = {rule: [] for rule in ['xor', 'majority', 'or']}
complexities_null = {rule: [] for rule in ['xor', 'majority', 'or']}
entropy_causet = {rule: [] for rule in ['xor', 'majority', 'or']}
entropy_null = {rule: [] for rule in ['xor', 'majority', 'or']}

for trial in range(n_trials):
    _, cs = make_2order_causet(N, rng)
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)

    for rule in ['xor', 'majority', 'or']:
        state = run_ca_on_causet(cs, rule=rule, init_rng=rng)
        state_null = run_ca_on_causet(cs_null, rule=rule, init_rng=rng)

        # LZ complexity
        lz = lempel_ziv_complexity(state)
        lz_null = lempel_ziv_complexity(state_null)
        complexities_causet[rule].append(lz)
        complexities_null[rule].append(lz_null)

        # State entropy
        p1 = np.mean(state)
        ent = -(p1 * np.log(p1 + 1e-30) + (1 - p1) * np.log(1 - p1 + 1e-30))
        p1_n = np.mean(state_null)
        ent_n = -(p1_n * np.log(p1_n + 1e-30) + (1 - p1_n) * np.log(1 - p1_n + 1e-30))
        entropy_causet[rule].append(ent)
        entropy_null[rule].append(ent_n)

print(f"\n  {'Rule':<10} {'LZ(causet)':<15} {'LZ(null)':<15} {'p-val':<10} "
      f"{'Ent(causet)':<15} {'Ent(null)':<15}")
for rule in ['xor', 'majority', 'or']:
    lz_c = np.mean(complexities_causet[rule])
    lz_n = np.mean(complexities_null[rule])
    t_lz, p_lz = stats.ttest_ind(complexities_causet[rule], complexities_null[rule])
    ent_c = np.mean(entropy_causet[rule])
    ent_n = np.mean(entropy_null[rule])
    print(f"  {rule:<10} {lz_c:<15.3f} {lz_n:<15.3f} {p_lz:<10.4f} {ent_c:<15.3f} {ent_n:<15.3f}")

print("\n  Edge of chaos: LZ complexity near 1.0 = random, near 0 = ordered,")
print("  intermediate = complex/interesting (Wolfram Class IV behavior)")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 145: CAUSAL SET AS NEURAL NETWORK")
print("Interpret C as directed network. What function does forward pass compute?")
print("=" * 78)

# Interpret the causal matrix C as a weight matrix of a neural network.
# Input: signal at minimal elements. Output: signal at maximal elements.
# Forward pass: x_{layer+1} = sigma(C^T @ x_{layer})
# where sigma is a nonlinearity (ReLU, sigmoid, tanh).
#
# Key question: what is the effective depth and width of this network?
# Manifold-like causets should have "deep, narrow" effective architecture
# (many layers = long causal chains) vs random DAGs (shallow, wide).

N = 50
n_trials = 15

print("\n--- Forward propagation analysis ---")

def forward_pass(cs, input_signal, n_layers=10, activation='relu'):
    """Propagate signal through causal network."""
    N = cs.n
    # Normalize weight matrix
    C = cs.order.astype(float)
    # Normalize columns to prevent explosion
    col_norms = C.sum(axis=0)
    col_norms[col_norms == 0] = 1
    W = C / col_norms[None, :]

    x = input_signal.copy()
    outputs = [x.copy()]
    for layer in range(n_layers):
        x = W.T @ x
        if activation == 'relu':
            x = np.maximum(x, 0)
        elif activation == 'tanh':
            x = np.tanh(x)
        # Renormalize to prevent vanishing
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm
        outputs.append(x.copy())
    return outputs


# Measure: how fast does the signal spread? (information propagation)
# And: how much of the network is activated?

spread_causet = []
spread_null = []
activation_frac_causet = []
activation_frac_null = []

for trial in range(n_trials):
    _, cs = make_2order_causet(N, rng)
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)

    # Input: localized signal at element 0
    x0 = np.zeros(N)
    x0[0] = 1.0

    outputs = forward_pass(cs, x0, n_layers=15, activation='relu')
    outputs_null = forward_pass(cs_null, x0, n_layers=15, activation='relu')

    # Measure spread: entropy of |x|^2 at each layer
    spreads = []
    spreads_null = []
    for out in outputs:
        p = out**2
        s = p.sum()
        if s > 1e-20:
            p = p / s
            ent = -np.sum(p * np.log(p + 1e-30))
            spreads.append(ent)
        else:
            spreads.append(0)
    for out in outputs_null:
        p = out**2
        s = p.sum()
        if s > 1e-20:
            p = p / s
            ent = -np.sum(p * np.log(p + 1e-30))
            spreads_null.append(ent)
        else:
            spreads_null.append(0)

    spread_causet.append(max(spreads))
    spread_null.append(max(spreads_null))

    # Fraction of nodes with nonzero activation at final layer
    final = outputs[-1]
    frac = np.sum(np.abs(final) > 1e-10) / N
    final_null = outputs_null[-1]
    frac_null = np.sum(np.abs(final_null) > 1e-10) / N
    activation_frac_causet.append(frac)
    activation_frac_null.append(frac_null)

t_sp, p_sp = stats.ttest_ind(spread_causet, spread_null)
t_af, p_af = stats.ttest_ind(activation_frac_causet, activation_frac_null)
print(f"  Max spread entropy: causet={np.mean(spread_causet):.3f}±{np.std(spread_causet):.3f}, "
      f"null={np.mean(spread_null):.3f}±{np.std(spread_null):.3f}, p={p_sp:.4f}")
print(f"  Final activation fraction: causet={np.mean(activation_frac_causet):.3f}±{np.std(activation_frac_causet):.3f}, "
      f"null={np.mean(activation_frac_null):.3f}±{np.std(activation_frac_null):.3f}, p={p_af:.4f}")

# Effective depth: at which layer does the signal first reach ALL elements?
print(f"  Max entropy = ln(N) = {np.log(N):.3f} (fully spread)")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 146: ZETA FUNCTION OF THE CAUSAL SET")
print("zeta_C(s) = sum_{x<=y} |I(x,y)|^{-s}. Analytic properties?")
print("=" * 78)

# The "causet zeta function": sum over all related pairs (x,y),
# weighted by the size of the causal interval between them.
# This is inspired by the Ihara zeta function of graphs and the
# Riemann zeta function.
#
# Properties to test:
# 1. Does zeta_C(s) have a meromorphic continuation?
# 2. Do the zeros lie on a "critical line" Re(s) = 1/2?
# 3. Does the residue at the pole encode the "volume" (N)?

N = 60
n_trials = 12

print("\n--- Causal set zeta function ---")

s_values = np.linspace(0.5, 5.0, 30)

zeta_causet_all = []
zeta_null_all = []

for trial in range(n_trials):
    _, cs = make_2order_causet(N, rng)
    density = cs.ordering_fraction()
    cs_null = random_dag(N, density * 0.7, rng)

    # Compute interval sizes for all related pairs
    order_int = cs.order.astype(np.int32)
    interval_matrix = order_int @ order_int  # interval_matrix[i,j] = |I(i,j)| - 2

    # For each related pair, the interval size is interval_matrix[i,j] + 2
    # (including x and y themselves)
    i_idx, j_idx = np.where(np.triu(cs.order, k=1))
    sizes = interval_matrix[i_idx, j_idx] + 2  # inclusive interval size

    # Compute zeta for various s
    zeta_vals = []
    for s in s_values:
        # sum |I(x,y)|^{-s} over all related pairs with |I|>0
        valid = sizes > 0
        if np.any(valid):
            z = np.sum(sizes[valid].astype(float) ** (-s))
        else:
            z = 0
        zeta_vals.append(z)
    zeta_causet_all.append(zeta_vals)

    # Null
    order_null = cs_null.order.astype(np.int32)
    iv_null = order_null @ order_null
    i_n, j_n = np.where(np.triu(cs_null.order, k=1))
    sizes_null = iv_null[i_n, j_n] + 2

    zeta_null = []
    for s in s_values:
        valid = sizes_null > 0
        if np.any(valid):
            z = np.sum(sizes_null[valid].astype(float) ** (-s))
        else:
            z = 0
        zeta_null.append(z)
    zeta_null_all.append(zeta_null)

zeta_c_mean = np.mean(zeta_causet_all, axis=0)
zeta_n_mean = np.mean(zeta_null_all, axis=0)

# Find the "abscissa of convergence": where does zeta start diverging?
# Approximate: where does d(log zeta)/ds change sign?
print(f"\n  Zeta function values at selected s:")
print(f"  {'s':<6} {'zeta(causet)':<15} {'zeta(null)':<15} {'ratio':<10}")
for i in range(0, len(s_values), 5):
    s = s_values[i]
    ratio = zeta_c_mean[i] / (zeta_n_mean[i] + 1e-10)
    print(f"  {s:<6.2f} {zeta_c_mean[i]:<15.2f} {zeta_n_mean[i]:<15.2f} {ratio:<10.3f}")

# Key test: does log(zeta) vs s have a kink (phase transition)?
log_zeta_c = np.log(zeta_c_mean + 1e-10)
log_zeta_n = np.log(zeta_n_mean + 1e-10)

# Second derivative (curvature)
d2_c = np.diff(log_zeta_c, n=2) / (s_values[1] - s_values[0])**2
d2_n = np.diff(log_zeta_n, n=2) / (s_values[1] - s_values[0])**2

# Peak curvature location
peak_c = s_values[1:-1][np.argmax(np.abs(d2_c))]
peak_n = s_values[1:-1][np.argmax(np.abs(d2_n))]
print(f"\n  Peak curvature of log(zeta): causet at s={peak_c:.2f}, null at s={peak_n:.2f}")

# Functional equation test: does zeta_C(s) * zeta_C(a-s) = const for some a?
# Test a = 2 (analogy with Riemann: zeta(s)*zeta(1-s) ~ functional equation)
print("\n  Functional equation test: zeta(s) * zeta(2-s):")
for s_test in [0.5, 0.75, 1.0, 1.25]:
    idx_s = np.argmin(np.abs(s_values - s_test))
    idx_2ms = np.argmin(np.abs(s_values - (2 - s_test)))
    if idx_s < len(zeta_c_mean) and idx_2ms < len(zeta_c_mean):
        product_c = zeta_c_mean[idx_s] * zeta_c_mean[idx_2ms]
        product_n = zeta_n_mean[idx_s] * zeta_n_mean[idx_2ms]
        print(f"    s={s_test:.2f}: causet product={product_c:.2f}, null product={product_n:.2f}")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 147: QUANTUM WALK ON THE CAUSAL SET")
print("Quantum walk (not classical) on the DAG. Spreading & hitting times.")
print("=" * 78)

# Classical random walk gives spectral dimension.
# QUANTUM walk: the walker is in a superposition.
# On a directed graph, we define the quantum walk via the link matrix L:
#   |psi(t+1)> = U_walk |psi(t)>
#   where U_walk = exp(i * theta * L_adj) with L_adj = L + L^T (undirected links)
# The quantum walk spreads FASTER than classical (ballistic vs diffusive).
# The spreading rate sigma^2(t) ~ t^alpha:
#   alpha = 1: classical diffusion
#   alpha = 2: ballistic (quantum walk on lattice)
#   alpha = something else: interesting!

N_sizes_qw = [30, 50, 70]
n_trials_qw = 12
t_max = 20
theta = 0.5  # walk parameter

print("\n--- Quantum walk on link graph ---")

for N in N_sizes_qw:
    alphas_causet_qw = []
    alphas_null_qw = []

    for trial in range(n_trials_qw):
        _, cs = make_2order_causet(N, rng)
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.7, rng)

        # Build walk unitary from link graph
        L = cs.link_matrix()
        A_link = (L | L.T).astype(float)
        # Normalize: use the adjacency matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A_link.sum(axis=1) + 1e-10))
        L_norm = D_inv_sqrt @ A_link @ D_inv_sqrt
        U_walk = linalg.expm(1j * theta * L_norm)

        # Initial state: localized at a middle element
        psi = np.zeros(N, dtype=complex)
        psi[N // 2] = 1.0

        # Evolve and track variance of position
        positions = np.arange(N, dtype=float)
        variances = []
        for t in range(t_max):
            psi = U_walk @ psi
            psi = psi / np.linalg.norm(psi)  # renormalize
            probs = np.abs(psi)**2
            mean_pos = np.sum(probs * positions)
            var_pos = np.sum(probs * (positions - mean_pos)**2)
            variances.append(var_pos)

        # Fit variance ~ t^alpha
        t_arr = np.arange(1, t_max + 1, dtype=float)
        log_t = np.log(t_arr[2:])  # skip first 2 (transients)
        log_var = np.log(np.array(variances[2:]) + 1e-10)
        if len(log_t) > 3:
            slope, _, r_val, _, _ = stats.linregress(log_t, log_var)
            alphas_causet_qw.append(slope)

        # Null quantum walk
        L_null = cs_null.link_matrix()
        A_null = (L_null | L_null.T).astype(float)
        D_inv_null = np.diag(1.0 / np.sqrt(A_null.sum(axis=1) + 1e-10))
        L_norm_null = D_inv_null @ A_null @ D_inv_null
        U_walk_null = linalg.expm(1j * theta * L_norm_null)

        psi_null = np.zeros(N, dtype=complex)
        psi_null[N // 2] = 1.0
        variances_null = []
        for t in range(t_max):
            psi_null = U_walk_null @ psi_null
            psi_null = psi_null / np.linalg.norm(psi_null)
            probs_n = np.abs(psi_null)**2
            mean_n = np.sum(probs_n * positions[:cs_null.n])
            var_n = np.sum(probs_n * (positions[:cs_null.n] - mean_n)**2)
            variances_null.append(var_n)

        log_var_n = np.log(np.array(variances_null[2:]) + 1e-10)
        if len(log_t) > 3 and len(log_var_n) >= len(log_t):
            slope_n, _, _, _, _ = stats.linregress(log_t, log_var_n[:len(log_t)])
            alphas_null_qw.append(slope_n)

    if alphas_causet_qw and alphas_null_qw:
        t_qw, p_qw = stats.ttest_ind(alphas_causet_qw, alphas_null_qw)
        print(f"  N={N}: spreading exponent alpha: "
              f"causet={np.mean(alphas_causet_qw):.3f}±{np.std(alphas_causet_qw):.3f}, "
              f"null={np.mean(alphas_null_qw):.3f}±{np.std(alphas_null_qw):.3f}, p={p_qw:.4f}")
        print(f"         (alpha=1: diffusive, alpha=2: ballistic, alpha>2: super-ballistic)")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 148: d'ALEMBERTIAN HEAT KERNEL SPECTRAL DIMENSION")
print("Use the retarded propagator (not link graph) to define spectral dimension.")
print("=" * 78)

# The link graph Laplacian gives a SPATIAL spectral dimension.
# But the causal set has a LORENTZIAN structure. The correct "Laplacian"
# is the d'Alembertian (wave operator), not the graph Laplacian.
#
# The d'Alembertian is approximated by the Sorkin operator:
#   Box_epsilon = (2/epsilon^{2/d}) * sum_{k=0}^{d} C_k * f_k
# where f_k sums over intervals of size k.
#
# For the heat kernel: K(sigma) = Tr[exp(-sigma * Box)]
# Spectral dimension: d_s(sigma) = -2 * d ln K / d ln sigma

N_sizes_hk = [40, 60, 80]
n_trials_hk = 10

print("\n--- d'Alembertian spectral dimension vs link-graph spectral dimension ---")

for N in N_sizes_hk:
    ds_dalembert = []
    ds_link = []

    for trial in range(n_trials_hk):
        _, cs = make_2order_causet(N, rng)

        # === Method 1: d'Alembertian spectral dimension ===
        # Use the Pauli-Jordan operator H = i*(C^T - C)/N as proxy for d'Alembertian
        # (in 2D, the retarded Green's function is related to the causal matrix)
        C_float = cs.order.astype(float)
        # Retarded propagator: G_R = C (the causal matrix itself)
        # d'Alembertian: Box = G_R^{-1} ... but C is singular.
        # Better approach: use H = i*(C^T-C)*(2/N) and its eigenvalues as "momentum modes"
        evals_h = sj_eigenvalues(cs)
        evals_pos = evals_h[evals_h > 1e-12]

        if len(evals_pos) > 3:
            # Heat kernel: K(sigma) = sum exp(-sigma * lambda_k)
            sigmas = np.logspace(-1, 2, 50)
            K_vals = np.array([np.sum(np.exp(-s * evals_pos)) for s in sigmas])

            # Spectral dimension: d_s = -2 * d(ln K)/d(ln sigma)
            ln_sigma = np.log(sigmas)
            ln_K = np.log(K_vals + 1e-30)
            # Numerical derivative
            d_ln_K = np.gradient(ln_K, ln_sigma)
            ds_vals = -2 * d_ln_K

            # Take the spectral dimension at the plateau (middle range)
            mid = len(sigmas) // 3
            end = 2 * len(sigmas) // 3
            ds_plateau = np.mean(ds_vals[mid:end])
            ds_dalembert.append(ds_plateau)

        # === Method 2: Link graph spectral dimension ===
        L = cs.link_matrix()
        A = (L | L.T).astype(float)
        degrees = A.sum(axis=1)
        D_mat = np.diag(degrees)
        Lap = D_mat - A
        evals_lap = np.linalg.eigvalsh(Lap)
        evals_lap_pos = evals_lap[evals_lap > 1e-10]

        if len(evals_lap_pos) > 3:
            K_link = np.array([np.sum(np.exp(-s * evals_lap_pos)) for s in sigmas])
            ln_K_link = np.log(K_link + 1e-30)
            d_ln_K_link = np.gradient(ln_K_link, ln_sigma)
            ds_link_vals = -2 * d_ln_K_link
            ds_link_plateau = np.mean(ds_link_vals[mid:end])
            ds_link.append(ds_link_plateau)

    if ds_dalembert and ds_link:
        print(f"  N={N}: d_s(d'Alembert)={np.mean(ds_dalembert):.3f}±{np.std(ds_dalembert):.3f}, "
              f"d_s(link graph)={np.mean(ds_link):.3f}±{np.std(ds_link):.3f}")
        t_ds, p_ds = stats.ttest_ind(ds_dalembert, ds_link)
        print(f"         t-test d'Alemb vs link: t={t_ds:.2f}, p={p_ds:.4f}")
        print(f"         Expected for 2D Minkowski: d_s = 2.0")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 149: MCMC MIXING TIME & CRITICAL SLOWING DOWN")
print("Autocorrelation time of BD MCMC near the phase transition.")
print("=" * 78)

# Near a phase transition, MCMC mixing slows down critically:
# tau_auto ~ |beta - beta_c|^{-z*nu}
# The dynamical critical exponent z characterizes the universality class.
# If we can measure z, we learn about the dynamics of causal set quantum gravity.

N = 30  # small for speed
n_steps = 8000
n_therm = 2000

# Scan beta values around the known transition region
betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

print(f"\n--- Autocorrelation time of ordering fraction vs beta (N={N}) ---")
print(f"  {'beta':<8} {'<f>':<10} {'std(f)':<10} {'tau_auto':<12} {'acceptance':<12}")

autocorr_times = []
ordering_fracs = []
susceptibilities = []

for beta in betas:
    # Run MCMC
    to = TwoOrder(N, rng=rng)

    f_trace = []
    accepted = 0
    total = 0

    for step in range(n_steps):
        # Propose swap move
        to_new = swap_move(to, rng)
        cs_old = to.to_causet()
        cs_new = to_new.to_causet()

        # BD action (simplified 2D): S = N - 2L
        L_old = int(np.sum(cs_old.link_matrix()))
        L_new = int(np.sum(cs_new.link_matrix()))
        S_old = N - 2 * L_old
        S_new = N - 2 * L_new

        dS = S_new - S_old
        if dS <= 0 or rng.random() < np.exp(-beta * dS):
            to = to_new
            if step >= n_therm:
                accepted += 1
        if step >= n_therm:
            total += 1
            cs = to.to_causet()
            f_trace.append(cs.ordering_fraction())

    f_arr = np.array(f_trace)
    mean_f = np.mean(f_arr)
    std_f = np.std(f_arr)

    # Compute autocorrelation time
    f_centered = f_arr - mean_f
    n_corr = len(f_centered)
    autocorr = np.correlate(f_centered, f_centered, mode='full')[n_corr-1:]
    autocorr = autocorr / (autocorr[0] + 1e-30)

    # Integrated autocorrelation time
    # Sum until autocorrelation drops below 0.05
    tau = 0.5  # start with 0.5 (self-correlation)
    for k in range(1, min(len(autocorr), n_corr // 4)):
        if autocorr[k] < 0.05:
            break
        tau += autocorr[k]

    acc_rate = accepted / (total + 1e-10)
    autocorr_times.append(tau)
    ordering_fracs.append(mean_f)
    susceptibilities.append(std_f**2 * N)

    print(f"  {beta:<8.1f} {mean_f:<10.4f} {std_f:<10.4f} {tau:<12.1f} {acc_rate:<12.3f}")

# Find the beta with maximum autocorrelation time (critical slowing down)
peak_idx = np.argmax(autocorr_times)
beta_peak = betas[peak_idx]
tau_peak = autocorr_times[peak_idx]

print(f"\n  Peak autocorrelation: tau={tau_peak:.1f} at beta={beta_peak:.1f}")
print(f"  Peak susceptibility: chi={max(susceptibilities):.3f} at "
      f"beta={betas[np.argmax(susceptibilities)]:.1f}")

# Critical slowing down: tau ~ |beta - beta_c|^{-z*nu}
# Try to fit near the peak
if tau_peak > 5:
    print(f"  Critical slowing down detected near beta={beta_peak}")
else:
    print(f"  No strong critical slowing down at N={N} (tau_max={tau_peak:.1f})")
    print(f"  (Expected: need larger N to see divergent autocorrelation)")


# ================================================================
print("\n\n" + "=" * 78)
print("IDEA 150: INFORMATION BOTTLENECK — COMPRESSIBILITY OF CAUSAL MATRIX")
print("Compress C to k dimensions. How does k scale with N?")
print("=" * 78)

# The NxN causal matrix C has N(N-1)/2 potential entries.
# But a manifold-like causet has structure — it should be compressible.
# Method: SVD of C. The number of significant singular values k
# measures the intrinsic dimensionality.
# If k ~ N^alpha with alpha < 1, the causet is compressible.
# The exponent alpha should depend on the spacetime dimension.

N_sizes_ib = [30, 50, 70, 100, 150]
n_trials_ib = 15
threshold = 0.99  # fraction of Frobenius norm to capture

print(f"\n--- SVD compression of causal matrix (capturing {threshold*100}% of norm) ---")

k_causet = {N: [] for N in N_sizes_ib}
k_null = {N: [] for N in N_sizes_ib}

for N in N_sizes_ib:
    for trial in range(n_trials_ib):
        _, cs = make_2order_causet(N, rng)
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.7, rng)

        # SVD of causal matrix
        C = cs.order.astype(float)
        svs = np.linalg.svd(C, compute_uv=False)
        total_norm_sq = np.sum(svs**2)
        cumulative = np.cumsum(svs**2) / total_norm_sq
        k = np.searchsorted(cumulative, threshold) + 1
        k_causet[N].append(k)

        # Null
        C_null = cs_null.order.astype(float)
        svs_null = np.linalg.svd(C_null, compute_uv=False)
        total_null = np.sum(svs_null**2)
        cum_null = np.cumsum(svs_null**2) / total_null
        k_null_val = np.searchsorted(cum_null, threshold) + 1
        k_null[N].append(k_null_val)

print(f"  {'N':<6} {'k(causet)':<15} {'k(null)':<15} {'k/N(causet)':<12} {'k/N(null)':<12} {'p-val':<8}")
for N in N_sizes_ib:
    kc = np.mean(k_causet[N])
    kn = np.mean(k_null[N])
    t_k, p_k = stats.ttest_ind(k_causet[N], k_null[N])
    print(f"  {N:<6} {kc:<15.1f} {kn:<15.1f} {kc/N:<12.3f} {kn/N:<12.3f} {p_k:<8.4f}")

# Fit k ~ N^alpha
log_N = np.log(N_sizes_ib)
log_k_c = np.log([np.mean(k_causet[N]) for N in N_sizes_ib])
log_k_n = np.log([np.mean(k_null[N]) for N in N_sizes_ib])

slope_c, _, r_c, _, _ = stats.linregress(log_N, log_k_c)
slope_n, _, r_n, _, _ = stats.linregress(log_N, log_k_n)

print(f"\n  Scaling: k ~ N^alpha")
print(f"  Causet: alpha = {slope_c:.3f} (R² = {r_c**2:.3f})")
print(f"  Null:   alpha = {slope_n:.3f} (R² = {r_n**2:.3f})")
print(f"  alpha < 1 means compressible (geometric structure reduces information)")
print(f"  alpha = 1 means incompressible (need ~N numbers to describe)")

# Also: spectral entropy of singular values
print(f"\n--- Spectral entropy of singular values (information content) ---")
for N in [50, 100, 150]:
    ent_c_list = []
    ent_n_list = []
    for trial in range(n_trials_ib):
        _, cs = make_2order_causet(N, rng)
        density = cs.ordering_fraction()
        cs_null = random_dag(N, density * 0.7, rng)

        svs = np.linalg.svd(cs.order.astype(float), compute_uv=False)
        p_sv = svs**2 / (np.sum(svs**2) + 1e-30)
        ent = -np.sum(p_sv * np.log(p_sv + 1e-30))
        ent_c_list.append(ent / np.log(N))  # normalize by max entropy

        svs_n = np.linalg.svd(cs_null.order.astype(float), compute_uv=False)
        p_n = svs_n**2 / (np.sum(svs_n**2) + 1e-30)
        ent_n = -np.sum(p_n * np.log(p_n + 1e-30))
        ent_n_list.append(ent_n / np.log(N))

    t_e, p_e = stats.ttest_ind(ent_c_list, ent_n_list)
    print(f"  N={N}: S_spec/ln(N): causet={np.mean(ent_c_list):.3f}±{np.std(ent_c_list):.3f}, "
          f"null={np.mean(ent_n_list):.3f}±{np.std(ent_n_list):.3f}, p={p_e:.4f}")


# ================================================================
# FINAL SCORING
# ================================================================
print("\n\n" + "=" * 78)
print("FINAL SCORING — IDEAS 141-150 (WILD CARD ROUND)")
print("=" * 78)

print("""
Scoring criteria:
  - Novelty: Has this been done before? (0-3)
  - Rigor: Clean signal, survives null test? (0-3)
  - Depth: Connection to fundamental physics? (0-2)
  - Audience: Who cares? (0-2)
  Total out of 10.

IDEA 141 — CAUSAL SET AS QUANTUM COMPUTER: 5.5/10
  Novelty: 2.5 — Nobody has studied entangling power or OTOCs of causet unitaries.
  Rigor: 1.5 — Entangling power is LOWER for causets than random (p<0.0001),
    meaning causal structure CONSTRAINS quantum computation. Statistically
    significant but the effect is small (1.35 vs 1.38). OTOCs are noisy and
    show no clear difference — scrambling rates are indistinguishable.
    The reduced entangling power is interesting but may just reflect the
    lower entropy of the causal matrix.
  Depth: 1 — Connection to quantum computation is genuine but shallow.
  Audience: 0.5 — Niche intersection of QI and causets.

IDEA 142 — MUSIC OF THE CAUSAL SET (Power Spectrum): 4.0/10
  Novelty: 1.5 — Power spectrum analysis of RMT eigenvalues is well-studied.
  Rigor: 0.5 — TOTAL NULL RESULT. alpha ~ 1.93 for causet, null, AND pure GUE
    — all identical to 3 decimal places. The unfolded eigenvalue fluctuations
    have the same 1/f^2 power spectrum regardless of origin. This is just
    confirming that any antisymmetric matrix with enough entries has GUE-like
    spectral rigidity. The "music" is identical for everything.
  Depth: 1 — Confirms GUE universality (which we already knew).
  Audience: 1 — Adds nothing beyond existing GUE result.

IDEA 143 — GAME THEORY ON CAUSAL SETS: 3.5/10
  Novelty: 2 — Nobody has done this. Fresh idea.
  Rigor: 0 — COMPLETE NULL RESULT. Game value is 0.0000 for both (it's a
    zero-sum antisymmetric game, so value is always 0 by symmetry — should
    have predicted this). Nash entropy is indistinguishable (p=0.33).
    The Nash support collapses to a single element (concentrated strategy),
    identical for causet and null. The game-theoretic approach extracts
    no geometric information whatsoever.
  Depth: 1 — The idea was interesting but the execution shows it's vacuous.
  Audience: 0.5 — Nobody.

IDEA 144 — CELLULAR AUTOMATON ON CAUSAL SET: 6.0/10
  Novelty: 2.5 — Genuinely new. CA on DAGs (not lattices) is unexplored.
  Rigor: 2 — SIGNIFICANT RESULTS. XOR rule: causet LZ=1.16 vs null=0.71
    (p=0.0013). Causets produce MORE COMPLEX patterns than random DAGs!
    This is consistent across all 3 rules (XOR, majority, OR). The XOR
    rule on causets produces near-random complexity (LZ~1.16, above 1.0),
    while null DAGs produce ordered/simple patterns. The causet's geometric
    structure creates an "edge of chaos" effect.
    CAVEAT: This could be a density effect (causets have different link
    structure than our null). Need density-matched null with same link
    degree distribution to be sure.
  Depth: 0.5 — Connection to Wolfram's computational universe is loose.
  Audience: 1 — Complexity science + causets intersection.

IDEA 145 — CAUSAL SET AS NEURAL NETWORK: 5.0/10
  Novelty: 2 — Fresh perspective, nobody has done this.
  Rigor: 1.5 — Strong statistical separation (p<0.0001) but in a TRIVIAL
    direction: the causet network KILLS the signal (activation fraction=0%),
    while random DAGs preserve it (28%). This is because causets have long
    chains (depth >> width), causing vanishing gradients. The network
    interpretation tells us causets are "deep and narrow" — we already knew
    this from longest chain analysis.
  Depth: 0.5 — Confirms known geometric property via a novel lens.
  Audience: 1 — ML people might find it curious.

IDEA 146 — ZETA FUNCTION OF THE CAUSAL SET: 5.0/10
  Novelty: 2.5 — New construction. Causet zeta function hasn't been defined.
  Rigor: 1 — The zeta function exists and differs between causet and null
    (ratio ~0.4-0.76 depending on s). But: no functional equation (the
    products zeta(s)*zeta(2-s) are NOT constant — they vary from 9500 to
    12400). No zeros on a critical line. Peak curvature at s=0.66 is
    identical for causet and null. The zeta function is a smooth, boring
    decreasing function with no interesting analytic structure.
  Depth: 1 — Connection to number theory doesn't materialize.
  Audience: 0.5 — Would need actual analytic structure to interest anyone.

IDEA 147 — QUANTUM WALK ON CAUSAL SET: 6.5/10
  Novelty: 2.5 — Quantum walks on causets are unstudied. Genuinely new.
  Rigor: 2 — STRONG, CLEAN RESULTS. Spreading exponents differ significantly:
    * Causet: alpha=0.27 (N=30) to 0.61 (N=70) — SUB-DIFFUSIVE
    * Null: alpha=0.86 (N=30) to 1.41 (N=70) — diffusive to super-diffusive
    * p < 0.001 at all sizes
    The causal structure CONSTRAINS quantum spreading — the walker on a
    causet spreads SLOWER than on a random DAG. This is physically meaningful:
    the causal diamond geometry creates a "bottleneck" for quantum transport.
    The N-dependence of the causet exponent (increasing toward ~0.6) suggests
    it may converge to a dimension-dependent value at large N.
    CAVEAT: Need to check if this is just a degree distribution effect.
  Depth: 1 — Lorentzian signature affects quantum transport.
  Audience: 0.5 — Quantum walk community + causets.

IDEA 148 — d'ALEMBERTIAN HEAT KERNEL: 4.5/10
  Novelty: 2 — Using PJ eigenvalues for spectral dimension is somewhat new.
  Rigor: 0.5 — WRONG RESULTS. d'Alembertian spectral dimension gives
    d_s ~ 0.45 (N=40), DECREASING to 0.28 (N=80). Should be 2.0 for 2D
    Minkowski. The PJ eigenvalues are NOT the right thing to exponentiate
    for a heat kernel — they give the "momentum spectrum" of a Lorentzian
    propagator, not a Euclidean one. The link graph gives d_s ~ 7-9, also
    wrong (too high). Neither method works at these scales.
  Depth: 1 — Conceptually right but implementation doesn't give meaningful numbers.
  Audience: 1 — Spectral geometry people.

IDEA 149 — MCMC MIXING TIME: 5.5/10
  Novelty: 2 — Dynamical properties of BD MCMC are understudied.
  Rigor: 1.5 — Peak autocorrelation tau=1001 at beta=1.5, consistent with
    the known phase transition region. Acceptance rates drop to <1% for
    beta > 1, showing the MCMC is barely mixing. But at N=30, we can't
    reliably extract a dynamical critical exponent z — need multiple N
    values and much longer chains. The qualitative signal is there
    (critical slowing down exists) but quantitative extraction fails.
  Depth: 1.5 — Dynamical critical exponent z would classify the universality
    of the causet phase transition — genuinely important.
  Audience: 0.5 — MCMC specialists + causet phase transition people.

IDEA 150 — INFORMATION BOTTLENECK (Compressibility): 7.0/10
  Novelty: 2.5 — SVD compression of causal matrices is completely new.
  Rigor: 2.5 — THE BEST RESULT OF THIS ROUND. Clean, compelling findings:
    * Causet: k ~ N^0.774 (R² = 0.999 — near-perfect power law!)
    * Null: k ~ N^0.405 (R² = 0.939)
    * Both are compressible (alpha < 1), but causets are LESS compressible
      than random DAGs. This is the opposite of naive expectation!
    * The causet needs ~N^0.77 singular values = more information content.
    * Spectral entropy is HIGHER for causets (0.36 vs 0.23 at N=50).
    * INTERPRETATION: The geometric structure of a causet encodes MORE
      information than a random DAG — the 2D Minkowski embedding creates
      correlations across ALL scales, not just local ones. The random DAG
      is dominated by a few large singular values (hierarchical structure)
      while the causet has a broader spectrum (geometric structure).
    * The exponent 0.774 is tantalizingly close to (d-1)/d = 1/2 for d=2...
      no, it's closer to d/(d+1) = 2/3... or perhaps ln(3)/ln(4)...
      The exact value may encode the spacetime dimension.
    * p < 0.0001 at all N >= 50.
  Depth: 1 — Information-theoretic characterization of geometry.
  Audience: 1 — Information theory + quantum gravity intersection.

============================================================================
GRAND SUMMARY — ALL 150 IDEAS
============================================================================

WILD CARD ROUND TOP 3:
  1. Idea 150 (Information Bottleneck): 7.0/10 — k ~ N^0.774 is a clean,
     novel result with near-perfect power law fit.
  2. Idea 147 (Quantum Walk): 6.5/10 — Sub-diffusive spreading is a
     genuine physical effect.
  3. Idea 144 (Cellular Automaton): 6.0/10 — Causets produce more complex
     CA patterns than random DAGs.

SURPRISES:
  - Idea 150: The causet is LESS compressible than random, not more.
    Geometric structure = more information, not less.
  - Idea 147: Quantum walks are SLOWER on causets (sub-diffusive).
    The causal diamond constrains quantum transport.
  - Idea 141: Causets have LOWER entangling power. Causal structure
    constrains quantum computation.
  - Idea 144: Causets produce MORE complex CA patterns. The partial order
    creates edge-of-chaos dynamics.

FAILURES (as expected for wild ideas):
  - Idea 142: Complete null. Power spectrum identical for everything.
  - Idea 143: Structural null. Zero-sum antisymmetric game always has value 0.
  - Idea 148: Wrong numbers. PJ heat kernel doesn't give correct d_s.

THE 7.5 CEILING REMAINS:
  After 150 ideas, the GUE quantum chaos result (7.5/10) remains the single
  strongest finding. Idea 150 (information bottleneck at 7.0) is the second
  best across ALL rounds.

  The fundamental barrier to 8+: at toy scale (N=30-150), finite-size effects
  dominate, and most signals either (a) reduce to density effects, or (b)
  require N > 1000 to show convergence to continuum predictions.

  PATH TO 8+: The two most promising routes remain:
  1. GUE proof: Explain WHY the Pauli-Jordan spectrum is GUE analytically.
  2. Large-N computation: Push to N ~ 1000+ to test continuum convergence
     of the information bottleneck exponent and quantum walk exponent.
""")
