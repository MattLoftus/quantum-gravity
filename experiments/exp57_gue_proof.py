"""
Experiment 57: Toward an analytic proof of GUE statistics for the Pauli-Jordan operator

The Pauli-Jordan operator on a 2-order causal set is:
    Delta = (2/N)(C^T - C)
where C is the causal (order) matrix. The Hermitian form H = i*Delta has real eigenvalues,
and Paper D showed that the level spacing ratio <r> ~ 0.56-0.62, matching GUE.

This experiment systematically investigates WHY GUE emerges:

Part 1: Signum matrix spectrum (the total order / chain limit)
Part 2: Crossover from signum to GUE as randomness increases
Part 3: Random signed matrix null hypothesis
Part 4: Eigenvalue density shape
Part 5: Universality test (independent entries with matched variance)
Part 6: Entry correlation structure
Part 7: N-scaling of <r> for all ensembles
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from causal_sets.two_orders import TwoOrder
from causal_sets.fast_core import FastCausalSet

np.set_printoptions(precision=4, suppress=True, linewidth=120)
rng = np.random.default_rng(42)


# ============================================================
# UTILITIES
# ============================================================

def level_spacing_ratio(evals):
    """Compute mean level spacing ratio <r> from sorted eigenvalues.
    GUE: <r> ~ 0.5996, GOE: <r> ~ 0.5307, Poisson: <r> ~ 0.3863
    """
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return np.nan
    s = spacings
    r_min = np.minimum(s[:-1], s[1:])
    r_max = np.maximum(s[:-1], s[1:])
    return np.mean(r_min / r_max)


def make_signum_matrix(N):
    """The N x N signum matrix: S[i,j] = sign(j-i).
    This is C^T - C for the total order (chain)."""
    idx = np.arange(N)
    S = np.sign(idx[None, :] - idx[:, None]).astype(float)
    return S


def pauli_jordan_evals(cs):
    """Eigenvalues of H = i * Delta = i * (2/N)(C^T - C)."""
    N = cs.n
    C = cs.order.astype(float)
    A = C.T - C
    H = 1j * (2.0 / N) * A
    return np.linalg.eigvalsh(H).real


def make_2order_causet(N, rng):
    """Generate a random 2-order causal set."""
    to = TwoOrder(N, rng=rng)
    return to.to_causet()


def make_sparse_antisym(N, density, rng):
    """Random sparse antisymmetric {-1,0,+1} matrix (vectorized).
    density = fraction of upper-triangular entries that are nonzero."""
    n_upper = N * (N - 1) // 2
    # Draw which entries are nonzero
    mask = rng.random(n_upper) < density
    # Draw signs for nonzero entries
    signs = rng.choice([-1.0, 1.0], size=n_upper)
    vals = mask * signs
    # Fill upper triangle
    A = np.zeros((N, N))
    A[np.triu_indices(N, k=1)] = vals
    A -= A.T
    return A


def make_dense_antisym_pm1(N, rng):
    """Random dense antisymmetric matrix with iid +-1 entries."""
    signs = rng.choice([-1.0, 1.0], size=(N * (N - 1) // 2,))
    A = np.zeros((N, N))
    A[np.triu_indices(N, k=1)] = signs
    A -= A.T
    return A


def make_gaussian_antisym(N, rng):
    """Random antisymmetric matrix with iid Gaussian entries."""
    vals = rng.standard_normal(N * (N - 1) // 2)
    A = np.zeros((N, N))
    A[np.triu_indices(N, k=1)] = vals
    A -= A.T
    return A


def hermitian_evals(A, N):
    """Eigenvalues of i*(2/N)*A."""
    H = 1j * (2.0 / N) * A
    return np.linalg.eigvalsh(H).real


# ============================================================
# PART 1: SIGNUM MATRIX SPECTRUM
# ============================================================
print("=" * 70)
print("PART 1: SIGNUM MATRIX SPECTRUM")
print("=" * 70)
print()

for N in [5, 10, 20, 50, 100, 200]:
    S = make_signum_matrix(N)
    evals = hermitian_evals(S, 1)  # No 2/N scaling for raw signum
    evals_sorted = np.sort(evals)
    r = level_spacing_ratio(evals_sorted)

    # Try known formula for signum matrix eigenvalues:
    # iS has eigenvalues +/- cot(pi(2k-1)/(4N+2)) for k=1,...,N/2
    # (for even N; odd N is similar with a zero eigenvalue)
    if N <= 50:
        if N % 2 == 0:
            formula_evals = []
            for k in range(1, N // 2 + 1):
                val = 1.0 / np.tan(np.pi * (2 * k - 1) / (4 * N + 2))
                formula_evals.extend([val, -val])
            formula_evals = np.sort(formula_evals)
            match = np.allclose(evals_sorted, formula_evals, atol=0.01)
        else:
            match = "N/A (odd)"
    else:
        match = "skipped"

    print(f"N={N:4d}: evals range [{evals_sorted[0]:.4f}, {evals_sorted[-1]:.4f}], "
          f"<r> = {r:.4f}, formula match: {match}")

    if N <= 10:
        print(f"  eigenvalues: {evals_sorted}")

print()
print("Reference: GUE <r>=0.5996, GOE <r>=0.5307, Poisson <r>=0.3863")
print()

# ============================================================
# PART 2: CROSSOVER FROM CHAIN TO RANDOM 2-ORDER
# ============================================================
print("=" * 70)
print("PART 2: CROSSOVER — chain to random 2-order")
print("=" * 70)
print()

N = 100
n_trials = 50

print(f"N={N}, {n_trials} trials per swap count")
print(f"{'n_swaps':>10s}  {'<r> mean':>8s}  {'<r> std':>8s}  {'ord_frac':>10s}")
print("-" * 50)

swap_counts = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]
for n_swaps in swap_counts:
    r_vals = []
    of_vals = []
    for trial in range(n_trials):
        u = np.arange(N)
        v = np.arange(N)

        # Apply n_swaps random transpositions to v
        if n_swaps > 0:
            pairs = rng.choice(N, size=(n_swaps, 2), replace=True)
            for s in range(n_swaps):
                i, j = pairs[s]
                if i != j:
                    v[i], v[j] = v[j], v[i]

        to = TwoOrder.from_permutations(u, v)
        cs = to.to_causet()
        evals = pauli_jordan_evals(cs)
        r_vals.append(level_spacing_ratio(evals))
        of_vals.append(cs.ordering_fraction())

    print(f"{n_swaps:10d}  {np.mean(r_vals):8.4f}  {np.std(r_vals):8.4f}  {np.mean(of_vals):10.4f}")

print()

# ============================================================
# PART 3: RANDOM SIGNED MATRIX NULL HYPOTHESIS
# ============================================================
print("=" * 70)
print("PART 3: RANDOM SIGNED MATRIX NULL")
print("=" * 70)
print()
print("Q: Does a random antisymmetric matrix with the same sparsity")
print("   as a 2-order produce the same <r>? If so, GUE comes from")
print("   the sparsity pattern alone, not the 2-order structure.")
print()

for N in [50, 100, 200]:
    n_trials_here = 100 if N <= 100 else 40

    r_2order = []
    r_null_matched = []
    r_null_dense = []
    of_vals = []

    for trial in range(n_trials_here):
        # 2-order
        cs = make_2order_causet(N, rng)
        evals = pauli_jordan_evals(cs)
        r_2order.append(level_spacing_ratio(evals))
        f = cs.ordering_fraction()
        of_vals.append(f)

        # Sparse null with matched density (2f of upper entries are nonzero)
        A_null = make_sparse_antisym(N, 2 * f, rng)
        r_null_matched.append(level_spacing_ratio(hermitian_evals(A_null, N)))

        # Dense antisymmetric ±1
        A_dense = make_dense_antisym_pm1(N, rng)
        r_null_dense.append(level_spacing_ratio(hermitian_evals(A_dense, N)))

    mean_of = np.mean(of_vals)
    print(f"N={N:3d} (ord_frac={mean_of:.3f}):")
    print(f"  2-order:          <r> = {np.mean(r_2order):.4f} +/- {np.std(r_2order):.4f}")
    print(f"  Sparse null:      <r> = {np.mean(r_null_matched):.4f} +/- {np.std(r_null_matched):.4f}")
    print(f"  Dense antisym:    <r> = {np.mean(r_null_dense):.4f} +/- {np.std(r_null_dense):.4f}")
    print()

# ============================================================
# PART 4: EIGENVALUE DENSITY SHAPE
# ============================================================
print("=" * 70)
print("PART 4: EIGENVALUE DENSITY — semicircle vs heavy tails")
print("=" * 70)
print()

N = 200
n_trials_density = 50

all_evals_2order = []
all_evals_null_sparse = []
all_evals_null_dense = []

# Signum
S = make_signum_matrix(N)
all_evals_signum = hermitian_evals(S, N)

for trial in range(n_trials_density):
    cs = make_2order_causet(N, rng)
    evals = pauli_jordan_evals(cs)
    all_evals_2order.extend(evals)
    f = cs.ordering_fraction()

    A_null = make_sparse_antisym(N, 2 * f, rng)
    all_evals_null_sparse.extend(hermitian_evals(A_null, N))

    A_dense = make_dense_antisym_pm1(N, rng)
    all_evals_null_dense.extend(hermitian_evals(A_dense, N))

all_evals_2order = np.array(all_evals_2order)
all_evals_null_sparse = np.array(all_evals_null_sparse)
all_evals_null_dense = np.array(all_evals_null_dense)

for label, evals in [("2-order", all_evals_2order),
                      ("Sparse null", all_evals_null_sparse),
                      ("Dense null", all_evals_null_dense),
                      ("Signum", all_evals_signum)]:
    evals_abs = np.abs(evals)
    print(f"{label:15s}: mean|lam|={np.mean(evals_abs):.4f}, "
          f"std={np.std(evals_abs):.4f}, "
          f"max={np.max(evals_abs):.4f}, "
          f"kurtosis={stats.kurtosis(evals):.2f}, "
          f"skewness={stats.skew(evals):.4f}")

print()
print("Semicircle reference: kurtosis = -1.0")
print("Gaussian reference:   kurtosis =  0.0")
print()

# ============================================================
# PART 5: UNIVERSALITY TEST — matched variance, independent entries
# ============================================================
print("=" * 70)
print("PART 5: UNIVERSALITY — independent entries with matched variance")
print("=" * 70)
print()
print("Test Erdos-Yau universality: if we match the variance profile")
print("of the 2-order matrix but make entries independent, do we still get GUE?")
print()

for N in [50, 100, 200]:
    n_trials_here = 80 if N <= 100 else 30

    # Measure variance profile of 2-order matrices
    var_sum = np.zeros((N, N))
    n_samples = min(n_trials_here, 50)
    for trial in range(n_samples):
        cs = make_2order_causet(N, rng)
        C = cs.order.astype(float)
        A = C.T - C
        var_sum += A ** 2
    var_profile = var_sum / n_samples  # E[A[i,j]^2]

    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    avg_var = np.mean(var_profile[upper_mask])
    # Extract upper triangle sigma values
    sigma_profile = np.sqrt(var_profile)

    r_2order = []
    r_indep_matched = []
    r_indep_uniform = []

    for trial in range(n_trials_here):
        # 2-order
        cs = make_2order_causet(N, rng)
        evals = pauli_jordan_evals(cs)
        r_2order.append(level_spacing_ratio(evals))

        # Independent Gaussian entries with matched variance profile (vectorized)
        n_upper = N * (N - 1) // 2
        sigmas_upper = sigma_profile[np.triu_indices(N, k=1)]
        vals_upper = rng.standard_normal(n_upper) * sigmas_upper
        A_indep = np.zeros((N, N))
        A_indep[np.triu_indices(N, k=1)] = vals_upper
        A_indep -= A_indep.T
        r_indep_matched.append(level_spacing_ratio(hermitian_evals(A_indep, N)))

        # Independent Gaussian with uniform variance
        sigma_unif = np.sqrt(avg_var)
        A_unif = make_gaussian_antisym(N, rng) * sigma_unif
        r_indep_uniform.append(level_spacing_ratio(hermitian_evals(A_unif, N)))

    print(f"N={N:3d} (avg entry variance = {avg_var:.4f}):")
    print(f"  2-order (correlated):     <r> = {np.mean(r_2order):.4f} +/- {np.std(r_2order):.4f}")
    print(f"  Indep matched variance:   <r> = {np.mean(r_indep_matched):.4f} +/- {np.std(r_indep_matched):.4f}")
    print(f"  Indep uniform variance:   <r> = {np.mean(r_indep_uniform):.4f} +/- {np.std(r_indep_uniform):.4f}")
    print()

# ============================================================
# PART 6: ENTRY CORRELATION STRUCTURE
# ============================================================
print("=" * 70)
print("PART 6: ENTRY CORRELATION STRUCTURE")
print("=" * 70)
print()
print("How correlated are entries A[i,j] and A[k,l] in 2-order matrices?")
print("If correlations decay fast, universality arguments apply.")
print()

N = 50
n_samples_corr = 500
A_samples = np.zeros((n_samples_corr, N, N))
for trial in range(n_samples_corr):
    cs = make_2order_causet(N, rng)
    C = cs.order.astype(float)
    A_samples[trial] = C.T - C

ref_entry = A_samples[:, 0, 1]
std_ref = np.std(ref_entry)

print(f"Correlations of A[0,1] with other entries (N={N}):")
print(f"{'(i,j)':>10s}  {'corr':>8s}  {'overlap':>10s}")
print("-" * 35)

test_pairs = [(0, 2), (0, 5), (0, 10), (0, 25), (0, 49),
              (1, 2), (1, 5), (1, 10),
              (5, 10), (10, 20), (25, 49),
              (2, 3), (20, 21), (48, 49)]

for (i, j) in test_pairs:
    other = A_samples[:, i, j]
    if np.std(other) < 1e-10 or std_ref < 1e-10:
        corr = 0.0
    else:
        corr = np.corrcoef(ref_entry, other)[0, 1]
    overlap = len({0, 1} & {i, j})
    print(f"({i:2d},{j:2d})      {corr:8.4f}  {overlap:10d}")

print()
print("Average |correlation| by index overlap:")
corr_by_overlap = {0: [], 1: []}
for _ in range(3000):
    i1, j1 = sorted(rng.choice(N, 2, replace=False))
    i2, j2 = sorted(rng.choice(N, 2, replace=False))
    if (i1, j1) == (i2, j2):
        continue
    overlap = len({i1, j1} & {i2, j2})
    if overlap > 1:
        continue
    a1 = A_samples[:, i1, j1]
    a2 = A_samples[:, i2, j2]
    s1, s2 = np.std(a1), np.std(a2)
    if s1 < 1e-10 or s2 < 1e-10:
        continue
    c = np.corrcoef(a1, a2)[0, 1]
    corr_by_overlap[overlap].append(abs(c))

for ovlp in [0, 1]:
    vals = corr_by_overlap[ovlp]
    if vals:
        print(f"  Overlap {ovlp}: mean |corr| = {np.mean(vals):.4f} "
              f"(max = {np.max(vals):.4f}, n={len(vals)})")

print()

# ============================================================
# PART 7: KEY DIAGNOSTIC — N-scaling of <r>
# ============================================================
print("=" * 70)
print("PART 7: N-SCALING OF <r> FOR ALL ENSEMBLES")
print("=" * 70)
print()

print(f"{'N':>5s}  {'2-order':>10s}  {'Sparse+-1':>10s}  {'Dense+-1':>10s}  {'Gauss':>10s}  {'Signum':>10s}")
print("-" * 60)

for N in [20, 30, 50, 80, 100, 150, 200, 300]:
    trials_here = 50 if N <= 100 else (30 if N <= 200 else 15)

    r_2o, r_sp, r_de, r_ga = [], [], [], []

    for trial in range(trials_here):
        cs = make_2order_causet(N, rng)
        evals = pauli_jordan_evals(cs)
        r_2o.append(level_spacing_ratio(evals))
        f = cs.ordering_fraction()

        A_null = make_sparse_antisym(N, 2 * f, rng)
        r_sp.append(level_spacing_ratio(hermitian_evals(A_null, N)))

        A_dense = make_dense_antisym_pm1(N, rng)
        r_de.append(level_spacing_ratio(hermitian_evals(A_dense, N)))

        A_gauss = make_gaussian_antisym(N, rng)
        r_ga.append(level_spacing_ratio(hermitian_evals(A_gauss, N)))

    S = make_signum_matrix(N)
    r_sig = level_spacing_ratio(hermitian_evals(S, N))

    print(f"{N:5d}  {np.mean(r_2o):10.4f}  {np.mean(r_sp):10.4f}  "
          f"{np.mean(r_de):10.4f}  {np.mean(r_ga):10.4f}  {r_sig:10.4f}")

print()
print("GUE = 0.5996, GOE = 0.5307, Poisson = 0.3863")
print()

# ============================================================
# PART 8: CRITICAL SPARSITY THRESHOLD
# ============================================================
print("=" * 70)
print("PART 8: CRITICAL SPARSITY — at what density does GUE emerge?")
print("=" * 70)
print()

N = 100
n_trials_sparse = 50

print(f"N={N}, {n_trials_sparse} trials per density")
print(f"{'density':>10s}  {'<r> sparse':>10s}  {'<r> std':>10s}")
print("-" * 35)

for density in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]:
    r_vals = []
    for trial in range(n_trials_sparse):
        A = make_sparse_antisym(N, density, rng)
        r_vals.append(level_spacing_ratio(hermitian_evals(A, N)))
    print(f"{density:10.2f}  {np.mean(r_vals):10.4f}  {np.std(r_vals):10.4f}")

print()
print("2-order ordering fraction ~ 0.25, so density ~ 0.50")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("SUMMARY & IMPLICATIONS FOR ANALYTIC PROOF")
print("=" * 70)
print("""
Key questions answered:
1. Does the signum matrix (total order) have GUE statistics?
2. How many random swaps are needed to reach GUE from a chain?
3. Does a random sparse antisymmetric matrix reproduce GUE?
   -> If YES: GUE is generic for antisymmetric matrices, not special to causets
   -> If NO: the causal structure plays a role
4. Do independent entries with matched variance give GUE?
   -> If YES: correlations don't matter, universality applies
5. How correlated are the entries of the causal matrix?
   -> If weakly correlated: Erdos-Yau universality should apply
6. At what sparsity does GUE emerge for random antisymmetric matrices?
   -> Gives the critical density for the universality transition

Implications:
- If sparse null matches: "The GUE statistics of the SJ vacuum are explained
  by random matrix universality for antisymmetric matrices with sufficient
  density of nonzero entries."
- If 2-order differs from null: "The causal structure imprints specific
  correlations that modify eigenvalue statistics beyond what random matrix
  universality predicts."
""")
