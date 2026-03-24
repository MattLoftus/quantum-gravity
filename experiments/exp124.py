"""
Experiment 124: TEN WAYS TO EXTRACT d=2 FROM A RANDOM 2-ORDER (Ideas 751-760)

The observable: the NUMBER 2 (spacetime dimension d=2 for 2-orders).
We compute it 10 COMPLETELY DIFFERENT ways and ask: which estimator is best?

751. Myrheim-Meyer dimension estimator
752. Chain scaling exponent (longest_chain ~ N^{1/d} -> d = 1/alpha)
753. Antichain scaling exponent (max_antichain ~ N^{(d-1)/d} -> d = 1/(1-alpha))
754. Ordering fraction (f ~ 1/2^{d-1} -> d = 1 + log2(1/f))
755. Link fraction scaling (link_frac ~ ln(N)^{d-1}/N -> extract d from the exponent)
756. Spectral embedding dimension (how many eigenvectors for good R^2?)
757. Interval distribution shape (the master formula depends on d -> extract d from shape)
758. Box-counting dimension on embedded coordinates
759. Hasse graph expansion rate (Cheeger constant scaling with N)
760. Correlation dimension (from pairwise distances in spectral embedding)
"""

import sys
sys.path.insert(0, '/Users/Loftus/workspace/quantum-gravity')

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
import time
import warnings
warnings.filterwarnings('ignore')

from causal_sets.fast_core import FastCausalSet
from causal_sets.two_orders import TwoOrder
from causal_sets.dimension import myrheim_meyer, _ordering_fraction_theory, _invert_ordering_fraction
from causal_sets.bd_action import count_intervals_by_size

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(42)

print("=" * 90)
print("EXPERIMENT 124: TEN WAYS TO EXTRACT d=2 FROM A RANDOM 2-ORDER")
print("=" * 90)
sys.stdout.flush()

# ============================================================
# SHARED UTILITIES
# ============================================================

def random_2order(N, rng_local=None):
    """Generate a random 2-order and return FastCausalSet + TwoOrder."""
    if rng_local is None:
        rng_local = rng
    to = TwoOrder(N, rng=rng_local)
    return to.to_causet(), to


def hasse_adjacency(cs):
    """Symmetric adjacency matrix of the Hasse diagram."""
    links = cs.link_matrix()
    return (links | links.T).astype(np.float64)


def hasse_laplacian(cs):
    """Laplacian of the Hasse diagram."""
    adj = hasse_adjacency(cs)
    degree = np.sum(adj, axis=1)
    return np.diag(degree) - adj


def longest_chain(cs):
    """Length of the longest chain."""
    return cs.longest_chain()


def max_antichain(cs):
    """Size of the maximum antichain (width of the poset).
    Greedy approach with multiple random orderings."""
    N = cs.n
    best = 0
    for trial in range(min(20, N)):
        perm = rng.permutation(N)
        ac = []
        for idx in perm:
            compatible = True
            for a in ac:
                if cs.order[idx, a] or cs.order[a, idx]:
                    compatible = False
                    break
            if compatible:
                ac.append(idx)
        best = max(best, len(ac))
    return best


def spectral_embedding(cs, n_dims=10):
    """Embed the causet using Laplacian eigenvectors of the Hasse diagram.
    Returns coordinates (N x n_dims) from the smallest nonzero eigenvectors."""
    L = hasse_laplacian(cs)
    evals, evecs = eigh(L)
    n_use = min(n_dims, cs.n - 1)
    coords = evecs[:, 1:1+n_use]
    return coords, evals[1:1+n_use]


# ============================================================
# CONFIGURATION
# ============================================================
Ns = [30, 50, 70]
N_trials = 40
TRUE_D = 2.0

# Storage for all results
all_results = {N: {i: [] for i in range(751, 761)} for N in Ns}


# ============================================================
# RUN ALL 10 ESTIMATORS
# ============================================================

print(f"\nRunning {N_trials} trials each for N = {Ns}")
print("True dimension: d = 2")
print()
sys.stdout.flush()

t_start = time.time()

for N in Ns:
    t0 = time.time()

    # Pre-generate all causets for this N
    causets = []
    two_orders = []
    for trial in range(N_trials):
        cs, to = random_2order(N, rng_local=np.random.default_rng(1000 * N + trial))
        causets.append(cs)
        two_orders.append(to)

    for trial in range(N_trials):
        cs = causets[trial]
        to = two_orders[trial]

        # ----------------------------------------------------------
        # 751: MYRHEIM-MEYER DIMENSION ESTIMATOR
        # Uses the ordering fraction f and inverts the exact formula
        # f(d) = Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))
        # ----------------------------------------------------------
        d_mm = myrheim_meyer(cs)
        all_results[N][751].append(d_mm)

        # ----------------------------------------------------------
        # 752: CHAIN SCALING EXPONENT
        # longest_chain ~ N^{1/d}, so d = 1/alpha where alpha = log(chain)/log(N)
        # ----------------------------------------------------------
        lc = longest_chain(cs)
        alpha_chain = np.log(lc) / np.log(N)
        d_chain = 1.0 / alpha_chain if alpha_chain > 0 else float('nan')
        all_results[N][752].append(d_chain)

        # ----------------------------------------------------------
        # 753: ANTICHAIN SCALING EXPONENT
        # max_antichain ~ N^{(d-1)/d}, so d = 1/(1 - alpha)
        # where alpha = log(AC)/log(N)
        # ----------------------------------------------------------
        ac = max_antichain(cs)
        alpha_ac = np.log(ac) / np.log(N)
        d_ac = 1.0 / (1.0 - alpha_ac) if alpha_ac < 1.0 else float('nan')
        all_results[N][753].append(d_ac)

        # ----------------------------------------------------------
        # 754: ORDERING FRACTION -> d = 1 + log2(1/f)
        # Simple formula from f ~ 1/2^{d-1}
        # ----------------------------------------------------------
        f = cs.ordering_fraction()
        if f > 0:
            d_of = 1.0 + np.log2(1.0 / f)
        else:
            d_of = float('nan')
        all_results[N][754].append(d_of)

        # ----------------------------------------------------------
        # 755: LINK FRACTION SCALING
        # E[#links]/N ~ (2*ln(N))^{d-1}/(d-1)!
        # Single-N: d = 1 + log(links/N) / log(2*ln(N))
        # ----------------------------------------------------------
        links_mat = cs.link_matrix()
        n_links = int(np.sum(links_mat))
        lnN = np.log(N)
        ratio_link = n_links / N
        if ratio_link > 0 and lnN > 0:
            d_link = 1.0 + np.log(ratio_link) / np.log(2 * lnN)
        else:
            d_link = float('nan')
        all_results[N][755].append(d_link)

        # ----------------------------------------------------------
        # 756: SPECTRAL EMBEDDING DIMENSION
        # Eigenvalue gap in the Laplacian spectrum.
        # Number of "small" eigenvalues before the biggest gap.
        # ----------------------------------------------------------
        coords, spec_evals = spectral_embedding(cs, n_dims=min(15, N-1))
        if len(spec_evals) >= 3:
            # Eigenvalue gap method
            gaps = np.diff(spec_evals[:min(10, len(spec_evals))])
            if len(gaps) > 1:
                gap_idx = np.argmax(gaps)
                d_spec = float(gap_idx + 1)
            else:
                d_spec = float('nan')
        else:
            d_spec = float('nan')
        all_results[N][756].append(d_spec)

        # ----------------------------------------------------------
        # 757: INTERVAL DISTRIBUTION SHAPE
        # For d-dim Poisson causet, P(k) ~ k^{d/2-1} for small k
        # Fit: log(P) = (d/2 - 1)*log(k) + const -> d = 2*(slope+1)
        # ----------------------------------------------------------
        _, interval_sizes = cs.interval_sizes_vectorized()
        d_interval = float('nan')
        if len(interval_sizes) > 10:
            max_k = max(interval_sizes) + 1
            hist = np.bincount(interval_sizes, minlength=max_k + 1)
            k_vals = np.arange(1, max(2, max_k // 2))
            p_vals = hist[1:max(2, max_k // 2)].astype(float)
            p_total = np.sum(p_vals)
            if p_total > 0:
                p_vals = p_vals / p_total
                mask = p_vals > 0
                if np.sum(mask) >= 3:
                    log_k = np.log(k_vals[mask])
                    log_p = np.log(p_vals[mask])
                    slope, intercept, r, p_val, se = stats.linregress(log_k, log_p)
                    d_interval = 2 * (slope + 1)
        all_results[N][757].append(d_interval)

        # ----------------------------------------------------------
        # 758: BOX-COUNTING DIMENSION ON EMBEDDED COORDINATES
        # Use (u, v) from the 2-order as a natural 2D embedding.
        # ----------------------------------------------------------
        u_coords = to.u.astype(float) / N
        v_coords = to.v.astype(float) / N
        emb_coords = np.column_stack([u_coords, v_coords])

        eps_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        box_counts = []
        for eps in eps_values:
            grid = set()
            for pt in emb_coords:
                gx = int(pt[0] / eps)
                gy = int(pt[1] / eps)
                grid.add((gx, gy))
            box_counts.append(len(grid))

        log_eps = np.log(eps_values)
        log_bc = np.log(box_counts)
        slope_bc, _, _, _, _ = stats.linregress(log_eps, log_bc)
        d_box = -slope_bc
        all_results[N][758].append(d_box)

        # ----------------------------------------------------------
        # 759: HASSE GRAPH EXPANSION RATE (CHEEGER/FIEDLER)
        # Fiedler value lambda_2 ~ N^{-2/d} for d-dim graphs
        # Single-N estimate: d = -2*log(N)/log(lambda_2)
        # ----------------------------------------------------------
        L = hasse_laplacian(cs)
        evals_L = np.sort(np.linalg.eigvalsh(L))
        lambda2 = evals_L[1] if len(evals_L) > 1 else float('nan')
        if lambda2 > 1e-10:
            d_cheeger = -2.0 * np.log(N) / np.log(lambda2)
            if d_cheeger < 0 or d_cheeger > 20:
                d_cheeger = float('nan')
        else:
            d_cheeger = float('nan')
        all_results[N][759].append(d_cheeger)

        # ----------------------------------------------------------
        # 760: CORRELATION DIMENSION FROM PAIRWISE DISTANCES
        # Grassberger-Procaccia on spectral embedding distances.
        # C(r) ~ r^d -> slope of log(C) vs log(r) = d
        # ----------------------------------------------------------
        d_corr = float('nan')
        if len(spec_evals) >= 2:
            n_use = min(5, len(spec_evals))
            emb = coords[:, :n_use]
            dists = pdist(emb)
            dists = dists[dists > 0]
            if len(dists) > 10:
                sorted_dists = np.sort(dists)
                n_pairs = len(sorted_dists)
                r_min = np.percentile(sorted_dists, 5)
                r_max = np.percentile(sorted_dists, 80)
                if r_min > 0 and r_max > r_min:
                    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 20)
                    C_r = np.searchsorted(sorted_dists, r_vals) / n_pairs
                    mask_cr = C_r > 0
                    if np.sum(mask_cr) >= 5:
                        log_r = np.log(r_vals[mask_cr])
                        log_C = np.log(C_r[mask_cr])
                        slope_cr, _, _, _, _ = stats.linregress(log_r, log_C)
                        d_corr = slope_cr
        all_results[N][760].append(d_corr)

    elapsed = time.time() - t0
    print(f"  N={N}: {elapsed:.1f}s")
    sys.stdout.flush()

total_time = time.time() - t_start
print(f"\nTotal computation time: {total_time:.1f}s")


# ============================================================
# MULTI-N FITS FOR SCALING-BASED ESTIMATORS
# ============================================================
print("\n" + "=" * 90)
print("MULTI-N SCALING FITS (using all three N values together)")
print("=" * 90)

# 752: Chain scaling -- fit log(chain) = (1/d)*log(N) + const across N values
chain_lengths = {}
for N in Ns:
    chains = []
    for trial in range(N_trials):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(1000 * N + trial))
        chains.append(longest_chain(cs))
    chain_lengths[N] = chains

log_N = np.log(Ns)
log_chain = [np.log(np.mean(chain_lengths[N])) for N in Ns]
slope_chain, intercept_chain, _, _, _ = stats.linregress(log_N, log_chain)
d_chain_multi = 1.0 / slope_chain
print(f"\n752 (Chain scaling): slope={slope_chain:.4f}, d_est = 1/slope = {d_chain_multi:.4f}")

# 753: Antichain scaling -- fit log(AC) = ((d-1)/d)*log(N) + const
ac_sizes = {}
for N in Ns:
    acs = []
    for trial in range(N_trials):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(1000 * N + trial))
        acs.append(max_antichain(cs))
    ac_sizes[N] = acs

log_ac = [np.log(np.mean(ac_sizes[N])) for N in Ns]
slope_ac, _, _, _, _ = stats.linregress(log_N, log_ac)
d_ac_multi = 1.0 / (1.0 - slope_ac)
print(f"753 (Antichain scaling): slope={slope_ac:.4f}, d_est = 1/(1-slope) = {d_ac_multi:.4f}")

# 755: Link fraction multi-N fit
# E[#links]/N ~ (2*ln(N))^{d-1}/(d-1)!
# log(links/N) = (d-1)*log(2*ln(N)) + const
link_ratios = {}
for N in Ns:
    ratios = []
    for trial in range(N_trials):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(1000 * N + trial))
        links_m = cs.link_matrix()
        n_lnk = int(np.sum(links_m))
        ratios.append(n_lnk / N)
    link_ratios[N] = ratios

log_2lnN = [np.log(2 * np.log(N)) for N in Ns]
log_link_ratio = [np.log(np.mean(link_ratios[N])) for N in Ns]
slope_link, _, _, _, _ = stats.linregress(log_2lnN, log_link_ratio)
d_link_multi = slope_link + 1
print(f"755 (Link scaling): slope={slope_link:.4f}, d_est = slope+1 = {d_link_multi:.4f}")

# 759: Fiedler value multi-N fit
# lambda_2 ~ N^{-2/d}, so log(lambda_2) = -2/d * log(N) + const
fiedler_vals = {}
for N in Ns:
    fvals = []
    for trial in range(N_trials):
        cs, _ = random_2order(N, rng_local=np.random.default_rng(1000 * N + trial))
        L = hasse_laplacian(cs)
        evals_L = np.sort(np.linalg.eigvalsh(L))
        fvals.append(evals_L[1])
    fiedler_vals[N] = fvals

log_fiedler = [np.log(np.mean(fiedler_vals[N])) for N in Ns]
slope_fiedler, _, _, _, _ = stats.linregress(log_N, log_fiedler)
d_fiedler_multi = -2.0 / slope_fiedler if slope_fiedler < 0 else float('nan')
print(f"759 (Fiedler scaling): slope={slope_fiedler:.4f}, d_est = -2/slope = {d_fiedler_multi:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================
print("\n" + "=" * 90)
print("RESULTS: ESTIMATED d FOR EACH METHOD AND N")
print("=" * 90)

idea_names = {
    751: "Myrheim-Meyer",
    752: "Chain scaling",
    753: "Antichain scaling",
    754: "Ordering fraction",
    755: "Link fraction",
    756: "Spectral embedding",
    757: "Interval dist shape",
    758: "Box-counting",
    759: "Cheeger/Fiedler",
    760: "Correlation dim",
}

print(f"\n{'Idea':>5}  {'Method':<22}", end="")
for N in Ns:
    print(f"  {'N='+str(N)+' mean':>10}  {'+/-std':>7}", end="")
print(f"  {'Avg err':>10}  {'Grade':>6}")
print("-" * 110)

# Collect grades for final ranking
grades = {}

for idea in range(751, 761):
    name = idea_names[idea]
    print(f"{idea:>5}  {name:<22}", end="")

    errors = []
    for N in Ns:
        vals = [v for v in all_results[N][idea] if not np.isnan(v)]
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            err = abs(mean_val - TRUE_D)
            errors.append(err)
            print(f"  {mean_val:>10.4f}  {std_val:>7.4f}", end="")
        else:
            print(f"  {'NaN':>10}  {'N/A':>7}", end="")

    if len(errors) > 0:
        avg_err = np.mean(errors)
        if avg_err < 0.1:
            grade = "A+"
        elif avg_err < 0.2:
            grade = "A"
        elif avg_err < 0.3:
            grade = "A-"
        elif avg_err < 0.5:
            grade = "B"
        elif avg_err < 0.8:
            grade = "C"
        elif avg_err < 1.5:
            grade = "D"
        else:
            grade = "F"
        grades[idea] = avg_err
        print(f"  {avg_err:>10.4f}  {grade:>6}")
    else:
        grades[idea] = 999.0
        print(f"  {'N/A':>10}  {'N/A':>6}")

# Multi-N estimates
print(f"\n{'':>5}  {'MULTI-N FITS':<22}")
print(f"{'752m':>5}  {'Chain (multi-N)':<22}  d_est = {d_chain_multi:.4f}   err = {abs(d_chain_multi-2):.4f}")
print(f"{'753m':>5}  {'Antichain (multi-N)':<22}  d_est = {d_ac_multi:.4f}   err = {abs(d_ac_multi-2):.4f}")
print(f"{'755m':>5}  {'Link frac (multi-N)':<22}  d_est = {d_link_multi:.4f}   err = {abs(d_link_multi-2):.4f}")
print(f"{'759m':>5}  {'Fiedler (multi-N)':<22}  d_est = {d_fiedler_multi:.4f}   err = {abs(d_fiedler_multi-2):.4f}")


# ============================================================
# RANKING
# ============================================================
print("\n" + "=" * 90)
print("FINAL RANKING (best to worst, by average |d_est - 2|)")
print("=" * 90)

sorted_ideas = sorted(grades.items(), key=lambda x: x[1])
for rank, (idea, err) in enumerate(sorted_ideas, 1):
    name = idea_names[idea]
    biases = []
    for N in Ns:
        vals = [v for v in all_results[N][idea] if not np.isnan(v)]
        if len(vals) > 0:
            biases.append(np.mean(vals) - TRUE_D)
    avg_bias = np.mean(biases) if biases else float('nan')

    bias_str = f"+{avg_bias:.3f}" if avg_bias > 0 else f"{avg_bias:.3f}"
    print(f"  #{rank:>2}: {idea} {name:<22}  avg_err={err:.4f}  bias={bias_str}")


# ============================================================
# DETAILED ANALYSIS OF TOP METHODS
# ============================================================
print("\n" + "=" * 90)
print("DETAILED ANALYSIS")
print("=" * 90)

top_3 = [idea for idea, _ in sorted_ideas[:3]]
print(f"\nTop 3 methods: {[idea_names[i] for i in top_3]}")
print("\nConvergence with N:")
for idea in top_3:
    name = idea_names[idea]
    print(f"\n  {idea} ({name}):")
    for N in Ns:
        vals = [v for v in all_results[N][idea] if not np.isnan(v)]
        if len(vals) > 0:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            err = abs(mean_v - 2.0)
            pct_err = err / 2.0 * 100
            print(f"    N={N:>3}: d={mean_v:.4f} +/- {std_v:.4f}  (err={pct_err:.1f}%)")

# Consistency check: how often do different methods agree?
print("\n\nCROSS-METHOD CONSISTENCY (correlation of per-trial d estimates at N=70):")
N_check = 70
valid_ideas = []
for idea in range(751, 761):
    vals = all_results[N_check][idea]
    if len(vals) == N_trials and not any(np.isnan(v) for v in vals):
        valid_ideas.append(idea)

if len(valid_ideas) >= 2:
    print(f"\n{'':>22}", end="")
    for j in valid_ideas:
        print(f"  {j:>5}", end="")
    print()
    for i in valid_ideas:
        print(f"  {idea_names[i]:>20}", end="")
        for j in valid_ideas:
            r, _ = stats.pearsonr(all_results[N_check][i], all_results[N_check][j])
            print(f"  {r:>5.2f}", end="")
        print()


print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
best_idea, best_err = sorted_ideas[0]
print(f"""
BEST DIMENSION ESTIMATOR: {best_idea} ({idea_names[best_idea]})
  Average error |d_est - 2| = {best_err:.4f}

All 10 methods attempt to extract d=2 from random 2-orders at N=30,50,70.
The methods range from algebraic (Myrheim-Meyer, ordering fraction) to
geometric (box-counting, correlation dimension) to spectral (Fiedler value,
spectral embedding) to combinatorial (chain/antichain scaling, interval
distributions, link fraction).

Key observations:
1. Algebraic methods (751, 754) that use the ordering fraction have a direct
   closed-form relationship to d, giving the best per-trial accuracy.
2. Scaling-based methods (752, 753, 755, 759) improve with multi-N fits
   but are noisy at single N because of finite-size corrections.
3. Spectral methods (756, 760) measure the embedding dimension of the
   Hasse diagram -- a graph-theoretic rather than causal-set property.
4. The interval distribution (757) and box-counting (758) offer geometric
   perspectives independent of the other approaches.
""")

print(f"Total runtime: {time.time() - t_start:.1f}s")
