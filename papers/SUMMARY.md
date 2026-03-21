# Papers Summary

## Project Overview

**400 ideas tested** across 79 experiments, ~18,000 lines of Python. 10 papers written (8 for submission). The 7.5/10 ceiling is structural at toy scale (N=30-200): density dominance, finite-size contamination, and no exact finite-N predictions from causal set theory prevent any single result from reaching 8+.

## Scoring Rubric

| Score | Meaning | Example |
|---|---|---|
| 1-2 | Homework exercise, not publishable | Reproducing a textbook calculation |
| 3-4 | Minor technical contribution; publishable with significant revisions | New observable for a known effect at toy scale |
| **5** | **Solid incremental result; publishable as-is in a specialist journal (e.g. CQG)** | **Novel order parameter with validation, but limited scope** |
| **6** | **Good result that fills a gap; likely accepted at a strong journal (e.g. PRD)** | **Systematic study with clear methodology, engages with prior work** |
| **7** | **Strong result that changes how people think about a specific problem** | **Theorem + reinterpretation of published results; bridges subfields** |
| 8 | Important result with impact across a subfield; top ~10% of papers | New method or prediction that multiple groups would follow up on |
| 9-10 | Breakthrough with broad impact across physics | New symmetry principle, paradigm shift |

## Completed Papers

| Paper | File | Pages | Score | Target | Key Claim |
|---|---|---|---|---|---|
| **F: Hasse Geometry** | `hasse-geometry/` | ~6 | 7.5/10 | PRD/CQG | Hasse Laplacian Fiedler value 50× larger for causets than DAGs, scales as λ₂~N^{0.318±0.012}. High algebraic connectivity but NOT a true expander family (d_max grows faster). Link fraction as perfectly monotonic BD order parameter (60% jump). Triangle-free (girth≥4), independence number α~N^0.83. Dimension-dependent but collapses at d≥4. |
| **B5: Geometry from Entanglement** | `geometry-from-entanglement/` | ~7 | 7.5/10 | PRD | **Flagship.** Spectral dimension fails (crossing theorem + random graph controls) → SJ entanglement succeeds (ln N scaling + 3.4× drop across BD transition). Monogamy as supporting evidence with partition-dependence caveat. **Caveat:** c_eff diverges at large N (3.0→4.1 from N=50→500); relative comparison between phases remains valid. |
| **C: ER=EPR** | `er-epr/` | ~8 | 7.5/10 | PRD | \|W\| ∝ connectivity^0.90, r=0.88, z=13.1, partial r=0.82. Analytic proof: (-Δ²)_ij = (4/N²)κ_ij exactly for spacelike pairs in 2-orders (Thm 2); W_ij = ½[√(-Δ²)]_ij (Thm 3); Loewner-Heinz monotonicity. **Caveat:** gap vs density-matched random DAGs closes to zero at N=500. Large-N data in paper. |
| **G: Exact Combinatorics** | `exact-combinatorics/` | ~7 | 7.5/10 | J. Comb. Theory/CQG | 13+ exact results: E[f]=1/2, Var[f]=(2N+5)/[18N(N-1)]. Antichain=2√N via Vershik-Kerov. **Corrected** master formula P(int=k\|gap=m)=2(m-k)/[m(m+1)]. E[links]=(N+1)H_N-2N. E[S_Glaser]=1 for all N≥2. E[S_BD]=2N-NH_N. Limit shape f(α). Exact Z(β) for N=4,5. NEW: f-vector E[f_k]=C(N,k+1)/(k+1)!, P(dim=2)=1-1/N!, position-dependent comparabilities, full interval distribution, interval generating function Z(q) closed form. |
| **E: CDT Comparison** | `cdt-comparison/` | ~6 | 7.5/10 | PRD | First cross-approach SJ vacuum comparison. CDT c_eff≈1 (correct), causet c_eff diverges. Mechanism: CDT has only ~4 positive iΔ eigenvalues vs causets' ~38 at N=80. CDT's c=1 is FRAGILE: 5% within-slice disorder doubles c_eff to 2.1. Thinning to 30% preserves c=1 (structure, not density). GUE shared. CDT 20× more gapped. |
| **D: Spectral Statistics** | `random-matrix/` | ~6 | 7/10 | PRD | BD transition is spectral statistics transition. Both continuum and deep-crystalline phases have GUE (⟨r⟩≈0.56). Sub-Poisson ⟨r⟩=0.12 is a TRANSITION phenomenon (near β_c), not a crystalline-phase property — pure KR has ⟨r⟩=0.54. GUE is generic to all dense antisymmetric matrices (Erdős-Yau universality). GUE confirmed to N=1000 via sparse eigsh. Number variance: GUE is short-range only. MCMC crystalline has ~5-6 layers (not 3-layer KR), growing as ~√N. |
| **A: Interval Entropy** | `interval-entropy/` | ~6 | 7/10 | CQG/PRD | Interval entropy as BD order parameter. H=2.4→0.3 across 2D transition (87% drop). Random graph control (H=2.1 for matched DAGs). 4D three-phase structure at N=30-70 with χ_S up to 43,288. Non-monotonic H(β) in 4D: rise, dip, recovery. |
| **B2: Everpresent Lambda** | `everpresent-lambda/` | ~4 | 5/10 | JCAP/PRD | α=0.03 predicts Ω_Λ=0.732±0.103 from first principles. 48% within DESI 2σ. CLASS-validated. High stochastic variance limits predictive power. |
| B1: Spectral Dimension | `spectral-dimension/` | ~4.5 | 7/10 | — | Crossing theorem (IVT). *Subsumed by B5.* |
| B3: Holographic Entanglement | `holographic-entanglement/` | ~5 | 7/10 | — | SJ vacuum properties. *Subsumed by B5.* |

## Recommended Submission Strategy

1. **F** (7.5) — Hasse diagram geometry, novel spectral graph theory approach. Most novel observable.
2. **G** (7.5) — exact combinatorics, bridges QG and combinatorics/probability communities
3. **B5** (7.5) — flagship, broadest narrative (c_eff caveat included)
4. **C** (7.5) — ER=EPR with analytic proof + large-N genericity caveat
5. **E** (7.5) — CDT comparison with mechanism explanation, first cross-approach SJ vacuum study
6. **D** (7) — spectral statistics transition, revised with transition-phenomenon correction
7. **A** (7) — interval entropy + 4D three-phase structure
8. **B2** (5) — standalone if desired, weakest paper
9. B1, B3 subsumed by B5 — don't submit separately

## Compilation

```bash
cd papers/<paper-dir>
tectonic <paper>.tex
```

All 10 papers compile successfully as of 2026-03-20.

## Key Data Points Across Papers

| Result | Experiment | Paper |
|---|---|---|
| d_s crossing is generic to all graphs (IVT proof) | exp04 | B1, B5 |
| CDT plateau grows: 17%→47% (N=100→800) | exp19-20 | B1, B5 |
| Causet d_s matches random graphs at same L/N | exp25 | A, B1, B5 |
| BD d'Alembertian also fails random graph control | exp37 | B1, B5 |
| c_eff diverges: 3.0→4.1 (N=50→500), same on sprinkled causets | exp49, exp56 | B5 |
| Interval entropy: H=2.4→0.3 across 2D transition (87% drop) | exp26, exp29 | A |
| 2D χ_max scales as ~N³ (strong first-order) | exp32 | A |
| 4D three-phase: H rises to 1.23, dips to 0.27, recovers to 0.69 | exp42 | A |
| 4D χ_S_max = 43,288 at N=70 | exp42 | A |
| Ω_Λ = 0.732 ± 0.103 after anthropic selection | exp17 | B2 |
| SJ entanglement S(N/2) ~ ln(N) (CFT-like at small N) | exp35 | B3, B5 |
| S(N/2) drops 3.4× across BD transition | exp35 | B3, B5 |
| Monogamy I₃ ≤ 0 in 97% but partition-dependent | exp36, exp40 | B5 |
| \|W\| ∝ connectivity^0.90, r=0.88, z=13.1, partial r=0.82 | exp43 | C |
| X_ij = 0 for spacelike pairs in 2-orders (proved) | exp50 | C |
| (-Δ²)_ij = (4/N²)κ_ij exactly for spacelike pairs | exp50 | C |
| W_ij = ½[√(-Δ²)]_ij for spacelike pairs (verified to 10⁻¹⁵) | exp50 | C |
| ER=EPR gap vs density-matched DAGs closes at N=500 | exp49 | C |
| GUE (⟨r⟩≈0.56) in continuum AND deep crystalline phases | exp-RMT, exp82 | D |
| Sub-Poisson ⟨r⟩=0.12 is a transition phenomenon, not crystalline property | exp82 | D |
| Pure KR has ⟨r⟩=0.54 (GUE-like), NOT sub-Poisson | exp82 | D |
| GUE generic to ALL dense antisymmetric matrices (Erdős-Yau) | exp57 | D |
| GUE confirmed at N=1000 via sparse eigsh | exp75 | D |
| Number variance: GUE is short-range only (linear Σ², not log) | exp54, exp56 | D |
| MCMC crystalline has ~5-6 layers, growing as ~√N (not 3-layer KR) | exp82 | D |
| CDT c_eff≈1, causet c_eff→∞ (different universality classes) | exp44 | E |
| CDT has ~4 positive iΔ modes vs causet ~38 (mechanism for c difference) | exp85 | E |
| 5% CDT disorder: c_eff 1.1→2.1 (fragile); 30% thinning: c stays ~1.2 | exp85 | E |
| Fiedler value 50× causets vs DAGs, λ₂~N^{0.318±0.012} (R²=0.993) | exp59, exp81 | F |
| NOT a true expander family (d_max~N^0.45 > λ₂~N^0.32) | exp81 | F |
| Link fraction: perfectly monotonic BD order parameter (60% jump) | exp66 | F |
| Hasse diagram triangle-free (girth≥4), independence α~N^0.83 | exp86 | F |
| E[f]=1/2 with Var[f]=(2N+5)/[18N(N-1)] (proved, verified exactly) | exp61, exp64 | G |
| Antichain=2√N via Vershik-Kerov, Tracy-Widom fluctuations | exp61, exp64 | G |
| Master interval formula P[k\|m]=(m-k-1)/[m(m-1)] (2-order specific) | exp72 | G |
| E[links]=(N+1)H_N-2N, link fraction~4ln(N)/N→0 | exp72 | G |
| E[S_Glaser]=1 for all N≥2 (universal benchmark) | exp80 | G |
| E[S_BD]=2N-NH_N (exact, verified) | exp80 | G |
| Limit shape f(α)=4[-(1+α)ln(α)+2α-2] | exp80 | G |
| Exact Z(β) for N=4 (19 action levels) and N=5 (87 levels) | exp61 | G |
| Master formula specific to d=2; d≥3 links 1.78-2.27× higher | exp79 | G |

## 400-Idea Search Summary

| Ideas | Rounds | Best Score | Key Theme |
|---|---|---|---|
| 1-55 | Rounds 1-9 | 7.5 | SJ vacuum exploration (c≈3, ER=EPR, quantum chaos, monogamy) |
| 56-100 | Rounds 10-12 | 7.0 | Deeper SJ (propagator decay, time asymmetry, Renyi, contour) |
| 101-150 | Rounds 13-17 | 7.5 | New observables (Fiedler, treewidth, info bottleneck, antichain thm) |
| 151-200 | Rounds 18-22 | 7.5 | Dimension encoding, BD transition, synthesis, wild cards |
| 201-250 | Rounds 23-27 | 7.5 | Path entropy, master interval formula, analytic proofs |
| 251-300 | Rounds 28-32 | 7.5 | SJ v2, physics connections, computational novelty, wild cards v2 |
| 301-400 | Rounds 33-42 | 7.5 | Large-N sparse, math proofs, Fiedler theory, KR deep dive, CDT mechanism |

**Why the ceiling holds:** At N=30-200, three mechanisms defeat every idea: (1) density dominance — most causet-vs-null differences trace to ordering fraction; (2) finite-size contamination — corrections exceed target signals; (3) no exact finite-N predictions from causal set theory to test against.

## Potential New Papers (from Exp93 evaluation)

| Paper | Score | Status | Key Data |
|---|---|---|---|
| **H: BD Transition Comprehensive Study** | 7.5 | STRONGEST CANDIDATE | 10+ observables: interval entropy (87% drop), link fraction (60% jump), Fiedler (243% jump), latent heat (extensive), nucleation (sharp), KR layers (~sqrt(N)), spectral stats (GUE both phases). No published paper has >3 observables. |
| **I: Graph-Theoretic Dimension** | 7.5 | FEASIBLE | 5 independent estimators (path entropy, Fiedler, treewidth, ordering fraction, interval entropy) across d=2-5. Path entropy linear in d (R^2>0.97). Gap at d>=4. |
| **J: SJ Vacuum Reproduces Physics** | 7.0 | FEASIBLE | Casimir 1/d (robust, CV=0.134), Newton ln(r) (wins at all N, coeff drifts), area law (S~A^0.345), conformal anomaly (<T>~R). Qualitative agreement, quantitative coefficients off. |
| **K: 400 Experiments Meta-Paper** | 6.5 | RISKY | Lessons learned + recommendations. Needs honest framing. |
