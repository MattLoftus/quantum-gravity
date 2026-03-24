# Ideas Log — 850 Computational Experiments in Causal Set Quantum Gravity

Definitive searchable index of all 850 ideas tested in this research programme, spanning experiments exp01 through exp133. Each idea is scored 1-10 per the project rubric (see CLAUDE.md). Scores marked (?) are estimated from context where exact scores were not recorded.

## Summary Statistics
- Total ideas: 850
- Ideas scoring 8+ (honest): 0 — the 7.5 ceiling held across all 850 ideas
- Ideas scoring 7+: ~80 (by honest assessment)
- Best single result: Paper E Kronecker product theorem (8.0 at the paper level, not any individual idea)
- Mean score (programme): ~5.5
- Note: Individual agent scores shown below are often inflated (agents scored their own work 8-9.5). Honest re-scoring consistently places all individual ideas at ≤7.5. The 8.0 paper score for Paper E comes from the *combination* of multiple 7.5-level results (Kronecker theorem + exact eigenvalues + Wightman factorization + fragility), not from any single idea.
- Categories: SJ_VACUUM (~120), GEOMETRY (~180), GRAPH (~100), RMT (~40), ANALYTIC (~80), CDT (~50), COSMOLOGY (~30), PHYSICS (~60), INFO (~50), COMPUTE (~40), DIMENSION (~40), TRANSITION (~60), CREATIVE (~100)

## Category Key
- **SJ_VACUUM**: Sorkin-Johnston vacuum, entanglement entropy, Wightman function
- **GEOMETRY**: Pure causal set geometry (ordering fraction, chains, antichains, intervals)
- **GRAPH**: Graph theory on the Hasse diagram (Fiedler, treewidth, expansion, diameter)
- **RMT**: Random matrix theory, spectral statistics, level spacing
- **ANALYTIC**: Exact theorems, proofs, closed-form formulas
- **CDT**: Cross-approach comparison with CDT
- **COSMOLOGY**: Everpresent Lambda, dark energy
- **PHYSICS**: Connections to known physics (Newton's law, Casimir, Bekenstein)
- **INFO**: Information theory (complexity, compressibility, channel capacity)
- **COMPUTE**: Computational methods (sparse eigensolvers, large-N, algorithms)
- **DIMENSION**: Dimension estimation and encoding
- **TRANSITION**: BD phase transition observables
- **CREATIVE**: Wild card, cross-disciplinary, chaotic methodology ideas

---

## Pre-numbered Experiments (exp01-exp45): Foundation Building

These experiments predate the numbered idea system. Each experiment explored a specific topic.

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| — | 6.0 | DIMENSION | Myrheim-Meyer dimension estimator validation (exp01) | Works correctly; errors < 0.03 for 2D, < 0.16 for 4D |
| — | 5.0 | GEOMETRY | CSG emergent dimension via transitive percolation (exp02) | Too 1-dimensional; global MM dim ~ 1.75 even at low coupling |
| — | 5.0 | DIMENSION | Spectral dimension flow on CSG causets (exp03) | Peak d_s ~ 3.5-5.8; pass through d_s ~ 2 at intermediate scales |
| — | 7.0 | GEOMETRY | Null hypothesis test for d_s = 2 crossing (exp04) | d_s ~ 2 crossing is UNIVERSAL across all graph types — not physically meaningful alone |
| — | 6.0 | COSMOLOGY | Cosmological constant from volume fluctuations (exp05) | CSG fluctuations sub-Poisson; partly confounded by saturation |
| — | 7.0 | COSMOLOGY | Link density fluctuations and Sorkin's Lambda (exp06) | Sorkin's Lambda ~ 1/sqrt(N) prediction SURVIVES in dynamical models |
| — | 7.5 | GEOMETRY | CSG coupling constant scan (exp07) | Step k_max=1 gives MM dim ~ 4.3 — only CSG coupling near 4D |
| — | 7.0 | GEOMETRY | Deep investigation of 4D CSG candidate (exp08) | MM dim stable at 4.3 but spectral dim only 1.15 — tree-like, not manifold-like |
| — | 6.0 | COSMOLOGY | Everpresent Lambda cosmological simulation (exp09) | Lambda fluctuations produce w=-1 crossings; 14% phantom-to-quintessence |
| — | 5.0 | SJ_VACUUM | Holographic codes with magic — preliminary (exp10) | Phase transition from zero to full entanglement with magic; coupling too crude |
| — | 7.0 | SJ_VACUUM | Proper holographic codes (HaPPY + Clifford+T) (exp11) | Page curve emerges with magic; scrambling reduces bulk-boundary coupling |
| — | 7.5 | COSMOLOGY | Everpresent Lambda vs DESI — proper normalization (exp12) | Sorkin prediction confirmed: alpha=0.03 gives Omega_Lambda ~ 0.74 |
| — | 6.0 | GEOMETRY | Systematic CSG coupling scan — 274 configs (exp13) | MAJOR NEGATIVE: no coupling produces manifold-like causets at any dimension |
| — | 7.5 | COSMOLOGY | Bayesian analysis of everpresent Lambda (exp14) | MAP alpha=0.025; fits DESI better than LCDM but penalized by variance |
| — | 6.0 | SJ_VACUUM | Larger holographic codes (11-14 qubits) (exp15) | Weak positive evidence for Preskill effect in chain geometry |
| — | 6.0 | TRANSITION | Benincasa-Dowker action MCMC (exp16) | BD-weighted MCMC more manifold-like than CSG but MM dim undershoots |
| — | 7.5 | COSMOLOGY | Cosmic variance analysis — 1000 realizations (exp17) | After anthropic selection: Omega_Lambda = 0.732 +/- 0.103; obs value at 31st percentile |
| — | 5.0 | TRANSITION | Extended BD action MCMC (exp18) | Moves toward manifold-like action but doesn't converge to manifold geometry |
| — | 7.5 | CDT | 2D CDT spectral dimension comparison (exp19) | CDT peak d_s = 2.54 with plateau; causets show only transient crossing |
| — | 7.5 | CDT | CDT vs causal set systematic comparison (exp20) | CDT plateau GROWS with system size; causet plateau SHRINKS |
| — | 5.0 | TRANSITION | BD phase transition initial study (exp22-23) | Transition is first-order; spectral dimension INVARIANT across transition |
| — | 6.0 | TRANSITION | BD calibration + deep crystalline phase (exp24) | Crystalline phase reached at beta >= 0.5; normalization mismatch with Surya |
| — | 7.0 | TRANSITION | BD phase transition at correct epsilon=0.05 (exp25) | d_s jumps but trivially explained by link density — cannot distinguish phases |
| — | 8.0 | TRANSITION | Alternative observables across BD transition (exp26) | Interval entropy is best single observable; 3 pass random graph control |
| — | 6.5 | TRANSITION | Finite-size scaling of interval entropy (exp27) | H drops from ~2.5 to ~0.5-1.0 across transition at all sizes |
| — | 7.0 | TRANSITION | Calibrated BD transition (exp28) | beta_c EXACTLY matches Glaser; chi_max scales linearly with N |
| — | 8.0 | TRANSITION | Full phase diagram — definitive result (exp29) | Action and interval entropy transitions coincide at beta ~ 10.5; publishable |
| — | 7.0 | TRANSITION | 4D phase transition via 4-orders (exp31) | First 4D BD transition measurement; non-monotonic interval entropy |
| — | 7.5 | COMPUTE | Large-N parallel tempering N=50-200 (exp32) | chi_S_max ~ N^4.2; parallel tempering enables crystalline phase exploration |
| — | 6.0 | SJ_VACUUM | Holographic entanglement at BD transition (exp34) | Page-curve shape for I(A:B); continuum more connected per boundary link |
| — | 7.5 | SJ_VACUUM | SJ vacuum entanglement entropy at BD transition (exp35) | S ~ ln(N) with c_eff ~ 3; 3.4x drop at crystalline transition |
| — | 5.0 | SJ_VACUUM | SJ spectral dimension (exp38) | Causal Laplacian gives d_s ~ 1.93 at N=50 but overshoots at larger N |
| — | 6.5 | PHYSICS | Black hole entropy from causal set entanglement (exp39) | S ~ 0.58 * ln(N); volume law in 2D as expected |
| — | 5.0 | SJ_VACUUM | De Sitter static patch entropy (exp41) | dS has 4.7x less entanglement; static patch too small for horizon effects |
| — | 7.5 | TRANSITION | 4D intermediate phase study (exp42) | Three regimes confirmed in 4D; H dip stable across N; chi_S explodes |
| — | 7.5 | SJ_VACUUM | ER=EPR deep investigation (exp43) | r=0.86-0.89 stable; partial r=0.80-0.84 after controls; null z=13.1 |
| — | 7.5 | CDT | SJ vacuum on CDT deep (exp44) | CDT c_eff converges toward c=1; causets at c=2.86-3.43; different universality |
| — | 5.0 | SJ_VACUUM | Ten moonshot ideas (exp45) | No 8+ results; SSA satisfied perfectly; BW theorem FAILS on causets |

---

## Ideas 1-100: SJ Vacuum Exploration and 8+ Search

Ideas 1-45 were embedded in the pre-numbered experiments above. Ideas 46-100 were the systematic search for an 8+ paper from the SJ vacuum.

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 1 | 7.0 | SJ_VACUUM | Entanglement thermodynamics — S_ent vs S_BD first law | S_ent = 0.24 * S_BD + 3.1 with r=0.986; coefficient ~ 1/4 suggestive |
| 2 | ~5 (?) | SJ_VACUUM | Curved spacetime SJ entanglement entropy | — |
| 3 | 5.5 | SJ_VACUUM | Information scrambling time on causal sets | Power law decay |W| ~ d^{-0.43}; partial scrambling in 7-9 elements |
| 4 | 6.5 | SJ_VACUUM | RG flow via coarse-graining | All observables flow monotonically; ordering fraction scale-invariant |
| 5 | 4.0 | SJ_VACUUM | QEC properties of BD continuum phase | Trivially explained: bulk-boundary weakly coupled; need different decomposition |
| 6 | 5.5 | SJ_VACUUM | Discrete c-theorem via coarse-graining | c_eff decreases under coarse-graining but ALSO in crystalline — finite-size effect |
| 7 | 4.0 | SJ_VACUUM | Nested diamond mutual information (RT test) | I grows with inner size — volume-like, not area-like; RT needs d >= 3 |
| 8 | 6.0 | SJ_VACUUM | Vacuum phase transition in SJ eigenstates | W eigenvalue distribution changes: ~6 modes (continuum) -> ~2 (crystalline) |
| 9 | 3.0 | SJ_VACUUM | Second law for SJ entropy | Trivially true — entropy of growing region always increases |
| 10 | 4.0 | SJ_VACUUM | Entanglement dimension estimator from S(N) scaling | Cannot distinguish exponents at N=10-30; finite-size dominates |
| 11 | 5.5 | SJ_VACUUM | Massive scalar SJ vacuum | ER=EPR weakens with mass (r: 0.88 -> 0.36); Yukawa short-range |
| 12 | 7.0 | CDT | SJ vacuum on CDT — first computation | CDT c=1.55 vs causet c=3.59; different universality class |
| 13 | 3.0 | SJ_VACUUM | Lambda from SJ vacuum energy | Vacuum energy ~ N^1.29 — standard UV divergence, NOT sqrt(N) |
| 14 | 6.0 | TRANSITION | ML phase classification | 100% accuracy; top feature is link fraction — rediscovers interval entropy |
| 15 | 6.5 | SJ_VACUUM | Discrete ER=EPR | r=0.879; partial r=0.817 after controlling distance; partly generic (DAG r=0.71) |
| 16 | 5.5 | DIMENSION | Universal properties across dimensions | H less discriminating at d >= 5 (saturates near 0) |
| 17 | 7.0 | RMT | Spectral gap of Pauli-Jordan operator | Continuum: gapless (gap ~ 1/N); crystalline: gap opens 100x; gap-opening transition |
| 18 | ~5 (?) | SJ_VACUUM | Phase transition in SJ vacuum itself | — |
| 19 | 5.5 | SJ_VACUUM | Tensor network of SJ vacuum | Bond dim grows polynomially; 6 modes capture 90% of W |
| 20 | 6.0 | SJ_VACUUM | Time-space entanglement | I(T:S)=0.46 for causets vs 0.05 for random DAGs (9x stronger) |
| 21 | 6.5 | SJ_VACUUM | Curvature detection via SJ vacuum | 3.4sigma difference flat vs curved; but density effect, not curvature |
| 22 | 5.5 | SJ_VACUUM | Topology detection from SJ entanglement | Diamond vs cylinder distinguishable (t=9) but fails null test |
| 23 | ~5 (?) | SJ_VACUUM | — | — |
| 24 | ~5 (?) | SJ_VACUUM | — | — |
| 25 | ~5 (?) | SJ_VACUUM | — | — |
| 26 | 4.0 | SJ_VACUUM | Self-similarity under coarse-graining | Similarity=0.16 — NOT a conformal fixed point |
| 27 | 4.0 | SJ_VACUUM | Phase prediction from SJ entanglement | Inconclusive — all samples fell in same phase |
| 28 | ~5 (?) | SJ_VACUUM | — | — |
| 29 | ~5 (?) | SJ_VACUUM | — | — |
| 30 | 3.0 | DIMENSION | Information dimension from SJ vacuum | d_info=5-14 for true d=2-5 — WRONG; finite-size dominates |
| 31 | 6.5 | SJ_VACUUM | Entanglement entropy scaling fit | S = 0.131|bdy| + 0.332*ln|A| - 0.261 (r=0.999); mixed area/log law |
| 32 | ~5 (?) | SJ_VACUUM | — | — |
| 33 | 5.0 | SJ_VACUUM | Monotonicity of SJ vacuum | 0/200 violations; valid quantum state |
| 34 | 5.0 | SJ_VACUUM | S(beta_c) scaling | S(beta_c)/ln(N) converges toward ~1.0 — same as beta=0 |
| 35 | ~5 (?) | SJ_VACUUM | — | — |
| 36 | ~5 (?) | SJ_VACUUM | — | — |
| 37 | ~5 (?) | SJ_VACUUM | — | — |
| 38 | 6.5 | SJ_VACUUM | Contiguous vs random region entropy | Contiguous regions have LESS entropy (t=-8 to -16); causal density explains it |
| 39 | 6.0 | SJ_VACUUM | W eigenvalue restructuring at transition | w1~w2 increase; w3~w4 collapse; spectral concentration |
| 40 | 5.0 | SJ_VACUUM | Participation ratio vs dimension | Increases with d (6.1 -> 8.2) but saturates |
| 41 | ~5 (?) | SJ_VACUUM | — | — |
| 42 | 5.0 | SJ_VACUUM | S vs |A| fit quality | Near-perfect r=0.998 BUT random matrices also give r=0.993 |
| 43 | 5.0 | SJ_VACUUM | S vs PJ eigenvalue sum | S ~ 1.656 * Sum(PJ evals); 2.5% CV but likely same as #42 |
| 44 | 4.5 | SJ_VACUUM | Formula coefficients across dimensions | Not universal — b: 0.028 -> 0.324 from 2D to 4D |
| 45 | ~5 (?) | SJ_VACUUM | — | — |
| 46 | ~5 (?) | SJ_VACUUM | — | — |
| 47 | 4.0 | SJ_VACUUM | Causal/spacelike |W| ratio | Ratio ~ 1.10, stable with N; weak distinction |
| 48 | ~5 (?) | SJ_VACUUM | — | — |
| 49 | 5.5 | SJ_VACUUM | 2-orders vs sprinkled c_eff comparison | Different S values (t=4.9) — not equivalent representations |
| 50 | 4.0 | SJ_VACUUM | Minimum-entropy partition vs RT | 0.8% fewer cross-relations — consistent with RT but negligible |
| 51 | ~5 (?) | SJ_VACUUM | Lorentz invariance violation from SJ propagator | — |
| 52 | ~5 (?) | SJ_VACUUM | — | — |
| 53 | ~5 (?) | SJ_VACUUM | — | — |
| 54 | ~5 (?) | SJ_VACUUM | — | — |
| 55 | 5.0 | TRANSITION | Non-monotonic H(beta) across dimensions | Appears in ALL d >= 3; thermalization issues |
| 56 | 3.0 | SJ_VACUUM | Modular Hamiltonian / Bisognano-Wichmann | BW theorem FAILS on causets; random matrix gives r=0.996 |
| 57 | 4.0 | RMT | Spectral form factor | No clean dip-ramp-plateau; messy ramp (later FIXED in idea 595) |
| 58 | 5.0 | SJ_VACUUM | Mutual information decay law | I ~ d^{-0.75}; much slower than CFT's d^{-2} |
| 59 | 4.0 | SJ_VACUUM | Entanglement contour | Weak boundary peaking (r=0.27); marginal area law signal |
| 60 | 5.0 | RMT | Renyi entropy spectrum | S2/S1=0.57 vs CFT 0.75; all ratios deviate 20-35% |
| 61 | 3.0 | SJ_VACUUM | Reflected entropy | S_R/I=32; ratio unreasonably large |
| 62 | 5.0 | SJ_VACUUM | Topological entanglement entropy | S_topo=-0.072 (causet) vs +0.013 (random); negative = monogamy (known) |
| 63 | 5.0 | SJ_VACUUM | Entanglement temperature | beta_ent=1.373 +/- 0.011 — remarkably stable but meaning unclear |
| 64 | 2.0 | GEOMETRY | Ollivier-Ricci curvature | TOTAL FAILURE: all kappas = 0.000; Wasserstein distance collapsed |
| 65 | 3.0 | TRANSITION | Correlation length at BD transition | No divergence at beta_c; confirms first-order |
| 66 | 2.0 | PHYSICS | Deficit angle / curvature from intervals | Flat ~ curved distributions (ratios 0.92-1.07); no signal |
| 67 | 4.0 | RMT | d'Alembertian spectrum | Gap 6000x lattice Laplacian; very different from continuum |
| 68 | 5.0 | SJ_VACUUM | Conditional mutual info / Markov property | I(A:C|B)/S(A) ~ 0.05 — approximately Markov but same as random DAGs |
| 69 | 3.0 | TRANSITION | Fisher information vs beta | Monotonically increasing; no peak at beta_c |
| 70 | 4.0 | SJ_VACUUM | Residual tangle / tripartite entanglement | tau=8.0 (causet) < 15.7 (random); causets have LESS tripartite entanglement |
| 71 | 6.0 | SJ_VACUUM | Propagator decay — timelike vs spacelike | Timelike |W| ~ d^{-0.70}, spacelike ~ d^{-1.40}; distinct exponents |
| 72 | 3.0 | SJ_VACUUM | Relative entropy between phases | S_rel/N ~ 0.001; phases nearly identical at small N |
| 73 | 5.0 | RMT | Eigenstate thermalization hypothesis test | S_eigenstate/S_thermal ~ 0 — completely non-thermal; TENSION with GUE |
| 74 | 5.0 | SJ_VACUUM | Linear response theory | delta_S/S0 = 2.75*epsilon + 0.03, r=0.988; beautiful but not novel |
| 75 | 6.0 | SJ_VACUUM | Time asymmetry of W | Time: marginal (t=2.4); space: perfect symmetry (t=0.0) |
| 76 | 5.0 | SJ_VACUUM | Sprinkled vs 2-order c_eff | Sprinkled ~5% lower but both diverge with N |
| 77 | 2.0 | SJ_VACUUM | SJ weight vs volume of past | Null model has HIGHER correlation — artifact |
| 78 | 3.0 | SJ_VACUUM | Causal matrix C spectrum vs SJ spectrum | ~47% positive for both causets and random DAGs |
| 79 | 4.0 | SJ_VACUUM | W(i,j) vs causal distance | Power law |W| ~ d^{-0.61} but p=0.23 (not significant) |
| 80 | 3.0 | SJ_VACUUM | Timelike vs spacelike entanglement | Ratio S_t/S_x = 1.001 — no asymmetry (Lorentz invariance) |
| 81 | 4.0 | SJ_VACUUM | SJ vacuum fidelity under MCMC | Mean F=0.97, tau ~ 1 step — unsurprising at small N |
| 82 | 6.0 | INFO | Causal entropy (row entropy of C) | KS p<0.003 — discriminates manifold from random DAG |
| 83 | 5.0 | SJ_VACUUM | Mutual information geometry | MI ~ sep^{-0.27}; much slower than CFT's sep^{-2} |
| 84 | 3.0 | RMT | Spectral gap of SJ Hermitian | Gap ~ N^{-1.1}; no difference between causet types |
| 85 | 6.0 | GRAPH | Euler characteristic from links | chi/N: sprinkled=-1.8, DAG=-0.7; KS p=0.008 |
| 86 | 4.0 | GEOMETRY | Longest chain scaling (Ulam problem) | Exponent 0.385 (not Ulam's 0.5); no change across BD transition |
| 87 | 4.0 | GEOMETRY | Interval size distribution shape | Different from DAG (KS p=0) but expected from geometry |
| 88 | 5.0 | RMT | Eigenvalue density of iDelta | Kurtosis diverges (15 -> 56 with N); non-semicircular |
| 89 | 3.0 | TRANSITION | BD action gap scaling | Gap negative, small (~0.005), shrinks with N |
| 90 | 4.0 | DIMENSION | SJ spectral dimension via Weyl law | d_spec=1.3 (not 2.0); Weyl law doesn't apply to iDelta |
| 91 | 3.0 | SJ_VACUUM | Causal diamond entropy vs area | c_eff=6.4 but null gives 6.9 — KILLED by null model |
| 92 | 5.0 | RMT | Number variance Sigma^2(L) | Linear growth (not log) — GUE universality is SHORT-RANGE ONLY |
| 93 | 4.0 | GEOMETRY | Antichain width at transition | Width/sqrt(N) ~ 1.09; no jump; noisy greedy algorithm |
| 94 | 5.0 | RMT | Mass gap from SJ propagator | m_eff ~ 2.3 (real); null gives 0; but m*sqrt(N) not constant |
| 95 | 5.0 | RMT | Spectral zeta function | Ratios grow near s=1 but poles at same location; descriptive only |
| 96 | 5.0 | RMT | C+C^T spectrum vs RMT universality | Kurtosis and TW edge differ from null but just encode ordering fraction |
| 97 | 5.5 | ANALYTIC | Lee-Yang zeros of interval Z(q) | Zeros do NOT cluster on unit circle; no Lee-Yang structure |
| 98 | 4.5 | INFO | Dynamical entropy growth during building | Growth exponents indistinguishable (causet 0.52 vs null 0.49, p=0.24) |
| 99 | 7.0 | GEOMETRY | Longest antichain / sqrt(N) universal constant | AC/sqrt(N) converges to ~1.7 (CV=4.5%); exponent=0.58 vs expected 0.50 |
| 100 | 6.0 | GRAPH | Link Laplacian Fiedler value as mass gap | Fiedler GROWS with N (~ N^0.51); causets are good expanders |

---

## Ideas 101-200: New Observables and Dimension Encoding

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 101 | 5.5 | GEOMETRY | Persistent homology — Betti numbers of order complex | Betti_1 = 0 for all causets (trivial topology); descriptive |
| 102 | 5.0 | INFO | Causal set as quantum channel — channel capacity | Capacity grows sub-linearly; descriptive |
| 103 | 6.0 | RMT | Full eigenvalue spacing distribution shape | Non-Wigner shape; excess small spacings from near-zero modes |
| 104 | 4.0 | TRANSITION | Specific heat exponent at BD transition | Normalization trap: C_V decreases due to beta_c^2 ~ 1/N^2 |
| 105 | 7.0 | GRAPH | Treewidth of comparability graph | tw/N ~ 0.27; linear treewidth; structural complexity measure |
| 106 | 4.0 | ANALYTIC | Interval size distribution mod p | No number-theoretic structure detected |
| 107 | 6.5 | RMT | Entanglement spectrum level statistics | Non-standard; initial near-integer ratios were SINGLE-REALIZATION FLUKE |
| 108 | 5.0 | GRAPH | Ihara zeta function of link graph | Functional determinant computable but no clean physical interpretation |
| 109 | 3.0 | SJ_VACUUM | Mutual information decay across causal diamonds | Decay too slow for useful comparison |
| 110 | 5.5 | RMT | Local spectral measure — eigenvalue density vs position | Position-dependent density detected but weak signal |
| 111 | 5.5 | TRANSITION | BD action fluctuations across 2-order ensemble | Var[S] ~ N; approximately Gaussian |
| 112 | 6.0 | TRANSITION | MCMC autocorrelation and dynamical critical exponent | Autocorrelation time increases near beta_c |
| 113 | 6.5 | GRAPH | Hasse diagram graph invariants | Multiple invariants computed; triangle-free confirmed |
| 114 | 6.5 | DIMENSION | Cross-dimensional scaling d=2,3,4,5 | Observable scaling exponents match d-dependent predictions |
| 115 | 5.5 | INFO | Information-theoretic properties of C | Shannon entropy of rows; distinguishes causet types |
| 116 | 5.0 | GRAPH | Percolation on the link graph | Percolation threshold dimension-dependent |
| 117 | 5.0 | GEOMETRY | Approximate symmetries — causal neighborhood classes | Few symmetry classes; rigid structure |
| 118 | 5.0 | INFO | Causal matrix as quantum channel — capacity scaling | — |
| 119 | 8.0 | SJ_VACUUM | SJ vacuum on MCMC continuum phase | c_eff STILL diverges; intrinsic to SJ construction on 2-orders |
| 120 | ~5 (?) | TRANSITION | — | — |
| 121 | 6.5 | DIMENSION | Chain/antichain ratio as dimension estimator | h/w ~ N^{(2-d)/d}; works but known result |
| 122 | 5.5 | GEOMETRY | Interval distribution moments and N-scaling | Variance ~ N; higher moments characterize shape |
| 123 | 5.5 | DIMENSION | Order dimension approximation for d-dim sprinklings | Dimension correctly estimated |
| 124 | 5.5 | TRANSITION | Causal set entropy from BD partition function | Computable but noisy |
| 125 | 5.5 | TRANSITION | Width of BD transition vs N | Width narrows with N |
| 126 | 6.5 | TRANSITION | Latent heat — action jump at beta_c vs N | Jump scales with N; consistent with first-order |
| 127 | 5.0 | TRANSITION | Metastability / hysteresis at BD transition | Weak hysteresis detected |
| 128 | 5.5 | TRANSITION | Correlation between action and geometric observables | BD action correlates with ordering fraction and link count |
| 129 | 4.5 | GEOMETRY | Complement causal set properties | — |
| 130 | 5.5 | TRANSITION | Finite-size scaling exponents at BD transition | Consistent with first-order |
| 131 | 7.0 | ANALYTIC | Exact ordering fraction = 1/3 for 2-orders | PROVED: E[f] = 1/3; verified by enumeration |
| 132 | 7.0 | ANALYTIC | Signum matrix eigenvalues — exact cotangent formula | DERIVED: eigenvalues are cot(pi*k/(2N+1)) |
| 133 | 6.0 | ANALYTIC | Exact partition function Z(beta) for N=4,5 | Complete action spectra: N=4 has 19 levels, N=5 has 87 |
| 134 | 6.0 | RMT | Spectral gap * N bound for Pauli-Jordan | Chain: gap*N -> pi/2 (PROVED); 2-order: gap*N ~ 5.5 |
| 135 | 6.5 | ANALYTIC | Expected number of k-intervals for random 2-order | E[N_k] formula derived; links most common |
| 136 | 5.5 | ANALYTIC | Mean BD action at beta=0 | E[S_2D] formula derived; large-N limit trivial |
| 137 | 5.0 | ANALYTIC | BD action distribution Gaussianity | Approximately Gaussian by CLT; overdispersion grows as O(N) |
| 138 | 5.5 | TRANSITION | Interval entropy monotonicity with beta | — |
| 139 | 6.5 | ANALYTIC | Longest antichain scaling for 2-orders (Erdos-Szekeres) | — |
| 140 | 5.5 | ANALYTIC | MI I(u;v) between permutations under BD | — |
| 141 | 5.5 | CREATIVE | Causal set as quantum computer — computational power | Measured circuit complexity; moderate scaling |
| 142 | 4.0 | CREATIVE | Music of the causal set — power spectrum of eigenvalues | Pink noise spectrum; not novel |
| 143 | 3.5 | CREATIVE | Game theory on causal sets | Nash equilibrium analysis; weak results |
| 144 | 6.0 | CREATIVE | Cellular automaton on the causal set | Complex dynamics; edge-of-chaos |
| 145 | 5.0 | CREATIVE | Causal set as neural network | Activation patterns dimension-dependent |
| 146 | 5.0 | ANALYTIC | Zeta function of the causal set | Riemann zeta-like structure; descriptive |
| 147 | 6.5 | CREATIVE | Quantum walk on the causal set | Spreading rate dimension-dependent |
| 148 | 4.5 | CREATIVE | — | — |
| 149 | 5.5 | TRANSITION | MCMC mixing time and critical slowing down | Mixing time diverges near transition |
| 150 | 7.0 | INFO | Information bottleneck — compressibility | Causets 26% more compressible than random DAGs |
| 151 | 8.0 | DIMENSION | Fiedler value vs dimension d | Perfect anticorrelation (r=-1.0); d=2: 0.78, d=5: 0.00 |
| 152 | 5.0 | DIMENSION | Fiedler scaling with N at each d | Exponents unstable for d >= 4 |
| 153 | 6.0 | DIMENSION | Treewidth/N vs dimension d | tw/N decreases with d; doesn't match (d-1)/d |
| 154 | 6.5 | DIMENSION | SVD compressibility vs d | Alpha increases with d (0.78 -> 1.03); monotonic |
| 155 | 8.5 | DIMENSION | Geometric fingerprint classifier | Cohen's d > 2.9 for ALL adjacent dimension pairs |
| 156 | 6.0 | RMT | Spectral gap vs d | Gap decreases with d |
| 157 | 7.0 | DIMENSION | Link density scaling links/N ~ N^beta | beta matches (d-1)/d for d >= 3 |
| 158 | 6.5 | TRANSITION | Fiedler across BD transition | Jumps from 0.62 to ~1.5-2.2 |
| 159 | 8.0 | DIMENSION | Chain height scaling h ~ N^{1/d} | R^2 > 0.99 across d=2-5 |
| 160 | 8.5 | DIMENSION | Antichain width scaling w ~ N^{(d-1)/d} | R^2 > 0.99; dual exponents sum to 1 |
| 161 | 6.0 | GRAPH | Fiedler value of Hasse diagram | lambda_2 ~ N^0.34; partial Cheeger proof |
| 162 | 5.0 | GRAPH | Treewidth scaling proof | tw/N ~ 0.27; proof sketch only |
| 163 | 5.0 | INFO | Compressibility exponent | k_90 ~ N^0.30; conjecture alpha=3/4 |
| 164 | 7.0 | ANALYTIC | Exact link formula | P(link) via double sum over gap sizes |
| 165 | 6.0 | RMT | Spectral gap * N bound | Chain gap*N -> pi/2 (PROVED) |
| 166 | 8.0 | ANALYTIC | Tracy-Widom antichain fluctuations — THEOREM | (AC-2sqrt(N))/N^{1/6} -> TW_2; verified to N=2000 |
| 167 | 5.0 | ANALYTIC | Exact expected BD action | Large-N limit trivial |
| 168 | 5.0 | ANALYTIC | Interval distribution convergence | NOT Binomial; CLT to Gaussian |
| 169 | 7.5 | ANALYTIC | Ordering fraction variance — PROVED EXACTLY | Var[f] = (2N+5)/(18N(N-1)) |
| 170 | 5.0 | ANALYTIC | Exact free energy F(beta) for N=4,5 | Cv peak at ~3x beta_c |
| 171 | ~5 (?) | DIMENSION | — | — |
| 172 | 6.0 | DIMENSION | Treewidth scaling tw/N^{(d-1)/d} | Approximate match |
| 173 | ~5 (?) | DIMENSION | — | — |
| 174 | 7.0 | DIMENSION | Chain scaling L_max / N^{1/d} across d | Confirmed dual exponent |
| 175 | 7.0 | DIMENSION | Antichain scaling A_max / N^{(d-1)/d} across d | Confirmed dual exponent |
| 176 | 6.0 | DIMENSION | Ordering fraction f(d) characterization | f ~ 1/d! for sprinkled |
| 177-180 | ~5 (?) | DIMENSION | Additional dimension tests | — |
| 181 | 6.5 | TRANSITION | Fiedler value across BD phase transition | Jumps at transition |
| 182 | 5.5 | GRAPH | Treewidth via greedy min-degree | Descriptive |
| 183 | 5.0 | RMT | Significant singular values of C | Dimension-dependent but noisy |
| 184 | 6.0 | GRAPH | Diameter of Hasse across BD | 21% drop; ~ N^0.155 |
| 185 | 5.0 | CREATIVE | Quantum walk spreading on Hasse DAG | — |
| 186 | 6.0 | GEOMETRY | Links / total relations across BD | Link fraction doubles in crystalline |
| 187 | 4.0 | GRAPH | Clique number of Hasse graph | — |
| 188 | 5.5 | INFO | Mean row entropy of causal matrix | Distinguishes phases |
| 189 | 5.5 | GEOMETRY | Antichain / sqrt(N) across BD | — |
| 190 | 6.5 | DIMENSION | PCA geometric fingerprint | PCA separates dimensions |
| 191 | 6.5 | DIMENSION | Geometric dimension from multiple observables | Combined estimator works |
| 192 | 6.0 | INFO | Phase classification from information theory | — |
| 193 | 6.0 | RMT | Random matrix + graph theory synthesis | — |
| 194 | 6.0 | TRANSITION | Universality class of BD transition | Exponent ~0.047 |
| 195 | ~5 (?) | TRANSITION | — | — |
| 196 | 5.5 | SJ_VACUUM | SJ vacuum vs pure geometry meta-analysis | Pure geometry scored higher |
| 197 | 5.5 | TRANSITION | FSS collapse for interval entropy | Approximate collapse |
| 198 | 5.5 | ANALYTIC | Lee-Yang zeros from exact Z(beta) | Zeros in complex beta-plane |
| 199 | ~5 (?) | CREATIVE | — | — |
| 200 | 6.0 | CREATIVE | Meta-analysis — best paper from 200 ideas | Geometric fingerprint best |

---

## Ideas 201-300: Pure Geometry, Physics, and Wild Cards

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 201 | 6.5 | GEOMETRY | Graph homomorphism density — subposet patterns | Dimension-dependent |
| 202 | 6.0 | GEOMETRY | Width profile — antichain at each height | Encodes dimension |
| 203 | 5.5 | GEOMETRY | Ramsey-type bounds for 2-orders | Tradeoff quantified |
| 204 | 6.0 | GEOMETRY | Forbidden subposet frequencies (N, Z-poset) | Varies with d |
| 205 | 4.5 | GEOMETRY | Automorphism group size | Rigid (trivial automorphisms) |
| 206 | 6.0 | ANALYTIC | Order polytope volume via linear extensions | Computable for small N |
| 207 | 6.0 | GEOMETRY | Chain decomposition (Dilworth) | Structural |
| 208 | 6.5 | GRAPH | Path enumeration in Hasse diagram | Path entropy dimension-dependent |
| 209 | 5.5 | ANALYTIC | Mobius function statistics | Hall's theorem verified |
| 210 | 5.5 | GEOMETRY | Convex subposet density | — |
| 211 | 7.0 | TRANSITION | Latent heat scaling Delta_S ~ N^0.73 | Consistent with first-order |
| 212 | 5.5 | TRANSITION | Hysteresis loop at BD transition | Weak (~1% of action) |
| 213 | 6.0 | TRANSITION | Action histogram bimodality | Valley/peak ~ 0.67 |
| 214 | 5.0 | TRANSITION | Binder cumulant | Too few samples |
| 215 | 4.5 | TRANSITION | Metastable lifetime | No N-dependence detected |
| 216 | 6.5 | TRANSITION | Nucleation dynamics | SHARP jump in first ~50 steps |
| 217 | 7.5 | TRANSITION | KR phase structure | WIDE NOT LONG; ~6 layers, doubled link fraction |
| 218 | 5.5 | TRANSITION | Specific heat peak scaling | DECREASES with N — normalization trap |
| 219 | 5.0 | TRANSITION | Link fraction susceptibility | Same normalization issue |
| 220 | 6.5 | TRANSITION | Tricritical search in epsilon | No tricritical in [0.05, 0.25] |
| 221 | 4.0 | DIMENSION | Interval distribution shape vs d | Non-monotonic; mean tracks f |
| 222 | 7.0 | DIMENSION | Link-to-relation ratio L/R vs d | Monotonic 0.15->0.98; works at d>=4 |
| 223 | 4.0 | DIMENSION | BD action density S/N vs d | Not meaningful across d |
| 224 | 5.0 | RMT | Spectral gap of C+C^T vs d | No dimension info |
| 225 | 7.5 | DIMENSION | Chain/antichain ratio h/w scaling | d=3: -0.346 vs theory -0.333 |
| 226 | 5.5 | GEOMETRY | Maximal/minimal fractions vs d | Obvious |
| 227 | 6.5 | GEOMETRY | Interval entropy rate H/ln(N) vs d | Monotonic decrease |
| 228 | 6.0 | DIMENSION | Layer width distribution vs d | n_layers ~ N^{1/d} |
| 229 | 5.5 | GRAPH | Percolation threshold on Hasse vs d | p_c decreases |
| 230 | 6.5 | TRANSITION | MCMC on 4-orders: phase structure | Smooth crossover at N=20 |
| 231 | 5.5 | INFO | Lempel-Ziv complexity of causal matrix | Dimension-dependent |
| 232 | 5.0 | INFO | Conditional entropy H(row_j | row_i) | — |
| 233 | 5.5 | INFO | MI between past and future cones | — |
| 234 | 5.5 | INFO | Info-theoretic dimension estimator | Approximate |
| 235 | 5.0 | INFO | Source coding bound | — |
| 236 | 5.0 | INFO | Rate-distortion for causal matrices | — |
| 237 | 5.0 | INFO | Information geometry — Fisher info vs beta | — |
| 238 | 5.0 | INFO | Kolmogorov structure function | — |
| 239-240 | ~5 (?) | INFO | — | — |
| 241 | 7.5 | ANALYTIC | E[links/N] exact for random 2-orders | E[L] = (N+1)H_N - 2N |
| 242 | 7.0 | ANALYTIC | Link fraction exact formula | Derived and verified |
| 243 | 5.5 | GRAPH | Hasse diameter of 2-orders | — |
| 244 | 6.0 | ANALYTIC | Expected BD action at beta=0 | — |
| 245 | 7.0 | ANALYTIC | Ordering fraction variance 1/(9N) | Verified |
| 246 | 5.5 | ANALYTIC | Distinct 2-orders up to isomorphism | — |
| 247 | 6.0 | ANALYTIC | Joint (chain, antichain) distribution | Neg correlated |
| 248-250 | ~5 (?) | ANALYTIC | — | — |
| 251 | 5.5 | SJ_VACUUM | SJ with Sorkin-Yazdi truncation | — |
| 252 | 5.5 | SJ_VACUUM | SJ with 1/2 normalization | — |
| 253 | 5.5 | SJ_VACUUM | Massive scalar — mass gap emergence | — |
| 254 | 5.0 | SJ_VACUUM | SJ on regular causets | — |
| 255 | 4.0 | SJ_VACUUM | SJ two-point in momentum space | No clean p-space |
| 256 | 5.5 | SJ_VACUUM | Entanglement of causal diamonds | — |
| 257 | 5.0 | SJ_VACUUM | SJ fidelity between nearby causets | — |
| 258 | 5.0 | SJ_VACUUM | SJ state purity | — |
| 259-260 | ~5 (?) | SJ_VACUUM | — | — |
| 261 | 4.0 | PHYSICS | Hawking temperature from SJ | FAILED; no valid thermal fits |
| 262 | 6.5 | PHYSICS | Bekenstein entropy S=A/4 | S ~ N^0.345; S/A=0.08 |
| 263 | 5.0 | PHYSICS | Regge calculus vs BD action | 2D gravity is topological |
| 264 | 4.0 | PHYSICS | Geodesic deviation | Correct sign; p=0.49 |
| 265 | 2.0 | PHYSICS | Ollivier-Ricci curvature | TOTAL FAILURE |
| 266 | 3.0 | PHYSICS | BD thermodynamic entropy | S ~ N^1.6 (not Bekenstein) |
| 267 | 6.0 | PHYSICS | Particle creation | 1.2 particles; thermal character |
| 268 | 4.0 | PHYSICS | Penrose diagram of crystalline phase | No new physics |
| 269 | 6.5 | PHYSICS | GW from interval distribution | h=0.5 -> +12% link change |
| 270 | 7.0 | PHYSICS | Newton's law from SJ correlations | Log fit WINS; 3.6x better |
| 271 | 6.5 | COMPUTE | Sparse SJ vacuum N=1000+ | Working sparse SVD |
| 272 | 5.5 | COMPUTE | Randomized SVD of C | — |
| 273 | 6.5 | DIMENSION | Graph features predict dimension | — |
| 274 | 7.0 | COMPUTE | MCMC with parallel tempering | Swap acceptance 50-85% |
| 275 | 5.5 | COMPUTE | Spectral clustering of elements | — |
| 276 | 5.5 | GRAPH | Community detection on Hasse | — |
| 277 | 5.5 | GRAPH | PageRank on causal set | — |
| 278 | 5.0 | GRAPH | Graph wavelets on Hasse Laplacian | — |
| 279 | 6.0 | GRAPH | Persistent homology via chain-distance | — |
| 280 | 5.0 | GRAPH | Tensor decomposition of C*C | — |
| 281-290 | ~5 (?) | CREATIVE | (Ideas 281-290 not individually documented) | — |
| 291 | 7.0 | CREATIVE | Price of Anarchy on causal sets | PoA increases with d (2.30->3.18) |
| 292 | 6.5 | INFO | Kolmogorov complexity of 2-order | 26% more compressible |
| 293 | 7.5 | SJ_VACUUM | Quantum complexity of SJ state | AREA LAW: S ~ N^0.453 |
| 294 | 6.0 | INFO | TDA of 2-order config space | Effective dim=1.0 vs 18 |
| 295 | 5.5 | DIMENSION | Fractal dimension of Hasse | Decreases with manifold dim |
| 296 | 7.0 | CREATIVE | Boolean network dynamics | ALL near edge of chaos |
| 297 | 6.5 | SJ_VACUUM | RG flow via coarse-graining | f flows to total order |
| 298 | 3.0 | CREATIVE | Causal matrix as Markov chain | TOTAL FAILURE; absorbs in 1 step |
| 299 | 6.5 | INFO | Max-flow min-cut | Dimension-dependent |
| 300 | 7.5 | GEOMETRY | Emergent spatial topology | f, n_min, n_max distinguish topologies at p<0.001 |

---

## Ideas 301-400: Large-N, Hauptvermutung, Physics, CDT, Graph Theory

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 301 | 6.0 | COMPUTE | SJ c_eff at large N sparse SVD | Truncation artifact at k=50 |
| 302 | 7.0 | COMPUTE | ER=EPR gap reopens at large N | +32% at N=500; reverses Exp49 |
| 303 | 6.0 | RMT | Degenerate top singular values of PJ | Rotation symmetry |
| 304 | 6.5 | RMT | Eigenvalue density at large N | NOT semicircle; ~ rank^{-0.95} |
| 305 | 5.5 | RMT | Positive modes n_pos/N | ~ 0.475-0.480 |
| 306 | 6.0 | GRAPH | Fiedler at large N | ~ N^0.27; nearly saturating |
| 307 | 6.5 | GEOMETRY | Interval entropy stability large N | H -> ~3.0 (CV=0.009) |
| 308 | 6.5 | GEOMETRY | Antichain/sqrt(N) at large N | Stable ~1.0 to N=5000 |
| 309 | 6.0 | GEOMETRY | Link fraction at large N | ~ N^{-0.72} |
| 310 | 5.5 | GEOMETRY | Ordering fraction variance large N | Formula approximate |
| 311-340 | ~5 (?) | GEOMETRY | (Ideas 311-340 — intermediate experiments) | — |
| 341 | 6.5 | TRANSITION | Pure KR order — all observables | chain/N=0.11, AC/sqrt(N)=2.88 |
| 342 | 6.0 | TRANSITION | KR degeneracy | exp(N) distinct configs |
| 343 | 5.5 | TRANSITION | KR entropy | Grows linearly with N |
| 344 | 6.0 | SJ_VACUUM | SJ vacuum on pure KR | Dramatically different |
| 345 | 6.0 | TRANSITION | MCMC cooling dynamics | Layer structure emerges |
| 346-350 | ~5 (?) | TRANSITION | — | — |
| 351 | 7.5 | GEOMETRY | Coordinate reconstruction via MDS | R^2=0.79 at N=80 |
| 352 | 7.5 | DIMENSION | Dimension estimation — 3 methods | MM: MAE=0.037 |
| 353 | 4.0 | GEOMETRY | Topology detection | 8.9% accuracy; features overlap |
| 354 | 5.5 | PHYSICS | Curvature via BD action | Noisy |
| 355 | 6.0 | TRANSITION | Phase ID without beta | 70.8% accuracy |
| 356 | 8.0 | GEOMETRY | Sprinkled vs CSG discrimination | 100% accuracy |
| 357 | 7.0 | GEOMETRY | Same-spacetime test | d=0.11 same vs d=0.80 diff |
| 358 | 6.5 | GRAPH | Spectral embedding for coordinates | R^2=0.73 |
| 359 | 8.0 | GEOMETRY | Hauptvermutung: C encodes metric | All p<0.0001 |
| 360 | 8.5 | GEOMETRY | Blind spacetime identification | 96% from 7 observables |
| 361 | 6.0 | PHYSICS | Equivalence principle | Exactly Lorentz-invariant |
| 362 | 4.0 | PHYSICS | Gravitational redshift | Not Tolman relation |
| 363 | 4.0 | PHYSICS | Cosmological particle creation | Conformal coordinates issue |
| 364 | 6.0 | PHYSICS | Hawking radiation / horizon | Horizon affects vacuum (p=0.004) |
| 365 | 7.0 | PHYSICS | Casimir effect | 1/d WINS; matches 1+1D |
| 366 | 7.0 | PHYSICS | Conformal anomaly | Linear in R (R^2=0.88); 40x too small |
| 367 | 6.0 | PHYSICS | Brown-York stress tensor | E density constant (CV=7.1%) |
| 368 | 5.0 | PHYSICS | Fluctuation-dissipation theorem | Exactly but tautological |
| 369 | 3.0 | PHYSICS | KMS condition | Zero slope; needs complex W |
| 370 | 7.0 | PHYSICS | Discrete Dirac operator | Novel; chiral symmetry breaking |
| 371-380 | 5-7 | CDT | Deep CDT comparison (eigenvalues, gaps, Fiedler) | Comprehensive CDT analysis |
| 381-385 | ~5 (?) | GRAPH | — | — |
| 386 | 6.0 | GRAPH | Matching number of Hasse | — |
| 387 | 6.0 | GRAPH | Vertex connectivity | — |
| 388 | 6.5 | GRAPH | Cheeger constant | 4.3x larger than DAGs |
| 389 | 5.0 | GRAPH | Hamiltonian path | — |
| 390 | 5.0 | GRAPH | Planarity | — |
| 391-400 | ~5 (?) | GRAPH | — | — |

---

## Ideas 401-500: Paper Strengthening and Synthesis

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 401 | 7.5 | ANALYTIC | Corrected master interval formula | P(int=k|gap=m)=2(m-k)/[m(m+1)] |
| 402 | 7.5 | ANALYTIC | f-vector theorem E[f_k]=C(N,k+1)/(k+1)! | Clean proof |
| 403 | 7.0 | ANALYTIC | Poset dimension P(dim=2)=1-1/N! | Exact |
| 404 | 6.5 | ANALYTIC | Position-dependent comparabilities | Formula derived |
| 405 | 7.0 | ANALYTIC | Full interval distribution | Right-skewed (skew ~1.8) |
| 406 | 6.5 | ANALYTIC | Interval generating function Z(q) | Complete formula |
| 407 | 6.0 | ANALYTIC | Joint (height, width) distribution | Neg correlated (r~-0.24) |
| 408 | 6.5 | ANALYTIC | Expected maximal chains/antichains | Exact for N=3,4,5 |
| 409 | 6.0 | ANALYTIC | Mobius function exact values | — |
| 410 | 7.0 | ANALYTIC | P(Hasse connected)=P(comp connected) PROVED | Always identical |
| 411 | 6.5 | GRAPH | Fiedler lower bound | ~ 0.20*N^0.33 |
| 412 | 6.0 | GRAPH | Spectral gap ratio | ~ 0.06 |
| 413 | 7.5 | GRAPH | Fiedler eigvec = spatial partition | r=0.55 spatial |
| 414 | 7.0 | GRAPH | Cheeger constant measured | Causets 4.3x DAGs |
| 415 | 6.0 | TRANSITION | Link fraction FSS | Does NOT sharpen |
| 416 | 6.5 | TRANSITION | Fiedler jump FSS | ~50% jump |
| 417 | 6.5 | TRANSITION | Path entropy across BD | 16% drop |
| 418 | 6.0 | GEOMETRY | Geometric health index | Moderate discrimination |
| 419 | 6.5 | GRAPH | Diameter across BD | 21% drop |
| 420 | 7.5 | GEOMETRY | 2-orders vs sprinkled universality | Match within 3-14% |
| 421 | 7.0 | CDT | Why ~4 positive modes on CDT? | Eigenvalue structure |
| 422 | 7.0 | CDT | CDT SJ at larger N | c->1 (1.84->1.10) |
| 423 | 7.0 | CDT | n_pos/N CDT vs causets | CDT fewer |
| 424 | 7.0 | CDT | c_eff vs T at fixed N | T controls c |
| 425 | 6.5 | CDT | SJ on CDT through transition | — |
| 426 | 6.0 | CDT | Interval entropy on CDT transition | — |
| 427 | 7.0 | CDT | CDT -> causet via rewiring | c changes |
| 428 | 6.0 | CDT | Hasse of CDT | — |
| 429 | ~5 (?) | CDT | — | — |
| 430 | 8.0 | CDT | Kronecker product theorem | iDelta = A_T tensor J; EXACT |
| 431 | 8.0 | RMT | Fine beta scan — no sub-Poisson | <r>=0.57-0.60 everywhere |
| 432 | 7.5 | RMT | Min <r> vs N | All consistent with GUE |
| 433 | 6.5 | RMT | MCMC dynamics of <r> | Stable mean ~0.58 |
| 434 | 7.0 | RMT | Full P(s) distribution | Not Wigner surmise |
| 435 | 6.0 | RMT | <r> vs action bimodality | No correlation |
| 436 | 7.5 | RMT | Mixture test — mechanism proved | Mixing GUE -> sub-Poisson |
| 437 | 7.0 | RMT | W <r> statistics | Both GUE |
| 438 | 6.0 | RMT | 4D BD spectral stats | Unreliable (stuck chain) |
| 439 | 7.0 | RMT | Epsilon dependence of <r> | No effect |
| 440 | 7.0 | RMT | Total order spectral transition | Chain <r>=0.82; 50% removal -> GUE |
| 441 | 7.0 | DIMENSION | Path entropy as dim estimator | Monotonic; survives null |
| 442 | 6.5 | PHYSICS | Newton's law stability vs N | Log wins; coeff drifts |
| 443 | 7.5 | PHYSICS | Casimir 1/d robustness | Most robust physics result |
| 444 | 7.0 | TRANSITION | KR layer count scaling | sqrt(N); R^2=0.93 |
| 445 | 7.0 | GEOMETRY | Topology detection breadth | 5/6 pairs distinguished |
| 446 | 7.5 | DIMENSION | Graph-theoretic dim estimators feasibility | 5 independent estimators |
| 447 | 7.0 | PHYSICS | SJ reproduces physics feasibility | Qualitative on 5 results |
| 448 | 7.5 | TRANSITION | BD transition comprehensive feasibility | HIGHLY FEASIBLE |
| 449 | 4.0 | COMPUTE | Methods paper feasibility | LOW VALUE |
| 450 | 6.5 | CREATIVE | 400 experiments meta-paper | Risky framing |
| 451 | 5.0 | PHYSICS | Weyl's law for Hasse | Overshoots |
| 452 | 4.0 | DIMENSION | Minkowski box-counting | Biased |
| 453 | 7.0 | DIMENSION | Hausdorff dimension of Hasse | CONVERGING to 2.0 |
| 454 | 8.0 | PHYSICS | SJ Green's function vs continuum | r=0.90, rho=0.93 |
| 455 | 3.0 | PHYSICS | Propagator poles in p-space | No clean p-space |
| 456 | 8.5 | DIMENSION | Spectral dim from Hasse heat kernel | CONVERGES TO 2.0! |
| 457 | 6.0 | GEOMETRY | Volume-distance scaling | Close to r^2 |
| 458 | 3.5 | GEOMETRY | Euler char via Gauss-Bonnet | Triangle-free defeats it |
| 459 | 4.0 | PHYSICS | Geodesic deviation analogue | Chains too short |
| 460 | 4.5 | PHYSICS | Scalar curvature from BD | No sensitivity |
| 461 | 5.5 | GEOMETRY | 2-order vs 3-order in 2D | — |
| 462 | 5.5 | GEOMETRY | Diamond vs rectangle vs strip | — |
| 463 | 5.5 | COMPUTE | MCMC move comparison | — |
| 464 | 5.5 | TRANSITION | BD at different epsilon | — |
| 465 | 5.0 | GEOMETRY | Finite vs periodic BC | — |
| 466 | 5.5 | SJ_VACUUM | Normalization dependence | — |
| 467 | 5.5 | SJ_VACUUM | Partition dependence | — |
| 468 | 5.0 | COMPUTE | Seed dependence | — |
| 469-470 | ~5 (?) | COMPUTE | — | — |
| 471 | 7.5 | ANALYTIC | Hasse connectivity -> 1 PROVED | E[deg]~2ln(N); P(conn)=1 for N>=160 |
| 472 | 6.0 | RMT | Eigenvalue density exact? | Kurtosis ~ N/4; OPEN |
| 473 | 7.0 | GRAPH | Fiedler -> infinity conditional proof | Via Cheeger; ~ N^0.28 |
| 474 | 8.0 | ANALYTIC | EXACT link fraction formula | 4((N+1)H_N-2N)/(N(N-1)); zero parameters |
| 475 | 6.5 | ANALYTIC | E[S_BD] at beta=0 | Off by ~2x |
| 476 | 7.0 | ANALYTIC | Chain-antichain independence CONFIRMED | Baik-Rains 2001 |
| 477 | 7.5 | ANALYTIC | Tracy-Widom convergence rate | Mean->-1.77, var->0.81 |
| 478 | 7.0 | ANALYTIC | Interval dist E[N_k] monotone PROVED | Strictly decreasing |
| 479 | 6.5 | RMT | Spectral gap ~ N^{-1.98} | Max/gap ~ N^2 |
| 480 | 5.0 | ANALYTIC | BD partition function zeros | Z>0 for real beta |
| 481 | 9.0 | COSMOLOGY | Cosmological constant prediction | Within O(1); ONLY approach |
| 482 | 6.5 | DIMENSION | Spectral dimension flow | Shared with CDT, AS |
| 483 | 5.0 | SJ_VACUUM | Central charge prediction | c~12.7 (expected 1) |
| 484 | 7.5 | PHYSICS | Bekenstein-Hawking coefficient | alpha~0.41 vs BH 0.25 |
| 485 | 6.5 | PHYSICS | Gravitational decoherence rate | Consistent with Diosi-Penrose |
| 486 | 8.0 | COSMOLOGY | Number variance of Lambda | sigma^2*N ~ const; 5 orders of mag |
| 487 | 4.0 | PHYSICS | Graviton propagator | 10^{-30} at LHC; unfalsifiable |
| 488 | 6.0 | SJ_VACUUM | Matter content from c_eff | 3D: 1.07 closest to c=1 |
| 489 | 8.0 | PHYSICS | Information recovery across horizon | Early radiation compensates |
| 490 | 6.5 | PHYSICS | Newton's constant from causets | G not free parameter |
| 491 | 8.0 | GEOMETRY | 20-observable comparison table | 21/22 significant at p<0.001 |
| 492 | 7.0 | CREATIVE | Rank all 500 ideas | Top: geometry, wild card |
| 493 | 7.5 | CREATIVE | Category analysis | Wild card highest ceiling |
| 494 | 7.0 | CREATIVE | Diminishing returns | Discovery ACCELERATED |
| 495 | 7.0 | CREATIVE | Meta-paper abstract | Complete |
| 496 | 7.0 | PHYSICS | Most important open question | BH S=A/4G from SJ |
| 497 | 6.0 | COMPUTE | Project statistics | 66K lines, 98 files |
| 498 | 7.0 | CREATIVE | Retrospective | Pure geometry first |
| 499 | 7.0 | CREATIVE | Will 7.5 ceiling break? | YES via analytic proof |
| 500 | 7.5 | PHYSICS | Bekenstein-Hawking in 3D | S~N^0.735; S/A=0.052 |

---

## Ideas 501-600: Post-500 Deepening and Programme Finale

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 501 | 7.0 | CDT | Approximate Kronecker for causets | Factorization CDT-specific |
| 502 | 7.5 | ANALYTIC | Master formula for d-orders | alpha grows super-linearly with d |
| 503 | 8.0 | ANALYTIC | Link fraction correction | Matches exact to 10 sig figs |
| 504 | 8.5 | GRAPH | Spectral embedding (19 eigvecs) | R^2=0.83-0.91; 90% of geometry |
| 505 | 6.5 | TRANSITION | Variance of Glaser action | Self-averaging but slow |
| 506 | 6.0 | TRANSITION | BD transition width scaling | ~ N^{-0.84} |
| 507 | 7.5 | RMT | Spectral compressibility | chi->0 confirms GUE long-range |
| 508 | 7.0 | GRAPH | Hasse girth distribution | Girth=4; 4-cycles ~ N^2.19 |
| 509 | 7.5 | ANALYTIC | Chain fluctuations -> TW_2 | CONFIRMED |
| 510 | 7.0 | ANALYTIC | Interval gen function zeros | Ring structure |
| 511 | 6.0 | TRANSITION | Interval entropy N=100-150 4D | Ergodicity issues |
| 512 | 7.0 | GEOMETRY | Interval entropy sprinkled 4D | Continuum-like phase |
| 513 | 4.0 | TRANSITION | FSS collapse of H(beta) | MCMC too short |
| 514 | 5.0 | TRANSITION | H vs link fraction sharpness | H 7-10x higher chi |
| 515 | 7.5 | COSMOLOGY | Tighten alpha constraint | alpha=0.035-0.070 |
| 516 | 7.0 | COSMOLOGY | DESI DR2 w(z) prediction | Qualitative match; huge scatter |
| 517 | 6.5 | COSMOLOGY | Bayes factor EPL vs LCDM | LCDM favored 3.8x |
| 518 | 8.0 | RMT | Phase-mixing artifact quantified | <r>=0.12 explained |
| 519 | 8.5 | RMT | GUE on 3D and 4D | CONFIRMED all dims and methods |
| 520 | 9.5 | ANALYTIC | Gram identity universal | EXACT to 10^{-17} for ALL causets |
| 521 | 6.0 | CREATIVE | VAE/PCA on causal matrices | r=0.88 with dim |
| 522 | 6.5 | CREATIVE | Hasse braiding/writhe | Manifold > random |
| 523 | 7.0 | TRANSITION | BD universality class | ~Ising exponent |
| 524 | 6.0 | INFO | Channel capacity scaling | ~ N^0.40 |
| 525 | 6.5 | GRAPH | Small-world test | NOT small-world (clustering=0) |
| 526 | 7.0 | GEOMETRY | Order complex Euler char | Dimension-dependent |
| 527 | 5.5 | CREATIVE | Category theory / functors | Re-derives known facts |
| 528 | 7.5 | SJ_VACUUM | QMI distribution | SSA satisfied; time<random |
| 529 | 7.0 | CREATIVE | Lyapunov exponents of MCMC | Uniformly chaotic |
| 530 | 8.0 | CREATIVE | Real-space RG (maximal matching) | Novel; trivial fixed point |
| 531 | 6.0 | COMPUTE | SJ entropy N=1000 | S ~ N^0.295 |
| 532 | 6.0 | SJ_VACUUM | ER=EPR N=1000 | Gap negative |
| 533 | 6.0 | GRAPH | Fiedler large N | Nearly constant ~1.5 |
| 534 | 6.0 | GEOMETRY | Interval entropy large N | H->3.00 |
| 535 | 7.0 | GEOMETRY | Antichain N=1000-20000 | AC/sqrt(N) ~ 1.0 |
| 536 | 7.0 | GEOMETRY | Link fraction large N | c=3.14 (close to pi!) |
| 537 | 6.0 | GEOMETRY | Ordering frac var large N | Formula approximate |
| 538 | 6.0 | TRANSITION | BD action mean large N | Boundary dominates |
| 539 | 7.0 | GEOMETRY | Chain length large N | ~ 1.35*N^0.541 |
| 540 | 6.0 | GRAPH | Hasse diameter large N | =4-6; ~ N^0.109 |
| 541-550 | 5.5-6.5 | ANALYTIC | Paper G exact results (maximal, minimal, chromatic, etc.) | Multiple exact formulas |
| 551 | 7.0 | CDT | Kronecker non-uniform CDT | n_pos ~ floor(T/2) |
| 552 | 7.5 | CDT | c_eff vs T at fixed s | T controls c |
| 553 | 8.0 | GRAPH | Fiedler on trans-closed DAGs | Geometry essential |
| 554 | 7.5 | ANALYTIC | Master formula at beta_c | Robust to BD weighting |
| 555 | 9.0 | SJ_VACUUM | ER=EPR at d=4, N=50 | r=0.91; STRONGER than d=2 |
| 556 | 7.5 | RMT | Erdos-Yau binary matrices | GUE validated |
| 557 | 7.0 | TRANSITION | 4D MCMC acceptance | >10% everywhere |
| 558 | 7.0 | SJ_VACUUM | c_eff divergence reframing | Scaling difference |
| 559 | 8.0 | CDT | Density-preserving rewiring | Structure, not density |
| 560 | 7.0 | GRAPH | Fiedler d>=4 response | Collapse IS physics |
| 561 | 9.0 | CDT | Kronecker -> spectrum EXACTLY | Error ~ 10^{-15} |
| 562 | 8.0 | ANALYTIC | Master formula -> entropy | <2% error |
| 563 | 7.0 | GRAPH | Fiedler -> entanglement | Partial r=0.10 |
| 564 | 6.5 | GRAPH | Link frac -> ER=EPR | Partial r=0.42 |
| 565 | 7.5 | GEOMETRY | Antichain -> spatial extent | Converging to 1 |
| 566 | 8.0 | ANALYTIC | E[S_Glaser]=1 -> beta_c | Fluctuation argument |
| 567 | 8.0 | CDT | Kronecker -> CDT entanglement | n_pos exact; S ~75% |
| 568 | 6.5 | TRANSITION | Phase-mixing all observables | Mild (0.8-6.5%) |
| 569 | 5.5 | ANALYTIC | Exact Z -> beta_c extrapolation | Too broad |
| 570 | 8.5 | CREATIVE | Unified 8-paper narrative | All bridges identified |
| 571 | 7.0 | DIMENSION | Paper F on 3-orders | Higher link, lower path entropy |
| 572 | 7.5 | ANALYTIC | Paper G on 3/4-orders | E[f] != 1/d! |
| 573 | 6.5 | SJ_VACUUM | SJ on 4-orders | c_eff ~ 0.3 |
| 574 | 6.5 | TRANSITION | BD on 4-orders | Visible at N=30 |
| 575 | 7.5 | GRAPH | Hasse Laplacian d-dependence | d=4: 0-10% connected |
| 576 | 8.0 | DIMENSION | Antichain scaling d=2-5 | Within 6% of theory |
| 577 | 7.5 | DIMENSION | Chain scaling d=2,3 | Confirmed |
| 578 | 7.0 | ANALYTIC | E[S_BD_4D]/N | Grows neg |
| 579 | 7.0 | GEOMETRY | Link fraction d-dependent | c_d measured |
| 580 | 8.5 | CDT | Kronecker all spatial dims | n_pos=floor(T/2) for ALL d |
| 581 | 8.5 | CDT | Kronecker proof rigorous | Formal proof |
| 582 | 7.5 | CDT | Non-uniform CDT extension | Block structure |
| 583 | 7.5 | CDT | Eigenvalues of A_T | Closed form |
| 584 | 7.0 | CDT | Derive c_eff(CDT) | — |
| 585 | 7.0 | CDT | When c_eff reaches 1 | Predicted |
| 586 | 7.5 | CDT | Min perturbation breaks Kronecker | Sharp increase |
| 587 | 7.0 | CDT | Entanglement profile on CDT | — |
| 588-590 | ~5 (?) | CDT | — | — |
| 591 | 9.0 | ANALYTIC | E[f]=1/2^{d-1} THEOREM | Conjectured d! WRONG |
| 592 | 8.0 | ANALYTIC | Exact Z(beta) 4D BD N=4 | 11 action levels |
| 593 | 7.0 | ANALYTIC | MI I(u;v) between permutations | Peak at intermediate beta |
| 594 | 7.5 | ANALYTIC | 4D BD ground state | "Diamond+tail" |
| 595 | 7.0 | RMT | Spectral form factor fixed | Polynomial unfolding works |
| 596 | 8.0 | DIMENSION | Spectral dim large N | Converging to d_s=2 |
| 597 | 7.5 | CDT | SJ entropy on CDT | c: 1.84->1.10 |
| 598 | 7.0 | CREATIVE | Master 6-panel figure | Layout specified |
| 599 | 8.0 | CREATIVE | Review paper abstract | Complete |
| 600 | 8.5 | CREATIVE | Final assessment | Programme 8.0/10 |

---

## Ideas 601-700: Papers, Large-N, Analytics, Field Impact

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 601-610 | 6-7.5 | CDT | Kronecker extensions (exact c, profiles, MI, perturbation) | Comprehensive CDT theory |
| 611-620 | 6.5-7.5 | CREATIVE | Paper-level synthesis (outline, merge, review, impact) | Full publication planning |
| 621-630 | 6-7 | COMPUTE | Large-N verification N=1000-50000 | All scaling laws verified |
| 631 | 8.0 | ANALYTIC | E[maximal elements]=H_N PROVED | Harmonic number |
| 632 | 7.0 | ANALYTIC | E[k-antichains]=C(N,k)/k! PROVED | — |
| 633 | 7.0 | ANALYTIC | P(Hasse connected) bounds | — |
| 634 | 8.0 | ANALYTIC | E[f^2] exact second moment | Complete |
| 635 | 8.0 | ANALYTIC | E[c_k] chains exact | Clean formula |
| 636 | 7.0 | ANALYTIC | E[N_k] monotone PROVED | — |
| 637 | 7.0 | ANALYTIC | Cov(chain, antichain) | Negative |
| 638 | 6.0 | ANALYTIC | Generating function G(z) | — |
| 639 | 8.0 | ANALYTIC | Link fraction constant = 4 (not pi) | Settled |
| 640 | 9.0 | ANALYTIC | E[S_BD] vs epsilon complete | Full formula |
| 641-650 | 6-7.5 | CDT | Deep CDT theory (c_eff, Toeplitz, MPS, continuum) | Analytic CDT results |
| 651-656 | 6.5-7 | TRANSITION | Paper A/F strengthening (4D, universal H, Cheeger) | — |
| 661 | 7.5 | SJ_VACUUM | Foliation projection vacuum | c_eff fix attempt |
| 662 | 7.0 | TRANSITION | Modified BD with background subtraction | Improved action |
| 663 | 7.5 | COMPUTE | Fiedler-filtered MCMC | Selection criterion |
| 664 | 6.5 | RMT | Kurtosis excess observable | — |
| 665 | 6.0 | GRAPH | Diameter saturation | — |
| 666 | 7.5 | CREATIVE | Textbook chapter | Publication-ready |
| 667 | 7.0 | CREATIVE | Phase-mixing short note | Standalone note |
| 668 | 8.5 | CREATIVE | c_eff divergence programme | Best new direction |
| 669 | 7.0 | CREATIVE | Expert perspective | — |
| 670 | 7.5 | CREATIVE | Workshop talk recommendation | Kronecker theorem |
| 671-680 | 5-7 | CREATIVE | Export methodology (Ising, brain, 3-order, social, exact) | Cross-field applications |
| 681-690 | 6.5-7 | SJ_VACUUM | Open questions attacked (c_eff, link const, diameter, etc.) | 10 key questions |
| 691 | 7.0 | SJ_VACUUM | SJ entropy across horizon | — |
| 692 | 7.5 | CDT | Kronecker c_eff prediction | Exact |
| 693 | 7.0 | ANALYTIC | E[S_BD] from master formula | Complete |
| 694 | 7.0 | GRAPH | Fiedler bound attempt | — |
| 695 | 7.0 | DIMENSION | Complete dimension table d=2-6 | Reference |
| 696 | 7.0 | TRANSITION | Phase diagram (beta, eps) | Mapped |
| 697 | 7.0 | CREATIVE | Cross-paper figure | Visual synthesis |
| 698 | 7.5 | CREATIVE | Review paper intro | Draft |
| 699 | 9.5 | CREATIVE | Best/worst 10 retrospective | Complete ranking |
| 700 | 8.0 | CREATIVE | Most important lesson | "Comparison is king" |

---

## Ideas 701-800: Creative Methodologies

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 701 | 7.5 | SJ_VACUUM | c_eff fix — optimal clamping | Clamping approach |
| 702 | 7.5 | SJ_VACUUM | Foliated vacuum with disorder | Modified prescription |
| 703 | 7.5 | TRANSITION | BD dynamical critical exponent z | — |
| 704 | 8.0 | GRAPH | Hasse diameter analytic bound | O(1) proved |
| 705 | 7.5 | ANALYTIC | Link fraction 2nd-order expansion | Full expansion |
| 706 | 6.0 | CREATIVE | Poker hands as causal sets | Partial orders |
| 707 | 6.5 | CREATIVE | Recipe DAGs as causal sets | Dependency graphs |
| 708 | 6.9 | CREATIVE | Weather 2-orders | Correlated permutations |
| 709 | 6.5 | CREATIVE | Musical causets | Harmonic structure |
| 710 | 7.0 | CREATIVE | Tournament causets | Transitivity fraction |
| 711-720 | 5.5-6.5 | CREATIVE | Random walk to causets (Fibonacci, p-adic, tropical, etc.) | 10 connection paths |
| 721-730 | 5-6.5 | CREATIVE | Bad idea inversions (temperature, speed, rotation, etc.) | Inversion methodology |
| 731-740 | 5.5-7 | CREATIVE | Cross-disciplinary analogues (biology, econ, CS, ecology) | 10 field connections |
| 741-750 | 5.5-6.5 | CREATIVE | Dream logic (perturbation, attraction, amnesia, etc.) | Creative exploration |
| 751-760 | 6-7 | DIMENSION | 10 ways to extract d=2 from a random 2-order | Comprehensive methods |
| 761-770 | 5.5-6.5 | ANALYTIC | Perturbation theory on exact results | 10 perturbations |
| 771-780 | 5.5-6.5 | GEOMETRY | Symmetry breaking in 2-orders | 10 symmetry-broken ensembles |
| 781-790 | 6-6.5 | CREATIVE | Emergent phenomena (dimension, locality, time, etc.) | 10 emergence tests |
| 791-800 | 5.5-7 | CREATIVE | Maximum chaos methodology (crystallize, ferment, etc.) | Die-roll approaches |

---

## Ideas 801-850: Grand Finale

| # | Score | Category | Summary | Key Finding |
|---|-------|----------|---------|-------------|
| 801-810 | 5.5-6.5 | CREATIVE | Synesthesia (taste, feel, see, hear, smell, touch) | Sensory analogies |
| 811 | 7.0 | ANALYTIC | Random 2-order contains every 4-element poset | Universality |
| 812 | 6.5 | SJ_VACUUM | SJ vacuum positive semidefinite | Basic property |
| 813 | 7.0 | ANALYTIC | No two random 2-orders N>=10 isomorphic | Uniqueness |
| 814 | 6.5 | ANALYTIC | BD action at beta=0 always positive | Sign theorem |
| 815 | 7.0 | GRAPH | Hasse never a tree for N>=6 | Cycle existence |
| 816 | 6.0 | GEOMETRY | Every element in some maximal antichain | Existence |
| 817 | 6.5 | ANALYTIC | Ordering fraction never exactly 1/2 | Impossibility |
| 818 | 6.5 | GEOMETRY | >= log2(N) distinct layer widths | Lower bound |
| 819 | 6.0 | SJ_VACUUM | SJ entropy bounded by N/2 | Upper bound |
| 820 | 6.0 | GEOMETRY | Two 2-orders share >=1 common relation | Overlap |
| 821-830 | 5.5-6.5 | CREATIVE | Time-reversed research (inverse problems) | 10 inverse problems |
| 831-840 | 5.5-6.5 | CREATIVE | Evolutionary dynamics (evolve H, Fiedler, c_eff, etc.) | Genetic algorithms |
| 841-850 | 5.5-6 | CREATIVE | Ultimate randomness (shuffle, dart, coin, dice, etc.) | Random physical actions |

---

## Top Ideas by Honest Score

**IMPORTANT:** The scores in the per-idea tables above are the agents' self-assessments, which are systematically inflated. When I (the supervising model) re-scored each result honestly, NO individual idea exceeded 7.5/10. The 8.0 score for Paper E comes from the *combination* of multiple 7.5-level results, not from any single idea.

The following ideas represent the honestly strongest results across 850 experiments:

| Rank | # | Honest Score | Category | Summary |
|------|---|-------------|----------|---------|
| 1 | 430 | 7.5 | CDT | Kronecker product theorem C^T-C = A_T⊗J |
| 2 | 583 | 7.5 | CDT | Exact eigenvalue formula μ_k = cot(π(2k-1)/(2T)) |
| 3 | 588 | 7.5 | CDT | Wightman depends only on temporal indices |
| 4 | 250 | 7.5 | ANALYTIC | Master interval formula (later corrected) |
| 5 | 241 | 7.5 | ANALYTIC | E[links] = (N+1)H_N - 2N proved |
| 6 | 245 | 7.5 | ANALYTIC | Var[f] = (2N+5)/[18N(N-1)] proved |
| 7 | 139 | 7.5 | ANALYTIC | Antichain = 2√N via Vershik-Kerov |
| 8 | 119 | 7.5 | GRAPH | Hasse Laplacian Fiedler value 50× causets vs DAGs |
| 9 | 186 | 7.5 | TRANSITION | Link fraction as perfectly monotonic BD order parameter |
| 10 | 640 | 7.5 | ANALYTIC | E[S_BD(ε)] exact formula |
| 11 | 639 | 7.5 | ANALYTIC | Link fraction = 2ln(N)/N exactly (undirected) |
| 12 | 635 | 7.5 | ANALYTIC | Chain-antichain symmetry E[c_k]=C(N,k)/k! |
| 13 | 541 | 7.5 | ANALYTIC | E[maximal elements] = H_N |
| 14 | 474 | 7.5 | ANALYTIC | Link fraction exact formula verified to 0.06% |
| 15 | 507 | 7.5 | RMT | Spectral compressibility χ→0 (full GUE with unfolding) |

**Pattern:** 10 of the top 15 are ANALYTIC (exact theorems). Proved results consistently outscore numerical observations.

---

## Key Lessons

1. **Pure geometry > SJ vacuum**: Wild card and analytic approaches far outperformed SJ vacuum (6.1% hit rate for 7+)
2. **Theorems beat data**: Exact proofs consistently score 7-7.5, numerical fits plateau at 5-6
3. **Null models essential**: ~30% of initial results killed by random graph controls
4. **Agent scores are inflated**: Supervising re-scoring consistently reduces scores by 0.5-2.0 points
5. **The 7.5 ceiling is structural**: 850 ideas across systematic, chaotic, adversarial, evolutionary, and dream-logic methods all hit the same ceiling
5. **Comparison is king**: Best ideas COMPARE across dimensions/methods/approaches
6. **7.5 ceiling structural at toy N**: Breaking through requires proofs, cross-approach comparisons, or N=1000+
