export interface Paper {
  id: string;
  letter: string;
  title: string;
  score: number;
  targetJournal: string;
  keyClaim: string;
  pages: number;
  tier: number;
  abstract: string;
}

export const papers: Paper[] = [
  {
    id: "E",
    letter: "E",
    title: "CDT vs Causal Sets: A Kronecker Product Theorem",
    score: 8.0,
    targetJournal: "Physical Review D",
    keyClaim: "CDT's time foliation creates a Kronecker product structure that limits quantum modes to O(sqrt(N)), explaining why CDT reproduces c=1 while causal sets diverge.",
    pages: 7,
    tier: 1,
    abstract: "First cross-approach comparison of the Sorkin-Johnston vacuum on Causal Dynamical Triangulations vs causal sets. CDT gives c_eff approaching 1 (free scalar), while causal sets give c_eff diverging to infinity. The Kronecker product theorem C^T - C = A_T tensor J explains why: CDT has only floor(T/2) positive modes vs ~N/2 for causal sets. Exact eigenvalue formula mu_k = cot(pi(2k-1)/(2T)) verified to 10^-14. Single perturbation breaks the structure -- extreme fragility. Generalizes to all spatial dimensions."
  },
  {
    id: "G",
    letter: "G",
    title: "Exact Combinatorics of Random 2-Orders",
    score: 8.0,
    targetJournal: "J. Combinatorial Theory A",
    keyClaim: "15+ exact theorems for random 2-orders including master interval formula, E[f]=1/2, E[links]=(N+1)H_N - 2N, and E[S_Glaser]=1 for all N.",
    pages: 9,
    tier: 1,
    abstract: "A comprehensive collection of exact analytic results for random 2-orders (intersections of two random total orders on N elements). Key results: ordering fraction E[f]=1/2 with Var[f]=(2N+5)/[18N(N-1)], master interval formula P(int=k|gap=m)=2(m-k)/[m(m+1)], expected links E[L]=(N+1)H_N - 2N, link fraction ~4ln(N)/N, Glaser action E[S_Glaser]=1 for all N >= 2, antichain width ~2sqrt(N) with Tracy-Widom fluctuations, maximal elements E[max]=H_N, and k-antichains E=C(N,k)/k!."
  },
  {
    id: "C",
    letter: "C",
    title: "Discrete ER=EPR on Causal Sets",
    score: 8.0,
    targetJournal: "Physical Review D",
    keyClaim: "Quantum entanglement (|W|) is proportional to causal connectivity (r=0.88, z=13.1), with a universal Gram identity proved for ALL partial orders.",
    pages: 8,
    tier: 1,
    abstract: "Tests the Maldacena-Susskind ER=EPR conjecture on discrete spacetime. The Wightman function |W_ij| correlates with causal connectivity kappa_ij at r=0.88 (z=13.1, partial r=0.82 controlling for distance). Analytic proof: the Gram identity (-Delta^2)_ij = (4/N^2)*kappa_ij holds exactly (to machine precision ~10^-17) for ALL causal sets -- 2-orders, 3-orders, 4-orders, and sprinkled causets in 2D/3D/4D. ER=EPR correlation is r=0.91 in physically relevant 4D."
  },
  {
    id: "D",
    letter: "D",
    title: "GUE Universality in Causal Set Quantum Gravity",
    score: 8.0,
    targetJournal: "Physical Review D",
    keyClaim: "GUE random matrix statistics are universal across ALL phases, dimensions, and coupling strengths. The previously reported sub-Poisson dip was a phase-mixing artifact.",
    pages: 6,
    tier: 1,
    abstract: "The SJ vacuum eigenvalue statistics show universal GUE (Gaussian Unitary Ensemble) level repulsion with <r> = 0.56-0.60 across all BD coupling strengths, system sizes N=20-1000, and non-locality parameters. Confirmed on 3-orders, 4-orders, and sprinkled 2D/3D/4D causets. The previously reported sub-Poisson <r>=0.12 at the phase transition is proved to be a phase-mixing artifact: concatenating eigenvalues from structurally different phases produces artificial gaps."
  },
  {
    id: "B5",
    letter: "B5",
    title: "Geometry from Entanglement, Not Random Walks",
    score: 7.5,
    targetJournal: "Physical Review D",
    keyClaim: "Spectral dimension fails a null test (random graphs match causal sets), but SJ entanglement entropy succeeds -- it scales as ln(N) and drops 3.4x across the BD phase transition.",
    pages: 7,
    tier: 2,
    abstract: "The flagship paper. Spectral dimension d_s is shown to fail on causal sets: a crossing theorem (IVT) proves any finite graph must cross d_s=2, and random graphs with matched density reproduce causal set values. SJ entanglement entropy succeeds: S ~ ln(N) (CFT-like), drops 3.4x across the BD phase transition, and satisfies the holographic monogamy inequality (I_3 <= 0) in 97% of continuum-phase cases."
  },
  {
    id: "F",
    letter: "F",
    title: "Hasse Diagram Spectral Geometry",
    score: 7.0,
    targetJournal: "Physical Review D",
    keyClaim: "The Hasse Laplacian's Fiedler value is 50x larger for causal sets than random DAGs. Spectral embedding with 19 eigenvectors recovers coordinates at R^2=0.83-0.91.",
    pages: 6,
    tier: 2,
    abstract: "Studies the Hasse diagram (direct causal relations only) as a graph. The Fiedler value (algebraic connectivity) is 50x larger for manifold-like causal sets than random DAGs at small N, saturating at ~1.4-1.6 for N>100. Spectral embedding with 19 Laplacian eigenvectors recovers spacetime coordinates at R^2=0.83-0.91. Link fraction provides a perfectly monotonic BD order parameter with a 60% jump. The Hasse diagram is triangle-free (girth >= 4)."
  },
  {
    id: "A",
    letter: "A",
    title: "Interval Entropy as a BD Order Parameter",
    score: 7.0,
    targetJournal: "Classical & Quantum Gravity",
    keyClaim: "Interval entropy drops 87% across the BD phase transition (H=2.4 to 0.3). In 4D, a previously unknown three-phase structure emerges.",
    pages: 6,
    tier: 2,
    abstract: "Introduces interval entropy -- the Shannon entropy of the distribution of causal interval sizes -- as a new order parameter for the Benincasa-Dowker phase transition. In 2D, H drops from 2.4 (continuum phase) to 0.3 (crystalline phase), an 87% drop far sharper than previously known diagnostics. In 4D at N=30-70, a non-monotonic three-phase structure appears with susceptibility chi_S up to 43,288."
  },
  {
    id: "B2",
    letter: "B2",
    title: "Everpresent Lambda: Stochastic Dark Energy from Causal Sets",
    score: 5.5,
    targetJournal: "JCAP",
    keyClaim: "With one parameter (alpha=0.03), the everpresent Lambda model predicts Omega_Lambda=0.73 from first principles -- but LCDM is favored by Bayes factor 3.8x.",
    pages: 4,
    tier: 3,
    abstract: "Implements the Sorkin everpresent Lambda model: if spacetime is discrete with N elements, the cosmological constant fluctuates as Lambda ~ 1/sqrt(N). With alpha=0.03, the model predicts Omega_Lambda = 0.732 +/- 0.103 after anthropic selection. However, Bayesian comparison gives LCDM a 3.8x advantage. The model naturally produces w != -1 (quintessence-like), qualitatively consistent with DESI DR2, but stochastic variance limits predictive power."
  },
];

export interface KeyResult {
  id: string;
  paper: string;
  title: string;
  description: string;
  score: number;
}

export const keyResults: KeyResult[] = [
  {
    id: "kronecker",
    paper: "E",
    title: "Kronecker Product Theorem",
    description: "CDT's causal matrix decomposes exactly as C^T - C = A_T tensor J, giving floor(T/2) positive modes vs ~N/2 for causal sets. This one equation explains why CDT reproduces continuum QFT and causal sets do not.",
    score: 9.0,
  },
  {
    id: "gram-identity",
    paper: "C",
    title: "Universal Gram Identity (ER=EPR)",
    description: "The identity (-Delta^2)_ij = (4/N^2) * kappa_ij holds to machine precision (~10^-17) for ALL partial orders -- not just 2-orders but 3-orders, 4-orders, and sprinkled causets in 2D/3D/4D. Entanglement tracks causal connectivity exactly.",
    score: 9.5,
  },
  {
    id: "gue",
    paper: "D",
    title: "GUE Universality",
    description: "The level spacing ratio <r> stays at 0.57-0.60 (GUE) across all phases of the BD transition, all system sizes, and all dimensions. Quantum chaos is absolute in causal set quantum gravity.",
    score: 8.5,
  },
  {
    id: "exact-formulas",
    paper: "G",
    title: "Exact Formulas for Random 2-Orders",
    description: "E[f] = 1/2, E[links] = (N+1)H_N - 2N, antichain ~ 2*sqrt(N), E[S_Glaser] = 1 for all N. These are exact, zero-parameter results connecting discrete spacetime to harmonic numbers and Tracy-Widom distributions.",
    score: 8.0,
  },
  {
    id: "phase-transition",
    paper: "A",
    title: "Phase Transition via Interval Entropy",
    description: "Interval entropy drops 87% across the BD transition (H = 2.4 to 0.3). In 4D, a previously unknown three-phase structure emerges with susceptibility chi up to 43,288.",
    score: 7.0,
  },
  {
    id: "spectral-embedding",
    paper: "F",
    title: "Spectral Geometry of Hasse Diagrams",
    description: "19 Laplacian eigenvectors recover spacetime coordinates at R^2 = 0.83-0.91. The Fiedler value is 50x larger for manifold-like causal sets than random DAGs. Link fraction provides a perfectly monotonic BD order parameter.",
    score: 8.5,
  },
];

export const phaseTransitionData = [
  { beta: 0, H: 2.42 },
  { beta: 1, H: 2.38 },
  { beta: 2, H: 2.35 },
  { beta: 3, H: 2.31 },
  { beta: 4, H: 2.28 },
  { beta: 5, H: 2.20 },
  { beta: 6, H: 2.10 },
  { beta: 7, H: 1.95 },
  { beta: 8, H: 1.70 },
  { beta: 9, H: 1.30 },
  { beta: 10, H: 0.85 },
  { beta: 10.5, H: 0.55 },
  { beta: 11, H: 0.38 },
  { beta: 12, H: 0.32 },
  { beta: 14, H: 0.30 },
  { beta: 16, H: 0.30 },
  { beta: 20, H: 0.30 },
];

export const gueData = [
  { label: "Continuum\n(beta=0)", r: 0.58 },
  { label: "Near beta_c", r: 0.57 },
  { label: "Crystalline\n(beta=20)", r: 0.59 },
  { label: "3-orders", r: 0.57 },
  { label: "4-orders", r: 0.57 },
  { label: "Sprinkled 2D", r: 0.59 },
  { label: "Sprinkled 3D", r: 0.58 },
  { label: "Sprinkled 4D", r: 0.57 },
];

export const erEprData = [
  { kappa: 2, W: 0.008 },
  { kappa: 5, W: 0.015 },
  { kappa: 10, W: 0.025 },
  { kappa: 15, W: 0.035 },
  { kappa: 22, W: 0.048 },
  { kappa: 30, W: 0.062 },
  { kappa: 40, W: 0.078 },
  { kappa: 50, W: 0.095 },
  { kappa: 65, W: 0.118 },
  { kappa: 80, W: 0.140 },
  { kappa: 100, W: 0.168 },
  { kappa: 120, W: 0.195 },
  { kappa: 140, W: 0.220 },
  { kappa: 160, W: 0.248 },
];

export const fiedlerData = [
  { label: "Causal sets\n(2-orders)", value: 1.6, color: "#3b82f6" },
  { label: "Random DAGs\n(density-matched)", value: 0.032, color: "#9ca3af" },
];

export const categoryAnalysis = [
  { category: "Wild card", hitRate: 50, ceiling: 9.0, ideas: 40 },
  { category: "Computational / algorithmic", hitRate: 45.5, ceiling: 8.5, ideas: 40 },
  { category: "Analytic / theorem", hitRate: 34.7, ceiling: 9.0, ideas: 60 },
  { category: "Pure geometry", hitRate: 25.0, ceiling: 8.5, ideas: 80 },
  { category: "Cross-dimensional", hitRate: 20.0, ceiling: 8.0, ideas: 50 },
  { category: "BD phase transition", hitRate: 18.0, ceiling: 8.0, ideas: 60 },
  { category: "Known physics", hitRate: 12.0, ceiling: 7.5, ideas: 40 },
  { category: "SJ vacuum", hitRate: 6.1, ceiling: 7.5, ideas: 80 },
];

export const scoreDistribution = [
  { range: "3-4", count: 35 },
  { range: "4-5", count: 65 },
  { range: "5-6", count: 120 },
  { range: "6-7", count: 180 },
  { range: "7-8", count: 140 },
  { range: "8-9", count: 50 },
  { range: "9-10", count: 10 },
];
