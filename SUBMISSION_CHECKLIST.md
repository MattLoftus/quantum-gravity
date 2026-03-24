# Submission Checklist

## Paper Status

| Paper | Score | Target Journal | Figures | Peer Reviewed | Compiled | Cover Letter | References Verified | ArXiv | Submitted |
|---|---|---|---|---|---|---|---|---|---|
| **E: CDT Comparison** | **8.0** | PRD | N/A (tables) | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **B5: Geometry from Entanglement** | **7.5** | PRD | ✅ 2 figs | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **C: ER=EPR** | **7.5** | PRD | ✅ 1 fig | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **G: Exact Combinatorics** | **7.5** | J. Comb. Theory A | N/A (tables) | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **D: Spectral Statistics** | **7.5** | PRD | N/A (tables) | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **F: Hasse Geometry** | **7.0** | PRD/CQG | N/A (tables) | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **A: Interval Entropy** | **7.0** | CQG | ✅ 2 figs | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |
| **B2: Everpresent Lambda** | **5.5** | JCAP | N/A | ✅ | ✅ | ✅ | ✅ | [ ] | [ ] |

## Pre-Submission Tasks

### Completed
- [x] All papers written and compiled
- [x] All papers peer reviewed (12 HIGH issues found and fixed)
- [x] Figures generated for Papers A, B5, C
- [x] Paper D completely rewritten (sub-Poisson debunked)
- [x] Paper E updated with Kronecker product theorem + exact eigenvalues + Wightman factorization + S depends on T only + Toeplitz W_T + eigenvalue sum rule + MPS interpretation + continuum limit caveat
- [x] Paper F updated with Fiedler saturation correction (N^0.054, not N^0.32) and diameter correction (N^0.11, not sqrt(N))
- [x] Paper G updated with corrected master formula, new theorems, link_frac constant correction, E[S_BD(epsilon)], chain-antichain symmetry, unimodality, directed/undirected link fraction resolved
- [x] Paper C updated with universal Gram identity (all partial orders) + d=4 result
- [x] Paper D updated with 3-order/4-order/sprinkled GUE + spectral compressibility + artifact mechanism
- [x] Paper A updated with sprinkled 4D causet result
- [x] Paper B2 updated with optimal alpha range, Bayes factor, w(z) deviation
- [x] Layman summary (somewhat_laymans_terms.html) updated to 600 ideas
- [x] SUMMARY.md updated with honest scores and 600-idea summary
- [x] Reference verification (6 errors found and fixed)
- [x] Cover letters written (8 letters at `papers/cover-letters/`)
- [x] GitHub repo prepared (README, LICENSE, .gitignore, requirements.txt, git init, 151 files, initial commit)
- [x] ArXiv endorsement email drafted (`papers/arxiv_endorsement_email.txt`)

### Remaining
- [ ] Human proofread of each paper
- [ ] Upload to arXiv (8 papers)
- [ ] Submit to journals (8 submissions)
- [ ] Create GitHub repo on github.com/MattLoftus/quantum-gravity
- [ ] Push code to GitHub

## Submission Order (recommended)

### Tier 1: Submit simultaneously (different journals)
1. **E** (8.0) → PRD (strongest analytic result — Kronecker theorem + exact eigenvalues)
2. **G** (7.5) → J. Combinatorial Theory A (different audience — math/combinatorics)
3. **F** (7.0) → CQG (novel graph-theoretic approach, corrected scaling)

### Tier 2: Submit after Tier 1 accepted/in review
4. **C** (7.5) → PRD (universal Gram identity + honest caveats)
5. **D** (7.5) → PRD (universal GUE across all dimensions, artifact identification)
6. **B5** (7.5) → PRD (flagship, broadest narrative)

### Tier 3: Submit last
7. **A** (7.0) → CQG (clean, self-contained)
8. **B2** (5.5) → JCAP (optional, weakest)

## Key Caveats in Each Paper (for reviewer preparation)

| Paper | Known Weakness | How Addressed |
|---|---|---|
| E | CDT data from only 3 N values; c_eff → 0 in strict continuum limit | Acknowledged in text; Kronecker theorem is exact regardless; continuum limit caveat added |
| C | ER=EPR gap vanishes at N=500; Gram identity universal to ALL posets | Large-N table in paper; universality caveat explicitly stated |
| G | Some results known in combinatorics (Vershik-Kerov) | Erdos-Szekeres precedent acknowledged; QG connection is novel |
| D | GUE generic to antisymmetric matrices | Explicitly framed as algebraic property, not gravity-specific |
| B5 | c_eff diverges at large N | Caveat paragraph added; relative comparison still valid |
| F | **Fiedler saturates** at N>100 (N^0.054 not N^0.32); diameter N^0.11 not sqrt(N) | Corrected in abstract, results, discussion, and conclusion |
| A | FSS limited to N=50-150 | Acknowledged; parallel tempering used |
| B2 | Stochastic variance too large; Bayes factor 3.8x against | Main caveat clearly stated; framed as prediction not fit |

## Files

| Item | Location |
|---|---|
| Papers (.tex) | `papers/<name>/<name>.tex` |
| Figures (.pdf) | `papers/<name>/fig*.pdf` |
| Cover letters | `papers/cover-letters/` |
| Experiment code | `experiments/exp*.py` |
| Core library | `causal_sets/`, `cdt/`, `cosmology/`, `holographic/` |
| Layman summary | `somewhat_laymans_terms.html` |
| Paper summary | `papers/SUMMARY.md` |
| Findings log | `FINDINGS.md` |
| This checklist | `SUBMISSION_CHECKLIST.md` |

---
*Last updated: 2026-03-22 (Papers E and G updated with new results from Ideas 601-649)*
