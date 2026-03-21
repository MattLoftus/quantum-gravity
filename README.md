# Computational Quantum Gravity

Numerical experiments in causal set theory, CDT, and related approaches to quantum gravity. This repository contains ~90 experiments testing 400+ ideas, plus 10 research papers (8 submission-ready) covering causal set phase transitions, spectral dimension, holographic entanglement, discrete ER=EPR, random matrix universality, and exact combinatorics of random partial orders.

## Papers

| Paper | Directory | Description |
|---|---|---|
| **Hasse Geometry (F)** | `papers/hasse-geometry/` | Spectral graph theory of Hasse diagrams: Fiedler scaling, link fraction as BD order parameter |
| **Exact Combinatorics (G)** | `papers/exact-combinatorics/` | 13+ exact results for random 2-orders: comparability fraction, interval formula, link counts, partition function |
| **Geometry from Entanglement (B5)** | `papers/geometry-from-entanglement/` | Flagship paper: spectral dimension fails null test, SJ entanglement succeeds |
| **Discrete ER=EPR (C)** | `papers/er-epr/` | Wightman function correlates with connectivity (r=0.88), analytic proof for 2-orders |
| **CDT Comparison (E)** | `papers/cdt-comparison/` | First cross-approach SJ vacuum comparison: CDT c_eff=1 vs causet divergence |
| **Spectral Statistics (D)** | `papers/random-matrix/` | BD transition as spectral statistics transition, GUE universality |
| **Interval Entropy (A)** | `papers/interval-entropy/` | Interval entropy as BD order parameter, 4D three-phase structure |
| **Everpresent Lambda (B2)** | `papers/everpresent-lambda/` | Stochastic cosmological constant predicting Omega_Lambda=0.732 |

Papers B1 (Spectral Dimension) and B3 (Holographic Entanglement) are subsumed by B5.

## Installation

**Python 3.9+** with:

```bash
pip install numpy scipy matplotlib
```

For cosmological calculations (Paper B2 only):

```bash
pip install classy
```

For compiling LaTeX papers:

```bash
brew install tectonic   # macOS
# or see https://tectonic-typesetting.github.io for other platforms
```

## Quick Start

### Run an experiment

```bash
/usr/bin/python3 experiments/exp01_validate_dimension.py
```

Most experiments print results to stdout. Some generate matplotlib plots.

### Compile a paper

```bash
cd papers/hasse-geometry
tectonic hasse_geometry.tex
```

### Use the causal set library

```python
from causal_sets.sprinkle import sprinkle_minkowski
from causal_sets.dimension import myrheim_meyer_dimension

cs = sprinkle_minkowski(N=200, d=2)
d_est = myrheim_meyer_dimension(cs)
print(f"Estimated dimension: {d_est:.2f}")
```

## Project Structure

```
quantum-gravity/
├── causal_sets/          # Core library: sprinkling, dimension estimators,
│   │                     #   BD action, MCMC, SJ vacuum, growth models
│   ├── core.py           # CausalSet data structure
│   ├── fast_core.py      # Vectorized operations
│   ├── sprinkle.py       # Poisson sprinkling (Minkowski, de Sitter)
│   ├── dimension.py      # Myrheim-Meyer + spectral dimension
│   ├── bd_action.py      # Benincasa-Dowker action (2D, 4D)
│   ├── mcmc.py           # MCMC sampling weighted by BD action
│   ├── sj_vacuum.py      # Sorkin-Johnston vacuum state
│   └── ...
├── cdt/                  # Causal Dynamical Triangulations
│   └── triangulation.py  # 2D CDT with MCMC + spectral dimension
├── cosmology/
│   └── everpresent_lambda.py  # Stochastic Lambda FRW simulation
├── holographic/
│   ├── tensor_network.py # Tensor network infrastructure
│   └── happy_code.py     # [[5,1,3]] HaPPY code
├── experiments/          # ~90 numbered experiments (exp01-exp98)
├── papers/               # 10 research papers (LaTeX + figures)
├── FINDINGS.md           # Complete findings log
├── LESSONS_LEARNED.md    # Meta-lessons from the research process
└── somewhat_laymans_terms.html  # Plain-language summary
```

## Citation

If you use this code or results in your work, please cite the relevant paper(s). BibTeX entries are available in each paper's `.tex` file.

For the codebase itself:

```bibtex
@software{loftus2026quantumgravity,
  author = {Loftus, Matt},
  title = {Computational Quantum Gravity: Causal Sets, CDT, and Beyond},
  year = {2026},
  url = {https://github.com/MattLoftus/quantum-gravity}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This research was conducted with substantial assistance from Claude (Anthropic), which contributed to code development, mathematical derivations, paper writing, and experimental design. All results were validated computationally. See `LESSONS_LEARNED.md` for a candid account of the research process, including what worked and what didn't.
