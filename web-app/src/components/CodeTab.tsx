export function CodeTab() {
  return (
    <div className="max-w-3xl">
      <div className="mb-10">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Code & Data</h2>
        <p className="text-gray-500 text-sm">
          The full codebase is open source under the MIT license.
        </p>
      </div>

      {/* GitHub link */}
      <div className="bg-navy rounded-lg p-6 mb-8">
        <div className="flex items-center gap-3 mb-3">
          <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
          <a
            href="https://github.com/MattLoftus/quantum-gravity"
            target="_blank"
            rel="noopener noreferrer"
            className="text-white text-lg font-semibold hover:text-blue-light transition-colors"
          >
            github.com/MattLoftus/quantum-gravity
          </a>
        </div>
        <p className="text-gray-400 text-sm">
          ~72,000 lines of Python. MIT License.
        </p>
      </div>

      {/* Project structure */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Project Structure</h3>
        <div className="bg-gray-50 rounded-lg p-5 border border-gray-100 font-mono text-sm text-gray-600 whitespace-pre leading-relaxed">
{`quantum-gravity/
  causal_sets/          Core library
    core.py             CausalSet data structure
    fast_core.py        Vectorized operations
    sprinkle.py         Poisson sprinkling (Minkowski, de Sitter)
    dimension.py        Myrheim-Meyer + spectral dimension
    bd_action.py        Benincasa-Dowker action (2D, 4D)
    mcmc.py             MCMC sampling weighted by BD action
    sj_vacuum.py        Sorkin-Johnston vacuum state
    d_orders.py         d-orders (2D/3D/4D/5D embeddings)
    two_orders.py       2-order representation + MCMC
  cdt/
    triangulation.py    2D CDT with MCMC + spectral dimension
  cosmology/
    everpresent_lambda.py   Stochastic Lambda FRW simulation
  holographic/
    tensor_network.py   Tensor network infrastructure
    happy_code.py       [[5,1,3]] HaPPY code
  experiments/          ~110 numbered experiments
  papers/               10 research papers (LaTeX + figures)`}
        </div>
      </div>

      {/* How to run */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">How to Run Experiments</h3>
        <div className="space-y-4">
          <CodeBlock title="Install dependencies" code="pip install numpy scipy matplotlib" />
          <CodeBlock title="Run an experiment" code="python3 experiments/exp01_validate_dimension.py" />
          <CodeBlock title="Use the causal set library" code={`from causal_sets.sprinkle import sprinkle_minkowski
from causal_sets.dimension import myrheim_meyer_dimension

cs = sprinkle_minkowski(N=200, d=2)
d_est = myrheim_meyer_dimension(cs)
print(f"Estimated dimension: {d_est:.2f}")`} />
        </div>
      </div>

      {/* How to compile papers */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">How to Compile Papers</h3>
        <div className="space-y-4">
          <CodeBlock title="Install tectonic (LaTeX compiler)" code="brew install tectonic" />
          <CodeBlock title="Compile a paper" code={`cd papers/hasse-geometry
tectonic hasse_geometry.tex`} />
        </div>
      </div>

      {/* Stack */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Technology Stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {[
            { name: "Python 3.9+", desc: "Core language" },
            { name: "NumPy / SciPy", desc: "Numerical computation" },
            { name: "Matplotlib", desc: "Visualization" },
            { name: "CLASS", desc: "Cosmological Boltzmann code" },
            { name: "Tectonic", desc: "LaTeX compilation" },
            { name: "MCMC", desc: "Custom Markov chain Monte Carlo" },
          ].map((item) => (
            <div key={item.name} className="bg-gray-50 rounded p-3 border border-gray-100">
              <div className="text-sm font-semibold text-gray-700">{item.name}</div>
              <div className="text-xs text-gray-400">{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Citation */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Citation</h3>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-100 font-mono text-xs text-gray-600 whitespace-pre leading-relaxed">
{`@software{loftus2026quantumgravity,
  author = {Loftus, Matt},
  title  = {Computational Quantum Gravity:
            Causal Sets, CDT, and Beyond},
  year   = {2026},
  url    = {https://github.com/MattLoftus/quantum-gravity}
}`}
        </div>
      </div>
    </div>
  )
}

function CodeBlock({ title, code }: { title: string; code: string }) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1">{title}</div>
      <div className="bg-navy rounded-lg p-4 font-mono text-sm text-blue-light whitespace-pre overflow-x-auto">
        {code}
      </div>
    </div>
  )
}
