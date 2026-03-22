import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, LineChart, Line,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'
import { phaseTransitionData, gueData, erEprData, fiedlerData } from '../data'

function SectionHeading({ title, paper, children }: { title: string; paper: string; children: React.ReactNode }) {
  return (
    <div className="mb-12">
      <div className="flex items-center gap-3 mb-2">
        <span className="inline-flex items-center justify-center w-6 h-6 rounded bg-navy text-white text-[10px] font-semibold">
          {paper}
        </span>
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      {children}
    </div>
  )
}

export function KeyResultsTab() {
  return (
    <div>
      <div className="mb-10">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Key Results</h2>
        <p className="text-gray-500 text-sm">
          The top findings from 600 experiments, with interactive charts.
        </p>
      </div>

      {/* Kronecker Product Theorem */}
      <SectionHeading title="Kronecker Product Theorem" paper="E">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          CDT's antisymmetrized causal matrix decomposes exactly as a Kronecker (tensor) product.
          This single equation explains why CDT reproduces continuum quantum field theory while
          causal sets do not: CDT has only floor(T/2) active quantum modes compared to ~N/2 for causal sets.
        </p>
        <div className="bg-navy rounded-lg p-6 mb-4 overflow-x-auto">
          <div className="text-center font-mono text-blue-light text-lg md:text-xl">
            C<sup>T</sup> - C = A<sub>T</sub> &otimes; J
          </div>
          <div className="text-center text-gray-400 text-xs mt-3">
            where A<sub>T</sub> is the T x T antisymmetric tridiagonal matrix and J is the s x s all-ones matrix
          </div>
        </div>
        <div className="bg-navy rounded-lg p-6 mb-4">
          <div className="text-center font-mono text-blue-light text-base">
            &mu;<sub>k</sub> = cot(&pi;(2k-1) / (2T))
          </div>
          <div className="text-center text-gray-400 text-xs mt-3">
            Exact eigenvalue formula, verified to 10<sup>-14</sup> precision
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="bg-gray-50 rounded p-3 border border-gray-100">
            <div className="font-semibold text-gray-700">CDT modes</div>
            <div className="text-gray-500">floor(T/2) ~ O(sqrt(N))</div>
          </div>
          <div className="bg-gray-50 rounded p-3 border border-gray-100">
            <div className="font-semibold text-gray-700">Causal set modes</div>
            <div className="text-gray-500">~N/2 ~ O(N)</div>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          Fragility: adding just 5% disorder doubles c_eff from 1.1 to 2.1. But thinning
          (removing 70% of elements) keeps c ~ 1. It is the structure, not the density, that matters.
        </p>
      </SectionHeading>

      {/* Phase Transition */}
      <SectionHeading title="Phase Transition via Interval Entropy" paper="A">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          Interval entropy H (the Shannon entropy of causal interval sizes) drops from 2.4 to 0.3
          across the Benincasa-Dowker phase transition -- an 87% collapse. The continuum phase has
          diverse causal structure; the crystalline phase is dominated by direct links.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={phaseTransitionData} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="beta"
                label={{ value: "Coupling strength beta", position: "bottom", offset: 5, style: { fontSize: 12, fill: '#9ca3af' } }}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
              />
              <YAxis
                label={{ value: "Interval entropy H", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12, fill: '#9ca3af' } }}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                domain={[0, 2.8]}
              />
              <Tooltip
                contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }}
                formatter={(value) => [Number(value).toFixed(2), 'H']}
              />
              <Line
                type="monotone"
                dataKey="H"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', r: 3 }}
                activeDot={{ r: 5 }}
              />
              <ReferenceLine y={0.3} stroke="#059669" strokeDasharray="5 5" label={{ value: "Crystalline", position: "right", style: { fontSize: 10, fill: '#059669' } }} />
              <ReferenceLine y={2.4} stroke="#3b82f6" strokeDasharray="5 5" label={{ value: "Continuum", position: "right", style: { fontSize: 10, fill: '#3b82f6' } }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-gray-400 mt-2">
          In 4D, a previously unknown three-phase structure emerges: H drops, partially recovers,
          then drops again. Susceptibility chi_S reaches 43,288 at N=70.
        </p>
      </SectionHeading>

      {/* ER=EPR */}
      <SectionHeading title="Discrete ER=EPR" paper="C">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          Quantum entanglement (|W|) is nearly perfectly predicted by causal connectivity (kappa):
          r = 0.88 in 2D, r = 0.91 in 4D. The Gram identity (-Delta^2)_ij = (4/N^2)*kappa_ij
          holds exactly for ALL partial orders.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                type="number"
                dataKey="kappa"
                name="Connectivity"
                label={{ value: "Causal connectivity kappa", position: "bottom", offset: 5, style: { fontSize: 12, fill: '#9ca3af' } }}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
              />
              <YAxis
                type="number"
                dataKey="W"
                name="|W|"
                label={{ value: "|W| (entanglement)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12, fill: '#9ca3af' } }}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
              />
              <Tooltip
                contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }}
                formatter={(value) => [Number(value).toFixed(3)]}
              />
              <Scatter data={erEprData} fill="#3b82f6">
                {erEprData.map((_entry, index) => (
                  <Cell key={index} fill="#3b82f6" />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
          <div className="bg-gray-50 rounded p-3 border border-gray-100 text-center">
            <div className="font-semibold text-gray-700">r = 0.88</div>
            <div className="text-gray-400 text-xs">2D correlation</div>
          </div>
          <div className="bg-gray-50 rounded p-3 border border-gray-100 text-center">
            <div className="font-semibold text-gray-700">r = 0.91</div>
            <div className="text-gray-400 text-xs">4D correlation</div>
          </div>
          <div className="bg-gray-50 rounded p-3 border border-gray-100 text-center">
            <div className="font-semibold text-gray-700">z = 13.1</div>
            <div className="text-gray-400 text-xs">vs null model</div>
          </div>
        </div>
      </SectionHeading>

      {/* GUE Universality */}
      <SectionHeading title="GUE Universality" paper="D">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          The level spacing ratio {"<r>"} stays at 0.57-0.60 across all phases, all system sizes,
          and all dimensions. This matches the GUE (Gaussian Unitary Ensemble) prediction of 0.5996,
          indicating universal quantum chaos in causal set quantum gravity.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={gueData} margin={{ top: 10, right: 30, bottom: 40, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: '#9ca3af' }}
                interval={0}
                angle={-30}
                textAnchor="end"
                height={60}
              />
              <YAxis
                domain={[0.3, 0.65]}
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                label={{ value: "<r> (level spacing ratio)", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }}
                formatter={(value) => [Number(value).toFixed(2), '<r>']}
              />
              <ReferenceLine y={0.5996} stroke="#059669" strokeDasharray="5 5" label={{ value: "GUE = 0.60", position: "right", style: { fontSize: 10, fill: '#059669' } }} />
              <ReferenceLine y={0.39} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "Poisson = 0.39", position: "right", style: { fontSize: 10, fill: '#ef4444' } }} />
              <Bar dataKey="r" fill="#3b82f6" radius={[3, 3, 0, 0]} barSize={32} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </SectionHeading>

      {/* Exact Formulas */}
      <SectionHeading title="Exact Formulas for Random 2-Orders" paper="G">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          Zero-parameter exact results connecting discrete spacetime to harmonic numbers,
          Tracy-Widom distributions, and combinatorial identities.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          {[
            { formula: "E[f] = 1/2", label: "Ordering fraction", note: "Exact for all N. Var[f] = (2N+5)/[18N(N-1)]" },
            { formula: "E[links] = (N+1)H_N - 2N", label: "Expected links", note: "H_N = nth harmonic number" },
            { formula: "antichain ~ 2*sqrt(N)", label: "Maximal antichain width", note: "Tracy-Widom TW_2 fluctuations" },
            { formula: "E[S_Glaser] = 1", label: "Glaser BD action", note: "Universal constant for ALL N >= 2" },
            { formula: "link_frac = 4ln(N)/N", label: "Link fraction", note: "NOT the power law N^-0.72 previously fitted" },
            { formula: "E[max] = H_N", label: "Maximal elements", note: "Equals the harmonic number" },
          ].map((item) => (
            <div key={item.label} className="bg-navy rounded-lg p-4">
              <div className="font-mono text-blue-light text-sm mb-1">{item.formula}</div>
              <div className="text-white text-xs font-medium mb-1">{item.label}</div>
              <div className="text-gray-400 text-[11px]">{item.note}</div>
            </div>
          ))}
        </div>
      </SectionHeading>

      {/* Fiedler Value */}
      <SectionHeading title="Spectral Geometry of Hasse Diagrams" paper="F">
        <p className="text-sm text-gray-600 mb-4 leading-relaxed">
          The Fiedler value (algebraic connectivity) of Hasse diagrams is dramatically larger for
          manifold-like causal sets than for random DAGs, demonstrating that the minimal causal
          structure carries rich geometric information.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={fiedlerData} layout="vertical" margin={{ top: 10, right: 30, bottom: 10, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis
                type="category"
                dataKey="label"
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                width={140}
              />
              <Tooltip
                contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }}
                formatter={(value) => [Number(value).toFixed(3), 'Fiedler value']}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={28}>
                {fiedlerData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
          <div className="bg-gray-50 rounded p-3 border border-gray-100 text-center">
            <div className="font-semibold text-gray-700">50x</div>
            <div className="text-gray-400 text-xs">Fiedler ratio (causet vs DAG)</div>
          </div>
          <div className="bg-gray-50 rounded p-3 border border-gray-100 text-center">
            <div className="font-semibold text-gray-700">R^2 = 0.83-0.91</div>
            <div className="text-gray-400 text-xs">Spectral embedding accuracy</div>
          </div>
        </div>
      </SectionHeading>
    </div>
  )
}
