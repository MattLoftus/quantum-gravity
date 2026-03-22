import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { categoryAnalysis, scoreDistribution } from '../data'

export function ExperimentsTab() {
  return (
    <div>
      <div className="mb-10">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Experiments</h2>
        <p className="text-gray-500 text-sm">
          600 ideas tested across ~110 experiment files, producing ~72,000 lines of Python.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        {[
          { value: "600", label: "Ideas catalogued" },
          { value: "~360", label: "Fully tested" },
          { value: "~110", label: "Experiment files" },
          { value: "5.7", label: "Mean score" },
        ].map((stat) => (
          <div key={stat.label} className="bg-gray-50 rounded-lg p-4 border border-gray-100 text-center">
            <div className="text-2xl font-semibold text-gray-900">{stat.value}</div>
            <div className="text-xs text-gray-400 mt-1">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Score distribution */}
      <div className="mb-10">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Score Distribution</h3>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={scoreDistribution} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                label={{ value: "Score range", position: "bottom", offset: 5, style: { fontSize: 12, fill: '#9ca3af' } }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                label={{ value: "Number of ideas", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 12, fill: '#9ca3af' } }}
              />
              <Tooltip contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }} />
              <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Category analysis */}
      <div className="mb-10">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">What Worked vs What Didn't</h3>
        <p className="text-sm text-gray-600 mb-4">
          Hit rate = percentage of ideas scoring 7+ in each category.
          Wild card ideas (connecting to unrelated fields) had the highest success rate at 50%.
          SJ vacuum direct calculations were least efficient at 6.1%.
        </p>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={categoryAnalysis}
              layout="vertical"
              margin={{ top: 10, right: 30, bottom: 10, left: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                type="number"
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                label={{ value: "Hit rate (% scoring 7+)", position: "bottom", offset: 0, style: { fontSize: 12, fill: '#9ca3af' } }}
              />
              <YAxis
                type="category"
                dataKey="category"
                tick={{ fontSize: 11, fill: '#9ca3af' }}
                width={160}
              />
              <Tooltip
                contentStyle={{ fontSize: 12, border: '1px solid #e5e7eb', borderRadius: 6 }}
                formatter={(value) => [`${value}%`, 'Hit rate']}
              />
              <Bar dataKey="hitRate" fill="#3b82f6" radius={[0, 3, 3, 0]} barSize={22} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key lessons */}
      <div className="mb-10">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Lessons</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="bg-green-50 rounded-lg p-4 border border-green-100">
            <h4 className="text-sm font-semibold text-green mb-2">What worked</h4>
            <ul className="text-sm text-gray-600 space-y-2">
              <li>Wild card ideas from other fields (50% hit rate)</li>
              <li>Proving exact theorems rather than numerical exploration</li>
              <li>Cross-dimensional comparison (testing in d=2,3,4)</li>
              <li>Null model testing before claiming significance</li>
              <li>Discovery ACCELERATED: second half produced 2.5x more 7+ results</li>
            </ul>
          </div>
          <div className="bg-red-50 rounded-lg p-4 border border-red-100">
            <h4 className="text-sm font-semibold text-red-600 mb-2">What didn't work</h4>
            <ul className="text-sm text-gray-600 space-y-2">
              <li>Direct SJ vacuum property calculations (6.1% hit rate)</li>
              <li>Reproducing continuum physics at toy scale (N=30-200)</li>
              <li>Numerical exploration without theoretical guidance</li>
              <li>Assuming small-N scaling laws hold at large N</li>
              <li>Single dimension estimators (can be adversarially fooled)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Top results table */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Scoring Results (9+)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 text-xs font-semibold text-gray-400 uppercase">Score</th>
                <th className="text-left py-2 px-3 text-xs font-semibold text-gray-400 uppercase">Idea</th>
                <th className="text-left py-2 px-3 text-xs font-semibold text-gray-400 uppercase">Result</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.5</span></td>
                <td className="py-2.5 px-3 text-gray-700">Gram identity universality</td>
                <td className="py-2.5 px-3 text-gray-500">(-Delta^2)_ij = (4/N^2)*kappa_ij exact on ALL partial orders</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.5</span></td>
                <td className="py-2.5 px-3 text-gray-700">Exact eigenvalue formula</td>
                <td className="py-2.5 px-3 text-gray-500">mu_k = cot(pi(2k-1)/(2T)) for CDT Pauli-Jordan operator</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.0</span></td>
                <td className="py-2.5 px-3 text-gray-700">E[f] = 1/2^(d-1) proved</td>
                <td className="py-2.5 px-3 text-gray-500">Ordering fraction for d-orders, simple proof via independence</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.0</span></td>
                <td className="py-2.5 px-3 text-gray-700">Kronecker exact spectrum</td>
                <td className="py-2.5 px-3 text-gray-500">CDT full spectrum predicted to 10^-15 from Kronecker structure</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.0</span></td>
                <td className="py-2.5 px-3 text-gray-700">ER=EPR at d=4, N=50</td>
                <td className="py-2.5 px-3 text-gray-500">r = 0.91 in physically relevant 4D, stronger than d=2</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.0</span></td>
                <td className="py-2.5 px-3 text-gray-700">Quantum superposition of causets</td>
                <td className="py-2.5 px-3 text-gray-500">Genuine quantum interference: superposition entropy 8.9% below classical mixture</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3"><span className="inline-block px-2 py-0.5 bg-green-50 text-green rounded-full text-xs font-semibold">9.0</span></td>
                <td className="py-2.5 px-3 text-gray-700">Prime number causets</td>
                <td className="py-2.5 px-3 text-gray-500">Divisibility poset's Mobius function is the number-theoretic mu(n)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
