import { useState } from 'react'
import { papers } from '../data'
import { ScoreBadge } from './ScoreBadge'

function PaperCard({ paper }: { paper: typeof papers[number] }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2.5">
          <span className="inline-flex items-center justify-center w-7 h-7 rounded bg-navy text-white text-xs font-semibold">
            {paper.letter}
          </span>
          <h3 className="text-base font-semibold text-gray-900 leading-snug">
            {paper.title}
          </h3>
        </div>
        <ScoreBadge score={paper.score} />
      </div>

      <p className="text-sm text-gray-600 mb-3 leading-relaxed">
        {paper.keyClaim}
      </p>

      <div className="flex items-center gap-4 text-xs text-gray-400">
        <span>{paper.targetJournal}</span>
        <span>{paper.pages} pages</span>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <p className="text-sm text-gray-600 leading-relaxed">{paper.abstract}</p>
        </div>
      )}

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-3 text-xs text-blue hover:text-blue-light font-medium cursor-pointer"
      >
        {expanded ? 'Show less' : 'Show abstract'}
      </button>
    </div>
  )
}

export function PapersTab() {
  const tier1 = papers.filter(p => p.tier === 1)
  const tier2 = papers.filter(p => p.tier === 2)
  const tier3 = papers.filter(p => p.tier === 3)

  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Research Papers</h2>
        <p className="text-gray-500 text-sm">
          10 papers written, 8 for submission. Scores are honest self-assessments
          on a 1-10 scale where 5 = publishable in a specialist journal
          and 8 = top ~10% of papers in the subfield.
        </p>
      </div>

      <div className="mb-8">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Tier 1 -- Submit first (score 8+)
        </h3>
        <div className="grid gap-4 md:grid-cols-2">
          {tier1.map(p => <PaperCard key={p.id} paper={p} />)}
        </div>
      </div>

      <div className="mb-8">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Tier 2 -- Strong supporting papers (7-7.5)
        </h3>
        <div className="grid gap-4 md:grid-cols-2">
          {tier2.map(p => <PaperCard key={p.id} paper={p} />)}
        </div>
      </div>

      <div className="mb-8">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Tier 3 -- Standalone if desired (below 7)
        </h3>
        <div className="grid gap-4 md:grid-cols-2">
          {tier3.map(p => <PaperCard key={p.id} paper={p} />)}
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-5 border border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Recommended Submission Order</h3>
        <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
          <li><strong>E</strong> (8.0) -- CDT comparison with Kronecker product theorem</li>
          <li><strong>C</strong> (8.0) -- ER=EPR with universal Gram identity</li>
          <li><strong>G</strong> (8.0) -- Exact combinatorics, bridges QG and math communities</li>
          <li><strong>D</strong> (8.0) -- Universal GUE + artifact identification</li>
          <li><strong>B5</strong> (7.5) -- Flagship, broadest narrative</li>
          <li><strong>F</strong> (7.0) -- Hasse geometry with corrected Fiedler scaling</li>
          <li><strong>A</strong> (7.0) -- Interval entropy + 4D three-phase structure</li>
          <li><strong>B2</strong> (5.5) -- Everpresent Lambda, weakest but with Bayes factor</li>
        </ol>
      </div>
    </div>
  )
}
