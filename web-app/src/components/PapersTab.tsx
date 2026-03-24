import { useState } from 'react'
import { papers } from '../data'

function PaperCard({ paper }: { paper: typeof papers[number] }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start gap-2.5 mb-2">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded bg-navy text-white text-xs font-semibold shrink-0 mt-0.5">
          {paper.letter}
        </span>
        <h3 className="text-base font-semibold text-gray-900 leading-snug">
          {paper.title}
        </h3>
      </div>

      <p className="text-sm text-gray-600 mb-3 leading-relaxed">
        {paper.keyClaim}
      </p>

      <div className="text-xs text-gray-400">
        {paper.pages} pages
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
  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Research Papers</h2>
        <p className="text-gray-500 text-sm">
          10 papers written across causal set quantum gravity, covering the Sorkin-Johnston vacuum,
          Benincasa-Dowker phase transitions, spectral graph theory, exact combinatorics, and
          cross-approach comparisons with CDT.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {papers.map(p => <PaperCard key={p.id} paper={p} />)}
      </div>
    </div>
  )
}
