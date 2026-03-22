import { useState } from 'react'
import { Hero } from './components/Hero'
import { PapersTab } from './components/PapersTab'
import { KeyResultsTab } from './components/KeyResultsTab'
import { ExperimentsTab } from './components/ExperimentsTab'
import { GuideTab } from './components/GuideTab'
import { CodeTab } from './components/CodeTab'

const tabs = [
  { id: 'papers', label: 'Papers' },
  { id: 'results', label: 'Key Results' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'guide', label: 'Guide' },
  { id: 'code', label: 'Code & Data' },
] as const

type TabId = typeof tabs[number]['id']

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('papers')

  return (
    <div className="min-h-screen bg-white">
      <Hero />

      <nav className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-5xl mx-auto px-4">
          <div className="flex gap-0 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-5 py-3.5 text-sm font-medium whitespace-nowrap transition-colors border-b-2 cursor-pointer ${
                  activeTab === tab.id
                    ? 'text-blue border-blue'
                    : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 py-10">
        {activeTab === 'papers' && <PapersTab />}
        {activeTab === 'results' && <KeyResultsTab />}
        {activeTab === 'experiments' && <ExperimentsTab />}
        {activeTab === 'guide' && <GuideTab />}
        {activeTab === 'code' && <CodeTab />}
      </main>

      <footer className="border-t border-gray-200 py-8 mt-16">
        <div className="max-w-5xl mx-auto px-4 text-center text-sm text-gray-400">
          <p>Computational Quantum Gravity Research Project</p>
          <p className="mt-1">600 experiments -- 10 papers -- 72,000 lines of Python</p>
          <p className="mt-1">Matt Loftus, 2024-2026</p>
        </div>
      </footer>
    </div>
  )
}

export default App
