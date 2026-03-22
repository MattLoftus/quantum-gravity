export function ScoreBadge({ score }: { score: number }) {
  let bg: string, text: string
  if (score >= 8) {
    bg = 'bg-green-50'
    text = 'text-green'
  } else if (score >= 7) {
    bg = 'bg-blue-50'
    text = 'text-blue'
  } else {
    bg = 'bg-gray-100'
    text = 'text-gray-500'
  }

  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold ${bg} ${text}`}>
      {score.toFixed(1)}
    </span>
  )
}
