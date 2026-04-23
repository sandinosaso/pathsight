type Props = {
  probabilities: Record<string, number> | null;
};

export function ProbabilityBars({ probabilities }: Props) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-slate-500">Class probabilities</p>
      {entries.map(([k, v]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between text-xs text-slate-400">
            <span className="capitalize">{k}</span>
            <span>{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-600 to-cyan-400"
              style={{ width: `${Math.min(100, v * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
