type Props = {
  probabilities: Record<string, number> | null;
};

const ORDER = ["cancer", "no-cancer"];

const BAR_COLORS: Record<string, string> = {
  cancer: "bg-gradient-to-r from-red-700 to-rose-500",
  "no-cancer": "bg-gradient-to-r from-emerald-700 to-green-400",
};

export function ProbabilityBars({ probabilities }: Props) {
  if (!probabilities) return null;

  const entries = ORDER.map((key) => [key, probabilities[key] ?? 0] as [string, number])
    .concat(
      Object.entries(probabilities).filter(([k]) => !ORDER.includes(k)),
    );

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
              className={`h-full rounded-full ${BAR_COLORS[k] ?? "bg-gradient-to-r from-sky-600 to-cyan-400"}`}
              style={{ width: `${Math.min(100, v * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
