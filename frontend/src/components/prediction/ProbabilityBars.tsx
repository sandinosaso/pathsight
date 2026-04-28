type Props = {
  probabilities: Record<string, number> | null;
};

const ORDER = ["cancer", "no-cancer"];

const BAR_COLORS: Record<string, string> = {
  cancer: "bg-gradient-to-r from-red-700 to-rose-500",
  "no-cancer": "bg-gradient-to-r from-emerald-700 to-green-400",
};

export function ProbabilityBars({ probabilities }: Props) {
  if (!probabilities) {
    return (
      <div className="flex h-full min-h-[9rem] items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-base text-slate-500">
        Class probabilities will appear here.
      </div>
    );
  }

  const entries = ORDER.map((key) => [key, probabilities[key] ?? 0] as [string, number]).concat(
    Object.entries(probabilities).filter(([k]) => !ORDER.includes(k)),
  );

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-5">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Class probabilities</p>
      {entries.map(([k, v]) => (
        <div key={k}>
          <div className="mb-2 flex justify-between items-baseline">
            <span className="text-base font-medium capitalize text-slate-300">{k}</span>
            <span className="text-2xl font-bold tabular-nums text-slate-100">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full rounded-full transition-all duration-500 ${BAR_COLORS[k] ?? "bg-gradient-to-r from-sky-600 to-cyan-400"}`}
              style={{ width: `${Math.min(100, v * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
