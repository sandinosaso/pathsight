import type { PredictionResponse } from "@/types/api";

type Props = {
  data: PredictionResponse | null;
};

export function PredictionCard({ data }: Props) {
  if (!data) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4 text-sm text-slate-500">
        Run inference to see predicted label and confidence.
      </div>
    );
  }
  const isTumor = data.predicted_label === "tumor";
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">Prediction</p>
      <p className={`mt-1 text-2xl font-semibold ${isTumor ? "text-rose-300" : "text-emerald-300"}`}>
        {isTumor ? "Suspicious" : "Not suspicious"}
      </p>
      <p className="mt-1 text-sm text-slate-400">
        Display label: <span className="text-slate-200">{data.predicted_label}</span> · Confidence{" "}
        <span className="text-slate-200">{(data.confidence * 100).toFixed(1)}%</span>
      </p>
      <p className="mt-2 text-xs text-slate-500">Model: {data.meta.model_name}</p>
    </div>
  );
}
