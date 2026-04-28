import type { PredictionResponse } from "@/types/api";

type Props = {
  data: PredictionResponse | null;
};

export function PredictionCard({ data }: Props) {
  if (!data) {
    return (
      <div className="flex h-full min-h-[9rem] items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-base text-slate-500">
        Run inference to see the prediction.
      </div>
    );
  }
  const isCancer = data.predicted_label === "cancer";
  return (
    <div className={`rounded-xl border p-6 ${isCancer ? "border-rose-800/60 bg-rose-950/30" : "border-emerald-800/60 bg-emerald-950/30"}`}>
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Prediction</p>
      <p className={`mt-2 text-4xl font-extrabold tracking-tight ${isCancer ? "text-rose-300" : "text-emerald-300"}`}>
        {isCancer ? "Suspicious" : "Not suspicious"}
      </p>
      <div className="mt-4 flex items-baseline gap-1">
        <span className="text-sm text-slate-400">Confidence</span>
        <span className="text-2xl font-bold text-slate-100">{(data.confidence * 100).toFixed(1)}%</span>
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Model: <span className="text-slate-400">{data.meta.model_name}</span>
      </p>
      {data.meta.gradcam_layer && (
        <p className="mt-0.5 text-xs text-slate-600">
          Grad-CAM: <span className="font-mono">{data.meta.gradcam_layer}</span>
        </p>
      )}
    </div>
  );
}
