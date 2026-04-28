import { useState } from "react";
import { Button } from "@/components/common/Button";
import { cn } from "@/lib/utils";

type Summary = Record<string, unknown>;

type Props = {
  summary: Summary | null | undefined;
};

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(value: unknown, key?: string): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();

    // Learning rates and similar hyperparameters: show as plain decimal / scientific
    const lrKeys = ["learning_rate", "fine_tune_lr", "lr", "fine_tune_learning_rate"];
    if (key && lrKeys.some((k) => key.toLowerCase().includes(k))) {
      // e.g. 0.001 → "0.001000", 0.00001 → "1e-5"
      return value >= 0.0001 ? value.toPrecision(4) : value.toExponential(1);
    }

    // Timing values larger than 1 — plain decimal
    if (value >= 1) return value.toFixed(2);

    // Fractions / probabilities (0 < x < 1) — show as percentage
    return (value * 100).toFixed(2) + "%";
  }
  return String(value);
}

function labelOf(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

type Row = { label: string; value: string };

function flatRows(obj: Record<string, unknown>): Row[] {
  return Object.entries(obj).map(([k, v]) => ({ label: labelOf(k), value: fmt(v, k) }));
}

// Colour badge for metric rows — green / amber / red based on known metric names
const POSITIVE_METRICS = new Set(["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc", "specificity"]);
const NEGATIVE_METRICS = new Set(["fnr"]);

function metricColor(key: string, numericValue: number | null): string {
  if (numericValue === null) return "";
  const k = key.toLowerCase();
  if (NEGATIVE_METRICS.has(k)) {
    return numericValue < 0.1 ? "text-emerald-400" : numericValue < 0.2 ? "text-amber-400" : "text-rose-400";
  }
  if (POSITIVE_METRICS.has(k)) {
    return numericValue > 0.85 ? "text-emerald-400" : numericValue > 0.70 ? "text-amber-400" : "text-rose-400";
  }
  return "";
}

// ── sub-tables ────────────────────────────────────────────────────────────────

function MetricTable({ title, rows, colorFn }: {
  title: string;
  rows: Row[];
  colorFn?: (label: string, raw: number | null) => string;
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-500">
        {title}
      </h3>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(({ label, value }) => {
            const raw = value.endsWith("%") ? parseFloat(value) / 100 : parseFloat(value);
            const color = colorFn?.(label, isNaN(raw) ? null : raw) ?? "";
            return (
              <tr key={label} className="border-b border-slate-800 last:border-0">
                <td className="py-1.5 pr-4 text-slate-400">{label}</td>
                <td className={cn("py-1.5 text-right font-mono font-medium tabular-nums", color || "text-slate-200")}>
                  {value}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function ModelSummaryPanel({ summary }: Props) {
  const [open, setOpen] = useState(false);

  if (!summary) return null;

  // Top-level scalar fields
  const topRows: Row[] = [];
  if (summary.run_id)         topRows.push({ label: "Run ID",      value: String(summary.run_id) });
  if (summary.backbone)       topRows.push({ label: "Backbone",    value: String(summary.backbone) });
  if (summary.image_size)     topRows.push({ label: "Image size",  value: `${summary.image_size} px` });
  if (summary.params_total)   topRows.push({ label: "Parameters",  value: Number(summary.params_total).toLocaleString() });
  if (summary.test_threshold !== undefined) {
    topRows.push({ label: "Decision threshold", value: fmt(summary.test_threshold) });
  }

  const testRows   = summary.test     ? flatRows(summary.test     as Record<string, unknown>) : [];
  const threshRows = summary.thresholds ? flatRows(summary.thresholds as Record<string, unknown>) : [];
  const timingRows = summary.timing   ? flatRows(summary.timing   as Record<string, unknown>) : [];
  const configRows = summary.config   ? flatRows(summary.config   as Record<string, unknown>) : [];

  return (
    <div className="mt-1">
      <Button
        type="button"
        variant="ghost"
        onClick={() => setOpen((v) => !v)}
        className="gap-2 text-xs"
      >
        <span>{open ? "▲" : "▼"}</span>
        {open ? "Hide model metrics" : "Show model metrics"}
      </Button>

      {open && (
        <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="mb-4 text-sm font-semibold text-slate-200">
            Model summary
          </h2>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">

            {topRows.length > 0 && (
              <MetricTable title="Overview" rows={topRows} />
            )}

            {testRows.length > 0 && (
              <MetricTable
                title="Test metrics"
                rows={testRows}
                colorFn={(label, raw) =>
                  metricColor(label.toLowerCase().replace(/ /g, "_"), raw)
                }
              />
            )}

            {threshRows.length > 0 && (
              <MetricTable title="Thresholds" rows={threshRows} />
            )}

            {timingRows.length > 0 && (
              <MetricTable title="Timing" rows={timingRows} />
            )}

            {configRows.length > 0 && (
              <MetricTable title="Training config" rows={configRows} />
            )}

          </div>
        </div>
      )}
    </div>
  );
}
