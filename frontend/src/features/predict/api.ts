import { apiBase } from "@/lib/apiBase";
import type { PredictionResponse } from "@/types/api";

export async function postPredict(file: File): Promise<PredictionResponse> {
  const fd = new FormData();
  fd.append("img", file);
  const res = await fetch(`${apiBase()}/predict`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<PredictionResponse>;
}
