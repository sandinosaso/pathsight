import { useCallback, useState } from "react";
import { postPredict } from "./api";
import type { PredictionResponse } from "@/types/api";

export function usePrediction() {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const r = await postPredict(file);
      setData(r);
      return r;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Prediction failed";
      setError(msg);
      setData(null);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, loading, error, run, reset };
}
