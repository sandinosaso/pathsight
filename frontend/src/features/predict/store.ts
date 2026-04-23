import { createContext, useContext } from "react";
import type { PredictionResponse } from "@/types/api";

export type PredictContextValue = {
  prediction: PredictionResponse | null;
  setPrediction: (p: PredictionResponse | null) => void;
};

export const PredictContext = createContext<PredictContextValue | null>(null);

export function usePredictContext() {
  const v = useContext(PredictContext);
  if (!v) throw new Error("PredictContext missing");
  return v;
}
