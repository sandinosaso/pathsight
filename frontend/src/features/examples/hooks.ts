import { useEffect, useState } from "react";
import { fetchExamples } from "./api";
import type { ExampleItem } from "@/types/api";

export function useExamples() {
  const [items, setItems] = useState<ExampleItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchExamples();
        if (!cancelled) setItems(r.examples);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load examples");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return { items, loading, error };
}
