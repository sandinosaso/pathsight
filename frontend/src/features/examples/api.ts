import { apiBase } from "@/lib/apiBase";
import type { ExamplesResponse } from "@/types/api";

export async function fetchExamples(): Promise<ExamplesResponse> {
  const res = await fetch(`${apiBase()}/examples`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<ExamplesResponse>;
}
