/** API prefix: dev uses Vite `/api` proxy; Docker nginx rewrites `/api` to backend. */
export function apiBase(): string {
  const raw = import.meta.env.VITE_API_BASE;
  if (raw != null && String(raw).trim() !== "") {
    return String(raw).replace(/\/$/, "");
  }
  return "/api";
}
