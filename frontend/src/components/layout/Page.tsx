import type { ReactNode } from "react";

export function Page({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-white">PathSight</h1>
            <p className="text-sm text-slate-400">AI-assisted histology patch triage (demo)</p>
          </div>
          <span className="rounded-full bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-300">
            CNN · Patch Classifier
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-[1600px] px-4 py-6">{children}</main>
    </div>
  );
}
