import type { ReactNode } from "react";

export function Sidebar({ children }: { children: ReactNode }) {
  return (
    <aside className="flex w-full flex-col gap-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4 lg:w-[28rem]">
      {children}
    </aside>
  );
}
