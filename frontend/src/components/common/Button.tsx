import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
};

export function Button({ className, variant = "primary", ...props }: Props) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-lg px-3 py-2 text-sm font-medium transition",
        variant === "primary" &&
          "bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50",
        variant === "ghost" && "bg-slate-800 text-slate-100 hover:bg-slate-700",
        className,
      )}
      {...props}
    />
  );
}
