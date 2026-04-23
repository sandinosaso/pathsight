import { useEffect, useRef } from "react";
import { createImageViewer } from "@/lib/openseadragon";
import { cn } from "@/lib/utils";

type Props = {
  imageUrl: string | null;
  className?: string;
};

export function OSDViewer({ imageUrl, className = "" }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !imageUrl) return;
    el.innerHTML = "";
    const v = createImageViewer(el, imageUrl);
    return () => v.destroy();
  }, [imageUrl]);

  if (!imageUrl) {
    return (
      <div
        className={cn(
          "flex h-full min-h-[320px] items-center justify-center rounded-xl border border-slate-800 bg-slate-900/50 text-slate-500",
          className,
        )}
      >
        Upload an image to explore with zoom/pan.
      </div>
    );
  }

  return <div ref={ref} className={cn("pathsight-osd h-full min-h-[320px] w-full", className)} />;
}
