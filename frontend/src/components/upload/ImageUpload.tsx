import { useCallback, useRef } from "react";
import { Button } from "@/components/common/Button";
import { cn } from "@/lib/utils";

type Props = {
  onFile: (file: File) => void;
  disabled?: boolean;
};

export function ImageUpload({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = useCallback(() => inputRef.current?.click(), []);

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) onFile(f);
      e.target.value = "";
    },
    [onFile],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) onFile(f);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 p-8 text-center",
        disabled && "opacity-50",
      )}
    >
      <p className="text-sm text-slate-400">Drop a patch image (PNG / JPEG / WebP) or choose a file.</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={onChange}
        disabled={disabled}
      />
      <Button type="button" onClick={onPick} disabled={disabled}>
        Choose image
      </Button>
    </div>
  );
}
