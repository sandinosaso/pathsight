import { Button } from "@/components/common/Button";
import type { ExampleItem } from "@/types/api";

type Props = {
  items: ExampleItem[];
  loading: boolean;
  error: string | null;
  onSelect: (url: string, filename: string) => void;
};

export function ExampleGallery({ items, loading, error, onSelect }: Props) {
  if (loading) return <p className="text-xs text-slate-500">Loading examples…</p>;
  if (error) return <p className="text-xs text-rose-400">{error}</p>;
  if (!items.length) return <p className="text-xs text-slate-500">No examples in manifest.</p>;

  return (
    <div>
      <p className="mb-2 text-xs uppercase tracking-wide text-slate-500">Example patches</p>
      <div className="grid grid-cols-4 gap-2">
        {items.map((ex) => (
          <div key={ex.id} className="rounded-lg border border-slate-800 bg-slate-900/40 p-2">
            <img
              src={ex.image_url}
              alt={ex.filename}
              className="mb-1 h-16 w-full rounded object-cover"
            />
            <Button
              type="button"
              variant="ghost"
              className="w-full text-xs"
              onClick={() => onSelect(ex.image_url, ex.filename)}
            >
              Use sample
            </Button>
            {ex.label && <p className="mt-1 text-center text-[10px] text-slate-500">Label: {ex.label}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
