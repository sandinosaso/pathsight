import { Button } from "@/components/common/Button";
import { Slider } from "@/components/common/Slider";

type Props = {
  overlayOn: boolean;
  onToggleOverlay: () => void;
  opacity: number;
  onOpacity: (v: number) => void;
  onResetZoom: () => void;
};

export function ViewerControls({ overlayOn, onToggleOverlay, opacity, onOpacity, onResetZoom }: Props) {
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
      <Button type="button" variant="ghost" onClick={onToggleOverlay}>
        {overlayOn ? "Hide Grad-CAM" : "Show Grad-CAM"}
      </Button>
      <Button type="button" variant="ghost" onClick={onResetZoom}>
        Reset zoom
      </Button>
      <div className="min-w-[200px] flex-1">
        <Slider label="Overlay opacity" value={opacity} min={0} max={1} step={0.05} onChange={onOpacity} />
      </div>
    </div>
  );
}
