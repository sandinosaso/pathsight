import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type Transform = { zoom: number; x: number; y: number };

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const ZOOM_FACTOR = 1.05;

type Props = {
  imageUrl: string | null;
  overlayUrl: string | null;
  overlayOpacity: number;
  overlayVisible: boolean;
  onResetRef?: (resetFn: () => void) => void;
  className?: string;
};

export function ZoomableViewer({
  imageUrl,
  overlayUrl,
  overlayOpacity,
  overlayVisible,
  onResetRef,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState<Transform>({ zoom: 1, x: 0, y: 0 });
  const dragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  const reset = useCallback(() => setTransform({ zoom: 1, x: 0, y: 0 }), []);

  // Expose reset to parent via callback ref
  useEffect(() => {
    onResetRef?.(reset);
  }, [onResetRef, reset]);

  // Reset when a new image is loaded
  useEffect(() => {
    reset();
  }, [imageUrl, reset]);

  // Attach wheel listener as non-passive so preventDefault works
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      // Mouse position relative to container centre
      const mx = e.clientX - rect.left - rect.width / 2;
      const my = e.clientY - rect.top - rect.height / 2;

      setTransform((prev) => {
        const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev.zoom * factor));
        const scale = newZoom / prev.zoom;
        return {
          zoom: newZoom,
          // Zoom towards mouse: adjust pan so the point under the cursor stays fixed
          x: mx - scale * (mx - prev.x),
          y: my - scale * (my - prev.y),
        };
      });
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
  }, []);

  const stopDrag = useCallback(() => { dragging.current = false; }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden rounded-xl border border-slate-800 bg-slate-900 select-none",
        imageUrl ? "cursor-grab active:cursor-grabbing" : "",
        className,
      )}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
      onDoubleClick={reset}
      title="Scroll to zoom · Drag to pan · Double-click to reset"
    >
      {imageUrl ? (
        /* Both images share this transform container — they always move together */
        <div
          style={{
            position: "relative",
            width: "100%",
            height: "100%",
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.zoom})`,
            transformOrigin: "center center",
            willChange: "transform",
          }}
        >
          <img
            src={imageUrl}
            alt="Uploaded patch"
            draggable={false}
            className="h-full w-full object-contain"
          />
          {overlayUrl && overlayVisible && (
            <img
              src={overlayUrl}
              alt="Grad-CAM overlay"
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              style={{ opacity: overlayOpacity }}
            />
          )}
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-sm text-slate-500">
          Upload an image to see the Grad-CAM overlay.
        </div>
      )}
    </div>
  );
}
