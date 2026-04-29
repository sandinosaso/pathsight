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

  // Scroll-zoom is opt-in: activated by clicking inside, deactivated on mouse-leave.
  // This prevents the viewer from hijacking normal page scroll when the user
  // is just trying to scroll past it.
  const scrollZoomActive = useRef(false);
  const [zoomEngaged, setZoomEngaged] = useState(false); // mirrors the ref for rendering
  const [hovered, setHovered] = useState(false);

  const reset = useCallback(() => setTransform({ zoom: 1, x: 0, y: 0 }), []);

  useEffect(() => {
    onResetRef?.(reset);
  }, [onResetRef, reset]);

  useEffect(() => {
    reset();
  }, [imageUrl, reset]);

  // Attach wheel listener as non-passive so preventDefault can work.
  // We only cancel the event (and zoom) when the user has clicked inside.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (!scrollZoomActive.current) return; // let the page scroll normally

      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left - rect.width / 2;
      const my = e.clientY - rect.top - rect.height / 2;

      setTransform((prev) => {
        const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev.zoom * factor));
        const scale = newZoom / prev.zoom;
        return {
          zoom: newZoom,
          x: mx - scale * (mx - prev.x),
          y: my - scale * (my - prev.y),
        };
      });
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const engageZoom = useCallback(() => {
    scrollZoomActive.current = true;
    setZoomEngaged(true);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    engageZoom();
    dragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  }, [engageZoom]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
  }, []);

  const onMouseLeave = useCallback(() => {
    dragging.current = false;
    scrollZoomActive.current = false;
    setZoomEngaged(false);
    setHovered(false);
  }, []);

  const onMouseEnter = useCallback(() => {
    if (imageUrl) setHovered(true);
  }, [imageUrl]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden rounded-xl border bg-slate-900 select-none transition-[border-color] duration-150",
        zoomEngaged ? "border-blue-500/60" : "border-slate-800",
        imageUrl ? "cursor-grab active:cursor-grabbing" : "",
        className,
      )}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={() => { dragging.current = false; }}
      onMouseLeave={onMouseLeave}
      onMouseEnter={onMouseEnter}
      onDoubleClick={reset}
    >
      {imageUrl ? (
        <>
          {/* Transform container — both images always move together */}
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

          {/* Interaction hint chip */}
          {hovered && (
            <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-3 py-1 text-xs text-slate-300 backdrop-blur-sm transition-opacity">
              {zoomEngaged
                ? "Scroll to zoom · Drag to pan · Double-click to reset"
                : "Click to enable zoom"}
            </div>
          )}
        </>
      ) : (
        <div className="flex h-full items-center justify-center text-sm text-slate-500">
          Upload an image to see the Grad-CAM overlay.
        </div>
      )}
    </div>
  );
}
