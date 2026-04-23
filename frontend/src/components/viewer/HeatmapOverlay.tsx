type Props = {
  overlayUrl: string | null;
  opacity: number;
  visible: boolean;
};

export function HeatmapOverlay({ overlayUrl, opacity, visible }: Props) {
  if (!overlayUrl || !visible) return null;
  return (
    <img
      src={overlayUrl}
      alt="Grad-CAM overlay"
      className="pointer-events-none absolute inset-0 h-full w-full object-contain"
      style={{ opacity }}
    />
  );
}
