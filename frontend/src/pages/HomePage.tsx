import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Page } from "@/components/layout/Page";
import { Sidebar } from "@/components/layout/Sidebar";
import { ImageUpload } from "@/components/upload/ImageUpload";
import { ZoomableViewer } from "@/components/viewer/ZoomableViewer";
import { ViewerControls } from "@/components/viewer/ViewerControls";
import { PredictionCard } from "@/components/prediction/PredictionCard";
import { ProbabilityBars } from "@/components/prediction/ProbabilityBars";
import { ExampleGallery } from "@/components/prediction/ExampleGallery";
import { ModelSummaryPanel } from "@/components/prediction/ModelSummaryPanel";
import { Spinner } from "@/components/common/Spinner";
import { usePrediction } from "@/features/predict/hooks";
import { useExamples } from "@/features/examples/hooks";
import { base64ToObjectUrl, revokeUrl } from "@/lib/image";

export function HomePage() {
  const { data, loading, error, run } = usePrediction();
  const { items, loading: exLoading, error: exError } = useExamples();

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [overlayOn, setOverlayOn] = useState(false);
  const [opacity, setOpacity] = useState(0.45);
  const resetZoomRef = useRef<() => void>(() => {});

  useEffect(() => {
    return () => {
      revokeUrl(previewUrl);
      revokeUrl(originalUrl);
      revokeUrl(overlayUrl);
    };
  }, [previewUrl, originalUrl, overlayUrl]);

  useEffect(() => {
    if (!data) {
      setOriginalUrl((u) => { revokeUrl(u); return null; });
      setOverlayUrl((u) => { revokeUrl(u); return null; });
      return;
    }
    // Use the model-input-sized original (always 224×224) so both the base
    // image and the heatmap overlay share the exact same intrinsic dimensions.
    // This guarantees object-contain letterboxes them identically → perfect alignment.
    const orig = base64ToObjectUrl(data.original_base64);
    setOriginalUrl((prev) => { revokeUrl(prev); return orig; });

    const overlay = base64ToObjectUrl(data.overlay_base64);
    setOverlayUrl((prev) => { revokeUrl(prev); return overlay; });
  }, [data]);

  const handleFile = useCallback(
    async (file: File) => {
      revokeUrl(previewUrl);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      await run(file);
    },
    [previewUrl, run],
  );

  const handleExample = useCallback(
    async (url: string, filename: string) => {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || "image/png" });
      await handleFile(file);
    },
    [handleFile],
  );

  const probs = useMemo(() => data?.probabilities ?? null, [data]);

  return (
    <Page>
      <div className="grid gap-6 lg:grid-cols-[1fr_28rem]">

        {/* ── Left column: result → viewer → controls ── */}
        <div className="space-y-4">

          {/* Prediction result — always visible; shows empty state before first run */}
          <div className="grid gap-4 sm:grid-cols-2">
            <PredictionCard data={data} />
            <ProbabilityBars probabilities={probs} />
          </div>

          <ZoomableViewer
            imageUrl={originalUrl ?? previewUrl}
            overlayUrl={overlayUrl}
            overlayOpacity={opacity}
            overlayVisible={overlayOn}
            onResetRef={(fn) => { resetZoomRef.current = fn; }}
            className="h-[520px] w-full"
          />
          <ViewerControls
            overlayOn={overlayOn}
            onToggleOverlay={() => setOverlayOn((v) => !v)}
            opacity={opacity}
            onOpacity={setOpacity}
            onResetZoom={() => resetZoomRef.current()}
          />
          <ModelSummaryPanel summary={data?.model_summary} />
        </div>

        {/* ── Right column: upload → examples ── */}
        <Sidebar>
          <ImageUpload onFile={handleFile} disabled={loading} />
          {loading && (
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Spinner />
              Running model…
            </div>
          )}
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <ExampleGallery items={items} loading={exLoading} error={exError} onSelect={handleExample} />
        </Sidebar>

      </div>
    </Page>
  );
}
