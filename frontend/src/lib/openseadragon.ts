import OpenSeadragon from "openseadragon";

export type ViewerHandle = {
  destroy: () => void;
};

export function createImageViewer(
  element: HTMLElement,
  imageUrl: string,
): ViewerHandle {
  const viewer = OpenSeadragon({
    element,
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
    tileSources: {
      type: "image",
      url: imageUrl,
    },
    showNavigationControl: true,
    showZoomControl: true,
    showHomeControl: true,
    showFullPageControl: false,
    animationTime: 0.2,
    blendTime: 0.1,
    constrainDuringPan: true,
    maxZoomPixelRatio: 4,
    minZoomLevel: 0.2,
    visibilityRatio: 1,
    zoomPerScroll: 1.2,
  });

  return {
    destroy: () => {
      viewer.destroy();
    },
  };
}
