/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "openseadragon" {
  const OpenSeadragon: (options: Record<string, unknown>) => { destroy: () => void };
  export default OpenSeadragon;
}
