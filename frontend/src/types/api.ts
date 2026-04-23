export type PredictionMeta = {
  input_size: number[];
  model_name: string;
  gradcam_layer?: string | null;
};

export type PredictionResponse = {
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  heatmap_base64: string;
  overlay_base64: string;
  original_base64: string;
  meta: PredictionMeta;
};

export type ExampleItem = {
  id: string;
  filename: string;
  label?: string | null;
  description?: string | null;
  image_url: string;
};

export type ExamplesResponse = {
  examples: ExampleItem[];
};
