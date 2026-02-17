export type HealthResponse = {
  status: string;
  model_loaded: boolean;
};

export type PredictionItem = {
  class_id: number;
  class_name: string;
  confidence: number;
};

export type PredictResponse = {
  image_path: string;
  predictions: PredictionItem[];
};
