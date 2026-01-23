export type ModelType = "gaussian_nb";

export type RunCreateRequest = {
  datasetName: string;
  targetColumn: string;
  featureColumns: string[];
  testSize: number;
  randomState?: number;
  modelType: ModelType;
  gnbParams: {
    varSmoothing?: number;
  };
};

export type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
};

export type RunResult = {
  id: string;
  createdAt: string;
  request: RunCreateRequest;
  metrics: Metrics;
  confusionMatrix: number[][];
  labels: string[];
};

export type RunListResponse = {
  items: RunResult[];
  total: number;
};
