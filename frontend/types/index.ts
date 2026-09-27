export interface HealthStatus {
  status: string;
  project: string;
  full_name: string;
  version: string;
  environment: string;
  uptime_seconds: number;
  timestamp: string;
  device: string;
  models_loaded: string[];
  dataset_version: string;
}

export interface CityInfo {
  city_id: string;
  name: string;
  state: string;
  latitude: number;
  longitude: number;
  data_start: string;
  data_end: string;
  total_records: number;
  completeness_score: number;
}

export interface PollutantReadings {
  pm25?: number;
  pm10?: number;
  no?: number;
  no2?: number;
  nox?: number;
  nh3?: number;
  co?: number;
  so2?: number;
  o3?: number;
  benzene?: number;
  toluene?: number;
  xylene?: number;
}

export interface ApiErrorResponse {
  type: string;
  title: string;
  status: number;
  detail?: string;
  instance?: string;
  invalid_params?: Array<{ name: string; reason: string }>;
}
