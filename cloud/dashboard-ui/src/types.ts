export type JobSummary = {
  job_name: string;
  created_at?: string;
  vcpus?: string | number | null;
  memory_mib?: string | number | null;
  gpus?: string | number | null;
  use_spot?: boolean | null;
  status?: string | null;
  eta_seconds?: number | null;
  cost_to_date_usd?: number | null;
};

export type TimelineEvent = {
  timestamp: string;
  label: string;
  detail?: string;
};

export type JobDetail = {
  job_name: string;
  inputs?: Record<string, string>;
  inputs_console?: Record<string, string>;
  output_s3?: string | null;
  output_console_url?: string | null;
  output_objects?: Array<{
    key?: string;
    size_bytes?: number | null;
    last_modified?: string | null;
  }>;
  pism_args?: string | null;
  gpus?: string | number | null;
  vcpus?: string | number | null;
  memory_mib?: string | number | null;
  actual_instance_type?: string | null;
  actual_availability_zone?: string | null;
  status?: string | null;
  status_reason?: string | null;
  log_url?: string | null;
  hourly_rate_usd?: number | null;
  cost_to_date_usd?: number | null;
  budget_usd?: number | null;
  runtime_seconds?: number | null;
  progress_percent?: number | null;
  eta_seconds?: number | null;
  estimated_total_cost_usd?: number | null;
  timeline?: TimelineEvent[];
};

export type PreviewMeta = {
  available: boolean;
  variables?: string[];
  default_var?: string;
  source_key?: string;
  message?: string;
};
