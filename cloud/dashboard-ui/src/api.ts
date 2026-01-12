import type { JobDetail, JobSummary, PreviewMeta } from "./types";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function getJobs(): Promise<JobSummary[]> {
  return fetchJson<JobSummary[]>("/api/jobs");
}

export async function getJobDetail(jobName: string): Promise<JobDetail> {
  return fetchJson<JobDetail>(`/api/jobs/${encodeURIComponent(jobName)}`);
}

export async function getPreviewMeta(jobName: string): Promise<PreviewMeta> {
  return fetchJson<PreviewMeta>(`/api/jobs/${encodeURIComponent(jobName)}/preview/meta`);
}
