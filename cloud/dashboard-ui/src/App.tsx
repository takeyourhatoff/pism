import { useCallback, useEffect, useMemo, useState } from "react";
import { getJobDetail, getJobs, getPreviewMeta } from "./api";
import type { JobDetail, JobSummary, PreviewMeta } from "./types";
import {
  buildStatusLine,
  formatArgs,
  formatBytes,
  formatCurrency,
  formatDuration,
  formatEta,
  formatPercent,
  formatRate,
  formatResources,
  formatTimestamp,
  timelineLines,
} from "./utils";

const STATUS_STYLES: Record<string, string> = {
  RUNNING: "bg-emerald-100 text-emerald-700",
  SUCCEEDED: "bg-sky-100 text-sky-700",
  FAILED: "bg-rose-100 text-rose-700",
  PENDING: "bg-amber-100 text-amber-700",
  RUNNABLE: "bg-amber-100 text-amber-700",
  STARTING: "bg-amber-100 text-amber-700",
  SUBMITTED: "bg-slate-100 text-slate-700",
};

function statusBadge(status?: string | null) {
  const value = status || "UNKNOWN";
  const classes = STATUS_STYLES[value] || "bg-slate-100 text-slate-700";
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-semibold ${classes}`}>
      {value}
    </span>
  );
}

function useQueryJob(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get("job");
}

function setJobInUrl(jobName: string | null) {
  const url = new URL(window.location.href);
  if (jobName) {
    url.searchParams.set("job", jobName);
  } else {
    url.searchParams.delete("job");
  }
  window.history.replaceState({}, "", url.toString());
}

export default function App() {
  const initialJob = useQueryJob();
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJob, setSelectedJob] = useState<string | null>(initialJob);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [previewMeta, setPreviewMeta] = useState<PreviewMeta | null>(null);
  const [previewVar, setPreviewVar] = useState<string | null>(null);
  const [previewStamp, setPreviewStamp] = useState<number>(0);
  const [previewSource, setPreviewSource] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const refreshAll = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const jobsData = await getJobs();
      setJobs(jobsData);

      if (!selectedJob) {
        setDetail(null);
        setPreviewMeta(null);
        setRefreshing(false);
        return;
      }

      const jobExists = jobsData.some((job) => job.job_name === selectedJob);
      if (!jobExists) {
        setSelectedJob(null);
        setDetail(null);
        setPreviewMeta(null);
        setJobInUrl(null);
        setRefreshing(false);
        return;
      }

      const [detailData, meta] = await Promise.all([
        getJobDetail(selectedJob),
        getPreviewMeta(selectedJob),
      ]);
      setDetail(detailData);
      setPreviewMeta(meta);

      if (meta.available) {
        const variables = meta.variables || [];
        const nextVar =
          previewVar && variables.includes(previewVar)
            ? previewVar
            : meta.default_var || variables[0] || null;
        if (nextVar !== previewVar) {
          setPreviewVar(nextVar);
          if (nextVar) {
            setPreviewStamp(Date.now());
          }
        }
        const sourceKey = meta.source_key || null;
        if (sourceKey && sourceKey !== previewSource) {
          setPreviewSource(sourceKey);
          setPreviewStamp(Date.now());
        }
      } else {
        setPreviewVar(null);
      }
    } catch (err) {
      setError("Failed to refresh dashboard data.");
    } finally {
      setRefreshing(false);
    }
  }, [previewSource, previewVar, selectedJob]);

  useEffect(() => {
    refreshAll();
    const interval = window.setInterval(refreshAll, 15000);
    return () => window.clearInterval(interval);
  }, [refreshAll]);

  const handleSelectJob = (jobName: string) => {
    setSelectedJob(jobName);
    setPreviewVar(null);
    setPreviewMeta(null);
    setPreviewSource(null);
    setPreviewStamp(Date.now());
    setJobInUrl(jobName);
  };

  const selectedSummary = jobs.find((job) => job.job_name === selectedJob) || null;
  const previewUrl = useMemo(() => {
    if (!selectedJob || !previewVar) return null;
    return `/api/jobs/${encodeURIComponent(selectedJob)}/preview.png?var=${encodeURIComponent(
      previewVar,
    )}&v=${previewStamp}`;
  }, [previewStamp, previewVar, selectedJob]);

  const argsLines = useMemo(() => formatArgs(detail?.pism_args), [detail?.pism_args]);
  const timeline = useMemo(() => timelineLines(detail?.timeline), [detail?.timeline]);

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-10">
      <header className="mb-6 flex flex-col gap-4 rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-glow backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">PISM Cloud</p>
            <h1 className="text-3xl font-semibold text-slate-900">Simulation Dashboard</h1>
          </div>
          <div className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            {refreshing ? "Refreshing" : "Up to date"}
          </div>
        </div>
        {error ? (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        ) : null}
        {selectedSummary ? (
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
            {statusBadge(detail?.status || selectedSummary.status)}
            <span>{buildStatusLine(detail || undefined)}</span>
            <span>·</span>
            <span>{formatResources(detail || selectedSummary)}</span>
            {detail?.actual_instance_type ? (
              <span>
                · Actual {detail.actual_instance_type}
                {detail.actual_availability_zone ? ` (${detail.actual_availability_zone})` : ""}
              </span>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-slate-500">Select a job to see its current state.</p>
        )}
      </header>

      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <section className="rounded-3xl border border-slate-200/70 bg-white/70 p-4 shadow-glow backdrop-blur">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">Jobs</h2>
            <span className="text-xs text-slate-400">{jobs.length} total</span>
          </div>
          <div className="flex flex-col gap-3">
            {jobs.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-center text-sm text-slate-400">
                No jobs yet.
              </div>
            ) : (
              jobs.map((job) => {
                const isActive = job.job_name === selectedJob;
                return (
                  <button
                    key={job.job_name}
                    type="button"
                    onClick={() => handleSelectJob(job.job_name)}
                    className={`flex flex-col gap-2 rounded-2xl border px-4 py-3 text-left transition ${
                      isActive
                        ? "border-slate-900 bg-slate-900 text-white shadow-lg"
                        : "border-slate-200 bg-white text-slate-900 hover:border-slate-400"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className={`text-sm font-semibold ${isActive ? "text-white" : "text-slate-900"}`}>
                        {job.job_name}
                      </span>
                      {statusBadge(job.status)}
                    </div>
                    <div className={`text-xs ${isActive ? "text-slate-200" : "text-slate-500"}`}>
                      ETA {formatEta(job.eta_seconds)} · Cost {formatCurrency(job.cost_to_date_usd)}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </section>

        <section className="flex flex-col gap-6">
          <div className="grid gap-6 xl:grid-cols-2">
            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Status</h3>
              {detail ? (
                <div className="grid gap-3 text-sm text-slate-700">
                  <div className="flex items-center justify-between">
                    <span>Status</span>
                    <span className="font-semibold text-slate-900">{buildStatusLine(detail)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Instance</span>
                    <span className="font-semibold text-slate-900">
                      {detail.actual_instance_type
                        ? `${detail.actual_instance_type}${
                            detail.actual_availability_zone
                              ? ` (${detail.actual_availability_zone})`
                              : ""
                          }`
                        : "-"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Requested</span>
                    <span className="font-semibold text-slate-900">{formatResources(detail)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Spot</span>
                    <span className="font-semibold text-slate-900">
                      {selectedSummary?.use_spot ? "yes" : "no"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Runtime</span>
                    <span className="font-semibold text-slate-900">{formatDuration(detail.runtime_seconds)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Progress</span>
                    <span className="font-semibold text-slate-900">
                      {formatPercent(detail.progress_percent)} · ETA {formatEta(detail.eta_seconds)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Logs</span>
                    {detail.log_url ? (
                      <a
                        className="font-semibold text-sky-600 hover:text-sky-800"
                        href={detail.log_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        CloudWatch
                      </a>
                    ) : (
                      <span className="font-semibold text-slate-900">-</span>
                    )}
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  Pick a job to see status details.
                </div>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Cost</h3>
              {detail ? (
                <div className="grid gap-3 text-sm text-slate-700">
                  <div className="flex items-center justify-between">
                    <span>Cost per hour</span>
                    <span className="font-semibold text-slate-900">{formatRate(detail.hourly_rate_usd)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Cost so far</span>
                    <span className="font-semibold text-slate-900">{formatCurrency(detail.cost_to_date_usd)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Estimated total</span>
                    <span className="font-semibold text-slate-900">
                      {formatCurrency(detail.estimated_total_cost_usd)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Budget</span>
                    <span className="font-semibold text-slate-900">{formatCurrency(detail.budget_usd)}</span>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  Pick a job to see cost estimates.
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Data</h3>
              {detail ? (
                <div className="space-y-4 text-sm text-slate-700">
                  {detail.inputs && Object.keys(detail.inputs).length ? (
                    <div className="space-y-3">
                      {Object.entries(detail.inputs).map(([name, value]) => {
                        const link = detail.inputs_console?.[name] || value;
                        return (
                          <div key={name} className="flex flex-col gap-1">
                            <span className="text-xs uppercase tracking-[0.2em] text-slate-400">{name}</span>
                            <a
                              href={link}
                              target="_blank"
                              rel="noreferrer"
                              className="break-all font-mono text-xs text-sky-600 hover:text-sky-800"
                            >
                              {value}
                            </a>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="text-slate-400">No inputs listed.</p>
                  )}

                  {detail.output_s3 ? (
                    <div>
                      <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Output</span>
                      <a
                        href={detail.output_console_url || detail.output_s3}
                        target="_blank"
                        rel="noreferrer"
                        className="block break-all font-mono text-xs text-sky-600 hover:text-sky-800"
                      >
                        {detail.output_s3}
                      </a>
                    </div>
                  ) : null}

                  {detail.output_objects && detail.output_objects.length ? (
                    <div className="mt-4 space-y-2">
                      <div className="grid grid-cols-[1fr_auto_auto] gap-3 text-xs uppercase tracking-[0.2em] text-slate-400">
                        <span>Output file</span>
                        <span>Size</span>
                        <span>Modified</span>
                      </div>
                      {detail.output_objects.map((obj) => (
                        <div
                          key={obj.key}
                          className="grid grid-cols-[1fr_auto_auto] gap-3 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-2"
                        >
                          <span className="break-all font-mono text-xs text-slate-700">{obj.key}</span>
                          <span className="text-xs text-slate-600">{formatBytes(obj.size_bytes)}</span>
                          <span className="text-xs text-slate-400">{formatTimestamp(obj.last_modified)}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  Pick a job to see data paths.
                </div>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Preview</h3>
              {selectedJob && previewMeta ? (
                previewMeta.available ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                      <span className="uppercase tracking-[0.2em]">Variable</span>
                      <select
                        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold"
                        value={previewVar || ""}
                        onChange={(event) => {
                          const value = event.target.value;
                          setPreviewVar(value);
                          setPreviewStamp(Date.now());
                        }}
                      >
                        {(previewMeta.variables || []).map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                      {previewMeta.source_key ? (
                        <span className="font-mono text-[10px] text-slate-400">{previewMeta.source_key}</span>
                      ) : null}
                    </div>
                    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
                      {previewUrl ? (
                        <img
                          src={previewUrl}
                          alt="NetCDF preview"
                          className="h-[320px] w-full object-contain"
                        />
                      ) : (
                        <div className="flex h-[320px] items-center justify-center text-sm text-slate-400">
                          Select a variable to preview.
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                    {previewMeta.message || "No preview available yet."}
                  </div>
                )
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  Pick a job to see NetCDF previews.
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">History</h3>
              {timeline.length ? (
                <div className="max-h-72 space-y-2 overflow-auto rounded-2xl border border-slate-100 bg-slate-50 p-4 font-mono text-xs text-slate-600">
                  {timeline.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  No events yet.
                </div>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-5 shadow-glow backdrop-blur">
              <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Args</h3>
              {argsLines.length ? (
                <div className="max-h-72 overflow-auto rounded-2xl border border-slate-100 bg-slate-50 p-4 font-mono text-xs text-slate-600">
                  {argsLines.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-400">
                  No args recorded.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
