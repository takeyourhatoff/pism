import type { JobDetail, JobSummary, TimelineEvent } from "./types";

export function formatCurrency(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `$${Number(value).toFixed(2)}`;
}

export function formatRate(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `$${Number(value).toFixed(2)}/hr`;
}

export function formatPercent(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Number(value).toFixed(1)}%`;
}

export function formatBytes(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Number(value);
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(1)} ${units[index]}`;
}

export function formatMemoryGiB(mib?: number | string | null): string {
  if (mib === null || mib === undefined) return "-";
  const value = Number(mib);
  if (Number.isNaN(value)) return "-";
  return `${(value / 1024).toFixed(1)} GiB`;
}

export function formatResources(job?: JobSummary | JobDetail | null): string {
  if (!job) return "-";
  const vcpus = job.vcpus ?? "-";
  const gpus = job.gpus ?? 0;
  const mem = formatMemoryGiB(job.memory_mib);
  return `${vcpus} vCPU | ${mem} | ${gpus} GPU`;
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  const totalSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0) {
    return `${minutes}m${String(secs).padStart(2, "0")}s`;
  }
  return `${secs}s`;
}

export function formatEta(seconds?: number | null): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "-";
  const totalMinutes = Math.max(0, Math.round(seconds / 60));
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours > 0) {
    return `${hours}h${String(minutes).padStart(2, "0")}m`;
  }
  return `${minutes}m`;
}

export function formatTimestamp(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const hours = String(date.getUTCHours()).padStart(2, "0");
  const minutes = String(date.getUTCMinutes()).padStart(2, "0");
  const seconds = String(date.getUTCSeconds()).padStart(2, "0");
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} UTC`;
}

export function formatArgs(args?: string | null): string[] {
  if (!args) return [];
  const tokens = args.trim().split(/\s+/);
  if (!tokens.length) return [];
  const lines: string[] = [];
  let current = "";
  tokens.forEach((token) => {
    if (token.startsWith("-")) {
      if (current) lines.push(current);
      current = token;
    } else if (current) {
      current += ` ${token}`;
    } else {
      current = token;
    }
  });
  if (current) lines.push(current);
  return lines;
}

export function buildStatusLine(detail?: JobDetail | null): string {
  if (!detail) return "-";
  const statusLine = detail.status || "-";
  if (detail.status_reason) {
    return `${statusLine} (${detail.status_reason})`;
  }
  return statusLine;
}

export function timelineLines(timeline?: TimelineEvent[]): string[] {
  if (!timeline || !timeline.length) return [];
  return timeline.map((event) => {
    const stamp = formatTimestamp(event.timestamp);
    const detailText = event.detail ? ` - ${event.detail}` : "";
    return `${stamp} ${event.label}${detailText}`;
  });
}
