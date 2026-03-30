import type { RunIndex, RunSummary, ScenarioDetail } from "./types";

const detailCache = new Map<string, ScenarioDetail>();
const CACHE_MAX = 50;
const cacheOrder: string[] = [];

export async function fetchIndex(): Promise<RunIndex> {
  const res = await fetch("/data/index.json");
  if (!res.ok) return { runs: [] };
  return res.json();
}

export async function fetchSummary(
  runId: string
): Promise<RunSummary | null> {
  try {
    const res = await fetch(
      `/data/runs/${encodeURIComponent(runId)}/summary.json`
    );
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchDetail(
  runId: string,
  scenarioId: string
): Promise<ScenarioDetail | null> {
  const key = `${runId}::${scenarioId}`;
  if (detailCache.has(key)) return detailCache.get(key)!;

  try {
    const res = await fetch(
      `/api/detail/${encodeURIComponent(runId)}/${encodeURIComponent(scenarioId)}`
    );
    if (!res.ok) return null;
    const detail: ScenarioDetail = await res.json();

    detailCache.set(key, detail);
    cacheOrder.push(key);
    while (cacheOrder.length > CACHE_MAX) {
      const oldest = cacheOrder.shift()!;
      detailCache.delete(oldest);
    }
    return detail;
  } catch {
    return null;
  }
}

export function formatDuration(s: number | null | undefined): string {
  if (s == null) return "-";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

export function formatPercent(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${(v * 100).toFixed(1)}%`;
}
