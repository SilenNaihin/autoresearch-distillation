"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchIndex, fetchSummary } from "@/lib/data";
import type { RunSummary, TrainingRun } from "@/lib/types";
import { Card, Badge, Spinner, EmptyState } from "@/components/ui";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const CHART_COLORS = ["#0070f3", "#0cce6b", "#ee5d50", "#f5a623", "#a855f7", "#06b6d4"];

const tooltipStyle = {
  contentStyle: { background: "#fafafa", border: "1px solid #eaeaea", borderRadius: 8 },
  labelStyle: { color: "#666" },
};

export default function TrainingPage() {
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [summaries, setSummaries] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch training runs live from wandb API route
    Promise.all([
      fetch("/api/training").then((r) => r.ok ? r.json() : []),
      fetchIndex().then(async (idx) => {
        const evalRuns = idx.runs.filter((r) => r.training_step != null);
        const results = await Promise.all(
          evalRuns.map((r) => fetchSummary(r.run_id))
        );
        return results.filter(Boolean) as RunSummary[];
      }),
    ]).then(([runs, sums]) => {
      setTrainingRuns(
        (runs as TrainingRun[]).sort(
          (a, b) => (b.started_at ?? "").localeCompare(a.started_at ?? "")
        )
      );
      setSummaries(sums);
      setLoading(false);
    });
  }, []);

  const chartData = useMemo(() => {
    return summaries
      .filter((s) => s.training_step != null)
      .sort((a, b) => a.training_step! - b.training_step!)
      .map((s) => ({
        step: s.training_step,
        pass_rate: s.aggregates.overall.pass_rate,
        avg_tool_score: s.aggregates.overall.avg_tool_score,
        ...Object.fromEntries(
          Object.entries(s.aggregates.by_category).map(([cat, stats]) => [
            `cat_${cat}`,
            stats.pass_rate,
          ])
        ),
      }));
  }, [summaries]);

  const allCategories = useMemo(() => {
    const cats = new Set<string>();
    summaries.forEach((s) =>
      Object.keys(s.aggregates.by_category).forEach((c) => cats.add(c))
    );
    return [...cats].sort();
  }, [summaries]);

  const hasEvalData = chartData.length > 0;

  if (loading) return <Spinner />;

  if (trainingRuns.length === 0 && !hasEvalData) {
    return (
      <EmptyState
        title="No training runs"
        description="Run viewer_sync.py to discover training runs from GPU boxes."
      />
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-lg font-semibold">Training</h1>
        <p className="text-sm text-text-tertiary">
          Active and completed training runs
        </p>
      </div>

      {/* Training Runs */}
      {trainingRuns.length > 0 && (
        <div className="space-y-3">
          {trainingRuns.map((run) => (
            <TrainingRunCard key={run.run_id} run={run} />
          ))}
        </div>
      )}

      {/* Eval Progression Charts (only if checkpoint evals exist) */}
      {hasEvalData && (
        <>
          <div className="pt-4">
            <h2 className="text-sm font-semibold mb-1">GAIA2 Eval Progression</h2>
            <p className="text-xs text-text-tertiary">
              Pass rate and tool score from checkpoint evaluations
            </p>
          </div>

          <Card>
            <h2 className="text-sm font-medium mb-4">Pass Rate</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eaeaea" />
                <XAxis dataKey="step" stroke="#999" fontSize={12} />
                <YAxis stroke="#999" fontSize={12} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  {...tooltipStyle}
                  formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Pass Rate"]}
                />
                <Line type="monotone" dataKey="pass_rate" stroke="#0070f3" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <h2 className="text-sm font-medium mb-4">Average Tool Score</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eaeaea" />
                <XAxis dataKey="step" stroke="#999" fontSize={12} />
                <YAxis stroke="#999" fontSize={12} domain={[0, 1]} />
                <Tooltip
                  {...tooltipStyle}
                  formatter={(value) => [Number(value).toFixed(3), "Tool Score"]}
                />
                <Line type="monotone" dataKey="avg_tool_score" stroke="#0cce6b" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {allCategories.length > 0 && (
            <Card>
              <h2 className="text-sm font-medium mb-4">Per-Category Pass Rates</h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eaeaea" />
                  <XAxis dataKey="step" stroke="#999" fontSize={12} />
                  <YAxis stroke="#999" fontSize={12} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(value, name) => [
                      `${(Number(value) * 100).toFixed(1)}%`,
                      String(name).replace("cat_", ""),
                    ]}
                  />
                  <Legend formatter={(value: string) => value.replace("cat_", "")} />
                  {allCategories.map((cat, i) => (
                    <Line
                      key={cat}
                      type="monotone"
                      dataKey={`cat_${cat}`}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 2 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

const STATUS_VARIANT: Record<string, "success" | "error" | "warning" | "muted"> = {
  active: "success",
  running: "success",
  finished: "muted",
  complete: "muted",
  crashed: "error",
  failed: "error",
};

function TrainingRunCard({ run }: { run: TrainingRun }) {
  const startDate = run.started_at
    ? new Date(run.started_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : null;

  return (
    <Card>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-sm font-semibold truncate">{run.run_name || run.model_name}</h3>
            {run.wandb_run_id && (
              <span className="font-mono text-[11px] text-text-tertiary">{run.wandb_run_id}</span>
            )}
            <Badge variant={STATUS_VARIANT[run.status] ?? "muted"}>
              {run.status}
            </Badge>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-secondary">
            {run.box && <span>{run.box.replace("h100-dev-", "")}</span>}
            {run.gpu && <span className="text-text-tertiary">{run.gpu}</span>}
            {startDate && <span className="text-text-tertiary">{startDate}</span>}
            {run.config_summary && (
              <span className="text-text-tertiary">{run.config_summary}</span>
            )}
          </div>
        </div>
        {run.wandb_url && (
          <a
            href={run.wandb_url}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 rounded-md border border-border px-3 py-1.5 text-xs font-medium text-accent hover:bg-bg-hover transition-colors"
          >
            wandb
          </a>
        )}
      </div>
    </Card>
  );
}
