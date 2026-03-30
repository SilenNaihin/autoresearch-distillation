"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchIndex, fetchSummary, formatPercent } from "@/lib/data";
import type { RunIndex, RunSummary } from "@/lib/types";
import { Card, Spinner, EmptyState } from "@/components/ui";
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
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [summaries, setSummaries] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchIndex().then(async (idx) => {
      setIndex(idx);
      const trainingRuns = idx.runs.filter((r) => r.training_step != null);
      const results = await Promise.all(
        trainingRuns.map((r) => fetchSummary(r.run_id))
      );
      setSummaries(results.filter(Boolean) as RunSummary[]);
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

  if (loading) return <Spinner />;

  if (!index?.runs.some((r) => r.training_step != null)) {
    return (
      <EmptyState
        title="No training runs"
        description="No runs with training_step found. Training progression will appear here once training runs are synced."
      />
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-lg font-semibold">Training Progression</h1>
        <p className="text-sm text-text-tertiary">
          Performance metrics over training steps
        </p>
      </div>

      {/* Pass Rate Chart */}
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

      {/* Tool Score Chart */}
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

      {/* Per-Category Pass Rates */}
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

      {/* Data table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="border-b border-border px-4 py-3">
          <h2 className="text-sm font-medium">Training Steps</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-text-tertiary">
                <th className="px-4 py-2 font-medium">Step</th>
                <th className="px-4 py-2 font-medium">Pass Rate</th>
                <th className="px-4 py-2 font-medium">Tool Score</th>
                {allCategories.map((cat) => (
                  <th key={cat} className="px-4 py-2 font-medium">{cat}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {chartData.map((row) => (
                <tr key={row.step} className="border-b border-border last:border-0 hover:bg-bg-hover transition-colors">
                  <td className="px-4 py-2.5 tabular-nums font-medium">{row.step}</td>
                  <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                    {formatPercent(row.pass_rate)}
                  </td>
                  <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                    {row.avg_tool_score.toFixed(3)}
                  </td>
                  {allCategories.map((cat) => (
                    <td key={cat} className="px-4 py-2.5 tabular-nums text-text-secondary">
                      {(row as Record<string, unknown>)[`cat_${cat}`] != null
                        ? formatPercent((row as Record<string, number>)[`cat_${cat}`])
                        : "-"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
