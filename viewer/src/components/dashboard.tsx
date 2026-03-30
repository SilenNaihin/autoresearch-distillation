"use client";

import { useEffect, useState } from "react";
import { fetchIndex, fetchSummary, formatPercent } from "@/lib/data";
import type { RunIndex, RunSummary } from "@/lib/types";
import { StatCard, EmptyState, Spinner } from "./ui";

export function DashboardClient() {
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchIndex().then((idx) => {
      setIndex(idx);
      if (idx.runs.length > 0) {
        setSelectedRun(idx.runs[0].run_id);
      }
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedRun) {
      setSummary(null);
      return;
    }
    setLoading(true);
    fetchSummary(selectedRun).then((s) => {
      setSummary(s);
      setLoading(false);
    });
  }, [selectedRun]);

  if (loading && !index) return <Spinner />;

  if (!index?.runs.length) {
    return (
      <EmptyState
        title="No runs loaded"
        description="Sync runs from your dev boxes with: python scripts/viewer_sync.py --serve"
      />
    );
  }

  const overall = summary?.aggregates?.overall;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Dashboard</h1>
          <p className="text-sm text-text-tertiary">
            Model evaluation overview
          </p>
        </div>
        <select
          value={selectedRun}
          onChange={(e) => setSelectedRun(e.target.value)}
          className="rounded-md border border-border bg-bg-input px-3 py-1.5 text-sm text-text outline-none focus:border-accent"
        >
          {index.runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.model_name} ({r.timestamp?.slice(0, 10)})
            </option>
          ))}
        </select>
      </div>

      {loading ? (
        <Spinner />
      ) : summary ? (
        <>
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              label="Scenarios"
              value={summary.scenarios.length}
            />
            <StatCard
              label="Pass Rate"
              value={formatPercent(overall?.pass_rate ?? 0)}
            />
            <StatCard
              label="Avg Tool Score"
              value={(overall?.avg_tool_score ?? 0).toFixed(2)}
            />
            <StatCard
              label="Categories"
              value={
                Object.keys(summary.aggregates.by_category).length
              }
            />
          </div>

          {/* Category breakdown */}
          <div className="rounded-lg border border-border">
            <div className="border-b border-border px-4 py-3">
              <h2 className="text-sm font-medium">
                Performance by Category
              </h2>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-tertiary">
                  <th className="px-4 py-2 font-medium">Category</th>
                  <th className="px-4 py-2 font-medium">Scenarios</th>
                  <th className="px-4 py-2 font-medium">Pass Rate</th>
                  <th className="px-4 py-2 font-medium">
                    Avg Tool Score
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.aggregates.by_category).map(
                  ([cat, stats]) => (
                    <tr
                      key={cat}
                      className="border-b border-border last:border-0 hover:bg-bg-hover transition-colors"
                    >
                      <td className="px-4 py-2.5 font-medium">{cat}</td>
                      <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                        {stats.n}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                        {formatPercent(stats.pass_rate)}
                      </td>
                      <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                        {stats.avg_tool_score.toFixed(2)}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>

          {/* Failure types */}
          {Object.keys(summary.aggregates.failure_types).length > 0 && (
            <div className="rounded-lg border border-border">
              <div className="border-b border-border px-4 py-3">
                <h2 className="text-sm font-medium">Failure Types</h2>
              </div>
              <div className="grid grid-cols-3 gap-4 p-4">
                {Object.entries(summary.aggregates.failure_types)
                  .sort((a, b) => b[1] - a[1])
                  .map(([type, count]) => (
                    <div key={type} className="flex items-center justify-between text-sm">
                      <span className="text-text-secondary">{type}</span>
                      <span className="tabular-nums font-medium">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <EmptyState
          title="Failed to load"
          description="Could not load summary for this run."
        />
      )}
    </div>
  );
}
