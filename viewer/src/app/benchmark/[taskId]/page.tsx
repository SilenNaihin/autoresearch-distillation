"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import type { BenchmarkScenario, RunIndex, RunSummary } from "@/lib/types";
import { Card, Badge, EmptyState, Spinner } from "@/components/ui";

const CATEGORY_COLORS: Record<string, "success" | "warning" | "error" | "default" | "muted"> = {
  search: "success",
  time: "warning",
  execution: "error",
  adaptability: "default",
  ambiguity: "muted",
};

interface BenchmarkData {
  stats: Record<string, unknown>;
  scenarios: BenchmarkScenario[];
}

export default function BenchmarkDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();

  const [scenarios, setScenarios] = useState<BenchmarkScenario[]>([]);
  const [scenario, setScenario] = useState<BenchmarkScenario | null>(null);
  const [loading, setLoading] = useState(true);
  const [runIndex, setRunIndex] = useState<RunIndex | null>(null);
  const [runResults, setRunResults] = useState<
    { run_id: string; model_name: string; passed: boolean; tool_score: number }[]
  >([]);

  useEffect(() => {
    Promise.all([
      fetch("/api/benchmark").then((r) => r.json()),
      fetch("/data/index.json")
        .then((r) => (r.ok ? r.json() : { runs: [] }))
        .catch(() => ({ runs: [] })),
    ])
      .then(([benchData, idx]: [BenchmarkData, RunIndex]) => {
        const all = benchData.scenarios ?? [];
        setScenarios(all);
        const found = all.find((s: BenchmarkScenario) => s.task_id === taskId);
        setScenario(found ?? null);
        setRunIndex(idx);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [taskId]);

  useEffect(() => {
    if (!runIndex || !scenario) return;
    const results: typeof runResults = [];

    Promise.all(
      runIndex.runs.map(async (run) => {
        try {
          const res = await fetch(
            `/data/runs/${encodeURIComponent(run.run_id)}/summary.json`
          );
          if (!res.ok) return;
          const summary: RunSummary = await res.json();
          const match = summary.scenarios.find(
            (s) => s.scenario_id === scenario.task_id || s.scenario_id === scenario.scenario_id
          );
          if (match) {
            results.push({
              run_id: run.run_id,
              model_name: run.model_name,
              passed: match.passed,
              tool_score: match.tool_score,
            });
          }
        } catch {
          // skip
        }
      })
    ).then(() => setRunResults(results));
  }, [runIndex, scenario]);

  if (loading) return <Spinner text="Loading scenario..." />;

  if (!scenario) {
    return (
      <EmptyState
        title="Scenario not found"
        description={`No benchmark scenario with task_id "${taskId}"`}
      />
    );
  }

  const currentIdx = scenarios.findIndex((s) => s.task_id === taskId);
  const prevId = currentIdx > 0 ? scenarios[currentIdx - 1].task_id : null;
  const nextId =
    currentIdx < scenarios.length - 1
      ? scenarios[currentIdx + 1].task_id
      : null;

  return (
    <div>
      {/* Nav */}
      <div className="mb-6 flex items-center justify-between">
        <Link
          href="/benchmark"
          className="text-sm text-text-tertiary hover:text-text-secondary"
        >
          &larr; Benchmark
        </Link>
        <div className="flex items-center gap-3">
          {prevId ? (
            <Link
              href={`/benchmark/${encodeURIComponent(prevId)}`}
              className="text-sm text-text-tertiary hover:text-text-secondary"
            >
              &larr; Prev
            </Link>
          ) : (
            <span className="text-sm text-text-tertiary/40">
              &larr; Prev
            </span>
          )}
          <span className="text-xs text-text-tertiary">
            {currentIdx + 1} / {scenarios.length}
          </span>
          {nextId ? (
            <Link
              href={`/benchmark/${encodeURIComponent(nextId)}`}
              className="text-sm text-text-tertiary hover:text-text-secondary"
            >
              Next &rarr;
            </Link>
          ) : (
            <span className="text-sm text-text-tertiary/40">
              Next &rarr;
            </span>
          )}
        </div>
      </div>

      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold">Scenario</h1>
          <Badge variant={CATEGORY_COLORS[scenario.category] ?? "default"}>
            {scenario.category}
          </Badge>
        </div>
        <p className="mt-1 font-mono text-xs text-text-tertiary">
          {scenario.scenario_id}
        </p>
      </div>

      {/* Question */}
      <Card className="mb-4">
        <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
          Task Prompt
        </p>
        <p className="text-sm leading-relaxed text-text-secondary">
          {scenario.question || <span className="italic text-text-tertiary">No task prompt available</span>}
        </p>
      </Card>

      {/* Expected Answer */}
      {scenario.expected_answer && (
        <Card className="mb-4">
          <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
            Expected Answer
          </p>
          <p className="font-mono text-sm text-text-secondary">
            {scenario.expected_answer}
          </p>
        </Card>
      )}

      {/* Environment */}
      <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Card>
          <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
            Available Apps ({scenario.apps.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {scenario.apps.map((app) => (
              <span
                key={app}
                className="rounded-md bg-bg-hover px-2 py-0.5 text-xs text-text-secondary"
              >
                {app}
              </span>
            ))}
          </div>
        </Card>

        {scenario.oracle_actions.length > 0 && (
          <Card>
            <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
              Oracle Actions ({scenario.oracle_actions.length})
            </p>
            <div className="space-y-1">
              {scenario.oracle_actions.map((a, i) => (
                <p key={i} className="font-mono text-xs text-text-secondary">
                  {a.app}.{a.function}
                </p>
              ))}
            </div>
          </Card>
        )}

        {scenario.tags.length > 0 && (
          <Card>
            <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
              Tags
            </p>
            <div className="flex flex-wrap gap-1.5">
              {scenario.tags.map((tag) => (
                <Badge key={tag} variant="default">
                  {tag}
                </Badge>
              ))}
            </div>
          </Card>
        )}

        {scenario.hints.length > 0 && (
          <Card>
            <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary mb-2">
              Hints
            </p>
            <ul className="space-y-1 text-sm text-text-secondary">
              {scenario.hints.map((h, i) => (
                <li key={i}>{h}</li>
              ))}
            </ul>
          </Card>
        )}
      </div>

      {/* Model Results */}
      {runResults.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-medium text-text">Model Results</h2>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-tertiary">
                  <th className="px-4 py-3 font-medium">Run</th>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">Result</th>
                  <th className="px-4 py-3 font-medium">Tool Score</th>
                </tr>
              </thead>
              <tbody>
                {runResults.map((r) => (
                  <tr
                    key={r.run_id}
                    className="border-b border-border last:border-b-0 hover:bg-bg-hover"
                  >
                    <td className="px-4 py-3">
                      <Link
                        href={`/scenarios?run=${encodeURIComponent(r.run_id)}`}
                        className="font-mono text-xs text-accent hover:underline"
                      >
                        {r.run_id.slice(0, 12)}...
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-text-secondary">
                      {r.model_name}
                    </td>
                    <td className="px-4 py-3">
                      <Badge variant={r.passed ? "success" : "error"}>
                        {r.passed ? "PASS" : "FAIL"}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 tabular-nums text-text-secondary">
                      {r.tool_score.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
