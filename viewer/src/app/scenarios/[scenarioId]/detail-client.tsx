"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchDetail, fetchSummary, formatDuration, formatPercent } from "@/lib/data";
import type { ScenarioDetail } from "@/lib/types";
import { Card, Badge, Spinner, EmptyState } from "@/components/ui";
import { ConversationTimeline } from "@/components/conversation";

export function ScenarioDetailClient({
  scenarioId,
  runId,
}: {
  scenarioId: string;
  runId: string;
}) {
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [scenarioIds, setScenarioIds] = useState<string[]>([]);

  useEffect(() => {
    if (!runId) { setLoading(false); return; }

    setLoading(true);
    const decoded = decodeURIComponent(scenarioId);

    Promise.all([
      fetchDetail(runId, decoded),
      fetchSummary(runId),
    ]).then(([d, s]) => {
      setDetail(d);
      if (s) setScenarioIds(s.scenarios.map((sc) => sc.scenario_id));
      setLoading(false);
    });
  }, [runId, scenarioId]);

  if (loading) return <Spinner />;

  if (!runId) {
    return <EmptyState title="No run selected" description="Navigate from the Scenarios page to view details." />;
  }

  if (!detail) {
    return <EmptyState title="Not found" description={`Scenario ${scenarioId} not found in this run.`} />;
  }

  const decoded = decodeURIComponent(scenarioId);
  const currentIdx = scenarioIds.indexOf(decoded);
  const prevId = currentIdx > 0 ? scenarioIds[currentIdx - 1] : null;
  const nextId = currentIdx < scenarioIds.length - 1 ? scenarioIds[currentIdx + 1] : null;

  const passed = detail.validation_decision === "correct";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link
            href={`/scenarios?run=${encodeURIComponent(runId)}`}
            className="text-xs text-text-tertiary hover:text-text-secondary transition-colors"
          >
            &larr; Back to scenarios
          </Link>
          <h1 className="mt-1 text-lg font-semibold font-mono">{decoded}</h1>
        </div>
        <div className="flex items-center gap-2">
          {prevId && (
            <Link
              href={`/scenarios/${encodeURIComponent(prevId)}?run=${encodeURIComponent(runId)}`}
              className="rounded-md border border-border px-3 py-1.5 text-xs text-text-tertiary hover:text-text-secondary hover:border-border-hover transition-colors"
            >
              &larr; Prev
            </Link>
          )}
          {nextId && (
            <Link
              href={`/scenarios/${encodeURIComponent(nextId)}?run=${encodeURIComponent(runId)}`}
              className="rounded-md border border-border px-3 py-1.5 text-xs text-text-tertiary hover:text-text-secondary hover:border-border-hover transition-colors"
            >
              Next &rarr;
            </Link>
          )}
        </div>
      </div>

      <div className="grid grid-cols-[1fr_300px] gap-6">
        {/* Main content */}
        <div className="space-y-6 min-w-0">
          {/* Task */}
          <Card>
            <h2 className="text-sm font-medium mb-3">Task</h2>
            <div className="rounded border border-border bg-bg p-4 text-sm">
              {detail.task_prompt}
            </div>
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-text-tertiary mb-1">Expected Answer</p>
                <p className="text-sm text-text-secondary">{detail.expected_answer}</p>
              </div>
              <div>
                <p className="text-xs text-text-tertiary mb-1">Model Answer</p>
                <p className="text-sm">{detail.model_answer}</p>
              </div>
            </div>
          </Card>

          {/* Conversation */}
          <div>
            <h2 className="text-sm font-medium mb-3">Conversation</h2>
            <ConversationTimeline messages={detail.conversation} />
          </div>

          {/* Oracle comparison */}
          {detail.per_tool.length > 0 && (
            <Card>
              <h2 className="text-sm font-medium mb-3">Tool Usage Comparison</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-text-tertiary">
                    <th className="px-4 py-2 font-medium">Tool</th>
                    <th className="px-4 py-2 font-medium">Agent</th>
                    <th className="px-4 py-2 font-medium">Oracle</th>
                    <th className="px-4 py-2 font-medium">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.per_tool.map((t) => (
                    <tr
                      key={t.tool}
                      className={`border-b border-border last:border-0 ${
                        t.delta > 0
                          ? "bg-success/5"
                          : t.delta < 0
                            ? "bg-error/5"
                            : ""
                      }`}
                    >
                      <td className="px-4 py-2.5 font-medium">{t.tool}</td>
                      <td className="px-4 py-2.5 tabular-nums text-text-secondary">{t.agent}</td>
                      <td className="px-4 py-2.5 tabular-nums text-text-secondary">{t.oracle}</td>
                      <td className={`px-4 py-2.5 tabular-nums font-medium ${
                        t.delta > 0 ? "text-success" : t.delta < 0 ? "text-error" : "text-text-secondary"
                      }`}>
                        {t.delta > 0 ? "+" : ""}{t.delta}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>

        {/* Right sidebar */}
        <div className="space-y-4">
          <Card>
            <div className="flex justify-center mb-4">
              <span className={`inline-flex items-center rounded-full px-4 py-1.5 text-sm font-semibold ${
                passed ? "bg-success/10 text-success" : "bg-error/10 text-error"
              }`}>
                {passed ? "PASS" : "FAIL"}
              </span>
            </div>
            <div className="flex justify-center">
              <Badge>{detail.category}</Badge>
            </div>
          </Card>

          <Card>
            <h3 className="text-xs text-text-tertiary uppercase tracking-wider mb-3">Metrics</h3>
            <dl className="space-y-2.5">
              {[
                { label: "Tool Score", value: detail.metrics.tool_score.toFixed(2) },
                { label: "Duration", value: formatDuration(detail.metrics.duration_s) },
                { label: "Total Tokens", value: detail.metrics.total_tokens.toLocaleString() },
                { label: "Prompt Tokens", value: detail.metrics.prompt_tokens.toLocaleString() },
                { label: "Completion Tokens", value: detail.metrics.completion_tokens.toLocaleString() },
                { label: "Tool Efficiency", value: formatPercent(detail.metrics.tool_efficiency) },
                { label: "Failure Type", value: detail.metrics.failure_type || "-" },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <dt className="text-xs text-text-tertiary">{item.label}</dt>
                  <dd className="text-sm tabular-nums">{item.value}</dd>
                </div>
              ))}
            </dl>
          </Card>

          <Card>
            <h3 className="text-xs text-text-tertiary uppercase tracking-wider mb-3">Judgment</h3>
            <div className="space-y-2">
              <div>
                <p className="text-xs text-text-tertiary">Decision</p>
                <p className="text-sm font-medium">{detail.validation_decision}</p>
              </div>
              <div>
                <p className="text-xs text-text-tertiary">Rationale</p>
                <p className="text-sm text-text-secondary">{detail.validation_rationale}</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
