"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchIndex, fetchSummary, fetchDetail, formatPercent } from "@/lib/data";
import type { RunIndex, RunSummary, ScenarioDetail } from "@/lib/types";
import { Card, Badge, Spinner, EmptyState } from "@/components/ui";
import { ConversationTimeline } from "@/components/conversation";

export default function ComparePage() {
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [loading, setLoading] = useState(true);

  const [runIds, setRunIds] = useState<string[]>(["", ""]);
  const [summaries, setSummaries] = useState<(RunSummary | null)[]>([null, null]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [details, setDetails] = useState<(ScenarioDetail | null)[]>([null, null]);
  const [loadingDetails, setLoadingDetails] = useState(false);

  useEffect(() => {
    fetchIndex().then((idx) => {
      setIndex(idx);
      const ids = idx.runs.slice(0, 2).map((r) => r.run_id);
      while (ids.length < 2) ids.push("");
      setRunIds(ids);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    Promise.all(
      runIds.map((id) => (id ? fetchSummary(id) : Promise.resolve(null)))
    ).then(setSummaries);
  }, [runIds]);

  useEffect(() => {
    if (!selectedScenario) { setDetails([null, null]); return; }
    setLoadingDetails(true);
    Promise.all(
      runIds.map((id) =>
        id ? fetchDetail(id, selectedScenario) : Promise.resolve(null)
      )
    ).then((d) => {
      setDetails(d);
      setLoadingDetails(false);
    });
  }, [selectedScenario, runIds]);

  const commonScenarios = useMemo(() => {
    const sets = summaries
      .filter(Boolean)
      .map((s) => new Set(s!.scenarios.map((sc) => sc.scenario_id)));
    if (sets.length === 0) return [];
    let common = sets[0];
    for (let i = 1; i < sets.length; i++) {
      common = new Set([...common].filter((x) => sets[i].has(x)));
    }
    return [...common].sort();
  }, [summaries]);

  function updateRunId(idx: number, value: string) {
    setRunIds((prev) => {
      const next = [...prev];
      next[idx] = value;
      return next;
    });
  }

  function addRun() {
    if (runIds.length < 3) setRunIds([...runIds, ""]);
  }

  function removeRun(idx: number) {
    if (runIds.length > 2) setRunIds(runIds.filter((_, i) => i !== idx));
  }

  if (loading) return <Spinner />;
  if (!index?.runs.length) {
    return <EmptyState title="No runs" description="No evaluation runs found." />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Compare Runs</h1>
        <p className="text-sm text-text-tertiary">Side-by-side comparison of evaluation runs</p>
      </div>

      {/* Run selectors */}
      <div className="flex flex-wrap items-end gap-3">
        {runIds.map((rid, i) => (
          <div key={i} className="flex items-center gap-2">
            <div>
              <label className="text-xs text-text-tertiary block mb-1">Run {i + 1}</label>
              <select
                value={rid}
                onChange={(e) => updateRunId(i, e.target.value)}
                className="rounded-md border border-border bg-bg-input px-3 py-1.5 text-sm text-text outline-none focus:border-accent"
              >
                <option value="">Select run...</option>
                {index.runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.model_name} ({r.timestamp?.slice(0, 10)})
                  </option>
                ))}
              </select>
            </div>
            {runIds.length > 2 && (
              <button
                onClick={() => removeRun(i)}
                className="text-xs text-text-tertiary hover:text-error transition-colors mt-4"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        {runIds.length < 3 && (
          <button
            onClick={addRun}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-text-tertiary hover:text-text-secondary hover:border-border-hover transition-colors"
          >
            + Add run
          </button>
        )}
      </div>

      {/* Aggregate comparison table */}
      {summaries.some(Boolean) && (
        <Card>
          <h2 className="text-sm font-medium mb-4">Aggregate Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-tertiary">
                  <th className="px-4 py-2 font-medium">Run</th>
                  <th className="px-4 py-2 font-medium">Pass Rate</th>
                  <th className="px-4 py-2 font-medium">Avg Tool Score</th>
                  {(() => {
                    const cats = new Set<string>();
                    summaries.filter(Boolean).forEach((s) =>
                      Object.keys(s!.aggregates.by_category).forEach((c) => cats.add(c))
                    );
                    return [...cats].sort().map((cat) => (
                      <th key={cat} className="px-4 py-2 font-medium">{cat}</th>
                    ));
                  })()}
                </tr>
              </thead>
              <tbody>
                {summaries.map((s, i) => {
                  if (!s) return null;
                  const allCats = new Set<string>();
                  summaries.filter(Boolean).forEach((sm) =>
                    Object.keys(sm!.aggregates.by_category).forEach((c) => allCats.add(c))
                  );
                  return (
                    <tr key={i} className="border-b border-border last:border-0">
                      <td className="px-4 py-2.5 font-medium text-xs">{s.model_name}</td>
                      <td className="px-4 py-2.5 tabular-nums">{formatPercent(s.aggregates.overall.pass_rate)}</td>
                      <td className="px-4 py-2.5 tabular-nums">{s.aggregates.overall.avg_tool_score.toFixed(2)}</td>
                      {[...allCats].sort().map((cat) => (
                        <td key={cat} className="px-4 py-2.5 tabular-nums text-text-secondary">
                          {s.aggregates.by_category[cat]
                            ? formatPercent(s.aggregates.by_category[cat].pass_rate)
                            : "-"}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Scenario picker */}
      {commonScenarios.length > 0 && (
        <div>
          <label className="text-xs text-text-tertiary block mb-1">Compare scenario</label>
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
            className="rounded-md border border-border bg-bg-input px-3 py-1.5 text-sm text-text outline-none focus:border-accent w-full max-w-md"
          >
            <option value="">Select a scenario...</option>
            {commonScenarios.map((sid) => (
              <option key={sid} value={sid}>{sid}</option>
            ))}
          </select>
        </div>
      )}

      {/* Side-by-side details */}
      {selectedScenario && (
        loadingDetails ? (
          <Spinner text="Loading scenario details..." />
        ) : (
          <div className={`grid gap-4 ${runIds.filter(Boolean).length === 3 ? "grid-cols-3" : "grid-cols-2"}`}>
            {details.map((d, i) => {
              if (!d) return (
                <Card key={i}>
                  <EmptyState title="Not found" description="Scenario not available for this run." />
                </Card>
              );
              const toolCalls = d.conversation.filter(
                (m) => m.role === "assistant" && m.tool_calls?.length
              );
              return (
                <Card key={i}>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-medium text-text-tertiary">{d.scenario_id}</p>
                      <Badge variant={d.validation_decision === "correct" ? "success" : "error"}>
                        {d.validation_decision}
                      </Badge>
                    </div>

                    <div>
                      <p className="text-xs text-text-tertiary mb-1">Model Answer</p>
                      <p className="text-sm">{d.model_answer}</p>
                    </div>

                    <div>
                      <p className="text-xs text-text-tertiary mb-1">Expected</p>
                      <p className="text-sm text-text-secondary">{d.expected_answer}</p>
                    </div>

                    <div>
                      <p className="text-xs text-text-tertiary mb-1">Tool Score</p>
                      <p className="text-sm tabular-nums">{d.metrics.tool_score.toFixed(2)}</p>
                    </div>

                    {toolCalls.length > 0 && (
                      <div>
                        <p className="text-xs text-text-tertiary mb-2">Tool Calls ({toolCalls.length})</p>
                        <div className="space-y-2">
                          {toolCalls.map((m, j) =>
                            m.tool_calls?.map((tc, k) => (
                              <div key={`${j}-${k}`} className="rounded border border-border bg-bg p-2">
                                <p className="text-[11px] font-medium text-warning">
                                  {tc.function?.name || tc.name || "tool"}
                                </p>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        )
      )}
    </div>
  );
}
