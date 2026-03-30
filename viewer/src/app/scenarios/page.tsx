"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { fetchIndex, fetchSummary, formatDuration, formatPercent } from "@/lib/data";
import type { RunIndex, RunSummary, ScenarioSummary, SortConfig } from "@/lib/types";
import { Badge, ScoreBar, Spinner, EmptyState } from "@/components/ui";

type PassFilter = "all" | "pass" | "fail";

export default function ScenariosPage() {
  const router = useRouter();
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [selectedRun, setSelectedRun] = useState("");
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set());
  const [passFilter, setPassFilter] = useState<PassFilter>("all");
  const [failureType, setFailureType] = useState("");
  const [sort, setSort] = useState<SortConfig>({ field: "scenario_id", dir: "asc" });

  useEffect(() => {
    fetchIndex().then((idx) => {
      setIndex(idx);
      if (idx.runs.length > 0) setSelectedRun(idx.runs[0].run_id);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedRun) { setSummary(null); return; }
    setLoading(true);
    fetchSummary(selectedRun).then((s) => {
      setSummary(s);
      setLoading(false);
    });
  }, [selectedRun]);

  const categories = useMemo(() => {
    if (!summary) return [];
    return [...new Set(summary.scenarios.map((s) => s.category))].sort();
  }, [summary]);

  const failureTypes = useMemo(() => {
    if (!summary) return [];
    return [...new Set(summary.scenarios.map((s) => s.failure_type).filter(Boolean))].sort();
  }, [summary]);

  const filtered = useMemo(() => {
    if (!summary) return [];
    let items = summary.scenarios;

    if (search) {
      const q = search.toLowerCase();
      items = items.filter((s) => s.scenario_id.toLowerCase().includes(q));
    }
    if (selectedCategories.size > 0) {
      items = items.filter((s) => selectedCategories.has(s.category));
    }
    if (passFilter === "pass") items = items.filter((s) => s.passed);
    if (passFilter === "fail") items = items.filter((s) => !s.passed);
    if (failureType) items = items.filter((s) => s.failure_type === failureType);

    const dir = sort.dir === "asc" ? 1 : -1;
    return [...items].sort((a, b) => {
      const av = a[sort.field as keyof ScenarioSummary];
      const bv = b[sort.field as keyof ScenarioSummary];
      if (typeof av === "string" && typeof bv === "string") return av.localeCompare(bv) * dir;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      if (typeof av === "boolean" && typeof bv === "boolean") return (Number(av) - Number(bv)) * dir;
      return 0;
    });
  }, [summary, search, selectedCategories, passFilter, failureType, sort]);

  function toggleSort(field: string) {
    setSort((prev) =>
      prev.field === field
        ? { field, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { field, dir: "asc" }
    );
  }

  function toggleCategory(cat: string) {
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  function sortArrow(field: string) {
    if (sort.field !== field) return "";
    return sort.dir === "asc" ? " \u2191" : " \u2193";
  }

  if (loading && !index) return <Spinner />;

  if (!index?.runs.length) {
    return <EmptyState title="No runs" description="No evaluation runs found." />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Scenarios</h1>
          <p className="text-sm text-text-tertiary">Browse evaluation scenarios</p>
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
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="text"
              placeholder="Search scenarios..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="rounded-md border border-border bg-bg-input px-3 py-1.5 text-sm text-text outline-none focus:border-accent w-56"
            />

            <div className="flex items-center gap-1">
              {(["all", "pass", "fail"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setPassFilter(v)}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    passFilter === v
                      ? "bg-text-tertiary/20 text-text"
                      : "text-text-tertiary hover:text-text-secondary"
                  }`}
                >
                  {v === "all" ? "All" : v === "pass" ? "Passed" : "Failed"}
                </button>
              ))}
            </div>

            {failureTypes.length > 0 && (
              <select
                value={failureType}
                onChange={(e) => setFailureType(e.target.value)}
                className="rounded-md border border-border bg-bg-input px-3 py-1.5 text-xs text-text outline-none focus:border-accent"
              >
                <option value="">All failure types</option>
                {failureTypes.map((ft) => (
                  <option key={ft} value={ft}>{ft}</option>
                ))}
              </select>
            )}
          </div>

          {/* Category pills */}
          {categories.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => toggleCategory(cat)}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    selectedCategories.has(cat)
                      ? "bg-accent text-white"
                      : "bg-text-tertiary/10 text-text-secondary hover:bg-text-tertiary/20"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          )}

          <p className="text-xs text-text-tertiary">
            {filtered.length} of {summary.scenarios.length} scenarios
          </p>

          {/* Table */}
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-tertiary">
                  {[
                    { key: "scenario_id", label: "Scenario" },
                    { key: "category", label: "Category" },
                    { key: "passed", label: "Result" },
                    { key: "tool_score", label: "Tool Score" },
                    { key: "failure_type", label: "Failure Type" },
                    { key: "n_messages", label: "Messages" },
                    { key: "duration_s", label: "Duration" },
                  ].map((col) => (
                    <th
                      key={col.key}
                      onClick={() => toggleSort(col.key)}
                      className="px-4 py-2 font-medium cursor-pointer hover:text-text-secondary select-none"
                    >
                      {col.label}{sortArrow(col.key)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((s) => (
                  <tr
                    key={s.scenario_id}
                    onClick={() => router.push(`/scenarios/${encodeURIComponent(s.scenario_id)}?run=${encodeURIComponent(selectedRun)}`)}
                    className="border-b border-border last:border-0 hover:bg-bg-hover transition-colors cursor-pointer"
                  >
                    <td className="px-4 py-2.5 font-medium font-mono text-xs">
                      {s.scenario_id.length > 20 ? s.scenario_id.slice(0, 20) + "..." : s.scenario_id}
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge>{s.category}</Badge>
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge variant={s.passed ? "success" : "error"}>
                        {s.passed ? "Pass" : "Fail"}
                      </Badge>
                    </td>
                    <td className="px-4 py-2.5">
                      <ScoreBar value={s.tool_score} />
                    </td>
                    <td className="px-4 py-2.5 text-xs text-text-secondary">
                      {s.failure_type || "-"}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-text-secondary">
                      {s.n_messages}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-text-secondary text-xs">
                      {formatDuration(s.duration_s)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 && (
              <EmptyState title="No matches" description="Try adjusting your filters." />
            )}
          </div>
        </>
      ) : (
        <EmptyState title="Failed to load" description="Could not load summary for this run." />
      )}
    </div>
  );
}
