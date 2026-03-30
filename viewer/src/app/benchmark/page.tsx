"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import type { BenchmarkScenario } from "@/lib/types";
import { Card, Badge, StatCard, EmptyState, Spinner } from "@/components/ui";

interface BenchmarkData {
  stats: {
    total?: number;
    by_category?: Record<string, number>;
    with_answer?: number;
    with_oracle_actions?: number;
  };
  scenarios: BenchmarkScenario[];
}

type SortField = "task_id" | "question" | "category" | "apps" | "oracle_actions";
type SortDir = "asc" | "desc";

const CATEGORIES = ["search", "time", "execution", "adaptability", "ambiguity"];
const CATEGORY_COLORS: Record<string, "success" | "warning" | "error" | "default" | "muted"> = {
  search: "success",
  time: "warning",
  execution: "error",
  adaptability: "default",
  ambiguity: "muted",
};

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<Set<string>>(new Set());
  const [sortField, setSortField] = useState<SortField>("task_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  useEffect(() => {
    fetch("/api/benchmark")
      .then((res) => res.json())
      .then((d: BenchmarkData) => {
        if (!d.scenarios || d.scenarios.length === 0) {
          setError(true);
        } else {
          setData(d);
        }
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    if (!data) return [];
    let items = data.scenarios;

    if (search) {
      const q = search.toLowerCase();
      items = items.filter(
        (s) =>
          s.task_id.toLowerCase().includes(q) ||
          s.scenario_id.toLowerCase().includes(q) ||
          s.question.toLowerCase().includes(q)
      );
    }

    if (categoryFilter.size > 0) {
      items = items.filter((s) => categoryFilter.has(s.category));
    }

    items = [...items].sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "task_id":
          cmp = a.task_id.localeCompare(b.task_id);
          break;
        case "question":
          cmp = a.question.localeCompare(b.question);
          break;
        case "category":
          cmp = a.category.localeCompare(b.category);
          break;
        case "apps":
          cmp = a.apps.length - b.apps.length;
          break;
        case "oracle_actions":
          cmp = a.oracle_actions.length - b.oracle_actions.length;
          break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return items;
  }, [data, search, categoryFilter, sortField, sortDir]);

  function toggleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  }

  function toggleCategory(cat: string) {
    setCategoryFilter((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  function sortIndicator(field: SortField) {
    if (sortField !== field) return "";
    return sortDir === "asc" ? " \u2191" : " \u2193";
  }

  if (loading) return <Spinner text="Loading benchmark data..." />;

  if (error || !data) {
    return (
      <EmptyState
        title="No benchmark data"
        description="Run: HF_TOKEN=hf_xxx python scripts/sync_benchmark.py"
      />
    );
  }

  const stats = data.stats;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-lg font-semibold">GAIA2 Benchmark</h1>
        <p className="mt-1 text-sm text-text-tertiary">
          {stats.total ?? data.scenarios.length} scenarios across 5 categories from GAIA2 ARE
        </p>
      </div>

      {/* Stats row */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
        {CATEGORIES.map((cat) => (
          <StatCard
            key={cat}
            label={cat.charAt(0).toUpperCase() + cat.slice(1)}
            value={stats.by_category?.[cat] ?? 0}
          />
        ))}
      </div>

      {/* Filters */}
      <Card className="mb-4">
        <div className="flex flex-wrap items-center gap-4">
          <input
            type="text"
            placeholder="Search by ID or question..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 flex-1 min-w-[200px] rounded-md border border-border bg-bg px-3 text-sm text-text placeholder:text-text-tertiary focus:border-accent focus:outline-none"
          />
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <span className="text-text-tertiary">Category:</span>
            {CATEGORIES.map((cat) => (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors capitalize ${
                  categoryFilter.has(cat)
                    ? "bg-accent text-white"
                    : "bg-bg-hover text-text-tertiary hover:text-text-secondary"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Results count */}
      <p className="mb-3 text-xs text-text-tertiary">
        {filtered.length} scenario{filtered.length !== 1 ? "s" : ""}
      </p>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-tertiary">
              <th
                className="cursor-pointer px-4 py-3 font-medium hover:text-text-secondary"
                onClick={() => toggleSort("task_id")}
              >
                Scenario{sortIndicator("task_id")}
              </th>
              <th
                className="cursor-pointer px-4 py-3 font-medium hover:text-text-secondary"
                onClick={() => toggleSort("question")}
              >
                Question{sortIndicator("question")}
              </th>
              <th
                className="cursor-pointer px-4 py-3 font-medium hover:text-text-secondary"
                onClick={() => toggleSort("category")}
              >
                Category{sortIndicator("category")}
              </th>
              <th
                className="cursor-pointer px-4 py-3 font-medium hover:text-text-secondary"
                onClick={() => toggleSort("apps")}
              >
                Apps{sortIndicator("apps")}
              </th>
              <th className="px-4 py-3 font-medium">
                Answer
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((s) => (
              <tr
                key={s.task_id}
                className="border-b border-border last:border-b-0 transition-colors hover:bg-bg-hover"
              >
                <td className="px-4 py-3">
                  <Link
                    href={`/benchmark/${encodeURIComponent(s.task_id)}`}
                    className="font-mono text-xs text-accent hover:underline"
                  >
                    {s.scenario_id.replace("scenario_", "").slice(0, 20)}
                  </Link>
                </td>
                <td className="max-w-md px-4 py-3 text-text-secondary">
                  <Link href={`/benchmark/${encodeURIComponent(s.task_id)}`} className="hover:text-text">
                    {s.question.length > 100
                      ? s.question.slice(0, 100) + "..."
                      : s.question || <span className="text-text-tertiary italic">No question text</span>}
                  </Link>
                </td>
                <td className="px-4 py-3">
                  <Badge variant={CATEGORY_COLORS[s.category] ?? "default"}>
                    {s.category}
                  </Badge>
                </td>
                <td className="px-4 py-3 tabular-nums text-text-secondary">
                  {s.apps.length}
                </td>
                <td className="px-4 py-3 text-text-tertiary">
                  {s.expected_answer ? (
                    <span className="text-text-secondary text-xs">
                      {s.expected_answer.length > 40
                        ? s.expected_answer.slice(0, 40) + "..."
                        : s.expected_answer}
                    </span>
                  ) : (
                    <span className="text-text-tertiary">-</span>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-text-tertiary">
                  No scenarios match your filters
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
