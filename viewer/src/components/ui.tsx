import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-border bg-bg-card p-5 ${className}`}
    >
      {children}
    </div>
  );
}

export function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg-card p-5">
      <p className="text-xs text-text-tertiary uppercase tracking-wider">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-text-tertiary">{sub}</p>}
    </div>
  );
}

export function Badge({
  children,
  variant = "default",
}: {
  children: ReactNode;
  variant?: "default" | "success" | "error" | "warning" | "muted";
}) {
  const colors = {
    default:
      "bg-text-tertiary/10 text-text-secondary",
    success: "bg-success/10 text-success",
    error: "bg-error/10 text-error",
    warning: "bg-warning/10 text-warning",
    muted: "bg-text-tertiary/10 text-text-tertiary",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${colors[variant]}`}
    >
      {children}
    </span>
  );
}

export function ScoreBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
  const color =
    pct >= 70
      ? "bg-success"
      : pct >= 40
        ? "bg-warning"
        : "bg-error";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 rounded-full bg-border">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs tabular-nums text-text-secondary">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function Spinner({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="flex items-center justify-center py-20 text-text-tertiary">
      <svg
        className="mr-3 h-4 w-4 animate-spin"
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="3"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <span className="text-sm">{text}</span>
    </div>
  );
}

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="py-20 text-center">
      <h3 className="text-sm font-medium text-text">{title}</h3>
      <p className="mt-1 text-sm text-text-tertiary">{description}</p>
    </div>
  );
}
