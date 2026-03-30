"""Shared metric computation for GAIA2 ARE trace analysis."""

import json
import math
from collections import Counter
from pathlib import Path


def classify_failure(
    agent_tools: list[str],
    oracle_tools: list[str],
    format_errors: int,
    think_leaks: int,
    passed: bool,
) -> str:
    """Classify scenario failure type.

    Returns one of: 'success', 'all_tools_missing', 'some_tools_missing',
    'wrong_tool_args', 'correct_tools_wrong_answer', 'format_error', 'think_leak'
    """
    if passed:
        return "success"
    if think_leaks > 0:
        return "think_leak"
    if format_errors > 0:
        return "format_error"

    agent_unique = set(agent_tools)
    oracle_unique = set(oracle_tools)

    if oracle_unique and not agent_unique & oracle_unique:
        return "all_tools_missing"
    if oracle_unique - agent_unique:
        return "some_tools_missing"
    if oracle_unique and oracle_unique <= agent_unique:
        return "correct_tools_wrong_answer"

    return "wrong_tool_args"


def compute_tool_score(agent_tools: list[str], oracle_tools: list[str]) -> float:
    """Compare agent tool calls vs oracle. Returns 0.0-1.0 score.

    Score = matched_tools / max(len(oracle_tools), 1)
    A tool is matched if it appears in both lists (considering multiplicity).
    """
    if not oracle_tools:
        return 1.0

    oracle_counts = Counter(oracle_tools)
    agent_counts = Counter(agent_tools)

    matched = 0
    for tool, count in oracle_counts.items():
        matched += min(agent_counts.get(tool, 0), count)

    return matched / max(len(oracle_tools), 1)


def compute_per_tool_comparison(
    agent_tools: list[str], oracle_tools: list[str]
) -> list[dict]:
    """Per-tool breakdown: [{tool, agent, oracle, delta}]"""
    all_tools = sorted(set(agent_tools) | set(oracle_tools))
    agent_counts = Counter(agent_tools)
    oracle_counts = Counter(oracle_tools)

    result = []
    for tool in all_tools:
        a = agent_counts.get(tool, 0)
        o = oracle_counts.get(tool, 0)
        result.append({"tool": tool, "agent": a, "oracle": o, "delta": a - o})
    return result


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _extract_tool_calls(messages: list[dict]) -> list[str]:
    """Extract tool call names from conversation messages."""
    tools = []
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                name = tc.get("function", {}).get("name") or tc.get("name", "")
                if name:
                    tools.append(name)
    return tools


def _extract_oracle_tools(oracle_events: list) -> list[str]:
    """Extract tool names from oracle events."""
    tools = []
    for event in oracle_events or []:
        if isinstance(event, dict):
            name = event.get("tool") or event.get("function", "")
            if name:
                tools.append(name)
        elif isinstance(event, str):
            tools.append(event)
    return tools


def _count_format_errors(messages: list[dict]) -> int:
    """Count format errors in conversation."""
    count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and "format error" in content.lower():
            count += 1
    return count


def _count_think_leaks(messages: list[dict]) -> int:
    """Count think tag leaks in assistant messages."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and "<think>" in content:
                count += 1
    return count


def parse_trace(trace_path: str) -> dict:
    """Parse an ARE trace JSONL file into a structured result dict.

    Each line in the JSONL has: scenario_id, category, messages (conversation),
    oracle_events, expected_answer, model_answer, validation_decision,
    validation_rationale, duration_s, prompt_tokens, completion_tokens.

    Returns dict with all fields plus computed metrics (tool_score, failure_type, etc.)
    """
    path = Path(trace_path)
    lines = path.read_text().strip().splitlines()

    # A trace JSONL may contain one record (one scenario) or multiple.
    # We return the last complete record (some writers append progress lines).
    record = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

    if record is None:
        raise ValueError(f"No valid JSON found in {trace_path}")

    messages = record.get("messages", [])
    oracle_events = record.get("oracle_events", [])
    validation = record.get("validation_decision", "")
    passed = validation.lower() in ("valid", "true", "pass", "correct") if isinstance(validation, str) else bool(validation)

    agent_tools = _extract_tool_calls(messages)
    oracle_tools = _extract_oracle_tools(oracle_events)
    format_errors = _count_format_errors(messages)
    think_leaks = _count_think_leaks(messages)

    tool_score = compute_tool_score(agent_tools, oracle_tools)
    failure_type = classify_failure(agent_tools, oracle_tools, format_errors, think_leaks, passed)
    per_tool = compute_per_tool_comparison(agent_tools, oracle_tools)

    n_tool_calls = len(agent_tools)
    n_oracle_calls = len(oracle_tools)
    tool_efficiency = (n_oracle_calls / n_tool_calls) if n_tool_calls > 0 else (1.0 if n_oracle_calls == 0 else 0.0)

    prompt_tokens = record.get("prompt_tokens", 0)
    completion_tokens = record.get("completion_tokens", 0)

    return {
        "scenario_id": record.get("scenario_id", path.stem),
        "category": record.get("category", "unknown"),
        "task_prompt": _extract_task_prompt(messages),
        "expected_answer": record.get("expected_answer", ""),
        "model_answer": record.get("model_answer", ""),
        "oracle_events": oracle_events,
        "validation_decision": validation,
        "validation_rationale": record.get("validation_rationale", ""),
        "passed": passed,
        "tool_score": round(tool_score, 4),
        "failure_type": failure_type,
        "n_tool_calls": n_tool_calls,
        "n_messages": len(messages),
        "duration_s": record.get("duration_s", 0.0),
        "n_format_errors": format_errors,
        "n_think_leaks": think_leaks,
        "tool_efficiency": round(tool_efficiency, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "per_tool": per_tool,
        "conversation": messages,
    }


def _extract_task_prompt(messages: list[dict]) -> str:
    """Extract the task prompt from conversation messages."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""


def aggregate_results(scenarios: list[dict]) -> dict:
    """Compute aggregates: overall stats, by_category, failure_types, top_missing_tools.

    Returns the 'aggregates' dict matching the summary.json schema.
    """
    if not scenarios:
        return {
            "overall": {"n": 0, "pass_rate": 0.0, "avg_tool_score": 0.0},
            "by_category": {},
            "failure_types": {},
            "top_missing_tools": [],
        }

    n = len(scenarios)
    n_passed = sum(1 for s in scenarios if s.get("passed"))
    tool_scores = [s.get("tool_score", 0.0) for s in scenarios]

    overall = {
        "n": n,
        "pass_rate": round(n_passed / n, 4),
        "avg_tool_score": round(sum(tool_scores) / n, 4),
    }

    # By category
    by_category: dict[str, list[dict]] = {}
    for s in scenarios:
        cat = s.get("category", "unknown")
        by_category.setdefault(cat, []).append(s)

    category_stats = {}
    for cat, cat_scenarios in sorted(by_category.items()):
        cn = len(cat_scenarios)
        cp = sum(1 for s in cat_scenarios if s.get("passed"))
        cts = [s.get("tool_score", 0.0) for s in cat_scenarios]
        category_stats[cat] = {
            "n": cn,
            "pass_rate": round(cp / cn, 4),
            "avg_tool_score": round(sum(cts) / cn, 4),
        }

    # Failure types
    failure_counts: Counter = Counter()
    for s in scenarios:
        failure_counts[s.get("failure_type", "unknown")] += 1

    # Top missing tools
    missing_counter: Counter = Counter()
    for s in scenarios:
        per_tool = s.get("per_tool", [])
        for t in per_tool:
            if t.get("oracle", 0) > 0 and t.get("agent", 0) < t["oracle"]:
                missing_counter[t["tool"]] += t["oracle"] - t["agent"]

    top_missing = missing_counter.most_common(20)

    return {
        "overall": overall,
        "by_category": category_stats,
        "failure_types": dict(failure_counts),
        "top_missing_tools": top_missing,
    }
