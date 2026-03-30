// === Index / Run manifest ===
export interface RunEntry {
  run_id: string;
  model_name: string;
  timestamp: string;
  training_step: number | null;
  n_scenarios: number;
  overall_pass_rate: number;
  overall_avg_tool_score: number;
  box?: string;
  wandb_url?: string | null;
  status?: "active" | "complete";
}

export interface TrainingRun {
  run_id: string;
  run_name: string;
  model_name: string;
  box: string;
  wandb_url: string;
  wandb_run_id?: string;
  wandb_project: string;
  started_at: string;
  status: string;
  checkpoints: number[];
  gpu: string;
  config_summary?: string;
}

export interface RunIndex {
  runs: RunEntry[];
  training_runs?: TrainingRun[];
}

// === Summary (per-run) ===
export interface CategoryStats {
  n: number;
  pass_rate: number;
  avg_tool_score: number;
}

export interface Aggregates {
  overall: CategoryStats;
  by_category: Record<string, CategoryStats>;
  failure_types: Record<string, number>;
  top_missing_tools: [string, number][];
}

export interface ScenarioSummary {
  scenario_id: string;
  category: string;
  passed: boolean;
  tool_score: number;
  failure_type: string;
  n_tool_calls: number;
  n_messages: number;
  duration_s: number;
  n_format_errors: number;
  n_think_leaks: number;
  tool_efficiency: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface RunSummary {
  run_id: string;
  model_name: string;
  training_step: number | null;
  timestamp: string;
  aggregates: Aggregates;
  scenarios: ScenarioSummary[];
}

// === Detail (per-scenario, lazy loaded) ===
export interface OracleEvent {
  app?: string;
  function?: string;
  tool?: string;
  args?: Record<string, unknown>;
}

export interface ToolComparison {
  tool: string;
  agent: number;
  oracle: number;
  delta: number;
}

export interface ConversationMessage {
  role: "system" | "user" | "assistant" | "tool-response" | "tool" | "tool_response";
  content: string;
  tool_calls?: {
    function?: { name?: string; arguments?: string };
    name?: string;
  }[];
}

export interface ScenarioDetail {
  scenario_id: string;
  category: string;
  task_prompt: string;
  expected_answer: string;
  model_answer: string;
  oracle_events: OracleEvent[];
  validation_decision: string;
  validation_rationale: string;
  per_tool: ToolComparison[];
  conversation: ConversationMessage[];
  metrics: {
    tool_score: number;
    total_tokens: number | number[];
    prompt_tokens: number | number[];
    completion_tokens: number | number[];
    duration_s: number;
    tool_efficiency: number;
    failure_type: string;
  };
}

// === GAIA2 Benchmark (from meta-agents-research-environments/gaia2) ===
export interface BenchmarkScenario {
  task_id: string;
  scenario_id: string;
  category: string;
  question: string;
  expected_answer: string;
  split: string;
  tags: string[];
  apps: string[];
  oracle_actions: { app: string; function: string }[];
  start_time: number | null;
  hints: string[];
}

// === Filter state ===
export interface Filters {
  categories: string[];
  passed: boolean | null;
  failureTypes: string[];
  search: string;
}

export interface SortConfig {
  field: string;
  dir: "asc" | "desc";
}
