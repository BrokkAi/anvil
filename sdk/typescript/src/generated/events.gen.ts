// Generated from openapi/anvil.v1.events.schema.json (Anvil Agent API contract 1.0.0).
// Generator: json-schema-to-typescript 15.0.4. Do not edit by hand.

/**
 * Contract for the JSON payload carried in each Server-Sent Event on GET /v1/runs/{run_id}/events. The SSE `id` field is the decimal `seq`; the SSE `event` field equals `type`. Every event shares the envelope fields and one payload variant selected by `type`. The stream ends after a terminal `run.completed`, `run.cancelled`, or `run.failed` event. The synthetic `events.gap` notice is emitted without an envelope `seq` when a reconnect falls behind the bounded replay buffer.
 */
export type AnvilRunEventStreamV1 =
  | RunCreated
  | MessageDelta
  | ThoughtDelta
  | PlanUpdated
  | ToolCallStarted
  | ToolCallBlocked
  | ToolCallInProgress
  | ToolCallFailed
  | ToolCallCompleted
  | PermissionRequested
  | PermissionResolved
  | RunTerminal
  | EventsGap;
export type RunCreated = Envelope & {
  type?: 'run.created';
  prompt_chars: number;
  [k: string]: unknown;
};
export type MessageDelta = Envelope & {
  type?: 'message.delta';
  text: string;
  [k: string]: unknown;
};
export type ThoughtDelta = Envelope & {
  type?: 'thought.delta';
  text: string;
  [k: string]: unknown;
};
export type PlanUpdated = Envelope & {
  type?: 'plan.updated';
  plan: {
    explanation?: string | null;
    plan: {
      step: string;
      status: 'pending' | 'in_progress' | 'completed';
    }[];
  };
  [k: string]: unknown;
};
export type ToolCallStarted = ToolCallBase & {
  type?: 'tool_call.started';
  input: unknown;
  /**
   * Present and true when the rendered card would hide the input (adapters use a static title).
   */
  oversized?: boolean;
  [k: string]: unknown;
};
export type ToolCallBase = Envelope & {
  call_id: string;
  tool_name: string;
  [k: string]: unknown;
};
export type ToolCallBlocked = ToolCallBase & {
  type?: 'tool_call.blocked';
  input: unknown;
  reason: string;
  [k: string]: unknown;
};
export type ToolCallInProgress = ToolCallBase & {
  type?: 'tool_call.in_progress';
  [k: string]: unknown;
};
export type ToolCallFailed = ToolCallBase & {
  type?: 'tool_call.failed';
  reason: string;
  permission_notice?: string | null;
  /**
   * Present for post-execution failures; null for pre-execution rejections.
   */
  input?: {
    [k: string]: unknown;
  };
  [k: string]: unknown;
};
export type ToolCallCompleted = ToolCallBase & {
  type?: 'tool_call.completed';
  input: unknown;
  output: string;
  diff?: null | {
    path: string;
    old_text: string | null;
    new_text: string;
  };
  permission_notice?: string | null;
  [k: string]: unknown;
};
export type PermissionRequested = Envelope & {
  type?: 'permission.requested';
  id?: string;
  permission_id: string;
  tool_name: string;
  tool_call_id: string;
  input?: unknown;
  permission_notice?: string | null;
  options: {
    id: string;
    label: string;
    kind: 'allow_once' | 'allow_always' | 'reject_once';
  }[];
  created_at_ms?: number;
  [k: string]: unknown;
};
export type PermissionResolved = Envelope & {
  type?: 'permission.resolved';
  permission_id: string;
  tool_call_id?: string;
  tool_name: string;
  /**
   * The selected option id, or `cancelled` / `unsupported`.
   */
  decision: string;
  [k: string]: unknown;
};
export type RunTerminal = Envelope & {
  type?: 'run.completed' | 'run.cancelled' | 'run.failed';
  stop_reason: 'end_turn' | 'max_turns' | 'time_limit' | 'cancelled' | 'error' | null;
  error?: string | null;
  result_text?: string | null;
  structured_output?: unknown;
  usage?: Usage;
  cumulative_usage?: Usage;
  [k: string]: unknown;
};

export interface Envelope {
  type: string;
  run_id: string;
  session_id: string;
  seq: number;
  ts_ms: number;
  [k: string]: unknown;
}
export interface Usage {
  input_tokens: number;
  output_tokens: number;
  thought_tokens: number;
  cached_read_tokens: number;
  cached_write_tokens: number;
}
export interface EventsGap {
  type: 'events.gap';
  run_id: string;
  missed_through_seq: number;
}
