// Generated from openapi/anvil.v1.yaml and openapi/anvil.v1.events.schema.json.
// Generator: @hey-api/openapi-ts 0.99.0. Do not edit by hand.

export type ClientOptions = {
    baseUrl: 'http://127.0.0.1:26845' | (string & {});
};

/**
 * Stable envelope for every non-2xx JSON response. The `request_id` is also echoed on every response in the `x-request-id` header.
 */
export type ErrorEnvelope = {
    error: {
        code: 'invalid_argument' | 'unauthorized' | 'forbidden' | 'not_found' | 'conflict' | 'internal';
        message: string;
        /**
         * Machine-readable context (offending field, supported values, workspace roots, ...).
         */
        details?: unknown;
    };
    request_id: string | null;
};

export type Health = {
    status: 'ok';
    name: 'anvil';
    version: string;
    models_discovered: number;
};

export type ModelsResponse = {
    models: Array<Model>;
    default_model: string | null;
};

/**
 * One discovered model. Ids are wire-tagged `<source>::<id>` (for example `codex::gpt-5-codex`, `ollama::llama3:latest`).
 */
export type Model = {
    id: string;
    default_reasoning_level: string | null;
    supported_reasoning_levels: Array<{
        effort: string;
        description: string;
    }>;
    service_tiers: Array<{
        id: string;
        name: string;
        description: string;
    }>;
    supports_images: boolean | null;
    context_length: number | null;
    pricing: null | {
        input_cost_per_token_usd: number;
        output_cost_per_token_usd: number;
    };
};

export type ToolsResponse = {
    tools: Array<Tool>;
};

export type Tool = {
    name: string;
    /**
     * Permission-gate classification (ACP tool kind).
     */
    kind: string;
    display_name: string;
    concurrency_safe: boolean;
    source: 'builtin' | 'mcp';
};

export type Usage = {
    input_tokens: number;
    output_tokens: number;
    thought_tokens: number;
    cached_read_tokens: number;
    cached_write_tokens: number;
};

export type Session = {
    id: string;
    cwd: string;
    additional_directories: Array<string>;
    model: string;
    behavior_mode: 'LUTZ' | 'PLAN';
    permission_mode: 'default' | 'auto' | 'acceptEdits' | 'readOnly' | 'bypassPermissions';
    reasoning_effort: string | null;
    service_tier: string | null;
    title: string | null;
    created_at_ms: number;
    modified_at_ms: number;
    /**
     * RFC 3339 rendering of `modified_at_ms`.
     */
    updated_at: string | null;
    history_turns: number;
    usage: Usage;
    usage_cost_usd: number | null;
    /**
     * Present only on `load` responses and `?include_history=true` inspections.
     */
    history?: Array<HistoryTurn>;
};

export type HistoryTurn = {
    user_prompt: string;
    agent_response: string;
    summary: string | null;
    tool_exchanges: Array<{
        call_id: string;
        tool_name: string;
        arguments: string;
        result: string;
        status: 'completed' | 'failed';
        permission_notice: string | null;
    }>;
};

export type SessionListResponse = {
    sessions: Array<{
        id: string;
        title: string | null;
        created_at_ms: number;
        modified_at_ms: number;
        updated_at: string | null;
        cwd: string | null;
        resident: boolean;
    }>;
};

/**
 * Anvil's internal MCP server configuration shape. Per-session servers are additive to the daemon's canonical setup.
 */
export type McpServerConfig = {
    name: string;
    transport?: 'stdio' | 'http' | 'sse';
    command?: string;
    url?: string | null;
    headers?: Array<McpEnvVar>;
    args?: Array<string>;
    env?: Array<McpEnvVar>;
    framing?: 'content-length' | 'line';
    enabled?: boolean;
};

export type McpEnvVar = {
    name: string;
    value: string;
};

/**
 * Client-owned per-session selectors, mirroring the ACP `SessionConfigOption` ids. At least one property is required.
 */
export type SessionConfigPatch = {
    model?: string;
    /**
     * A model-supported effort, `off` to omit reasoning controls, or empty string to clear back to the model default.
     */
    reasoning_effort?: string;
    /**
     * A provider tier id, or empty string to clear.
     */
    service_tier?: string;
    behavior_mode?: 'LUTZ' | 'PLAN';
    permission_mode?: 'default' | 'auto' | 'acceptEdits' | 'readOnly' | 'bypassPermissions';
};

export type CreateSessionRequest = {
    /**
     * Absolute working directory for the session.
     */
    cwd: string;
    additional_directories?: Array<string>;
    mcp_servers?: Array<McpServerConfig>;
    model?: string;
    reasoning_effort?: string;
    service_tier?: string;
    behavior_mode?: 'LUTZ' | 'PLAN';
    permission_mode?: 'default' | 'auto' | 'acceptEdits' | 'readOnly' | 'bypassPermissions';
};

export type LifecycleRequest = {
    /**
     * Must match the cwd the session was created under.
     */
    cwd: string;
    additional_directories?: Array<string>;
    mcp_servers?: Array<McpServerConfig>;
};

export type ConfigureSessionResponse = {
    session: Session;
    warnings: Array<string>;
};

export type DeleteSessionResponse = {
    deleted: boolean;
};

export type StructuredOutputSpec = {
    /**
     * JSON Schema the final response must satisfy.
     */
    schema: unknown;
    schema_name?: string;
    allow_coercion?: boolean;
    /**
     * Request basic JSON mode instead of strict `json_schema` on providers that support only the former.
     */
    prefer_json_object?: boolean;
};

export type CreateRunRequest = {
    prompt: string;
    structured_output?: null | StructuredOutputSpec;
};

export type Run = {
    id: string;
    session_id: string;
    status: 'running' | 'completed' | 'cancelled' | 'failed';
    stop_reason: 'end_turn' | 'max_turns' | 'time_limit' | 'cancelled' | 'error' | null;
    error: string | null;
    result_text: string | null;
    /**
     * Structured-output validation result (`status` = `success` / `coerced_success` / `validation_error`) when the run requested one.
     */
    structured_output: unknown;
    usage: null | Usage;
    created_at_ms: number;
    finished_at_ms: number | null;
    last_seq: number;
};

export type RunListResponse = {
    runs: Array<Run>;
};

export type PermissionOption = {
    id: string;
    label: string;
    kind: 'allow_once' | 'allow_always' | 'reject_once';
};

export type Permission = {
    id: string;
    run_id: string;
    session_id: string;
    tool_name: string;
    tool_call_id: string;
    /**
     * Raw tool-call arguments awaiting approval.
     */
    input: unknown;
    permission_notice: string | null;
    options: Array<PermissionOption>;
    created_at_ms: number;
    status: 'pending';
};

export type PermissionListResponse = {
    permissions: Array<Permission>;
};

/**
 * Pass exactly one of `option_id` or `cancel = true`.
 */
export type PermissionResponseRequest = {
    option_id?: string | null;
    cancel?: boolean;
};

export type PermissionRespondResult = {
    resolved: true;
    id: string;
};

export type EventEnvelope = {
    type: string;
    run_id: string;
    session_id: string;
    seq: number;
    ts_ms: number;
};

export type EventUsage = {
    input_tokens: number;
    output_tokens: number;
    thought_tokens: number;
    cached_read_tokens: number;
    cached_write_tokens: number;
};

export type EventToolCallBase = EventEnvelope & {
    call_id: string;
    tool_name: string;
};

export type EventRunCreated = EventEnvelope & {
    type?: 'run.created';
    prompt_chars: number;
};

export type EventMessageDelta = EventEnvelope & {
    type?: 'message.delta';
    text: string;
};

export type EventThoughtDelta = EventEnvelope & {
    type?: 'thought.delta';
    text: string;
};

export type EventPlanUpdated = EventEnvelope & {
    type?: 'plan.updated';
    plan: {
        explanation?: string | null;
        plan: Array<{
            step: string;
            status: 'pending' | 'in_progress' | 'completed';
        }>;
    };
};

export type EventToolCallStarted = EventToolCallBase & {
    type?: 'tool_call.started';
    input: unknown;
    /**
     * Present and true when the rendered card would hide the input (adapters use a static title).
     */
    oversized?: boolean;
};

export type EventToolCallBlocked = EventToolCallBase & {
    type?: 'tool_call.blocked';
    input: unknown;
    reason: string;
};

export type EventToolCallInProgress = EventToolCallBase & {
    type?: 'tool_call.in_progress';
};

export type EventToolCallFailed = EventToolCallBase & {
    type?: 'tool_call.failed';
    reason: string;
    permission_notice?: string | null;
    /**
     * Present for post-execution failures; null for pre-execution rejections.
     */
    input?: unknown;
};

export type EventToolCallCompleted = EventToolCallBase & {
    type?: 'tool_call.completed';
    input: unknown;
    output: string;
    diff?: null | {
        path: string;
        old_text: string | null;
        new_text: string;
    };
    permission_notice?: string | null;
};

export type EventPermissionRequested = EventEnvelope & {
    type?: 'permission.requested';
    id?: string;
    permission_id: string;
    tool_name: string;
    tool_call_id: string;
    input?: unknown;
    permission_notice?: string | null;
    options: Array<{
        id: string;
        label: string;
        kind: 'allow_once' | 'allow_always' | 'reject_once';
    }>;
    created_at_ms?: number;
};

export type EventPermissionResolved = EventEnvelope & {
    type?: 'permission.resolved';
    permission_id: string;
    tool_call_id?: string;
    tool_name: string;
    /**
     * The selected option id, or `cancelled` / `unsupported`.
     */
    decision: string;
};

export type EventRunTerminal = EventEnvelope & {
    type?: 'run.completed' | 'run.cancelled' | 'run.failed';
    stop_reason: 'end_turn' | 'max_turns' | 'time_limit' | 'cancelled' | 'error' | null;
    error?: string | null;
    result_text?: string | null;
    structured_output?: unknown;
    usage?: EventUsage;
    cumulative_usage?: EventUsage;
};

export type EventEventsGap = {
    type: 'events.gap';
    run_id: string;
    missed_through_seq: number;
};

/**
 * Anvil run event stream (v1)
 *
 * Contract for the JSON payload carried in each Server-Sent Event on GET /v1/runs/{run_id}/events. The SSE `id` field is the decimal `seq`; the SSE `event` field equals `type`. Every event shares the envelope fields and one payload variant selected by `type`. The stream ends after a terminal `run.completed`, `run.cancelled`, or `run.failed` event. The synthetic `events.gap` notice is emitted without an envelope `seq` when a reconnect falls behind the bounded replay buffer.
 */
export type AnvilRunEvent = EventRunCreated | EventMessageDelta | EventThoughtDelta | EventPlanUpdated | EventToolCallStarted | EventToolCallBlocked | EventToolCallInProgress | EventToolCallFailed | EventToolCallCompleted | EventPermissionRequested | EventPermissionResolved | EventRunTerminal | EventEventsGap;

export type SessionId = string;

export type RunId = string;

export type PermissionId = string;

export type GetHealthData = {
    body?: never;
    path?: never;
    query?: never;
    url: '/health';
};

export type GetHealthResponses = {
    /**
     * Daemon is serving.
     */
    200: Health;
};

export type GetHealthResponse = GetHealthResponses[keyof GetHealthResponses];

export type ListModelsData = {
    body?: never;
    path?: never;
    query?: {
        /**
         * Re-probe providers before answering. Discovery failures are never fatal; the cached catalog is served.
         */
        refresh?: boolean;
    };
    url: '/v1/models';
};

export type ListModelsErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
};

export type ListModelsError = ListModelsErrors[keyof ListModelsErrors];

export type ListModelsResponses = {
    /**
     * Discovered models.
     */
    200: ModelsResponse;
};

export type ListModelsResponse = ListModelsResponses[keyof ListModelsResponses];

export type ListToolsData = {
    body?: never;
    path?: never;
    query?: never;
    url: '/v1/tools';
};

export type ListToolsErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
};

export type ListToolsError = ListToolsErrors[keyof ListToolsErrors];

export type ListToolsResponses = {
    /**
     * Tool catalog.
     */
    200: ToolsResponse;
};

export type ListToolsResponse = ListToolsResponses[keyof ListToolsResponses];

export type ListSessionsData = {
    body?: never;
    path?: never;
    query?: {
        /**
         * Absolute workspace path whose on-disk session archives should be included.
         */
        cwd?: string;
    };
    url: '/v1/sessions';
};

export type ListSessionsErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
};

export type ListSessionsError = ListSessionsErrors[keyof ListSessionsErrors];

export type ListSessionsResponses = {
    /**
     * Session listing.
     */
    200: SessionListResponse;
};

export type ListSessionsResponse = ListSessionsResponses[keyof ListSessionsResponses];

export type CreateSessionData = {
    body: CreateSessionRequest;
    path?: never;
    query?: never;
    url: '/v1/sessions';
};

export type CreateSessionErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
};

export type CreateSessionError = CreateSessionErrors[keyof CreateSessionErrors];

export type CreateSessionResponses = {
    /**
     * Session created.
     */
    201: Session;
};

export type CreateSessionResponse = CreateSessionResponses[keyof CreateSessionResponses];

export type DeleteSessionData = {
    body?: never;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}';
};

export type DeleteSessionErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
};

export type DeleteSessionError = DeleteSessionErrors[keyof DeleteSessionErrors];

export type DeleteSessionResponses = {
    /**
     * Deletion outcome.
     */
    200: DeleteSessionResponse;
};

export type DeleteSessionResponse2 = DeleteSessionResponses[keyof DeleteSessionResponses];

export type GetSessionData = {
    body?: never;
    path: {
        session_id: string;
    };
    query?: {
        /**
         * Workspace to cold-load the session archive from when it is not resident. Defaults to the daemon's process cwd.
         */
        cwd?: string;
        include_history?: boolean;
    };
    url: '/v1/sessions/{session_id}';
};

export type GetSessionErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type GetSessionError = GetSessionErrors[keyof GetSessionErrors];

export type GetSessionResponses = {
    /**
     * Session resource.
     */
    200: Session;
};

export type GetSessionResponse = GetSessionResponses[keyof GetSessionResponses];

export type ConfigureSessionData = {
    body: SessionConfigPatch;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}';
};

export type ConfigureSessionErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type ConfigureSessionError = ConfigureSessionErrors[keyof ConfigureSessionErrors];

export type ConfigureSessionResponses = {
    /**
     * Updated session plus warnings.
     */
    200: ConfigureSessionResponse;
};

export type ConfigureSessionResponse2 = ConfigureSessionResponses[keyof ConfigureSessionResponses];

export type LoadSessionData = {
    body: LifecycleRequest;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}/load';
};

export type LoadSessionErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
    /**
     * State conflict — cwd mismatch, prompt already in flight, no model configured, or permission already resolved (`error.code = "conflict"`).
     */
    409: ErrorEnvelope;
};

export type LoadSessionError = LoadSessionErrors[keyof LoadSessionErrors];

export type LoadSessionResponses = {
    /**
     * Reopened session with `history` present.
     */
    200: Session;
};

export type LoadSessionResponse = LoadSessionResponses[keyof LoadSessionResponses];

export type ResumeSessionData = {
    body: LifecycleRequest;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}/resume';
};

export type ResumeSessionErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Refused by server policy — workspace roots or `bypassPermissions` gating (`error.code = "forbidden"`).
     */
    403: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
    /**
     * State conflict — cwd mismatch, prompt already in flight, no model configured, or permission already resolved (`error.code = "conflict"`).
     */
    409: ErrorEnvelope;
};

export type ResumeSessionError = ResumeSessionErrors[keyof ResumeSessionErrors];

export type ResumeSessionResponses = {
    /**
     * Reopened session.
     */
    200: Session;
};

export type ResumeSessionResponse = ResumeSessionResponses[keyof ResumeSessionResponses];

export type ListRunsData = {
    body?: never;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}/runs';
};

export type ListRunsErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
};

export type ListRunsError = ListRunsErrors[keyof ListRunsErrors];

export type ListRunsResponses = {
    /**
     * Runs for the session.
     */
    200: RunListResponse;
};

export type ListRunsResponse = ListRunsResponses[keyof ListRunsResponses];

export type CreateRunData = {
    body: CreateRunRequest;
    path: {
        session_id: string;
    };
    query?: never;
    url: '/v1/sessions/{session_id}/runs';
};

export type CreateRunErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
    /**
     * State conflict — cwd mismatch, prompt already in flight, no model configured, or permission already resolved (`error.code = "conflict"`).
     */
    409: ErrorEnvelope;
};

export type CreateRunError = CreateRunErrors[keyof CreateRunErrors];

export type CreateRunResponses = {
    /**
     * Run accepted and executing.
     */
    202: Run;
};

export type CreateRunResponse = CreateRunResponses[keyof CreateRunResponses];

export type GetRunData = {
    body?: never;
    path: {
        run_id: string;
    };
    query?: never;
    url: '/v1/runs/{run_id}';
};

export type GetRunErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type GetRunError = GetRunErrors[keyof GetRunErrors];

export type GetRunResponses = {
    /**
     * Run resource; terminal fields are populated once the run completes, fails, or is cancelled.
     */
    200: Run;
};

export type GetRunResponse = GetRunResponses[keyof GetRunResponses];

export type StreamRunEventsData = {
    body?: never;
    headers?: {
        'Last-Event-ID'?: string;
    };
    path: {
        run_id: string;
    };
    query?: {
        /**
         * Replay events with sequence numbers greater than this. The `Last-Event-ID` header wins when both are present.
         */
        after_seq?: number;
    };
    url: '/v1/runs/{run_id}/events';
};

export type StreamRunEventsErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type StreamRunEventsError = StreamRunEventsErrors[keyof StreamRunEventsErrors];

export type StreamRunEventsResponses = {
    /**
     * SSE stream of run events.
     */
    200: AnvilRunEvent;
};

export type StreamRunEventsResponse = StreamRunEventsResponses[keyof StreamRunEventsResponses];

export type CancelRunData = {
    body?: never;
    path: {
        run_id: string;
    };
    query?: never;
    url: '/v1/runs/{run_id}/cancel';
};

export type CancelRunErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type CancelRunError = CancelRunErrors[keyof CancelRunErrors];

export type CancelRunResponses = {
    /**
     * Current run resource.
     */
    200: Run;
};

export type CancelRunResponse = CancelRunResponses[keyof CancelRunResponses];

export type ListRunPermissionsData = {
    body?: never;
    path: {
        run_id: string;
    };
    query?: never;
    url: '/v1/runs/{run_id}/permissions';
};

export type ListRunPermissionsErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type ListRunPermissionsError = ListRunPermissionsErrors[keyof ListRunPermissionsErrors];

export type ListRunPermissionsResponses = {
    /**
     * Pending permission requests, oldest first.
     */
    200: PermissionListResponse;
};

export type ListRunPermissionsResponse = ListRunPermissionsResponses[keyof ListRunPermissionsResponses];

export type GetPermissionData = {
    body?: never;
    path: {
        permission_id: string;
    };
    query?: never;
    url: '/v1/permissions/{permission_id}';
};

export type GetPermissionErrors = {
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
};

export type GetPermissionError = GetPermissionErrors[keyof GetPermissionErrors];

export type GetPermissionResponses = {
    /**
     * Pending permission request.
     */
    200: Permission;
};

export type GetPermissionResponse = GetPermissionResponses[keyof GetPermissionResponses];

export type RespondPermissionData = {
    body: PermissionResponseRequest;
    path: {
        permission_id: string;
    };
    query?: never;
    url: '/v1/permissions/{permission_id}/respond';
};

export type RespondPermissionErrors = {
    /**
     * Invalid request (`error.code = "invalid_argument"`).
     */
    400: ErrorEnvelope;
    /**
     * Missing or invalid bearer token (`error.code = "unauthorized"`).
     */
    401: ErrorEnvelope;
    /**
     * Unknown resource (`error.code = "not_found"`).
     */
    404: ErrorEnvelope;
    /**
     * State conflict — cwd mismatch, prompt already in flight, no model configured, or permission already resolved (`error.code = "conflict"`).
     */
    409: ErrorEnvelope;
};

export type RespondPermissionError = RespondPermissionErrors[keyof RespondPermissionErrors];

export type RespondPermissionResponses = {
    /**
     * The request was resolved by this response.
     */
    200: PermissionRespondResult;
};

export type RespondPermissionResponse = RespondPermissionResponses[keyof RespondPermissionResponses];
