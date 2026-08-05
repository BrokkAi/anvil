// Generated from openapi/anvil.v1.yaml (Anvil Agent API contract 1.0.0).
// Generator: @hey-api/openapi-ts 0.99.0. Do not edit by hand.

import { client } from './client.gen.js';
import type { Client, ClientMeta, Options as Options2, RequestResult, ServerSentEventsResult, TDataShape } from './client/index.js';
import type { CancelRunData, CancelRunErrors, CancelRunResponses, ConfigureSessionData, ConfigureSessionErrors, ConfigureSessionResponses, CreateRunData, CreateRunErrors, CreateRunResponses, CreateSessionData, CreateSessionErrors, CreateSessionResponses, DeleteSessionData, DeleteSessionErrors, DeleteSessionResponses, GetHealthData, GetHealthResponses, GetPermissionData, GetPermissionErrors, GetPermissionResponses, GetRunData, GetRunErrors, GetRunResponses, GetSessionData, GetSessionErrors, GetSessionResponses, ListModelsData, ListModelsErrors, ListModelsResponses, ListRunPermissionsData, ListRunPermissionsErrors, ListRunPermissionsResponses, ListRunsData, ListRunsErrors, ListRunsResponses, ListSessionsData, ListSessionsErrors, ListSessionsResponses, ListToolsData, ListToolsErrors, ListToolsResponses, LoadSessionData, LoadSessionErrors, LoadSessionResponses, RespondPermissionData, RespondPermissionErrors, RespondPermissionResponses, ResumeSessionData, ResumeSessionErrors, ResumeSessionResponses, StreamRunEventsData, StreamRunEventsErrors, StreamRunEventsResponse, StreamRunEventsResponses } from './types.gen.js';

export type Options<TData extends TDataShape = TDataShape, ThrowOnError extends boolean = boolean, TResponse = unknown> = Options2<TData, ThrowOnError, TResponse> & {
    /**
     * You can provide a client instance returned by `createClient()` instead of
     * individual options. This might be also useful if you want to implement a
     * custom client.
     */
    client?: Client;
    /**
     * You can pass arbitrary values through the `meta` object. This can be
     * used to access values that aren't defined as part of the SDK function.
     */
    meta?: keyof ClientMeta extends never ? Record<string, unknown> : ClientMeta;
};

/**
 * Liveness, version, and model-discovery state.
 *
 * Always unauthenticated, including when a bearer token is configured, so orchestrators can probe liveness without credentials.
 */
export const getHealth = <ThrowOnError extends boolean = true>(options?: Options<GetHealthData, ThrowOnError>): RequestResult<GetHealthResponses, unknown, ThrowOnError, 'data'> => (options?.client ?? client).get<GetHealthResponses, unknown, ThrowOnError, 'data'>({
    responseStyle: 'data',
    url: '/health',
    ...options
});

/**
 * Model catalog and default model.
 */
export const listModels = <ThrowOnError extends boolean = true>(options?: Options<ListModelsData, ThrowOnError>): RequestResult<ListModelsResponses, ListModelsErrors, ThrowOnError, 'data'> => (options?.client ?? client).get<ListModelsResponses, ListModelsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/models',
    ...options
});

/**
 * Static harness tool catalog.
 *
 * The catalog of built-in and default MCP-loaded tools the harness supports. A session's live registry may expose fewer (server disabled) or more (extra MCP servers) at prompt time.
 */
export const listTools = <ThrowOnError extends boolean = true>(options?: Options<ListToolsData, ThrowOnError>): RequestResult<ListToolsResponses, ListToolsErrors, ThrowOnError, 'data'> => (options?.client ?? client).get<ListToolsResponses, ListToolsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/tools',
    ...options
});

/**
 * List sessions.
 *
 * Resident (in-memory) sessions, plus persisted sessions on disk under `cwd` when the query parameter is supplied. Sessions without a title (never prompted) are omitted, matching ACP `session/list`.
 */
export const listSessions = <ThrowOnError extends boolean = true>(options?: Options<ListSessionsData, ThrowOnError>): RequestResult<ListSessionsResponses, ListSessionsErrors, ThrowOnError, 'data'> => (options?.client ?? client).get<ListSessionsResponses, ListSessionsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions',
    ...options
});

/**
 * Create and configure a session.
 *
 * Creates the session, then applies the optional configuration selectors through the same validation path ACP uses. If any selector is invalid the session is rolled back and an error is returned, so a failed create leaves no state behind.
 */
export const createSession = <ThrowOnError extends boolean = true>(options: Options<CreateSessionData, ThrowOnError>): RequestResult<CreateSessionResponses, CreateSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).post<CreateSessionResponses, CreateSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});

/**
 * Delete a session (idempotent).
 *
 * Cancels any in-flight prompt, removes the persisted archive, and reports whether anything was deleted. Unknown ids report `deleted: false` rather than an error.
 */
export const deleteSession = <ThrowOnError extends boolean = true>(options: Options<DeleteSessionData, ThrowOnError>): RequestResult<DeleteSessionResponses, DeleteSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).delete<DeleteSessionResponses, DeleteSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}',
    ...options
});

/**
 * Inspect a session.
 */
export const getSession = <ThrowOnError extends boolean = true>(options: Options<GetSessionData, ThrowOnError>): RequestResult<GetSessionResponses, GetSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).get<GetSessionResponses, GetSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}',
    ...options
});

/**
 * Reconfigure session selectors.
 *
 * Applies selectors sequentially through the shared ACP validation path. On a validation failure, selectors already applied stay applied. Model changes that drop an unsupported reasoning effort or service tier pick surface a warning.
 */
export const configureSession = <ThrowOnError extends boolean = true>(options: Options<ConfigureSessionData, ThrowOnError>): RequestResult<ConfigureSessionResponses, ConfigureSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).patch<ConfigureSessionResponses, ConfigureSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});

/**
 * Reopen a persisted session with history.
 *
 * The HTTP analogue of ACP `session/load`; the response embeds the conversation history. The request cwd must match the cwd the session was created under.
 */
export const loadSession = <ThrowOnError extends boolean = true>(options: Options<LoadSessionData, ThrowOnError>): RequestResult<LoadSessionResponses, LoadSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).post<LoadSessionResponses, LoadSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}/load',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});

/**
 * Reopen a persisted session without history.
 */
export const resumeSession = <ThrowOnError extends boolean = true>(options: Options<ResumeSessionData, ThrowOnError>): RequestResult<ResumeSessionResponses, ResumeSessionErrors, ThrowOnError, 'data'> => (options.client ?? client).post<ResumeSessionResponses, ResumeSessionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}/resume',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});

/**
 * List a session's runs, newest first.
 */
export const listRuns = <ThrowOnError extends boolean = true>(options: Options<ListRunsData, ThrowOnError>): RequestResult<ListRunsResponses, ListRunsErrors, ThrowOnError, 'data'> => (options.client ?? client).get<ListRunsResponses, ListRunsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}/runs',
    ...options
});

/**
 * Start an asynchronous prompt run.
 *
 * Returns the run resource promptly; the turn executes in the background. One prompt may be in flight per session: a duplicate run is rejected with 409 until the active run reaches a terminal state or is cancelled. Progress is observable via the run's SSE event stream and by polling the run resource.
 *
 */
export const createRun = <ThrowOnError extends boolean = true>(options: Options<CreateRunData, ThrowOnError>): RequestResult<CreateRunResponses, CreateRunErrors, ThrowOnError, 'data'> => (options.client ?? client).post<CreateRunResponses, CreateRunErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/sessions/{session_id}/runs',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});

/**
 * Poll run state and result.
 */
export const getRun = <ThrowOnError extends boolean = true>(options: Options<GetRunData, ThrowOnError>): RequestResult<GetRunResponses, GetRunErrors, ThrowOnError, 'data'> => (options.client ?? client).get<GetRunResponses, GetRunErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/runs/{run_id}',
    ...options
});

/**
 * Stream run events as Server-Sent Events.
 *
 * Ordered `text/event-stream`. Each SSE message carries the event's sequence number as its `id`, the event type as its `event` name, and a JSON payload described by the companion event schema (`anvil.v1.events.schema.json`) as its `data`. Reconnect with the standard `Last-Event-ID` header (or the `after_seq` query parameter) to replay from a bounded per-run buffer; falling behind the buffer yields an `events.gap` notice. The stream ends after the terminal `run.*` event. Dropping the stream never cancels the run.
 *
 */
export const streamRunEvents = <ThrowOnError extends boolean = true>(options: Options<StreamRunEventsData, ThrowOnError, StreamRunEventsResponse>): Promise<ServerSentEventsResult<StreamRunEventsResponses>> => (options.client ?? client).sse.get<StreamRunEventsResponses, StreamRunEventsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/runs/{run_id}/events',
    ...options
});

/**
 * Cancel the active turn (idempotent).
 *
 * Cancels through the same token the runtime uses for ACP cancellation. A cancel racing natural completion loses gracefully; cancelling a terminal run returns its current resource.
 */
export const cancelRun = <ThrowOnError extends boolean = true>(options: Options<CancelRunData, ThrowOnError>): RequestResult<CancelRunResponses, CancelRunErrors, ThrowOnError, 'data'> => (options.client ?? client).post<CancelRunResponses, CancelRunErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/runs/{run_id}/cancel',
    ...options
});

/**
 * Pending interactive permission requests for a run.
 */
export const listRunPermissions = <ThrowOnError extends boolean = true>(options: Options<ListRunPermissionsData, ThrowOnError>): RequestResult<ListRunPermissionsResponses, ListRunPermissionsErrors, ThrowOnError, 'data'> => (options.client ?? client).get<ListRunPermissionsResponses, ListRunPermissionsErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/runs/{run_id}/permissions',
    ...options
});

/**
 * Inspect one pending permission request.
 */
export const getPermission = <ThrowOnError extends boolean = true>(options: Options<GetPermissionData, ThrowOnError>): RequestResult<GetPermissionResponses, GetPermissionErrors, ThrowOnError, 'data'> => (options.client ?? client).get<GetPermissionResponses, GetPermissionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/permissions/{permission_id}',
    ...options
});

/**
 * Approve, reject, or cancel a permission request.
 *
 * Resolves a pending request exactly once. `option_id` must be one of the ids offered on the request (for example `allow`, `allow_always`, `allow_outside_sandbox`, `reject`); alternatively pass `cancel: true` to cancel the request. Requests that were already resolved — including by run cancellation or completion — answer 404 (no longer pending) or 409 (racing resolution).
 *
 */
export const respondPermission = <ThrowOnError extends boolean = true>(options: Options<RespondPermissionData, ThrowOnError>): RequestResult<RespondPermissionResponses, RespondPermissionErrors, ThrowOnError, 'data'> => (options.client ?? client).post<RespondPermissionResponses, RespondPermissionErrors, ThrowOnError, 'data'>({
    responseStyle: 'data',
    security: [{ scheme: 'bearer', type: 'http' }],
    url: '/v1/permissions/{permission_id}/respond',
    ...options,
    headers: {
        'Content-Type': 'application/json',
        ...options.headers
    }
});
