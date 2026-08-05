import { createClient as createWireClient } from './generated/openapi/client/index.js';
import type { Client as WireClient } from './generated/openapi/client/index.js';
import {
  cancelRun as wireCancelRun,
  configureSession as wireConfigureSession,
  createRun as wireCreateRun,
  createSession as wireCreateSession,
  deleteSession as wireDeleteSession,
  getHealth as wireGetHealth,
  getPermission as wireGetPermission,
  getRun as wireGetRun,
  getSession as wireGetSession,
  listModels as wireListModels,
  listRunPermissions as wireListRunPermissions,
  listRuns as wireListRuns,
  listSessions as wireListSessions,
  listTools as wireListTools,
  loadSession as wireLoadSession,
  respondPermission as wireRespondPermission,
  resumeSession as wireResumeSession,
  streamRunEvents as wireStreamRunEvents,
} from './generated/openapi/sdk.gen.js';
import type {
  ConfigureSessionResponse,
  CreateRunRequest,
  CreateSessionRequest,
  DeleteSessionResponse,
  Health,
  LifecycleRequest,
  Model,
  Permission,
  PermissionRespondResult,
  PermissionResponseRequest,
  Run,
  Session,
  SessionConfigPatch,
  SessionListResponse,
  Tool,
} from './generated/openapi/types.gen.js';
import type {
  AnvilRunEventStreamV1,
  EventsGap,
  PermissionRequested,
} from './generated/events.gen.js';
import {
  AnvilApiError,
  AnvilPermissionRequiredError,
  AnvilStreamError,
} from './errors.js';

export const SDK_VERSION = '0.24.4';
export const DEFAULT_BASE_URL = 'http://127.0.0.1:26845';

export interface RetryOptions {
  /** Total attempts, including the first request. */
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  methods?: ReadonlyArray<string>;
  statuses?: ReadonlyArray<number>;
  sleep?: (delayMs: number) => Promise<void>;
}

export interface AnvilClientOptions {
  baseUrl?: string;
  token?: string | (() => Promise<string | undefined> | string | undefined);
  timeoutMs?: number | false;
  retry?: RetryOptions | false;
  fetch?: typeof fetch;
  headers?: HeadersInit;
  /** Set only in runtimes that permit the User-Agent header. Browsers ignore it. */
  userAgent?: string | false;
}

export interface RunEventStreamOptions {
  signal?: AbortSignal;
  afterSeq?: number;
  /** Reconnects after the initial connection. */
  maxReconnectAttempts?: number;
  reconnectDelayMs?: number;
  maxReconnectDelayMs?: number;
  sleep?: (delayMs: number) => Promise<void>;
  onReconnect?: (attempt: number, afterSeq: number | undefined) => void;
}

export type PermissionHandler = (
  permission: Permission,
) => PermissionResponseRequest | Promise<PermissionResponseRequest> | string | Promise<string>;

export interface WaitForRunOptions extends RunEventStreamOptions {
  timeoutMs?: number;
  onPermission?: PermissionHandler;
}

const DEFAULT_RETRY_METHODS = ['GET', 'HEAD'];
const DEFAULT_RETRY_STATUSES = [408, 429, 500, 502, 503, 504];
const TERMINAL_EVENT_TYPES = new Set(['run.completed', 'run.cancelled', 'run.failed']);

function isNodeRuntime(): boolean {
  return typeof globalThis === 'object' && 'process' in globalThis;
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
}

function assertNonNegativeNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative finite number`);
  }
}

function retryAfterMs(response: Response): number | undefined {
  const value = response.headers.get('retry-after');
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

function attemptSignal(
  original: AbortSignal | null,
  timeoutMs: number | false,
): { signal: AbortSignal; abort: (reason?: unknown) => void; cleanup: () => void } {
  if (timeoutMs !== false) assertNonNegativeNumber(timeoutMs, 'timeoutMs');
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const onAbort = () => controller.abort(original?.reason);

  if (original?.aborted) {
    controller.abort(original.reason);
  } else {
    original?.addEventListener('abort', onAbort, { once: true });
  }
  if (timeoutMs !== false) {
    timeout = setTimeout(
      () => controller.abort(new DOMException(`request timed out after ${timeoutMs}ms`, 'TimeoutError')),
      timeoutMs,
    );
  }

  return {
    signal: controller.signal,
    abort: (reason) => controller.abort(reason),
    cleanup: () => {
      if (timeout) clearTimeout(timeout);
      original?.removeEventListener('abort', onAbort);
    },
  };
}

function createPolicyFetch(
  fetchImpl: typeof fetch,
  timeoutMs: number | false,
  retry: RetryOptions | false,
): typeof fetch {
  const retryOptions = retry === false ? undefined : retry;
  const maxAttempts = retryOptions?.maxAttempts ?? (retry === false ? 1 : 3);
  assertNonNegativeInteger(maxAttempts, 'retry.maxAttempts');
  if (maxAttempts < 1) throw new RangeError('retry.maxAttempts must be at least 1');
  const baseDelayMs = retryOptions?.baseDelayMs ?? 250;
  const maxDelayMs = retryOptions?.maxDelayMs ?? 5_000;
  assertNonNegativeNumber(baseDelayMs, 'retry.baseDelayMs');
  assertNonNegativeNumber(maxDelayMs, 'retry.maxDelayMs');
  const retryMethods = new Set(
    (retryOptions?.methods ?? DEFAULT_RETRY_METHODS).map((method) => method.toUpperCase()),
  );
  const retryStatuses = new Set(retryOptions?.statuses ?? DEFAULT_RETRY_STATUSES);
  const wait = retryOptions?.sleep ?? sleep;

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const original = new Request(input, init);
    const canRetry = retryMethods.has(original.method.toUpperCase());
    let lastError: unknown;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const { signal, cleanup } = attemptSignal(original.signal, timeoutMs);
      try {
        const response = await fetchImpl(new Request(original.clone(), { signal }));
        if (!canRetry || !retryStatuses.has(response.status) || attempt === maxAttempts) {
          return response;
        }
        await response.body?.cancel();
        const exponential = Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs);
        await wait(retryAfterMs(response) ?? exponential);
      } catch (error) {
        lastError = error;
        if (!canRetry || original.signal.aborted || attempt === maxAttempts) throw error;
        await wait(Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs));
      } finally {
        cleanup();
      }
    }

    throw lastError;
  };
}

function isPermissionRequested(
  event: AnvilRunEventStreamV1,
): event is PermissionRequested {
  return event.type === 'permission.requested' && 'permission_id' in event;
}

function isEventsGap(event: AnvilRunEventStreamV1): event is EventsGap {
  return event.type === 'events.gap';
}

function isRetryableStreamError(error: unknown): boolean {
  if (!(error instanceof Error)) return true;
  const match = error.message.match(/SSE failed: (\d{3})/);
  if (!match) return true;
  const status = Number(match[1]);
  return status === 408 || status === 429 || status >= 500;
}

export class AnvilClient {
  readonly wire: WireClient;
  readonly baseUrl: string;
  private readonly streamWire: WireClient;

  constructor(options: AnvilClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    const fetchImpl = options.fetch ?? globalThis.fetch;
    if (!fetchImpl) throw new Error('a Fetch API implementation is required');
    const headers = new Headers(options.headers);
    const userAgent = options.userAgent === false ? undefined : options.userAgent ?? `@brokkai/anvil-sdk/${SDK_VERSION}`;
    if (userAgent && isNodeRuntime() && !headers.has('user-agent')) {
      headers.set('user-agent', userAgent);
    }
    const shared = {
      auth: options.token,
      baseUrl: this.baseUrl,
      headers,
      responseStyle: 'data' as const,
      throwOnError: true as const,
    };
    this.wire = createWireClient({
      ...shared,
      fetch: createPolicyFetch(fetchImpl, options.timeoutMs ?? 30_000, options.retry ?? {}),
    });
    this.streamWire = createWireClient({ ...shared, fetch: fetchImpl });
    const wrapError = (error: unknown, response: Response | undefined) =>
      error instanceof AnvilApiError ? error : new AnvilApiError(error, response?.status);
    this.wire.interceptors.error.use(wrapError);
    this.streamWire.interceptors.error.use(wrapError);
  }

  async health(): Promise<Health> {
    return wireGetHealth({ client: this.wire });
  }

  async models(): Promise<Model[]> {
    return (await wireListModels({ client: this.wire })).models;
  }

  async tools(): Promise<Tool[]> {
    return (await wireListTools({ client: this.wire })).tools;
  }

  async sessions(cwd?: string): Promise<SessionListResponse['sessions']> {
    return (await wireListSessions({ client: this.wire, query: cwd ? { cwd } : undefined })).sessions;
  }

  async createSession(request: CreateSessionRequest): Promise<SessionHandle> {
    const session = await wireCreateSession({ client: this.wire, body: request });
    return new SessionHandle(this, session.id);
  }

  session(session: Session | string): SessionHandle {
    return new SessionHandle(this, typeof session === 'string' ? session : session.id);
  }

  run(run: Run | string): RunHandle {
    return new RunHandle(this, typeof run === 'string' ? run : run.id);
  }

  async getSession(id: string, includeHistory = false): Promise<Session> {
    return wireGetSession({
      client: this.wire,
      path: { session_id: id },
      query: includeHistory ? { include_history: true } : undefined,
    });
  }

  async configureSession(id: string, patch: SessionConfigPatch): Promise<ConfigureSessionResponse> {
    return wireConfigureSession({ client: this.wire, path: { session_id: id }, body: patch });
  }

  async loadSession(id: string, request: LifecycleRequest): Promise<Session> {
    return wireLoadSession({ client: this.wire, path: { session_id: id }, body: request });
  }

  async resumeSession(id: string, request: LifecycleRequest): Promise<Session> {
    return wireResumeSession({ client: this.wire, path: { session_id: id }, body: request });
  }

  async deleteSession(id: string): Promise<DeleteSessionResponse> {
    return wireDeleteSession({ client: this.wire, path: { session_id: id } });
  }

  async createRun(sessionId: string, request: CreateRunRequest | string): Promise<RunHandle> {
    const body = typeof request === 'string' ? { prompt: request } : request;
    const run = await wireCreateRun({
      client: this.wire,
      path: { session_id: sessionId },
      body,
    });
    return new RunHandle(this, run.id);
  }

  async listRuns(sessionId: string): Promise<Run[]> {
    return (await wireListRuns({ client: this.wire, path: { session_id: sessionId } })).runs;
  }

  async getRun(id: string): Promise<Run> {
    return wireGetRun({ client: this.wire, path: { run_id: id } });
  }

  async cancelRun(id: string): Promise<Run> {
    return wireCancelRun({ client: this.wire, path: { run_id: id } });
  }

  async listRunPermissions(runId: string): Promise<Permission[]> {
    return (await wireListRunPermissions({ client: this.wire, path: { run_id: runId } })).permissions;
  }

  async getPermission(id: string): Promise<Permission> {
    return wireGetPermission({ client: this.wire, path: { permission_id: id } });
  }

  async respondPermission(
    id: string,
    response: PermissionResponseRequest | string,
  ): Promise<PermissionRespondResult> {
    const body = typeof response === 'string' ? { option_id: response } : response;
    return wireRespondPermission({ client: this.wire, path: { permission_id: id }, body });
  }

  async *streamRun(
    runId: string,
    options: RunEventStreamOptions = {},
  ): AsyncGenerator<AnvilRunEventStreamV1> {
    const maxReconnectAttempts = options.maxReconnectAttempts ?? 5;
    assertNonNegativeInteger(maxReconnectAttempts, 'maxReconnectAttempts');
    const reconnectDelayMs = options.reconnectDelayMs ?? 250;
    const maxReconnectDelayMs = options.maxReconnectDelayMs ?? 5_000;
    if (options.afterSeq !== undefined) assertNonNegativeInteger(options.afterSeq, 'afterSeq');
    assertNonNegativeNumber(reconnectDelayMs, 'reconnectDelayMs');
    assertNonNegativeNumber(maxReconnectDelayMs, 'maxReconnectDelayMs');
    const wait = options.sleep ?? sleep;
    let lastSeq = options.afterSeq;
    let reconnectAttempt = 0;
    let lastError: unknown;

    while (true) {
      if (options.signal?.aborted) throw options.signal.reason;
      lastError = undefined;
      const result = await wireStreamRunEvents({
        client: this.streamWire,
        path: { run_id: runId },
        query: lastSeq === undefined ? undefined : { after_seq: lastSeq },
        signal: options.signal,
        sseDefaultRetryDelay: 0,
        sseMaxRetryAttempts: 1,
        onSseError: (error) => {
          lastError = error;
        },
      });

      for await (const candidate of result.stream) {
        const event = candidate as unknown as AnvilRunEventStreamV1;
        if (isEventsGap(event)) {
          lastSeq = Math.max(lastSeq ?? 0, event.missed_through_seq);
        } else if ('seq' in event && typeof event.seq === 'number') {
          lastSeq = Math.max(lastSeq ?? 0, event.seq);
        }
        yield event;
        if (TERMINAL_EVENT_TYPES.has(event.type)) return;
      }

      if (options.signal?.aborted) throw options.signal.reason;
      if (lastError && !isRetryableStreamError(lastError)) {
        throw new AnvilStreamError('run event stream failed', lastError);
      }
      if (reconnectAttempt >= maxReconnectAttempts) {
        throw new AnvilStreamError(
          `run event stream ended before a terminal event after ${reconnectAttempt} reconnects`,
          lastError,
        );
      }
      reconnectAttempt += 1;
      options.onReconnect?.(reconnectAttempt, lastSeq);
      await wait(Math.min(reconnectDelayMs * 2 ** (reconnectAttempt - 1), maxReconnectDelayMs));
    }
  }

  async waitForRun(runId: string, options: WaitForRunOptions = {}): Promise<Run> {
    const combined = attemptSignal(options.signal ?? null, options.timeoutMs ?? false);
    try {
      for await (const event of this.streamRun(runId, { ...options, signal: combined.signal })) {
        if (!isPermissionRequested(event)) continue;
        const permission = await this.getPermission(event.permission_id);
        if (!options.onPermission) throw new AnvilPermissionRequiredError(permission);
        await this.respondPermission(permission.id, await options.onPermission(permission));
      }
      return this.getRun(runId);
    } finally {
      combined.abort(new DOMException('run wait finished', 'AbortError'));
      combined.cleanup();
    }
  }
}

export class SessionHandle {
  constructor(
    private readonly client: AnvilClient,
    readonly id: string,
  ) {}

  get(includeHistory = false): Promise<Session> {
    return this.client.getSession(this.id, includeHistory);
  }

  configure(patch: SessionConfigPatch): Promise<ConfigureSessionResponse> {
    return this.client.configureSession(this.id, patch);
  }

  load(request: LifecycleRequest): Promise<Session> {
    return this.client.loadSession(this.id, request);
  }

  resume(request: LifecycleRequest): Promise<Session> {
    return this.client.resumeSession(this.id, request);
  }

  runs(): Promise<Run[]> {
    return this.client.listRuns(this.id);
  }

  run(request: CreateRunRequest | string): Promise<RunHandle> {
    return this.client.createRun(this.id, request);
  }

  delete(): Promise<DeleteSessionResponse> {
    return this.client.deleteSession(this.id);
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.delete();
  }
}

export class RunHandle {
  constructor(
    private readonly client: AnvilClient,
    readonly id: string,
  ) {}

  get(): Promise<Run> {
    return this.client.getRun(this.id);
  }

  cancel(): Promise<Run> {
    return this.client.cancelRun(this.id);
  }

  permissions(): Promise<Permission[]> {
    return this.client.listRunPermissions(this.id);
  }

  events(options?: RunEventStreamOptions): AsyncGenerator<AnvilRunEventStreamV1> {
    return this.client.streamRun(this.id, options);
  }

  wait(options?: WaitForRunOptions): Promise<Run> {
    return this.client.waitForRun(this.id, options);
  }

  async cleanup(): Promise<Run> {
    const run = await this.get();
    return run.status === 'running' ? this.cancel() : run;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.cleanup();
  }
}
