import type { ErrorEnvelope, Permission } from './generated/openapi/types.gen.js';

function isErrorEnvelope(error: unknown): error is ErrorEnvelope {
  if (!error || typeof error !== 'object') return false;
  const envelope = error as Partial<ErrorEnvelope>;
  return Boolean(
    envelope.error &&
      typeof envelope.error === 'object' &&
      typeof envelope.error.code === 'string' &&
      typeof envelope.error.message === 'string',
  );
}

export class AnvilApiError extends Error {
  readonly code: ErrorEnvelope['error']['code'] | undefined;
  readonly details: unknown;
  readonly requestId: string | null | undefined;
  readonly status: number | undefined;
  readonly cause: unknown;

  constructor(error: unknown, status?: number) {
    const envelope = isErrorEnvelope(error) ? error : undefined;
    super(envelope?.error.message ?? (error instanceof Error ? error.message : String(error)));
    this.name = 'AnvilApiError';
    this.code = envelope?.error.code;
    this.details = envelope?.error.details;
    this.requestId = envelope?.request_id;
    this.status = status;
    this.cause = error;
  }
}

export class AnvilStreamError extends Error {
  readonly cause: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = 'AnvilStreamError';
    this.cause = cause;
  }
}

export class AnvilPermissionRequiredError extends Error {
  readonly permission: Permission;

  constructor(permission: Permission) {
    super(`run is waiting for permission ${permission.id} (${permission.tool_name})`);
    this.name = 'AnvilPermissionRequiredError';
    this.permission = permission;
  }
}
