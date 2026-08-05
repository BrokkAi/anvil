# `@brokkai/anvil-sdk`

Official TypeScript client for the versioned Anvil Agent HTTP API. It can run a complete Anvil session without ACP: create and configure a session, start a run, consume typed events, answer permission requests, cancel work, and collect structured output.

```bash
npm install @brokkai/anvil-sdk
```

The package requires Node.js 22.18 or newer. It is ESM-only; CommonJS callers should use dynamic `import()`. It ships JavaScript, TypeScript declarations, declaration maps, and source maps, with no runtime dependencies.

## Run a turn

Start Anvil's HTTP server first:

```bash
anvil serve --token "$ANVIL_TOKEN"
```

Then create a session and wait for the run:

```js
import { AnvilClient } from '@brokkai/anvil-sdk';

const client = new AnvilClient({
  baseUrl: 'http://127.0.0.1:26845',
  token: process.env.ANVIL_TOKEN,
});

const session = await client.createSession({
  cwd: '/absolute/path/to/workspace',
  permission_mode: 'acceptEdits',
});

try {
  const run = await session.run('Fix the failing tests and explain the result.');
  const result = await run.wait();
  console.log(result.result_text);
} finally {
  await session.delete();
}
```

## Stream tool, plan, and text events

`run.events()` reconnects an interrupted stream from its last sequence number. The iterator ends only after a terminal `run.completed`, `run.cancelled`, or `run.failed` event.

```js
const run = await session.run('Implement the requested change.');

for await (const event of run.events()) {
  switch (event.type) {
    case 'message.delta':
      process.stdout.write(event.text);
      break;
    case 'plan.updated':
      console.log(event.plan);
      break;
    case 'tool_call.started':
    case 'tool_call.completed':
    case 'tool_call.failed':
      console.log(event.type, event.tool_name);
      break;
  }
}
```

## Interactive permissions

Pass `onPermission` to `run.wait()`. The callback receives the generated `Permission` type and returns an offered option id or a generated `PermissionResponseRequest`.

```js
const result = await run.wait({
  onPermission(permission) {
    const safe = permission.tool_name === 'write_file';
    return safe ? 'allow' : { cancel: true };
  },
});
```

Without a callback, `wait()` throws `AnvilPermissionRequiredError` and exposes the pending permission as `error.permission`. You can also use `run.permissions()`, `client.getPermission()`, and `client.respondPermission()` directly.

## Cancellation and cleanup

```js
const controller = new AbortController();
const run = await session.run('Long-running task');

setTimeout(() => controller.abort(), 10_000);
try {
  await run.wait({ signal: controller.signal });
} finally {
  await run.cleanup(); // cancels only if the run is still active
}
```

`SessionHandle` and `RunHandle` also implement `Symbol.asyncDispose` for runtimes using `await using`.

## Structured output

```js
const run = await session.run({
  prompt: 'Return the release readiness decision.',
  structured_output: {
    schema_name: 'release_decision',
    schema: {
      type: 'object',
      required: ['ready', 'reason'],
      properties: {
        ready: { type: 'boolean' },
        reason: { type: 'string' },
      },
    },
  },
});

const result = await run.wait();
console.log(result.structured_output);
```

## Configuration

```js
const client = new AnvilClient({
  baseUrl: 'https://anvil.example.com',
  token: async () => refreshToken(),
  timeoutMs: 30_000,
  retry: {
    maxAttempts: 3,
    baseDelayMs: 250,
    maxDelayMs: 5_000,
  },
  fetch: instrumentedFetch,
  headers: { 'x-client-id': 'worker-1' },
  userAgent: 'my-service/1.0',
});
```

Retries default to safe `GET` and `HEAD` requests on network failures and status 408, 429, 500, 502, 503, or 504. Add methods explicitly only when your application can tolerate replay. `Retry-After` is honored. REST requests default to a 30-second timeout; run event streams use the iterator's signal and reconnect policy instead.

The client uses the standard Fetch, Request, Response, Streams, and AbortSignal APIs. Supply `fetch` when your runtime needs an adapter or instrumentation.

## Browser use

The package is browser-compatible, but the Anvil server must explicitly allow the browser origin with its CORS configuration. A browser cannot keep a bearer token secret: any script running on that origin can read it. Prefer a same-origin backend or short-lived, narrowly scoped credentials. Browsers also forbid setting `User-Agent`, so the `userAgent` option is ignored there.

## Generated wire API

All HTTP paths, request and response bodies, permissions, configuration, usage, error envelopes, and event payload types are generated from:

- `openapi/anvil.v1.yaml` (contract 1.0.0)
- `openapi/anvil.v1.events.schema.json` (contract 1.0.0)

The root package exports the generated low-level functions and types. The raw Fetch client is available as `createWireClient`; dedicated subpaths are also available:

```js
import { createSession } from '@brokkai/anvil-sdk/generated';
import { createClient } from '@brokkai/anvil-sdk/generated/client';

const wire = createClient({
  baseUrl: 'http://127.0.0.1:26845',
  auth: process.env.ANVIL_TOKEN,
  responseStyle: 'data',
  throwOnError: true,
});
const session = await createSession({ client: wire, body: { cwd: process.cwd() } });
```

Generated files are clearly marked and must not be edited. From the repository root, regenerate every SDK artifact with:

```bash
npm ci --prefix sdk/typescript
node scripts/generate-sdks.mjs
```

CI reruns generation and fails if it changes the checked-in output.

More runnable examples are in the repository's `sdk/typescript/examples` directory.

## License

`@brokkai/anvil-sdk` is licensed under `LGPL-3.0-only`. The published package includes the controlling LGPL version 3 text and its incorporated GPL version 3 text.
