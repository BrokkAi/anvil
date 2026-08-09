# `@brokkai/anvil-sdk`

Generated TypeScript client for the versioned Anvil Agent HTTP API.

```bash
npm install @brokkai/anvil-sdk
```

Every exported runtime API and wire type is generated from:

- `openapi/anvil.v1.yaml`
- `openapi/anvil.v1.events.schema.json`

The package contains no handwritten client implementation. It is ESM-only and
requires Node.js 22.18 or newer.

```js
import { createRun, createSession, getRun, streamRunEvents } from '@brokkai/anvil-sdk';
import { createClient } from '@brokkai/anvil-sdk/client';

const client = createClient({
  baseUrl: 'http://127.0.0.1:26845',
  auth: process.env.ANVIL_TOKEN,
  throwOnError: true,
});

const session = await createSession({
  client,
  body: { cwd: process.cwd(), permission_mode: 'acceptEdits' },
});
const run = await createRun({
  client,
  path: { session_id: session.id },
  body: { prompt: 'Fix the failing tests.' },
});

const { stream } = await streamRunEvents({
  client,
  path: { run_id: run.id },
});
for await (const event of stream) console.log(event);

console.log(await getRun({ client, path: { run_id: run.id } }));
```

The generated Fetch client supports configurable base URLs, bearer
authentication, request headers, interceptors, custom `fetch`, and SSE
reconnection with `Last-Event-ID`.

Do not expose a bearer token to untrusted browser code. Browser use also
requires an explicitly allowed CORS origin on the Anvil server.

From the repository root, regenerate every official SDK with:

```bash
node scripts/generate-sdks.mjs
```

Generated files are marked and must not be edited by hand. Release automation
regenerates them before building and publishing the package.

## License

`@brokkai/anvil-sdk` is licensed under `LGPL-3.0-only`.
