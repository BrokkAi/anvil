#!/usr/bin/env node
import assert from 'node:assert/strict';

import {
  configureSession,
  createRun,
  createSession,
  deleteSession,
  getHealth,
  getRun,
  getSession,
  listModels,
  listRuns,
  listTools,
  streamRunEvents,
} from '../dist/generated/openapi/index.js';
import { createClient } from '../dist/generated/openapi/client/index.js';

const [, , baseUrl, cwd] = process.argv;
if (!baseUrl || !cwd) throw new Error('usage: conformance.mjs BASE_URL CWD');

const client = createClient({ baseUrl, throwOnError: true });
assert.equal((await getHealth({ client })).status, 'ok');
assert.ok(Array.isArray((await listModels({ client })).models));
assert.ok(Array.isArray((await listTools({ client })).tools));

const session = await createSession({
  client,
  body: { cwd, permission_mode: 'acceptEdits' },
});
assert.equal((await getSession({ client, path: { session_id: session.id } })).cwd, cwd);
assert.equal(
  (await configureSession({
    client,
    path: { session_id: session.id },
    body: { behavior_mode: 'LUTZ' },
  })).session.id,
  session.id,
);

const run = await createRun({
  client,
  path: { session_id: session.id },
  body: { prompt: 'TypeScript SDK conformance turn' },
});
const { stream } = await streamRunEvents({
  client,
  path: { run_id: run.id },
  sseDefaultRetryDelay: 10,
  sseMaxRetryDelay: 50,
});
const eventTypes = [];
for await (const event of stream) eventTypes.push(event.type);
assert.ok(eventTypes.includes('run.completed'));

const terminal = await getRun({ client, path: { run_id: run.id } });
assert.equal(terminal.status, 'completed');
assert.equal(terminal.result_text, 'SDK conformance complete');
assert.ok(
  (await listRuns({ client, path: { session_id: session.id } })).runs.some(
    (candidate) => candidate.id === run.id,
  ),
);
assert.equal((await deleteSession({ client, path: { session_id: session.id } })).deleted, true);

process.stdout.write('TypeScript SDK conformance passed\n');
