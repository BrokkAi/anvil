#!/usr/bin/env node
import assert from 'node:assert/strict';

import { AnvilClient } from '../dist/index.js';

const [, , baseUrl, cwd] = process.argv;
if (!baseUrl || !cwd) throw new Error('usage: conformance.mjs BASE_URL CWD');

const client = new AnvilClient({ baseUrl, retry: { baseDelayMs: 10, maxDelayMs: 50 } });
assert.equal((await client.health()).status, 'ok');
assert.ok(Array.isArray(await client.models()));
assert.ok(Array.isArray(await client.tools()));

const session = await client.createSession({ cwd, permission_mode: 'acceptEdits' });
assert.equal((await session.get()).cwd, cwd);
assert.equal((await session.configure({ behavior_mode: 'LUTZ' })).session.id, session.id);

const run = await session.run('TypeScript SDK conformance turn');
const terminal = await run.wait({ reconnectDelayMs: 10 });
assert.equal(terminal.status, 'completed');
assert.equal(terminal.result_text, 'SDK conformance complete');
assert.ok((await session.runs()).some((candidate) => candidate.id === run.id));
assert.equal((await session.delete()).deleted, true);

process.stdout.write('TypeScript SDK conformance passed\n');
