import assert from 'node:assert/strict';
import test from 'node:test';

import { AnvilClient } from '../dist/index.js';

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

const session = {
  id: 'session-1',
  cwd: '/workspace',
  additional_directories: [],
  model: 'ollama::test',
  behavior_mode: 'LUTZ',
  permission_mode: 'default',
  reasoning_effort: null,
  service_tier: null,
  title: null,
  created_at_ms: 1,
  modified_at_ms: 1,
  updated_at: null,
  history_turns: 0,
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    thought_tokens: 0,
    cached_read_tokens: 0,
    cached_write_tokens: 0,
  },
  usage_cost_usd: null,
};

test('session handle drives the generated lifecycle and structured-output request', async () => {
  const requests = [];
  const client = new AnvilClient({
    retry: false,
    fetch: async (request) => {
      const url = new URL(request.url);
      const body = request.method === 'GET' || request.method === 'DELETE' ? undefined : await request.clone().json();
      requests.push({ method: request.method, path: url.pathname, body });
      if (request.method === 'POST' && url.pathname === '/v1/sessions') return json(session, 201);
      if (request.method === 'GET' && url.pathname === '/v1/sessions/session-1') return json(session);
      if (request.method === 'PATCH') return json({ session, warnings: [] });
      if (request.method === 'GET' && url.pathname.endsWith('/runs')) return json({ runs: [] });
      if (request.method === 'POST' && url.pathname.endsWith('/runs')) {
        return json({
          id: 'run-1',
          session_id: 'session-1',
          status: 'running',
          stop_reason: null,
          error: null,
          result_text: null,
          structured_output: null,
          usage: null,
          created_at_ms: 1,
          finished_at_ms: null,
          last_seq: 0,
        }, 202);
      }
      if (request.method === 'DELETE') return json({ deleted: true });
      throw new Error(`unexpected request: ${request.method} ${url}`);
    },
  });

  const handle = await client.createSession({ cwd: '/workspace' });
  assert.equal((await handle.get()).id, 'session-1');
  await handle.configure({ permission_mode: 'acceptEdits' });
  assert.deepEqual(await handle.runs(), []);
  const run = await handle.run({
    prompt: 'return JSON',
    structured_output: {
      schema_name: 'answer',
      schema: { type: 'object', properties: { answer: { type: 'number' } } },
    },
  });
  assert.equal(run.id, 'run-1');
  assert.equal((await handle.delete()).deleted, true);

  const createRunRequest = requests.find(
    (request) => request.method === 'POST' && request.path.endsWith('/runs'),
  );
  assert.equal(createRunRequest.body.structured_output.schema_name, 'answer');
  assert.deepEqual(
    requests.map(({ method, path }) => [method, path]),
    [
      ['POST', '/v1/sessions'],
      ['GET', '/v1/sessions/session-1'],
      ['PATCH', '/v1/sessions/session-1'],
      ['GET', '/v1/sessions/session-1/runs'],
      ['POST', '/v1/sessions/session-1/runs'],
      ['DELETE', '/v1/sessions/session-1'],
    ],
  );
});
