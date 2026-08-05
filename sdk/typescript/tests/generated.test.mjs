import assert from 'node:assert/strict';
import test from 'node:test';

import { createSession, getHealth, listModels } from '../dist/generated/openapi/index.js';
import { createClient } from '../dist/generated/openapi/client/index.js';

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

test('generated client supplies paths, serialization, and bearer authentication', async () => {
  const requests = [];
  const client = createClient({
    baseUrl: 'https://anvil.invalid',
    auth: 'secret',
    throwOnError: true,
    fetch: async (request) => {
      requests.push(request);
      const url = new URL(request.url);
      if (url.pathname === '/health') return json({ status: 'ok', version: 'test', models_ready: true });
      if (url.pathname === '/v1/models') return json({ models: [], default_model: null });
      if (url.pathname === '/v1/sessions') {
        assert.deepEqual(await request.clone().json(), { cwd: '/workspace' });
        return json({
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
        }, 201);
      }
      throw new Error(`unexpected request: ${request.method} ${request.url}`);
    },
  });

  assert.equal((await getHealth({ client })).status, 'ok');
  assert.deepEqual((await listModels({ client })).models, []);
  assert.equal((await createSession({ client, body: { cwd: '/workspace' } })).id, 'session-1');
  assert.equal(requests[0].headers.get('authorization'), null);
  assert.equal(requests[1].headers.get('authorization'), 'Bearer secret');
  assert.equal(requests[2].headers.get('authorization'), 'Bearer secret');
});
