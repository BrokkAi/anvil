import assert from 'node:assert/strict';
import test from 'node:test';

import { AnvilApiError, AnvilClient } from '../dist/index.js';

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function run(status = 'running', overrides = {}) {
  return {
    id: 'run-1',
    session_id: 'session-1',
    status,
    stop_reason: status === 'running' ? null : status === 'cancelled' ? 'cancelled' : 'end_turn',
    error: null,
    result_text: status === 'completed' ? 'done' : null,
    structured_output: null,
    usage: null,
    created_at_ms: 1,
    finished_at_ms: status === 'running' ? null : 2,
    last_seq: 1,
    ...overrides,
  };
}

test('adds bearer auth, retries safe requests, and preserves additive fields', async () => {
  let calls = 0;
  const client = new AnvilClient({
    token: async () => 'secret',
    retry: { baseDelayMs: 0, sleep: async () => {} },
    fetch: async (request) => {
      calls += 1;
      assert.equal(request.headers.get('authorization'), 'Bearer secret');
      assert.match(request.headers.get('user-agent'), /^@brokkai\/anvil-sdk\//);
      if (calls === 1) return json({ error: { code: 'internal', message: 'retry' } }, 503);
      return json({
        models: [
          {
            id: 'ollama::test',
            default_reasoning_level: null,
            supported_reasoning_levels: [],
            service_tiers: [],
            supports_images: null,
            context_length: null,
            pricing: null,
            future_capability: true,
          },
        ],
        default_model: 'ollama::test',
        future_top_level: { kept: true },
      });
    },
  });

  const models = await client.models();
  assert.equal(calls, 2);
  assert.equal(models[0].future_capability, true);
});

test('surfaces typed API error envelopes and auth failures', async () => {
  const client = new AnvilClient({
    retry: false,
    fetch: async () =>
      json(
        {
          error: { code: 'unauthorized', message: 'bad token', details: { scheme: 'bearer' } },
          request_id: 'req-401',
        },
        401,
      ),
  });

  await assert.rejects(client.models(), (error) => {
    assert.ok(error instanceof AnvilApiError);
    assert.equal(error.status, 401);
    assert.equal(error.code, 'unauthorized');
    assert.equal(error.requestId, 'req-401');
    assert.deepEqual(error.details, { scheme: 'bearer' });
    return true;
  });
});

test('enforces request timeouts', async () => {
  const client = new AnvilClient({
    timeoutMs: 5,
    retry: false,
    fetch: async (request) =>
      new Promise((resolve, reject) => {
        request.signal.addEventListener('abort', () => reject(request.signal.reason), { once: true });
      }),
  });

  await assert.rejects(client.models(), (error) => {
    assert.ok(error instanceof AnvilApiError);
    assert.equal(error.cause.name, 'TimeoutError');
    return true;
  });
});

test('cleanup cancels only a running run', async () => {
  const requests = [];
  const client = new AnvilClient({
    retry: false,
    fetch: async (request) => {
      requests.push([request.method, new URL(request.url).pathname]);
      if (request.method === 'GET') return json(run('running'));
      return json(run('cancelled'));
    },
  });

  const result = await client.run('run-1').cleanup();
  assert.equal(result.status, 'cancelled');
  assert.deepEqual(requests, [
    ['GET', '/v1/runs/run-1'],
    ['POST', '/v1/runs/run-1/cancel'],
  ]);
});
