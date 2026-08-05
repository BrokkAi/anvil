import assert from 'node:assert/strict';
import test from 'node:test';

import { AnvilClient, AnvilStreamError } from '../dist/index.js';

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function sse(...events) {
  const body = events
    .map(
      (event) =>
        `${'seq' in event ? `id: ${event.seq}\n` : ''}event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
    )
    .join('');
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

function envelope(type, seq, extra = {}) {
  return {
    type,
    run_id: 'run-1',
    session_id: 'session-1',
    seq,
    ts_ms: seq,
    ...extra,
  };
}

function terminalRun() {
  return {
    id: 'run-1',
    session_id: 'session-1',
    status: 'completed',
    stop_reason: 'end_turn',
    error: null,
    result_text: 'done',
    structured_output: { answer: 42 },
    usage: null,
    created_at_ms: 1,
    finished_at_ms: 2,
    last_seq: 2,
  };
}

test('reconnects an ended SSE stream from the last sequence and keeps unknown fields', async () => {
  const eventRequests = [];
  const client = new AnvilClient({
    fetch: async (request) => {
      const url = new URL(request.url);
      eventRequests.push(url);
      if (eventRequests.length === 1) {
        return sse(envelope('run.created', 1, { prompt_chars: 4, future_field: 'kept' }));
      }
      assert.equal(url.searchParams.get('after_seq'), '1');
      return sse(
        envelope('run.completed', 2, {
          stop_reason: 'end_turn',
          result_text: 'done',
          structured_output: null,
        }),
      );
    },
  });

  const events = [];
  for await (const event of client.run('run-1').events({
    reconnectDelayMs: 0,
    sleep: async () => {},
  })) {
    events.push(event);
  }

  assert.equal(eventRequests.length, 2);
  assert.equal(events[0].future_field, 'kept');
  assert.deepEqual(
    events.map((event) => event.type),
    ['run.created', 'run.completed'],
  );
});

test('wait handles an interactive permission and returns the terminal run', async () => {
  let streamCalls = 0;
  const responses = [];
  const permission = {
    id: 'permission-1',
    run_id: 'run-1',
    session_id: 'session-1',
    tool_name: 'write_file',
    tool_call_id: 'call-1',
    input: { path: 'hello.txt' },
    permission_notice: null,
    options: [{ id: 'allow', label: 'Allow', kind: 'allow_once' }],
    created_at_ms: 1,
    status: 'pending',
  };
  const client = new AnvilClient({
    fetch: async (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith('/events')) {
        streamCalls += 1;
        if (streamCalls === 1) {
          return sse(
            envelope('permission.requested', 1, {
              permission_id: permission.id,
              tool_name: permission.tool_name,
              tool_call_id: permission.tool_call_id,
              options: permission.options,
            }),
          );
        }
        assert.equal(url.searchParams.get('after_seq'), '1');
        return sse(
          envelope('run.completed', 2, {
            stop_reason: 'end_turn',
            result_text: 'done',
          }),
        );
      }
      if (url.pathname === `/v1/permissions/${permission.id}` && request.method === 'GET') {
        return json(permission);
      }
      if (url.pathname.endsWith('/respond')) {
        responses.push(await request.clone().json());
        return json({ resolved: true, id: permission.id });
      }
      if (url.pathname === '/v1/runs/run-1') return json(terminalRun());
      throw new Error(`unexpected request: ${request.method} ${url}`);
    },
  });

  const completed = await client.run('run-1').wait({
    reconnectDelayMs: 0,
    sleep: async () => {},
    onPermission: (requested) => requested.options[0].id,
  });

  assert.equal(completed.status, 'completed');
  assert.deepEqual(responses, [{ option_id: 'allow' }]);
});

test('does not retry a non-retryable SSE auth failure', async () => {
  let calls = 0;
  const client = new AnvilClient({
    fetch: async () => {
      calls += 1;
      return new Response('unauthorized', { status: 401 });
    },
  });

  await assert.rejects(
    async () => {
      for await (const _event of client.run('run-1').events()) {
        // no events expected
      }
    },
    (error) => error instanceof AnvilStreamError,
  );
  assert.equal(calls, 1);
});
