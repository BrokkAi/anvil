"""Installed-wheel tests: native Rust transport against a local SSE server."""
import asyncio
import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from anvil_client import Client, ClientClosedError, InvalidRequestError, InferenceTimeoutError, StructuredOutputError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        prompt = request["messages"][-1]["content"]
        if "slow" in prompt:
            time.sleep(0.5)
        output = {"wrong": True} if "invalid" in prompt else {"ok": True}
        events = [
            {"choices": [{"index": 0, "delta": {"content": json.dumps(output)}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        ]
        body = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


class ClientTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.env = patch.dict(os.environ, {"KIMI_API_KEY": "test-only",
            "KIMI_CODE_BASE_URL": f"http://127.0.0.1:{cls.server.server_port}/v1"})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    async def request(self, client, prompt="ok", **kwargs):
        return await client.infer(model="kimi::test", messages=[{"role": "user", "content": prompt}], schema=SCHEMA, **kwargs)

    async def test_native_transport_result_and_concurrency(self):
        async with Client() as client:
            results = await asyncio.gather(*(self.request(client) for _ in range(3)))
        for result in results:
            self.assertEqual(result.output, {"ok": True})
            self.assertEqual(result.model, "kimi::test")
            self.assertGreater(result.usage.input_tokens, 0)
        self.assertNotIn("tools", self.server.requests[-1])

    async def test_local_schema_failure(self):
        async with Client() as client:
            with self.assertRaises(StructuredOutputError):
                await self.request(client, "invalid", validation_retries=0)

    async def test_deadline_does_not_block_event_loop(self):
        async with Client() as client:
            with self.assertRaises(InferenceTimeoutError):
                await self.request(client, "slow", timeout=0.03)
            self.assertEqual((await self.request(client)).output, {"ok": True})

    async def test_python_cancellation(self):
        async with Client() as client:
            task = asyncio.create_task(self.request(client, "slow"))
            await asyncio.sleep(0.03)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((await self.request(client)).output, {"ok": True})

    async def test_close_cancels_inflight_and_future_requests(self):
        client = Client()
        task = asyncio.create_task(self.request(client, "slow"))
        await asyncio.sleep(0.03)
        client.close()
        with self.assertRaises(ClientClosedError):
            await task
        with self.assertRaises(ClientClosedError):
            await self.request(client)
        client.close()

    async def test_invalid_arguments(self):
        async with Client() as client:
            for timeout in [0, -1, float("inf"), float("nan")]:
                with self.assertRaises(InvalidRequestError):
                    await self.request(client, timeout=timeout)
            with self.assertRaises(InvalidRequestError):
                await client.infer(model="unknown::test", messages=[], schema=SCHEMA)


if __name__ == "__main__":
    unittest.main()

