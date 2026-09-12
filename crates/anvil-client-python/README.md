# brokk-anvil-client

Native asynchronous Python bindings for Anvil's standalone LLM client.
Install with `uv add brokk-anvil-client`. Binary wheels require no Rust toolchain
and do not download or launch the Anvil executable.

```python
import asyncio
from anvil_client import Client

async def main():
    async with Client() as client:
        result = await client.infer(
            model="deepseek::deepseek-v4-flash",
            messages=[{"role": "user", "content": "Return a short greeting."}],
            schema={"type": "object", "properties": {"greeting": {"type": "string"}},
                    "required": ["greeting"], "additionalProperties": False},
        )
        print(result.output, result.usage)

asyncio.run(main())
```

Models require an explicit `codex::`, `meta::`, `kimi::`, `grok::`, or `deepseek::`
prefix. Credentials and provider behavior are shared with `anvil infer`.
DeepSeek uses `DEEPSEEK_API_KEY` or Anvil's stored DeepSeek credentials.
The client supplies no tools and creates no agent sessions.

`Client.infer` accepts system/user messages, a JSON Schema, its `schema_name`,
optional `reasoning_effort` and `service_tier`, total `timeout` (120 seconds),
first-progress `idle_timeout` (60), inter-progress `stall_timeout` (30), and
`validation_retries` (one additional attempt). Results expose validated `output`,
`usage`, and effective request settings. All timeouts must be finite and positive.

`InferenceError.kind` classifies errors; specialized subclasses include
`AuthenticationError`, `RateLimitError`, `StructuredOutputError`, and
`InferenceTimeoutError`. Asyncio cancellation cancels the native future.
`close()` cancels active requests and makes the client unusable; it is idempotent.
Connections and lazily loaded credentials are reused for the client's lifetime.
Create a new client after changing credentials. Standard CPython 3.11+ is
supported; free-threaded CPython wheels are not currently published.

From a source checkout, run `uvx maturin develop --manifest-path
crates/anvil-client-python/Cargo.toml` in an activated virtual environment.
Binding builds depend only on `anvil-client`, not Anvil's agent or Wasm runtime.


Before building release wheels, run `python scripts/prepare-python-client.py`
from the repository root to stage generated third-party license notices.
