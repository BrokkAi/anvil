---
title: Python LLM Client
description: Native asynchronous inference from Python with Anvil's provider clients.
---

Install `brokk-anvil-client` with `uv add brokk-anvil-client`. This is a native
PyO3 binding to Anvil's standalone client, separate from the `brokk-anvil` CLI
launcher. Wheels do not require Rust or the Anvil executable. Standard CPython
3.11+ is supported on Linux x86-64/ARM64, macOS ARM64, and Windows x86-64.

```python
from anvil_client import Client

async def classify(text: str):
    async with Client() as client:
        result = await client.infer(
            model="deepseek::deepseek-v4-flash",
            messages=[{"role": "user", "content": text}],
            schema={"type": "object", "properties": {"relevant": {"type": "boolean"}},
                    "required": ["relevant"], "additionalProperties": False},
        )
        return result.output
```

The client accepts the same explicit hosted-provider routes and credentials as
[`anvil infer`](/providers/#tool-free-structured-inference): Codex, Meta, Kimi,
Grok, and DeepSeek. It reuses native authentication, retries, schema validation,
and usage accounting. No tools, agent sessions, or project instructions run.

DeepSeek structured inference uses its stateless Responses API with native
`text.format: json_schema` enforcement. Truncated responses are rejected even
when their partial text happens to be valid JSON. The schema is sent out of band,
so changing it does not prepend instructions ahead of your stable message prefix.
Other providers that require JSON-mode fallback receive the schema after your
messages; local validation still checks every provider's output. DeepSeek's normal
agent chat continues to use Chat Completions.

Keep one client for repeated calls to reuse provider connections. Use it as an
async context manager or call `close()`; closing cancels outstanding requests.
Asyncio task cancellation cancels native inference. Credentials are resolved
lazily and retained for the client's lifetime, so create a new client after
changing credentials.

`infer()` takes `model`, system/user `messages`, `schema`, and optional
`schema_name`, `reasoning_effort`, `service_tier`, `timeout` (120 seconds),
`idle_timeout` (60), `stall_timeout` (30), and `validation_retries` (1).
Timeouts must be finite and positive. Results provide `output`, `usage`, `model`,
`reasoning_effort`, and `service_tier`. Exceptions derive from `InferenceError`
and expose `kind`, with specific subclasses for invalid input, authentication,
rate limits, context length, transport, schema validation, and deadlines.

Maintainers build this package from `crates/anvil-client-python` using Maturin.
The package is versioned with Anvil. The Python client wheels workflow checks
installed wheels on Python 3.11 and 3.14, publishes tagged wheels as GitHub
release assets. A manual workflow dispatch with `publish_pypi` enabled publishes
to PyPI using the `pypi-publish` environment. For the initial project bootstrap,
a maintainer can publish downloaded, validated wheels with an existing PyPI
token; do not store that token in the repository.
Configure a PyPI trusted publisher for project `brokk-anvil-client`, owner
`BrokkAi`, repository `anvil`, workflow `python-client.yml`, environment
`pypi-publish` before its first publication. Wheels carry Anvil's LGPL license;
the tagged repository is their corresponding source.

