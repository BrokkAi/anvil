---
title: Model Providers and Setup
description: Discover, authenticate, select, and troubleshoot Anvil model providers.
---

Provider discovery is non-fatal. An unavailable source is logged and omitted while other ready sources remain usable. Models use collision-free wire IDs such as `codex::gpt-5-codex`, `ollama::llama3:latest`, and `openrouter::anthropic/claude-sonnet`.

Run `/setup` for the interactive setup home or `/setup choose` to select from ready providers.

## Codex and ChatGPT

Anvil reads `~/.codex/auth.json`, refreshes supported credentials, and can drive browser or device authorization:

```text
/setup codex
/setup codex device
/setup codex status
/setup codex disconnect
```

When the selected Codex model advertises a priority service tier, `/fast on` selects it for the current session and `/fast off` clears it. Fast mode can respond sooner but consumes subscription quota more aggressively.

### Tool-free structured inference

Automation that needs one schema-constrained model call without an ACP agent session can use `anvil infer`. It reads one request from stdin and writes one JSON result to stdout:

```bash
printf '%s' '{"messages":[{"role":"system","content":"Classify the item."},{"role":"user","content":"example"}],"schema_name":"classification","schema":{"type":"object","properties":{"label":{"type":"string"}},"required":["label"],"additionalProperties":false}}' \
  | anvil infer --model codex::gpt-5.5 --reasoning-effort medium
```

This path accepts only system and user text, supplies no tools, and bypasses ACP sessions, project instructions, skills, hooks, history, and the agent loop. The required model prefix selects the backend so no provider fallback can pick a different model: `codex::<model>`, `kimi::<model>`, `grok::<model>`, or `deepseek::<model>`. The corresponding credentials are the Codex auth file, Kimi Code credentials, Grok Build OAuth credentials, and the DeepSeek API key. Omit `--service-tier` to use the provider default. Transport diagnostics go to stderr, while successful stdout contains the validated `output`, aggregate token `usage`, and effective request settings. `--validation-retries` controls additional attempts after local JSON Schema validation fails.

## Local Models

Anvil probes Ollama at `http://localhost:11434/v1/models`. Start Ollama, then run:

```text
/setup local refresh
/setup local use
```

On macOS and Linux, a running `ds4-server` is discovered from its listening port. Set `DS4_BASE_URL` to point at a non-default or remote endpoint, then refresh local discovery.

## Hosted DeepSeek and Kimi Code

DeepSeek uses `DEEPSEEK_API_KEY` or credentials saved through `/setup deepseek`. Models use `deepseek::*` wire IDs.

Kimi Code uses `KIMI_API_KEY` or the OAuth credentials created by `kimi login`. Set `KIMI_CODE_BASE_URL` to override the default coding endpoint. Kimi models use `kimi::*` IDs.

## Grok Build OAuth

Anvil reuses first-party OAuth credentials created by the official Grok Build CLI. It does not use `XAI_API_KEY` and does not implement a separate Grok login flow:

```text
grok login --oauth
/setup grok refresh
/setup grok status
```

The credential is read from `$GROK_HOME/auth.json`, or `~/.grok/auth.json` when `GROK_HOME` is unset. Anvil refreshes expiring OAuth tokens using the same cross-process lock as the Grok CLI and writes rotated credentials atomically. Grok models use `grok::*` wire IDs and the Responses API.

## OpenRouter

Set `OPENROUTER_API_KEY` before starting Anvil or use:

```text
/setup openrouter
/setup openrouter status
/setup openrouter disconnect
```

## AWS Bedrock

Bedrock uses `AWS_BEARER_TOKEN_BEDROCK` or credentials saved through `/setup bedrock`; a legacy `~/.secrets/bedrock_api_key` fallback remains recognized. Configure and inspect it with:

```text
/setup bedrock
/setup bedrock region <region>
/setup bedrock model <model-id>
/setup bedrock catalog
/setup bedrock status
```

Anvil discovers inference profiles for the active region and normalizes base models to required profiles when possible. Native Anthropic-style models and Bedrock's OpenAI-compatible Responses endpoint have provider-specific routing and reasoning controls.

## Generic OpenAI-Compatible Profiles

Anvil reads `providers.json` once at startup from the Brokk config directory (`$BROKK_CONFIG_HOME`, or the OS config directory plus `brokk`):

```json
{
  "openai": {
    "deca": {
      "base_url": "https://api.example.com/deca",
      "api_key_env": "DECA_API_KEY"
    },
    "local-proxy": {
      "base_url": "http://127.0.0.1:8000"
    }
  }
}
```

Profile names must match `[a-z0-9][a-z0-9_-]{0,31}`. `base_url` must be HTTP(S) without credentials, query, or fragment. Anvil trims trailing slashes and appends `/v1` unless already present. Inline keys are not supported; `api_key_env` names an environment variable loaded at startup. Models use `openai::<profile>/<model-id>` IDs.

These profiles use baseline Chat Completions with streaming, tools, usage, and strict JSON Schema structured outputs. They do not add provider-specific reasoning controls or custom headers.

## Credentials and Setup Forms

Clients that advertise ACP elicitation forms receive out-of-transcript credential fields for OpenRouter, Bedrock, and DeepSeek. In a text-only client, commands such as `/setup openrouter key <key>` remain available but the pasted secret becomes part of the session transcript. Prefer environment variables or elicitation forms for sensitive credentials.

Provider priority for automatic selection is Bedrock, Codex, local models (Ollama then ds4), DeepSeek, Kimi, Grok, generic OpenAI-compatible profiles, then OpenRouter. Override it for the current session with `/setup model <wire-id>`.
