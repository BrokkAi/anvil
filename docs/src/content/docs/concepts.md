---
title: Core concepts
description: The main runtime boundaries in Anvil.
---

## Model routing

Anvil discovers configured hosted and local model providers and exposes collision-free wire IDs such as `codex::gpt-5-codex` and `ollama::llama3:latest`.

## Tooling and MCP

The runtime provides built-in filesystem, edit, search, and shell tools. MCP servers, skills, and subagents extend that base without requiring every ACP client to reimplement them.

## Permissions

Permission requests are ACP messages. The client presents the choice while Anvil enforces the selected permission mode and tool-specific safeguards.

## Sessions and context

Sessions persist conversation state and configuration. Context reporting and compaction keep long-running work useful within model limits.
