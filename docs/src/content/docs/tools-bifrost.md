---
title: Tools and Managed Bifrost
description: Understand built-in tools, code intelligence, and the evidence each tool provides.
---

Anvil combines built-in execution tools with tools supplied by MCP servers. The model-facing catalog is assembled per session and permission policy applies to every call.

## Built-In Tools

Core tools cover file reading and writing, targeted edits, directory listing, text search, web search, and shell execution. Path validation keeps built-in file operations inside the session working directory; shell execution follows the selected permission and sandbox strategy.

Use the tool that matches the question:

- file reads for a known non-source file or exact range;
- text search for strings, configuration, logs, and prose;
- Bifrost symbol tools for declarations, source, references, and callers; and
- shell commands for repository-native checks and workflows.

## Managed Bifrost

Anvil preconfigures an enabled local Bifrost MCP server:

```text
<managed-bifrost> --root {cwd} --mcp core --no-line-numbers
```

On first use, Anvil downloads the pinned release for the current platform, verifies its checksum, and caches it under the Anvil configuration directory. Network failure prevents that cold start; later sessions reuse the cached binary.

The managed toolset includes code-intelligence operations such as:

- `search_symbols` for declarations;
- `get_symbol_sources` for exact implementation blocks;
- `get_summaries` for API shape and orientation;
- `scan_usages_by_reference` or its compatible usage form for references and callers;
- `most_relevant_files` for repository relationships; and
- bounded filename and file-list discovery.

Tool names can evolve with the pinned Bifrost release. Use the advertised session catalog rather than assuming a tool exists from an older screenshot.

## Evidence Boundaries

A visible Bifrost call proves the configured analyzer answered that request; it does not prove every runtime target, path-sensitive behavior, or whole-program property. Inspect truncation, ambiguity, and diagnostics when present. A candidate importer file is not proof of an actual callsite, and text similarity is not declaration identity.

The [ten-minute evaluation](/evaluate-anvil/) deliberately requires named Bifrost tools so a plausible model answer cannot pass by ordinary file reading.

## Restrict the Catalog

An optional `allowed_tools` array in `setup.json` filters exact tool names after built-in, MCP, skill, and dynamic tools have been assembled:

```json
{
  "allowed_tools": ["read_file", "grep_search", "search_symbols", "task"]
}
```

Unknown names are ignored. Omitting the field preserves the full catalog; an empty array disables every model-callable tool.
