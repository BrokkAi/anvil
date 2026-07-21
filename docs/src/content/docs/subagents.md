---
title: Subagents
description: Author focused agents and delegate bounded work through the task tool.
---

Subagents are Markdown definitions that Anvil exposes through a dynamic `task` tool. They are useful for independent review, research, triage, and other work that benefits from a focused prompt.

## Define an agent

Create one flat `<name>.md` file under `.claude/agents/` or `.agents/agents/`:

```markdown
---
name: bug-hunter
description: Review a diff for concrete correctness bugs.
max_turns: 8
tools: Read, Grep, Glob
---

Flag concrete bugs only. Return path, line, and a short explanation.
```

`max_turns` is optional and can only lower the budget inherited from the parent. `maxTurns` is accepted for compatibility. `tools` is an optional allowlist; Claude-style names such as `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`, `WebSearch`, `WebFetch`, and `LS` map to Anvil's built-ins, and exact Anvil or MCP tool names are also accepted.

## Discovery and precedence

Anvil loads bundled agents, plugin agents, user agents under `~/.claude/agents/` and `~/.agents/agents/`, then project agents while walking from the Git root to the session directory. Later definitions win, so project agents override user and plugin definitions.

The `task` tool appears only when at least one valid agent is discovered. Invalid definitions are skipped with diagnostics on stderr.

## Execution contract

- Each subagent starts with a fresh transcript and receives only its own system instructions and delegated prompt.
- It shares the parent workspace, tool registry, session id, and permission gate.
- `task.permission_mode` defaults to `readOnly`. Consecutive read-only lanes may run concurrently, capped at six.
- `permission_mode: inherit` uses the parent permission behavior and remains serial.
- Delegation depth is one; subagents cannot spawn further subagents.
- Intermediate tokens and tool calls are silent. The parent receives the subagent's final text.
- Cancelling the parent prompt cancels all in-flight lanes cooperatively.

Read-only limits the available mutating tools; it does not create a separate filesystem or process sandbox. For ordering, timeout, observability, and result guarantees, read [Subagent Concurrency](../concurrency/).
