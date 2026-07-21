---
title: Skills and Plugins
description: Extend Anvil with reusable instructions, commands, hooks, agents, and tools.
---

Skills provide reusable instructions through `SKILL.md`. Claude Code-format plugins can bundle skills with commands, hooks, MCP servers, and subagents.

## Skill discovery

Anvil discovers skills from:

1. `$CODEX_HOME/skills` (or `~/.codex/skills`)
2. `~/.claude/skills` and `~/.agents/skills`
3. Project `.claude/skills` and `.agents/skills` directories
4. Enabled plugins

Project and user entries override plugin-provided entries. Built-in slash commands keep precedence if names collide. A discovered skill is exposed as a slash command and may also be activated by the model when its description matches the task.

A minimal skill looks like:

```text
my-skill/
├── SKILL.md
└── scripts/
```

```markdown
---
name: my-skill
description: Explain when this workflow should be used.
---

Follow these steps...
```

Instructions and scripts execute with the same workspace, permission, and sandbox boundaries as the parent session. Review a skill before installing or enabling it.

## Plugin management

Anvil discovers Claude Code installations and accepts Claude Code-format manifests at `.claude-plugin/plugin.json`.

```text
/plugin list
/plugin add <git-url | owner/repo | local-path> [marketplace-plugin]
/plugin update <name>
/plugin enable <name>
/plugin disable <name>
/plugin remove <name>
```

Plugins installed with `/plugin add` live in Anvil's configuration area. A plugin may add:

- skills and slash commands;
- subagent definitions;
- lifecycle hooks;
- MCP server processes.

That makes a plugin a code-and-process trust decision, not merely a prompt library. Inspect its manifest, hooks, executable commands, and MCP configuration before enabling it. See [Data and Trust Boundaries](../data-boundaries/).

## Hook execution

Plugins can register `UserPromptSubmit`, `PreToolUse`, and `PostToolUse` command hooks. Anvil runs them as `sh -c` on Unix or `cmd /C` on Windows, in the session working directory, with inherited host environment and operating-system authority. They are not routed through the model tool permission gate or the shell sandbox.

Exit code `0` accepts the hook and can contribute bounded stdout context. Exit code `2` blocks a prompt/tool or feeds post-tool feedback using stderr. Other exits and timeouts are logged as hook errors. Enabling a plugin therefore authorizes its hooks to execute at lifecycle events even when the model has not requested a shell tool.
