---
title: ACP clients
description: Connect editors, bots, TUIs, and automations to Anvil.
---

Anvil is designed to sit behind any client that speaks ACP over stdio. Existing examples include editor integrations, issue triage bots, review bots, and a small issue-writing TUI.

## Client responsibilities

Clients create or resume sessions, send prompts, render streamed updates, surface session controls, and answer permission requests.

## Example clients

The [`examples/`](https://github.com/BrokkAi/anvil/tree/master/examples) directory contains focused Rust clients that demonstrate the protocol shape without hiding it behind a framework.

Detailed client integration guides will be added here.
