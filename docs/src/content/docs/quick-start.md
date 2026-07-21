---
title: Quick start
description: Build Anvil and connect an ACP client.
---

## Build the server

The default build embeds the Wasm sandbox and requires the `wasm32-wasip2` target.

```bash
rustup target add wasm32-wasip2
cargo build --bin anvil
```

## Run Anvil

Most users configure an ACP client to launch the absolute path to `target/debug/anvil`. To confirm the stdio server starts, run it directly:

```bash
./target/debug/anvil
```

The first session provides setup guidance. Use `/setup` to connect a model provider, refresh discovery, or configure install-wide settings.

## Editor helpers

The repository includes helpers for Zed and JetBrains:

```bash
cargo xtask build-acp-for-zed
cargo xtask build-acp-for-jetbrains
```

More detailed installation and client walkthroughs will be added as the documentation develops.
