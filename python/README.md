# brokk-anvil

`brokk-anvil` installs the native [Anvil](https://anvil.brokk.ai/) ACP server
through Python package managers such as [uv](https://docs.astral.sh/uv/).
It does not compile Anvil and does not require a Rust toolchain.

```bash
uv tool install brokk-anvil
anvil --version
```

The Python package and native binary use the same version. On first run, the
launcher downloads that exact GitHub release archive and its `.sha256`
sidecar, verifies the archive, and caches only the expected `anvil` executable.
See the [installation guide](https://anvil.brokk.ai/install/#uv) for supported
platforms, upgrades, pins, cache behavior, and uninstall instructions.
