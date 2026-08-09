---
title: Neovim
description: Configure Anvil for CodeCompanion or Avante in Neovim.
---

Anvil can configure ACP integrations for
[CodeCompanion](https://github.com/olimorris/codecompanion.nvim) and
[Avante](https://github.com/yetone/avante.nvim). Install the selected plugin
before configuring Anvil.

## Install

Choose the plugin explicitly:

```bash
anvil install neovim --plugin codecompanion
anvil install neovim --plugin avante
```

When `--plugin` is omitted, Anvil prompts in an interactive terminal and uses
CodeCompanion in non-interactive use. The generated Lua module launches the
absolute path of the currently running Anvil executable. Move the executable to
its stable location first.

The generated modules are:

- `~/.config/nvim/lua/anvil/anvil_codecompanion.lua`
- `~/.config/nvim/lua/anvil/anvil_avante.lua`

If `init.lua` contains a matching `lazy.nvim` plugin block with the exact line
`opts = {},`, the installer safely replaces that line so the module loads
automatically. For customized plugin setup, the installer leaves `init.lua`
alone and prints the module to load manually.

Existing generated modules are not overwritten unless `--force` is supplied.
Restart Neovim after updating the configuration.
