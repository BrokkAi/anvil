# Corresponding Source

Each official Anvil artifact identifies its version as `X.Y.Z`. The complete
corresponding source for that artifact, including the build and release scripts,
is the Git tag `vX.Y.Z` in the Anvil repository:

https://github.com/BrokkAi/anvil/releases/tag/vX.Y.Z

Replace `X.Y.Z` with the version printed by `anvil --version`. The release page
provides source archives for that exact tag. The repository history is also
available at:

https://github.com/BrokkAi/anvil

Anvil is licensed under `LGPL-3.0-only`. `LICENSE` contains the GNU LGPL version
3 text, and `GPL-3.0.md` contains the incorporated GNU GPL version 3 text.
`THIRD_PARTY_LICENSES.html` and `SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt` contain
license information, standalone notices, and exact source-package links for the
locked Rust dependencies and vendored native components incorporated into
official binaries. This includes the `brokk-acp-sandbox` crate compiled into the
embedded `wasm32-wasip2` guest.

The workspace crate `crates/anvil-minimizer` is vendored from the oh-my-pi
project (https://github.com/can1357/oh-my-pi), `crates/pi-shell` at commit
`09a7c865636457c50ed75fc3b1a7cc21ef72c105`, under the MIT license. That code in
turn adapts MIT-licensed algorithms from `rtk-ai/rtk`. The upstream copyright
notices and license texts for both projects are reproduced in
`crates/anvil-minimizer/NOTICE`.
