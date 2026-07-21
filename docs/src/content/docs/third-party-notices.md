---
title: Third-Party Notices
description: Understand the license and source materials shipped with Anvil releases.
---

Anvil is `LGPL-3.0-only` and depends on components under additional open-source licenses. Release archives include the material needed to identify the covered Anvil version and its reviewed dependency graph.

## Shipped Material

- [`LICENSE`](https://github.com/BrokkAi/anvil/blob/master/LICENSE): GNU LGPL version 3 text.
- [`licenses/GPL-3.0.md`](https://github.com/BrokkAi/anvil/blob/master/licenses/GPL-3.0.md): incorporated GNU GPL version 3 text.
- [`licenses/SOURCE.md`](https://github.com/BrokkAi/anvil/blob/master/licenses/SOURCE.md): source and corresponding-source orientation.
- [`licenses/THIRD_PARTY_LICENSES.html`](https://github.com/BrokkAi/anvil/blob/master/licenses/THIRD_PARTY_LICENSES.html): generated Rust dependency report.
- [`licenses/SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt`](https://github.com/BrokkAi/anvil/blob/master/licenses/SUPPLEMENTAL_THIRD_PARTY_NOTICES.txt): standalone attributions and vendored native-code notices.

The generated report covers the locked native release targets and the `wasm32-wasip2` parser guest embedded by the default feature set. Release archives carry these files next to the executable.

## For Distributors

Do not substitute a notice report from another version. Preserve the files shipped with the exact archive you redistribute and make the corresponding source for that version available as required.

Dependency-policy decisions, regeneration commands, and reviewed exceptions belong to the maintainer workflow in [`CONTRIBUTING.md`](https://github.com/BrokkAi/anvil/blob/master/CONTRIBUTING.md#dependency-and-license-changes). The practical boundary between invoking, bundling, and linking Anvil is covered in [License and Use Cases](/license-use-cases/).
