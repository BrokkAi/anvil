---
title: License and Use Cases
description: Practical orientation for using and distributing Anvil under LGPL-3.0-only.
---

Anvil is available under the [GNU Lesser General Public License version 3](https://github.com/BrokkAi/anvil/blob/master/LICENSE) (`LGPL-3.0-only`). You may use it for research, internal work, and commercial products. Obligations depend mainly on whether you run it separately, link it into another program, modify it, or distribute a copy.

This page is practical orientation, not legal advice. The license text controls. It covers Anvil, not the separate Brokk product, trademarks, user data, model-provider terms, or third-party components under their own licenses.

## Start With the Integration Boundary

| How you use Anvil | Typical boundary | Distribution considerations |
| --- | --- | --- |
| Launch Anvil as a separate ACP stdio subprocess | Your client normally remains a separate program under its own terms. | If users install Anvil themselves, you do not distribute their copy. If you bundle it, satisfy the LGPL/GPL obligations for that executable. |
| Operate Anvil as a hosted service | LGPLv3 is not the AGPL; network use alone does not require publication of a private fork. | Shipping an on-premise image, container, VM, appliance, or executable can be distribution. |
| Refactor or fork Anvil into a dynamically linked Rust library | A combined-work analysis applies. Current releases expose a binary, not a supported Rust library API. | Preserve users' LGPL rights in the Anvil portion and the ability to replace or modify it. |
| Statically link or single-file bundle Anvil | Compliance is more involved. | Users generally need a practical way to modify Anvil and relink the application; obtain legal review. |
| Modify or fork Anvil | Private changes may remain private. | Recipients of a distributed fork or binary must receive the applicable LGPL freedoms and corresponding source. |

The cleanest proprietary-client boundary is usually to launch the released `anvil` executable through documented ACP stdio messages. The current crate exposes the `anvil` binary and no supported library target; linking or embedding requires a fork or refactor plus a separate combined-work analysis. A wrapper or container does not automatically guarantee separation if the programs exchange private internal structures or behave as inseparable halves.

## When You Give Someone a Copy

If you distribute an Anvil executable, linked application, container layer, or modified fork:

1. identify the exact Anvil version and mark modifications;
2. preserve copyright and license notices;
3. include the GNU LGPLv3 and incorporated GPLv3 texts;
4. provide the complete corresponding source for the exact covered binary in an allowed way;
5. preserve the recipient's ability to modify the Anvil portion and debug those modifications; and
6. review dependency licenses and shipped notices.

Official archives include the controlling license, GPLv3 text, source information, generated dependency report, and supplemental notices. See [Third-Party Notices](/third-party-notices/).

Source generally needs to be offered to people who receive the binary; LGPLv3 does not require every private fork to be published to the world. Recipients retain their redistribution rights.

## Research, Results, and Customer Code

The license on Anvil does not place customer source code, prompts, agent responses, tool output, or benchmark results under the LGPL merely because Anvil processes them. Rights in that material and the terms of the selected model provider remain separate questions.

Ask qualified counsel to review static linking, single-file bundling, appliances, EULA reverse-engineering restrictions, unusually coupled subprocess designs, or distribution across company boundaries.
