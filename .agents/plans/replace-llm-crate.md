# Replace the LLM package with a maintainer-owned client package

## Purpose

Replace the old LLM package with brokk-anvil-client, owned by foundev and shared with the same collaborators as brokk-anvil-minimizer. Preserve the implementation and standalone dependency boundary. Update Anvil and Mjolnir so neither active source nor resolved dependency graph references the old package.

## Progress

- [x] Confirm clean repositories, unused replacement name, and GitHub authentication as foundev.
- [x] Rename package, imports, directory, CI and publication references; prepare Anvil 0.28.1.
- [x] Validate, commit and push Anvil at 3e7ba29acf799658b7c91c528f850494b45fb093; pass master CI 34116751812.
- [x] Publish brokk-anvil-client 0.28.1 as foundev; add github:brokkai:brokk-eng ownership and invite jbellis and DavidBakerEffendi (invitations pending acceptance).
- [x] Configure replacement trusted publisher 19317 and read back exact repository, workflow and environment for all eleven Anvil/Mjolnir release crates.
- [x] Switch Mjolnir directly to the published registry version; pass extracted voice-package verification and final validation/CI, then push v2.1.0.
- [x] Push Anvil v0.28.1 after all pre-tag gates; publish all three crates successfully in workflow 34118993745 and documentation in 34118993685.
- [x] Complete Anvil GitHub platform release, npm publication 34120788795, and PyPI publication 34121256134; all publication jobs succeeded.
- [x] Complete Mjolnir v2.1.0 GitHub and npm releases and publish all eight crates, retaining voice on every release platform.

## Findings and Decisions

The old LLM package was owned solely by jbellis. The release identity published the minimizer but received HTTP 403 for the LLM package during v0.28.0. Leave that package and tag untouched. Use brokk-anvil-client, import anvil_client, and move crates/anvil-llm to crates/anvil-client. Prepare 0.28.1 rather than rewriting the pushed tag or republishing an immutable minimizer version.

Initially no local Cargo token or connected browser session was available. The user subsequently logged into Cargo, resolving bootstrap access. Actual replacement publication established foundev ownership; authenticated registry read-back then verified all expected trusted publishers before either tag was pushed. Ownership alone was not treated as proof of publisher authorization.

## Implementation and Validation

Mechanically rename tracked references while preserving behavior and license attribution. Synchronize Anvil package/dependency versions and Python launcher. Run cargo update --workspace and regenerate legal reports using cargo-about 0.9.1 and cargo-deny 0.20.2. Review the diff. Validate formatting, workspace tests, default and ACP-only Clippy, release build, Python tests, docs, and extracted client-package compilation. Run Cargo tests outside the restricted sandbox.

Mjolnir's workspace dependency, controller/chat/voice imports, license policy and lockfile now use published brokk-anvil-client 0.28.1 directly; no temporary Git pin was needed. Registry-only extracted voice-package verification passed. Voice remains enabled under this approach, superseding the interrupted voice-removal workaround.

## Recovery and Acceptance

Commit only task changes on current master branches. Push authorized source changes, but do not tag with unverified registry access. Do not remove owners, move tags, yank packages, or discard published versions. Completion requires foundev ownership of the replacement, collaborator access matching the other crates, working trusted publication, both projects using the replacement with no old package, and successful authorized release workflows. Local renaming alone is preparation.

## Outcomes

The replacement is published and trusted publishing is configured. Anvil 0.28.1 is published on GitHub (all five platform archives plus checksums), crates.io (all three crates), npm, and PyPI. All publication jobs succeeded. Mjolnir v2.1.0 is released at 80bbeb5e440b159c1808de00da805183ebfd7e8c after local checks and CI passed against the registry replacement; all three platform archives, npm packages and eight crates are published. Both repositories have removed the old package from active source and dependency graphs without removing voice. Collaborator invitations remain pending recipient acceptance.

Revision note: replaced obsolete bootstrap blockers with verified publication/ownership evidence and exact active release runs; release monitoring remains unfinished.

Revision note: recorded completed Anvil and Mjolnir releases and successful registry publications; the original voice dependency/publication blocker is resolved.
