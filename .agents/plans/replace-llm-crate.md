# Replace the LLM package with a maintainer-owned client package

## Purpose

Replace the old LLM package with brokk-anvil-client, owned by foundev and shared with the same collaborators as brokk-anvil-minimizer. Preserve the implementation and standalone dependency boundary. Update Anvil and Mjolnir so neither active source nor resolved dependency graph references the old package.

## Progress

- [x] Confirm clean repositories, unused replacement name, and GitHub authentication as foundev.
- [ ] Rename package, imports, directory, CI and publication references; prepare Anvil 0.28.1.
- [ ] Validate and commit Anvil, then pin Mjolnir to the exact replacement revision and validate.
- [ ] Establish foundev registry ownership, requested collaborator access, and verified trusted publishing.
- [ ] Publish the replacement, switch Mjolnir to its registry version, and complete authorized releases after all gates pass.

## Findings and Decisions

The old LLM package was owned solely by jbellis. The release identity published the minimizer but received HTTP 403 for the LLM package during v0.28.0. Leave that package and tag untouched. Use brokk-anvil-client, import anvil_client, and move crates/anvil-llm to crates/anvil-client. Prepare 0.28.1 rather than rewriting the pushed tag or republishing an immutable minimizer version.

No local Cargo token or connected browser session is available. Repository secrets list only the Discord webhook and the release environment has no listed secrets. Organization-secret listing is forbidden; this does not prove whether an organization publishing credential exists. Do not probe authorization with a real upload. Registry ownership bootstrap and verified per-package publishing authorization are gates before release publication.

## Implementation and Validation

Mechanically rename tracked references while preserving behavior and license attribution. Synchronize Anvil package/dependency versions and Python launcher. Run cargo update --workspace and regenerate legal reports using cargo-about 0.9.1 and cargo-deny 0.20.2. Review the diff. Validate formatting, workspace tests, default and ACP-only Clippy, release build, Python tests, docs, and extracted client-package compilation. Run Cargo tests outside the restricted sandbox.

Update Mjolnir's workspace dependency, voice-worker import, license policy and lockfile. Temporarily pin the exact validated Anvil commit until registry publication; this is not a publishable Mjolnir release candidate. Require registry-only extracted-package verification after switching to the published dependency. Voice remains enabled under this approach, superseding the interrupted voice-removal workaround.

## Recovery and Acceptance

Commit only task changes on current master branches. Push authorized source changes, but do not tag with unverified registry access. Do not remove owners, move tags, yank packages, or discard published versions. Completion requires foundev ownership of the replacement, collaborator access matching the other crates, working trusted publication, both projects using the replacement with no old package, and successful authorized release workflows. Local renaming alone is preparation.

## Outcomes

Preparation is in progress. No replacement package is published and no new release tag is created.
