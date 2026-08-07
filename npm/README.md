# npm packaging pipeline

This directory builds the npm distribution of Anvil ([issue #330](https://github.com/BrokkAi/anvil/issues/330)):

- `@brokkai/anvil` — root package: launcher + `anvil` bin entry, pins each
  platform package as an exact optional dependency.
- `@brokkai/anvil-darwin-universal`, `@brokkai/anvil-linux-x64`,
  `@brokkai/anvil-linux-arm64`, `@brokkai/anvil-android-arm64`,
  `@brokkai/anvil-win32-x64` — one package per released native target, each
  containing the complete checksum-verified GitHub release bundle.

The native binaries are never rebuilt here. Every payload is extracted from
the GitHub release zip for the requested tag after verifying its published
SHA-256 sidecar. Platform payloads use distinct package names — never
prerelease versions of the root package — so a platform payload can never
become `@brokkai/anvil`'s `latest`.

## Layout

| Path | Purpose |
| --- | --- |
| `launcher/anvil.js` | The root package's bin entry. Picks the installed platform package, execs the native binary, forwards args/stdio/signals/exit status. |
| `launcher/README.md` | README published with `@brokkai/anvil` (shown on npmjs.com). |
| `lib/common.mjs` | Shared target table and helpers. |
| `lib/registry.mjs` | Registry visibility checks shared with the TypeScript SDK publisher. Reads with `--prefer-online` and waits past npmjs' five-minute packument cache, so a just-published version is never mistaken for a failed publish. |
| `build-npm-packages.mjs` | Downloads (or reuses) release assets, verifies checksums, stages and packs all six tarballs, validates them, writes `dist/manifest.json`. |
| `test-npm-packages.mjs` | Installs the built tarballs into a throwaway global prefix with a cold cache and smoke-tests the real binary (version, arg forwarding, exit status, signal forwarding, one-shot `npm exec`). |
| `publish-npm-packages.mjs` | Publishes platform packages first, waits until all are publicly visible, then publishes the root. Dry run unless `--yes-publish`. Retry-safe. |

## Local build and test

```bash
node npm/build-npm-packages.mjs --tag v0.24.3
node npm/test-npm-packages.mjs
node npm/publish-npm-packages.mjs            # dry run, prints the plan
```

Requirements: Node 18+, `npm`, `unzip`, `tar`.

Validation performed before any registry write:

- every release zip is verified against its `.sha256` sidecar;
- extracted bundles must contain exactly the binary plus the documented
  license/readme files — any extra file (credentials, dev files, unintended
  payloads) fails the build;
- tarball contents are compared against an exact allowlist;
- native binaries must be present, plausibly sized, and executable;
- `package.json` names, versions, `os`/`cpu`, bin entries, and exact optional
  dependencies are asserted;
- neither package declares install scripts.

## CI

`.github/workflows/publish-npm.yml` (manual `workflow_dispatch`) packages an
existing release tag. It defaults to build-only: it downloads the release
assets, builds and validates all six tarballs, smoke-tests linux-x64 with the
real binary, and uploads the tarballs as workflow artifacts. Checking the
`publish` input additionally publishes via npm trusted publishing (OIDC,
`id-token: write`, no long-lived token). The run is safe to retry after a
partial publication: already-published versions are skipped.

## First-time publication (bootstrap)

npm trusted publishing can only be configured for packages that already
exist, so the first publication is manual, after the implementation PR is
merged:

1. `node npm/build-npm-packages.mjs --tag <tag>` and
   `node npm/test-npm-packages.mjs` — no registry writes.
2. `npm login` as a member of the `brokkai` npm organization, then
   `node npm/publish-npm-packages.mjs --yes-publish`. This bootstraps the
   five platform packages first, verifies each is publicly visible and
   installable, and only then bootstraps `@brokkai/anvil` — the root is never
   published while any exact optional platform dependency is unavailable.
3. Verify from clean environments (the publish script also does this):
   `npm install -g @brokkai/anvil && anvil --version` and
   `npx -y @brokkai/anvil --version`.
4. On npmjs.com, configure this repository's `publish-npm.yml` as the
   trusted publisher for all six package names, and disallow token
   publishing.

Subsequent releases publish through the workflow only.
