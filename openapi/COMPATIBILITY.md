# Anvil HTTP API compatibility policy

The files in this directory are the authoritative wire contract for the
Anvil Agent API:

- `anvil.v1.yaml` — OpenAPI 3.1 document for every `/v1` endpoint and
  `/health`.
- `anvil.v1.events.schema.json` — JSON Schema (draft 2020-12) for the
  payload of every Server-Sent Event on `GET /v1/runs/{run_id}/events`.

Official SDK wire layers are generated from these artifacts and record the
contract version (`info.version`) they were generated from. The black-box
conformance suite (`tests/http_conformance.rs`) starts a real `anvil serve`
daemon and validates its responses and event stream against these schemas;
CI fails when the implementation and the contract drift.

The generated packages live under `sdk/typescript`, `sdk/rust`, and
`sdk/python`. They contain no handwritten runtime client code. Generator
configuration and release scaffolding are maintained, but models, endpoint
methods, paths, authentication, and serialization are regenerated from the
two contract files.

From a clean checkout:

```bash
npm ci --prefix sdk/typescript
node scripts/generate-sdks.mjs
```

The command pins `@hey-api/openapi-ts` 0.99.0 and OpenAPI Generator 7.23.0,
combines the OpenAPI document with the event schema in a temporary derived
document, and regenerates all three language packages. CI and registry
publication run the same command and reject any generated diff.

## Versioning

The contract carries a semantic version in `info.version`, independent of
the crate version. The URL prefix (`/v1`) is the major version.

## What may change within `/v1`

Backwards-compatible changes bump the contract's **minor** version:

- Adding new endpoints or new optional request parameters/fields.
- Adding new response fields. Generated clients must be configured to
  ignore unknown response fields; hand-written clients must do the same.
- Adding new **event types** to the SSE stream. `type` is an open set:
  clients must skip events they do not recognize.
- Adding new values to enums documented as open (`error.code`, tool
  `kind`, permission option `id`).
- Documentation and example changes bump the **patch** version.

## What requires `/v2`

- Removing or renaming an endpoint, request field, or response field.
- Changing a field's type or nullability.
- Changing the meaning of a status code on an existing endpoint.
- Removing values from any enum, or adding values to the closed enums
  (`Run.status`, `Session.behavior_mode`, `Session.permission_mode`,
  `stop_reason`, permission option `kind`).
- Changing SSE ordering or terminal-event semantics.

A `/v2` ships alongside `/v1`; `/v1` is removed no earlier than the next
major release of Anvil after `/v2` becomes the default, and its endpoints
are marked `deprecated: true` in the contract for at least one minor
release beforehand.

## Editing rules

- The contract is hand-authored and reviewed; generated SDK code is never
  edited by hand. TypeScript output is under `sdk/typescript/src/generated`;
  the Rust and Python package trees are generated in full.
- Every change to `src/http_api/` that alters a wire shape must update
  these files and the conformance suite in the same pull request.
- `info.version` must be bumped in the same pull request that changes the
  contract.
