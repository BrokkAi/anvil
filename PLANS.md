# Bedrock Setup Fix — Plan

## Problems

1. **Bedrock missing from `/setup` "Pick one" list**: `render_setup_home_for_model` lists only Codex, local, and OpenRouter as setup options. Bedrock appears only in "Found now" status.

2. **No `/setup bedrock` handler**: `handle_setup` has no `"bedrock"` dispatch arm — typing `/setup bedrock` falls to the `other => "Unknown setup option"` error.

3. **Credentials inconsistent**: Bedrock uses `~/.secrets/` while Codex uses `~/.codex/auth.json` and OpenRouter uses `~/.config/brokk/openrouter.json`. Bedrock should additionally support `~/.config/brokk/bedrock.json`.

4. **No runtime install/uninstall**: `MultiBackend` has `install_codex`/`uninstall_codex` and `install_openrouter`/`uninstall_openrouter` but no bedrock equivalents. The Bedrock backend is static at startup.

5. **Can't paste key via setup**: No `/setup bedrock key <token>` command.

6. **Region/model not settable via setup**: Bedrock also needs a region (env only) and model override — these should be configurable via `/setup bedrock`.

## Plan

### Step 1: Create `src/bedrock_auth.rs` (mirror `openrouter_auth.rs`)

- On-disk credential file at `<config>/brokk/bedrock.json`
- Schema: `{ bearer_token, region?, default_model? }`
- Atomic write (tmp + rename + chmod 600)
- `CredentialState` for env vs file ownership
- Read/write/logout functions
- Precedence: env > file > ~/.secrets/ (legacy fallback)

### Step 2: Update `src/bedrock_client.rs`

- Add `bearer_token_from_brokk_config()` that reads from `bedrock_auth`
- Update `bearer_token_from_env_or_secrets()` to try brokk config first, then env, then secrets
- Add `region_from_config()` and `model_from_config()` that check the auth file
- Export `build_backend_from_config()` that assembles a `BedrockClient` from all sources

### Step 3: Add `install_bedrock` / `uninstall_bedrock` to `MultiBackend`

- Add `RwLock<Option<Arc<dyn LlmBackend>>>` for bedrock slot
- Add `install_bedrock()` / `uninstall_bedrock()` methods
- Update `fallback_source()` to consult the live bedrock slot
- Update `pick()` to read from the RwLock (not the static field)

### Step 4: Add `/setup bedrock` handler

- In `handle_setup`: add `"bedrock" =>` dispatch arm → calls `handle_setup_bedrock`
- `handle_setup_bedrock()`: like `handle_setup_openrouter` but simpler:
  - bare → render bedrock setup help
  - `key <token>` → write to bedrock_auth, install backend, refresh catalog
  - `status` → show credential state
  - `disconnect` → wipe creds, uninstall backend
  - `region <region>` → set region
  - `model <id>` → set default model
  - `refresh` → refresh catalog

### Step 5: Update `render_setup_home_for_model`

- Add `- /setup bedrock - Use AWS Bedrock.` to "Pick one" list

### Step 6: Update `src/main.rs`

- Wire `build_bedrock_backend()` to use `bedrock_auth` as an additional source
- Load bedrock auth at startup for the initial backend

### Step 7: Update messages (help text, advanced setup)

- Update `render_setup_models` Bedrock empty hint to mention `/setup bedrock`
- Update `render_setup_advanced` if needed

### Step 8: Tests

- `bedrock_auth.rs`: round-trip, logout, credential state, env precedence
- `handle_setup_bedrock` tests in agent tests
- MultiBackend bedrock install/uninstall tests