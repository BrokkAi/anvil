use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct SmokeCase {
    name: &'static str,
    prompt: String,
}

#[test]
fn slopcop_shaped_acp_path_does_not_abort() {
    let cases = [SmokeCase {
        name: "structured_readonly_empty_mcp_tool_followup",
        prompt: slopcop_sized_prompt(),
    }];

    for case in cases {
        run_smoke_case(&case);
    }
}

#[test]
fn auto_permission_prompt_cancel_does_not_abort() {
    let case = SmokeCase {
        name: "auto_permission_prompt_cancel",
        prompt: "Check the external cargo registry source if needed.".to_string(),
    };
    run_permission_cancel_case(&case, false);
}

#[test]
fn auto_permission_prompt_session_cancel_does_not_abort() {
    let case = SmokeCase {
        name: "auto_permission_prompt_session_cancel",
        prompt: "Check the external cargo registry source if needed.".to_string(),
    };
    run_permission_cancel_case(&case, true);
}

#[test]
fn relative_cwd_lifecycle_requests_return_invalid_params() {
    let case = SmokeCase {
        name: "relative_cwd_lifecycle_requests",
        prompt: String::new(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let mut child = spawn_anvil(&home, &config_home, &trace_path, None, 1);
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {
                    "readTextFile": false,
                    "writeTextFile": false
                },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);

    let relative_new = client.request(
        "session/new",
        json!({
            "cwd": "relative/repo",
            "mcpServers": []
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new",
        &relative_new,
        "cwd must be absolute",
        &client,
    );

    let new_session = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": []
        }),
    );
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let relative_load = client.request(
        "session/load",
        json!({
            "sessionId": session_id.clone(),
            "cwd": "relative/repo",
            "mcpServers": []
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load",
        &relative_load,
        "cwd must be absolute",
        &client,
    );

    let relative_resume = client.request(
        "session/resume",
        json!({
            "sessionId": session_id,
            "cwd": "relative/repo",
            "mcpServers": []
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/resume",
        &relative_resume,
        "cwd must be absolute",
        &client,
    );

    assert!(
        !client.exited(),
        "{}: anvil exited after relative cwd rejection; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn lifecycle_mcp_servers_applied_and_unsupported_rejected() {
    let case = SmokeCase {
        name: "lifecycle_mcp_servers",
        prompt: "Have a look around.".to_string(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    // A fake stdio MCP server supplied via session/load; spawning it leaves a
    // log line, proving the load-time mcpServers were applied (#145). It lives
    // in its own dir because `make_fake_bifrost_binary` writes a fixed
    // filename, which would otherwise clobber the bifrost script above.
    let extra_dir = temp.path().join("extra-mcp");
    std::fs::create_dir_all(&extra_dir).expect("create extra mcp dir");
    let extra_log = extra_dir.join("extra-mcp-spawn.log");
    let extra_server = make_fake_bifrost_binary(&extra_dir, &extra_log);
    // A second fake stdio server used to prove session/resume also applies a
    // newly-supplied server set, rebuilding the registry (#146).
    let extra2_dir = temp.path().join("extra2-mcp");
    std::fs::create_dir_all(&extra2_dir).expect("create extra2 mcp dir");
    let extra2_log = extra2_dir.join("extra2-mcp-spawn.log");
    let extra2_server = make_fake_bifrost_binary(&extra2_dir, &extra2_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    // Two text turns: one after the load-applied server, one after resume.
    let provider = start_openai_smoke_server(vec![
        text_sse_body("Looked around."),
        text_sse_body("Looked again."),
    ]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        2,
    );
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);
    // Anvil advertises stdio-only MCP support (#159).
    assert_eq!(
        initialize["result"]["agentCapabilities"]["mcpCapabilities"]["http"], false,
        "{}: should advertise mcpCapabilities.http=false: {initialize}",
        case.name
    );
    assert_eq!(
        initialize["result"]["agentCapabilities"]["mcpCapabilities"]["sse"], false,
        "{}: should advertise mcpCapabilities.sse=false: {initialize}",
        case.name
    );

    let http_server = json!({
        "type": "http",
        "name": "remote",
        "url": "https://example.com/mcp",
        "headers": []
    });

    // session/new with an HTTP transport is rejected, not silently dropped (#159).
    let new_http = client.request(
        "session/new",
        json!({ "cwd": cwd, "mcpServers": [http_server] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (http mcp)",
        &new_http,
        "unsupported 'http' transport",
        &client,
    );

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    // session/load and session/resume also reject unsupported transports (#159).
    let load_http = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [http_server] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (http mcp)",
        &load_http,
        "unsupported 'http' transport",
        &client,
    );
    let resume_http = client.request(
        "session/resume",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [http_server] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/resume (http mcp)",
        &resume_http,
        "unsupported 'http' transport",
        &client,
    );

    // session/load with a stdio MCP server applies it; the next prompt builds a
    // registry that spawns the server, leaving a spawn-log entry (#145).
    let load_stdio = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [ {
                "name": "extra",
                "command": extra_server,
                "args": [],
                "env": []
            } ]
        }),
    );
    assert_response_ok(&case, "session/load (stdio mcp)", &load_stdio, &client);
    assert!(
        !extra_log.exists(),
        "{}: stdio MCP server must not spawn until the next prompt",
        case.name
    );

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);
    assert!(
        extra_log.exists(),
        "{}: load-applied stdio MCP server was not spawned on the next prompt (#145); stderr:\n{}",
        case.name,
        client.stderr_text()
    );

    // session/resume with a different stdio server replaces the set and drops
    // the cached registry, so the next prompt spawns the new server (#146).
    let resume_stdio = client.request(
        "session/resume",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [ {
                "name": "extra2",
                "command": extra2_server,
                "args": [],
                "env": []
            } ]
        }),
    );
    assert_response_ok(&case, "session/resume (stdio mcp)", &resume_stdio, &client);
    let prompt2 = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": "again" } ]
        }),
    );
    assert_response_ok(&case, "session/prompt (after resume)", &prompt2, &client);
    assert!(
        extra2_log.exists(),
        "{}: resume-applied stdio MCP server was not spawned on the next prompt (#146); stderr:\n{}",
        case.name,
        client.stderr_text()
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during MCP lifecycle checks; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn session_list_without_cwd_and_cursor_semantics() {
    let case = SmokeCase {
        name: "session_list",
        prompt: "Investigate the repository structure.".to_string(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    // One plain text response: the prompt names the session (auto-rename) and
    // completes without tool calls.
    let provider = start_openai_smoke_server(vec![text_sse_body("Looked around.")]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        2,
    );
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);
    // The list capability must be advertised for this to mean anything.
    assert!(
        initialize["result"]["agentCapabilities"]["sessionCapabilities"]["list"].is_object(),
        "{}: initialize did not advertise sessionCapabilities.list: {initialize}",
        case.name
    );

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    // Prompt the session so it is auto-named (unnamed sessions are filtered out
    // of listings, mirroring the on-disk behavior).
    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);

    // session/list WITHOUT cwd returns the resident named session (#143).
    let list_no_cwd = client.request("session/list", json!({}));
    assert_response_ok(&case, "session/list (no cwd)", &list_no_cwd, &client);
    let ids_no_cwd: Vec<String> = list_no_cwd["result"]["sessions"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
        .collect();
    assert!(
        ids_no_cwd.contains(&session_id),
        "{}: session/list without cwd did not return the session: {list_no_cwd}",
        case.name
    );
    assert!(
        list_no_cwd["result"]["nextCursor"].is_null(),
        "{}: single-page list should omit nextCursor: {list_no_cwd}",
        case.name
    );

    // session/list WITH cwd still filters by cwd and finds it on disk.
    let list_cwd = client.request("session/list", json!({ "cwd": cwd }));
    assert_response_ok(&case, "session/list (cwd)", &list_cwd, &client);
    let ids_cwd: Vec<String> = list_cwd["result"]["sessions"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
        .collect();
    assert!(
        ids_cwd.contains(&session_id),
        "{}: session/list with cwd did not return the session: {list_cwd}",
        case.name
    );

    // An unrecognized cursor is a protocol error, not a silent first page (#144).
    let bad_cursor = client.request("session/list", json!({ "cursor": "not-a-real-cursor" }));
    assert_response_invalid_params_contains(
        &case,
        "session/list (invalid cursor)",
        &bad_cursor,
        "invalid session/list cursor",
        &client,
    );

    // A supplied cwd filter must be absolute, matching the other lifecycle
    // handlers (#143 keeps cwd optional, but a provided one is validated).
    let relative_cwd = client.request("session/list", json!({ "cwd": "relative/repo" }));
    assert_response_invalid_params_contains(
        &case,
        "session/list (relative cwd)",
        &relative_cwd,
        "cwd must be absolute",
        &client,
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during session/list checks; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn lifecycle_unknown_cwd_and_additional_dirs_return_invalid_params() {
    let case = SmokeCase {
        name: "lifecycle_validation",
        prompt: String::new(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    // A real-enough git marker (`.git/HEAD`) so the repo root is recognized and
    // a nested subdir resolves to the same session storage root -- required to
    // exercise the cold-reload cwd-mismatch path below.
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");
    std::fs::write(cwd.join(".git").join("HEAD"), "ref: refs/heads/main\n").expect("write HEAD");
    // A second, distinct repo root used to exercise cwd-mismatch rejection.
    let other_cwd = temp.path().join("other-repo");
    std::fs::create_dir_all(&other_cwd).expect("create other cwd");
    std::fs::create_dir_all(other_cwd.join(".git")).expect("create other git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let mut child = spawn_anvil(&home, &config_home, &trace_path, None, 1);
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);

    // session/new with additionalDirectories is rejected: Anvil does not
    // advertise the capability (#149).
    let new_with_dirs = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [other_cwd]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (additionalDirectories)",
        &new_with_dirs,
        "additionalDirectories is not supported",
        &client,
    );

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    // Unknown ids are protocol errors, not successful empty lifecycle
    // responses (#154).
    let unknown_load = client.request(
        "session/load",
        json!({ "sessionId": "does-not-exist", "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (unknown)",
        &unknown_load,
        "unknown session",
        &client,
    );
    let unknown_resume = client.request(
        "session/resume",
        json!({ "sessionId": "does-not-exist", "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/resume (unknown)",
        &unknown_resume,
        "unknown session",
        &client,
    );

    // Loading a warm session under a different cwd is rejected (#147).
    let mismatched_load = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": other_cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (cwd mismatch)",
        &mismatched_load,
        "does not match",
        &client,
    );
    let mismatched_resume = client.request(
        "session/resume",
        json!({ "sessionId": session_id, "cwd": other_cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/resume (cwd mismatch)",
        &mismatched_resume,
        "does not match",
        &client,
    );

    // additionalDirectories on load/resume is also rejected (#149).
    let load_with_dirs = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [other_cwd]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (additionalDirectories)",
        &load_with_dirs,
        "additionalDirectories is not supported",
        &client,
    );

    // additionalDirectories on resume is rejected too (#149).
    let resume_with_dirs = client.request(
        "session/resume",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [other_cwd]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/resume (additionalDirectories)",
        &resume_with_dirs,
        "additionalDirectories is not supported",
        &client,
    );

    // The matching-cwd load and resume still succeed (#147 regression guard).
    let ok_load = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/load (matching cwd)", &ok_load, &client);
    let ok_resume = client.request(
        "session/resume",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/resume (matching cwd)", &ok_resume, &client);

    // Cold path: close (evict) the session, then load it from disk under a
    // nested cwd that shares the same repo storage root. The persisted manifest
    // cwd must still drive the mismatch rejection (#147), proving the check
    // survives a cold reload and isn't merely a warm in-memory comparison.
    let close = client.request("session/close", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/close", &close, &client);
    let nested_cwd = cwd.join("nested");
    std::fs::create_dir_all(&nested_cwd).expect("create nested cwd");
    let cold_mismatch = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": nested_cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (cold cwd mismatch)",
        &cold_mismatch,
        "does not match",
        &client,
    );
    let cold_ok = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/load (cold matching cwd)", &cold_ok, &client);

    assert!(
        !client.exited(),
        "{}: anvil exited during lifecycle validation; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn mode_and_config_option_surfaces_stay_in_sync() {
    let case = SmokeCase {
        name: "mode_config_sync",
        prompt: String::new(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let mut child = spawn_anvil(&home, &config_home, &trace_path, None, 1);
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();
    // Discard any setup/usage updates from session/new before observing.
    let _ = client.take_update_kinds();

    // Changing the mode via config options also emits current_mode_update for
    // the legacy modes surface (#157).
    let set_behavior = client.request(
        "session/set_config_option",
        json!({ "sessionId": session_id, "configId": "behavior_mode", "value": "CODE" }),
    );
    assert_response_ok(
        &case,
        "session/set_config_option (behavior)",
        &set_behavior,
        &client,
    );
    let mode_update = client
        .take_update_of_kind("current_mode_update")
        .unwrap_or_else(|| {
            panic!(
                "{}: set_config_option(behavior_mode) did not emit current_mode_update",
                case.name
            )
        });
    assert_eq!(
        mode_update["currentModeId"].as_str(),
        Some("CODE"),
        "{}: current_mode_update carried the wrong mode: {mode_update}",
        case.name
    );

    // A non-mode config change must NOT emit a mode update (#157).
    let set_perm = client.request(
        "session/set_config_option",
        json!({ "sessionId": session_id, "configId": "permission_mode", "value": "auto" }),
    );
    assert_response_ok(
        &case,
        "session/set_config_option (permission)",
        &set_perm,
        &client,
    );
    let kinds = client.take_update_kinds();
    assert!(
        !kinds.iter().any(|k| k == "current_mode_update"),
        "{}: non-mode config change emitted an unnecessary current_mode_update: {kinds:?}",
        case.name
    );

    // Changing the mode via the legacy modes API also emits config_option_update
    // for the config-options surface (#156).
    let set_mode = client.request(
        "session/set_mode",
        json!({ "sessionId": session_id, "modeId": "ASK" }),
    );
    assert_response_ok(&case, "session/set_mode", &set_mode, &client);
    let config_update = client
        .take_update_of_kind("config_option_update")
        .unwrap_or_else(|| {
            panic!(
                "{}: set_mode did not emit config_option_update (#156)",
                case.name
            )
        });
    // The behavior-mode selector in the emitted set should reflect ASK.
    let options_json = config_update.to_string();
    assert!(
        options_json.contains("ASK"),
        "{}: config_option_update after set_mode did not reflect the new mode: {config_update}",
        case.name
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during mode/config sync checks; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn invalid_prompt_requests_return_invalid_params() {
    let case = SmokeCase {
        name: "invalid_prompt_requests",
        prompt: String::new(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let mut child = spawn_anvil(&home, &config_home, &trace_path, None, 1);
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false
            }
        }),
    );
    assert_response_ok(&case, "initialize", &initialize, &client);

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    // An empty prompt is an invalid request, not a completed end-turn (#155).
    let empty_prompt = client.request(
        "session/prompt",
        json!({ "sessionId": session_id, "prompt": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/prompt (empty)",
        &empty_prompt,
        "at least one",
        &client,
    );

    // An unknown session is a protocol error, not a successful end-turn (#155).
    let unknown_prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": "does-not-exist",
            "prompt": [ { "type": "text", "text": "hi" } ]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/prompt (unknown session)",
        &unknown_prompt,
        "unknown session",
        &client,
    );

    assert!(
        !client.exited(),
        "{}: anvil exited after invalid prompt rejection; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn run_smoke_case(case: &SmokeCase) {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("README.md"), "# smoke\n").expect("write readme");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = start_openai_smoke_server(vec![
        tool_call_sse_body(),
        text_sse_body(r#"{"answer":"Blocked write observed."}"#),
    ]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        2,
    );
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {
                    "readTextFile": false,
                    "writeTextFile": false
                },
                "terminal": false
            }
        }),
    );
    assert_response_ok(case, "initialize", &initialize, &client);
    assert_eq!(
        initialize["result"]["protocolVersion"], 1,
        "{}: initialize did not negotiate protocol version 1: {initialize}",
        case.name
    );
    assert!(
        initialize["result"]["agentCapabilities"]["promptCapabilities"].is_object(),
        "{}: initialize did not advertise promptCapabilities: {initialize}",
        case.name
    );
    assert_eq!(
        initialize["result"]["agentCapabilities"]["promptCapabilities"]["embeddedContext"], true,
        "{}: initialize did not advertise embedded prompt context support: {initialize}",
        case.name
    );
    assert_eq!(
        initialize["result"]["agentCapabilities"]["promptCapabilities"]["image"], true,
        "{}: initialize did not advertise image prompt support: {initialize}",
        case.name
    );
    assert!(
        initialize["result"]["agentCapabilities"]["sessionCapabilities"]["close"].is_object(),
        "{}: initialize did not advertise sessionCapabilities.close: {initialize}",
        case.name
    );

    let new_session = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": []
        }),
    );
    assert_response_ok(case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let config = client.request(
        "session/set_config_option",
        json!({
            "sessionId": session_id,
            "configId": "permission_mode",
            "value": "readOnly"
        }),
    );
    assert_response_ok(case, "session/set_config_option", &config, &client);

    let mut prompt_params = json!({
        "sessionId": session_id,
        "prompt": [
            {
                "type": "text",
                "text": case.prompt
            }
        ]
    });
    prompt_params["_meta"] = json!({
        "anvil": {
            "structuredOutput": {
                "schemaName": "slopcop_smoke",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "allowCoercion": false
            }
        }
    });

    let prompt = client.request("session/prompt", prompt_params);
    assert_response_ok(case, "session/prompt", &prompt, &client);
    assert!(
        !client.exited(),
        "{}: anvil exited after prompt; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    assert_structured_output_success(case, &prompt, &client);
    assert!(
        !cwd.join("blocked.txt").exists(),
        "{}: readOnly session allowed write_file to create blocked.txt",
        case.name
    );
    assert!(
        bifrost_log.exists(),
        "{}: explicit mcpServers: [] did not spawn persisted default Bifrost",
        case.name,
    );

    assert_eq!(
        provider.request_count(),
        2,
        "{}: expected provider to receive turn 0 and turn 1 requests",
        case.name
    );
    assert!(
        provider.request_bodies().get(1).is_some_and(
            |body| body.contains(r#""role":"tool""#) && body.contains("read-only mode forbids")
        ),
        "{}: turn-1 provider request did not include the readOnly blocked tool result; requests: {:?}",
        case.name,
        provider.request_bodies()
    );
    let trace = client.trace_text();
    assert!(
        trace_has_event_for_turn(&trace, "llm_request", 1),
        "{}: trace missing turn-1 llm_request after blocked tool result\ntrace:\n{}\nstderr:\n{}",
        case.name,
        trace,
        client.stderr_text()
    );
    assert!(
        trace_has_event_for_turn(&trace, "llm_response", 1),
        "{}: trace missing turn-1 llm_response after blocked tool result\ntrace:\n{}\nstderr:\n{}",
        case.name,
        trace,
        client.stderr_text()
    );

    let close = client.request(
        "session/close",
        json!({
            "sessionId": session_id,
        }),
    );
    assert_response_ok(case, "session/close", &close, &client);

    let close_again = client.request(
        "session/close",
        json!({
            "sessionId": session_id,
        }),
    );
    assert_response_error_contains(
        case,
        "session/close",
        &close_again,
        "already closed",
        &client,
    );

    let reload = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": []
        }),
    );
    assert_response_ok(case, "session/load after close", &reload, &client);

    let close_after_reload = client.request(
        "session/close",
        json!({
            "sessionId": session_id,
        }),
    );
    assert_response_ok(
        case,
        "session/close after session/load",
        &close_after_reload,
        &client,
    );

    let close_unknown = client.request(
        "session/close",
        json!({
            "sessionId": "missing-session",
        }),
    );
    assert_response_error_contains(
        case,
        "session/close",
        &close_unknown,
        "unknown session",
        &client,
    );

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn run_permission_cancel_case(case: &SmokeCase, send_session_cancel: bool) {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("README.md"), "# smoke\n").expect("write readme");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = start_openai_smoke_server(vec![
        tool_call_sse_body_for(
            "call_shell",
            "run_shell_command",
            r#"{"command":"sed -n '1,5p' ~/.cargo/config.toml"}"#,
        ),
        text_sse_body(r#"{"allow":false,"rationale":"outside the user request"}"#),
        text_sse_body("Permission prompt cancellation was handled."),
    ]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        2,
    );
    let (stdout_rx, stdout_join) = spawn_line_reader(child.stdout.take().expect("stdout"));
    let (stderr_rx, stderr_join) = spawn_line_reader(child.stderr.take().expect("stderr"));
    let mut stdin = child.stdin.take().expect("stdin");
    let mut client = JsonRpcClient::new(&mut stdin, stdout_rx, stderr_rx, child, trace_path)
        .with_permission_cancel_response(send_session_cancel);

    let initialize = client.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {
                    "readTextFile": false,
                    "writeTextFile": false
                },
                "terminal": false
            }
        }),
    );
    assert_response_ok(case, "initialize", &initialize, &client);

    let new_session = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": []
        }),
    );
    assert_response_ok(case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let config = client.request(
        "session/set_config_option",
        json!({
            "sessionId": session_id,
            "configId": "permission_mode",
            "value": "auto"
        }),
    );
    assert_response_ok(case, "session/set_config_option", &config, &client);

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [
                {
                    "type": "text",
                    "text": case.prompt
                }
            ]
        }),
    );
    assert_response_ok(case, "session/prompt", &prompt, &client);
    assert!(
        !client.exited(),
        "{}: anvil exited after cancelled permission prompt; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    if send_session_cancel {
        // A turn aborted via `session/cancel` MUST resolve its prompt with the
        // `cancelled` stop reason rather than a normal end-turn (#152).
        assert_eq!(
            prompt["result"]["stopReason"].as_str(),
            Some("cancelled"),
            "{}: cancelled prompt did not return stopReason=cancelled: {prompt}",
            case.name
        );
        assert_eq!(
            provider.request_count(),
            2,
            "{}: expected provider to receive only tool and classifier requests after session/cancel",
            case.name
        );
    } else {
        // Declining a single permission (without session/cancel) is not a
        // turn cancellation: the turn completes normally as end_turn (#152).
        assert_eq!(
            prompt["result"]["stopReason"].as_str(),
            Some("end_turn"),
            "{}: non-cancelled prompt did not return stopReason=end_turn: {prompt}",
            case.name
        );
        assert_eq!(
            provider.request_count(),
            3,
            "{}: expected provider to receive tool, classifier, and follow-up requests",
            case.name
        );
        assert!(
            provider
                .request_bodies()
                .get(2)
                .is_some_and(|body| body.contains("prompt was cancelled before the user responded")),
            "{}: follow-up request did not include cancelled permission result; requests: {:?}",
            case.name,
            provider.request_bodies()
        );
    }

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn spawn_anvil(
    home: &Path,
    config_home: &Path,
    trace_path: &Path,
    ollama_base_url: Option<&str>,
    max_turns: usize,
) -> Child {
    let bin = std::env::var_os("CARGO_BIN_EXE_anvil")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_anvil").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/anvil"));
    let max_turns = max_turns.to_string();
    let mut command = Command::new(bin);
    command
        .args([
            "--no-wasm-sandbox",
            "--transient-setup",
            "--default-model",
            "ollama::smoke",
            "--max-turns",
            &max_turns,
            "--llm-idle-timeout-secs",
            "1",
        ])
        .env("HOME", home)
        .env("CODEX_HOME", home.join(".codex"))
        .env("BROKK_CONFIG_HOME", config_home)
        .env("ANVIL_TRACE_JSONL", trace_path)
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("BEDROCK_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(url) = ollama_base_url {
        command.env("ANVIL_TEST_OLLAMA_BASE_URL", url);
    }
    command.spawn().expect("spawn anvil")
}

struct OpenAiSmokeServer {
    base_url: String,
    request_bodies: Arc<Mutex<Vec<String>>>,
}

impl OpenAiSmokeServer {
    fn request_count(&self) -> usize {
        self.request_bodies.lock().unwrap().len()
    }

    fn request_bodies(&self) -> Vec<String> {
        self.request_bodies.lock().unwrap().clone()
    }
}

fn start_openai_smoke_server(response_bodies: Vec<String>) -> OpenAiSmokeServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind smoke provider");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let request_bodies = Arc::new(Mutex::new(Vec::new()));
    let bodies_for_thread = request_bodies.clone();
    std::thread::spawn(move || {
        for (idx, stream) in listener.incoming().enumerate() {
            let Ok(stream) = stream else {
                break;
            };
            let Some(response_body) = response_bodies.get(idx) else {
                break;
            };
            handle_provider_connection(stream, response_body, &bodies_for_thread);
            if idx + 1 == response_bodies.len() {
                break;
            }
        }
    });
    OpenAiSmokeServer {
        base_url,
        request_bodies,
    }
}

fn handle_provider_connection(
    mut stream: TcpStream,
    response_body: &str,
    request_bodies: &Arc<Mutex<Vec<String>>>,
) {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 1024];
    loop {
        let read = stream.read(&mut buf).expect("read provider request");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..read]);
        if raw.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let header_end = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|pos| pos + 4)
        .unwrap_or(raw.len());
    let headers = String::from_utf8_lossy(&raw[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(key, value)| {
                key.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
        })
        .unwrap_or(0);
    while raw.len().saturating_sub(header_end) < content_length {
        let read = stream.read(&mut buf).expect("read provider body");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..read]);
    }
    let body = String::from_utf8_lossy(
        &raw[header_end..header_end + content_length.min(raw.len().saturating_sub(header_end))],
    )
    .to_string();
    request_bodies.lock().unwrap().push(body);

    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    );
    stream
        .write_all(response.as_bytes())
        .expect("write provider response");
    stream.flush().expect("flush provider response");
}

fn tool_call_sse_body() -> String {
    tool_call_sse_body_for(
        "call_write",
        "write_file",
        r#"{"file_path":"blocked.txt","content":"blocked"}"#,
    )
}

fn tool_call_sse_body_for(call_id: &str, tool_name: &str, raw_args: &str) -> String {
    let args = serde_json::to_string(raw_args).expect("encode args");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"{call_id}\",\"function\":{{\"name\":\"{tool_name}\",\"arguments\":{args}}}}}]}}}}]}}\n\
         \n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":4}}}}\n\
         \n\
         data: [DONE]\n\n"
    )
}

fn text_sse_body(text: &str) -> String {
    let text = serde_json::to_string(text).expect("encode text");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{text}}}}}]}}\n\
     \n\
     data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":5}}}}\n\
     \n\
     data: [DONE]\n\n"
    )
}

fn write_setup_with_fake_bifrost(config_home: &Path, temp: &Path, bifrost_log: &Path) {
    let fake_bifrost = make_fake_bifrost_binary(temp, bifrost_log);
    let setup = json!({
        "mcp_servers": [
            {
                "name": "bifrost",
                "command": fake_bifrost,
                "args": ["--root", "{cwd}", "--server", "core", "--no-line-numbers"],
                "framing": "line",
                "enabled": true
            }
        ]
    });
    std::fs::write(config_home.join("setup.json"), setup.to_string()).expect("write setup");
}

#[cfg(unix)]
fn make_fake_bifrost_binary(temp: &Path, bifrost_log: &Path) -> String {
    use std::os::unix::fs::PermissionsExt;

    let script = temp.join("fake-bifrost.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\necho spawned \"$@\" >> '{}'\n",
            bifrost_log.display()
        ),
    )
    .expect("write fake bifrost");
    let mut perms = std::fs::metadata(&script)
        .expect("stat fake bifrost")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).expect("chmod fake bifrost");
    script.display().to_string()
}

#[cfg(not(unix))]
fn make_fake_bifrost_binary(temp: &Path, bifrost_log: &Path) -> String {
    let script = temp.join("fake-bifrost.cmd");
    std::fs::write(
        &script,
        format!(
            "@echo off\r\necho spawned %* >> \"{}\"\r\n",
            bifrost_log.display()
        ),
    )
    .expect("write fake bifrost");
    script.display().to_string()
}

fn trace_has_event_for_turn(trace: &str, event_type: &str, turn: u64) -> bool {
    trace.lines().any(|line| {
        serde_json::from_str::<Value>(line)
            .ok()
            .is_some_and(|value| {
                value.get("type").and_then(Value::as_str) == Some(event_type)
                    && value.get("turn").and_then(Value::as_u64) == Some(turn)
            })
    })
}

fn assert_structured_output_success(
    case: &SmokeCase,
    response: &Value,
    client: &JsonRpcClient<'_>,
) {
    let structured = find_structured_output(response).unwrap_or_else(|| {
        panic!(
            "{}: prompt response missing structured-output metadata: {response}\nstderr:\n{}\ntrace:\n{}",
            case.name,
            client.stderr_text(),
            client.trace_text()
        )
    });
    assert_eq!(
        structured.get("status").and_then(Value::as_str),
        Some("success"),
        "{}: structured-output metadata was not successful: {structured}\nstderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    assert_eq!(
        structured
            .get("validated_output")
            .and_then(|value| value.get("answer"))
            .and_then(Value::as_str),
        Some("Blocked write observed."),
        "{}: structured-output metadata did not round-trip validated answer: {structured}",
        case.name
    );
}

fn find_structured_output(value: &Value) -> Option<&Value> {
    if let Some(found) = value
        .get("anvil")
        .and_then(|anvil| anvil.get("structuredOutput"))
    {
        return Some(found);
    }
    match value {
        Value::Array(items) => items.iter().find_map(find_structured_output),
        Value::Object(map) => map.values().find_map(find_structured_output),
        _ => None,
    }
}

fn spawn_line_reader<R>(reader: R) -> (mpsc::Receiver<String>, std::thread::JoinHandle<()>)
where
    R: std::io::Read + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let join = std::thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => {
                    let _ = tx.send(line);
                }
                Err(_) => break,
            }
        }
    });
    (rx, join)
}

struct JsonRpcClient<'a> {
    stdin: &'a mut std::process::ChildStdin,
    stdout: mpsc::Receiver<String>,
    stderr: mpsc::Receiver<String>,
    child: Child,
    trace_path: PathBuf,
    next_id: u64,
    stderr_lines: Vec<String>,
    cancel_permission_requests: bool,
    send_session_cancel_on_permission: bool,
    /// `params` of every `session/update` notification observed while waiting
    /// for responses, in arrival order.
    session_updates: Vec<Value>,
}

impl<'a> JsonRpcClient<'a> {
    fn new(
        stdin: &'a mut std::process::ChildStdin,
        stdout: mpsc::Receiver<String>,
        stderr: mpsc::Receiver<String>,
        child: Child,
        trace_path: PathBuf,
    ) -> Self {
        Self {
            stdin,
            stdout,
            stderr,
            child,
            trace_path,
            next_id: 1,
            stderr_lines: Vec::new(),
            cancel_permission_requests: false,
            send_session_cancel_on_permission: false,
            session_updates: Vec::new(),
        }
    }

    /// Drain the `session/update` notifications captured so far and return the
    /// `update.sessionUpdate` discriminator of each, in order.
    fn take_update_kinds(&mut self) -> Vec<String> {
        self.session_updates
            .drain(..)
            .filter_map(|params| {
                params
                    .get("update")
                    .and_then(|u| u.get("sessionUpdate"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    /// Drain captured `session/update` notifications and return the full
    /// `update` object of the first one matching `kind`, if any.
    fn take_update_of_kind(&mut self, kind: &str) -> Option<Value> {
        let found = self
            .session_updates
            .iter()
            .find(|params| {
                params
                    .get("update")
                    .and_then(|u| u.get("sessionUpdate"))
                    .and_then(Value::as_str)
                    == Some(kind)
            })
            .and_then(|params| params.get("update").cloned());
        self.session_updates.clear();
        found
    }

    fn with_permission_cancel_response(mut self, send_session_cancel: bool) -> Self {
        self.cancel_permission_requests = true;
        self.send_session_cancel_on_permission = send_session_cancel;
        self
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write request");
        self.stdin.flush().expect("flush request");
        self.wait_for_response(id, method)
    }

    fn wait_for_response(&mut self, id: u64, method: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            self.drain_stderr();
            if let Some(status) = self.child.try_wait().expect("poll child") {
                panic!(
                    "{method}: anvil exited before response id {id}: {status}\nstderr:\n{}\ntrace:\n{}",
                    self.stderr_text(),
                    self.trace_text()
                );
            }
            let now = Instant::now();
            assert!(
                now < deadline,
                "{method}: timed out waiting for response id {id}\nstderr:\n{}\ntrace:\n{}",
                self.stderr_text(),
                self.trace_text()
            );
            let remaining = deadline
                .saturating_duration_since(now)
                .min(Duration::from_millis(200));
            match self.stdout.recv_timeout(remaining) {
                Ok(line) => {
                    let value: Value = serde_json::from_str(&line)
                        .unwrap_or_else(|e| panic!("invalid json line from anvil: {e}: {line}"));
                    if value.get("id").and_then(Value::as_u64) == Some(id) {
                        return value;
                    }
                    // Capture session/update notifications (method, no id) so
                    // tests can assert on emitted updates.
                    if value.get("id").is_none()
                        && value.get("method").and_then(Value::as_str) == Some("session/update")
                    {
                        if let Some(params) = value.get("params") {
                            self.session_updates.push(params.clone());
                        }
                        continue;
                    }
                    if value.get("id").is_some() && value.get("method").is_some() {
                        if self.cancel_permission_requests
                            && value.get("method").and_then(Value::as_str)
                                == Some("session/request_permission")
                        {
                            self.respond_permission_cancelled(&value);
                        } else {
                            self.respond_error(&value);
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!(
                        "{method}: stdout closed before response id {id}\nstderr:\n{}\ntrace:\n{}",
                        self.stderr_text(),
                        self.trace_text()
                    );
                }
            }
        }
    }

    fn respond_error(&mut self, request: &Value) {
        let response = json!({
            "jsonrpc": "2.0",
            "id": request.get("id").cloned().unwrap_or(Value::Null),
            "error": {
                "code": -32601,
                "message": "smoke harness does not implement client request"
            }
        });
        writeln!(self.stdin, "{response}").expect("write client error response");
        self.stdin.flush().expect("flush client error response");
    }

    fn respond_permission_cancelled(&mut self, request: &Value) {
        if self.send_session_cancel_on_permission
            && let Some(session_id) = request
                .get("params")
                .and_then(|params| params.get("sessionId"))
                .and_then(Value::as_str)
        {
            let cancel = json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {
                    "sessionId": session_id
                }
            });
            writeln!(self.stdin, "{cancel}").expect("write session cancel notification");
            self.stdin
                .flush()
                .expect("flush session cancel notification");
        }
        let response = json!({
            "jsonrpc": "2.0",
            "id": request.get("id").cloned().unwrap_or(Value::Null),
            "result": {
                "outcome": {
                    "outcome": "cancelled"
                }
            }
        });
        writeln!(self.stdin, "{response}").expect("write permission cancel response");
        self.stdin
            .flush()
            .expect("flush permission cancel response");
    }

    fn drain_stderr(&mut self) {
        while let Ok(line) = self.stderr.try_recv() {
            self.stderr_lines.push(line);
        }
    }

    fn stderr_text(&self) -> String {
        self.stderr_lines.join("\n")
    }

    fn trace_text(&self) -> String {
        std::fs::read_to_string(&self.trace_path).unwrap_or_default()
    }

    fn exited(&mut self) -> bool {
        self.drain_stderr();
        self.child.try_wait().expect("poll child").is_some()
    }

    fn shutdown(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn assert_response_ok(
    case: &SmokeCase,
    method: &str,
    response: &Value,
    client: &JsonRpcClient<'_>,
) {
    assert!(
        response.get("error").is_none(),
        "{}: {method} returned error: {response}\nstderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
}

fn assert_response_error_contains(
    case: &SmokeCase,
    method: &str,
    response: &Value,
    expected: &str,
    client: &JsonRpcClient<'_>,
) {
    let reason = response["error"]["data"]["reason"]
        .as_str()
        .unwrap_or_default();
    assert!(
        reason.contains(expected),
        "{}: {method} expected error reason containing '{expected}', got: {response}\nstderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
}

fn assert_response_invalid_params_contains(
    case: &SmokeCase,
    method: &str,
    response: &Value,
    expected: &str,
    client: &JsonRpcClient<'_>,
) {
    assert_response_error_contains(case, method, response, expected, client);
    assert_eq!(
        response["error"]["code"],
        -32602,
        "{}: {method} expected invalid params error, got: {response}\nstderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
}

fn slopcop_sized_prompt() -> String {
    let mut prompt = String::from(
        "You are running a SlopCop ACP smoke review. Inspect the repository at a high level, \
         summarize likely risk areas, and return JSON matching the requested schema. Do not edit files.\n\n",
    );
    for idx in 0..80 {
        prompt.push_str(&format!(
            "- Smoke context line {idx}: repository risk signal, static-analysis lane, evidence receipt, readonly execution.\n"
        ));
    }
    prompt
}
