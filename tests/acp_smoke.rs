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

#[derive(Clone, Copy)]
enum AutoEscalationCase {
    ApproveOutside,
    ApproveNormalOnly,
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
fn auto_permission_classifier_denial_does_not_prompt_or_abort() {
    let case = SmokeCase {
        name: "auto_permission_classifier_denial",
        prompt: "Check the external cargo registry source if needed.".to_string(),
    };
    run_auto_classifier_denial_case(&case);
}

#[test]
fn auto_permission_classifier_can_approve_outside_sandbox_without_prompt() {
    if !os_shell_sandbox_available_for_smoke() {
        eprintln!("skipping outside-sandbox auto smoke: OS shell sandbox is unavailable");
        return;
    }

    let case = SmokeCase {
        name: "auto_permission_classifier_approves_outside_sandbox",
        prompt: "Create the requested outside-workspace marker file.".to_string(),
    };
    run_auto_classifier_escalation_case(&case, AutoEscalationCase::ApproveOutside);
}

#[test]
fn auto_permission_classifier_denies_unapproved_sandbox_escape_without_prompt() {
    if !os_shell_sandbox_available_for_smoke() {
        eprintln!("skipping outside-sandbox auto smoke: OS shell sandbox is unavailable");
        return;
    }

    let case = SmokeCase {
        name: "auto_permission_classifier_denies_outside_sandbox",
        prompt: "Create the requested outside-workspace marker file.".to_string(),
    };
    run_auto_classifier_escalation_case(&case, AutoEscalationCase::ApproveNormalOnly);
}

#[test]
fn complex_prompt_runs_directly_without_plan_permission() {
    let case = SmokeCase {
        name: "complex_prompt_runs_directly",
        prompt:
            "Implement the auth database migration, update tests, push the branch, and open a PR."
                .to_string(),
    };
    let (provider, permission_requests) = run_direct_prompt_case(&case);
    assert_eq!(
        provider.request_count(),
        1,
        "{}: complex prompt should make one direct LLM call",
        case.name
    );
    assert_eq!(
        permission_requests, 0,
        "{}: complex prompt should not request plan permission",
        case.name
    );
}

#[test]
fn p2t_replayed_prefix_tails_do_not_complete_without_llm_call() {
    let assistant_step = json!({
        "step": 1,
        "assistant_text": "I inspected calc.py; add() concatenates strings. I will fix and finish.",
        "tool_calls": [],
        "results": []
    });
    let raw_user_step = json!({
        "step": 2,
        "assistant_text": "",
        "tool_calls": [],
        "results": [],
        "messages": [{
            "role": "user",
            "content": "Still reproduces on my end. Could you take another look?"
        }]
    });

    run_p2t_prefix_tail_case(
        SmokeCase {
            name: "p2t_prefix_assistant_tail",
            prompt: "calc.py's add() mishandles string inputs; investigate and fix.".to_string(),
        },
        vec![assistant_step.clone()],
    );
    run_p2t_prefix_tail_case(
        SmokeCase {
            name: "p2t_prefix_raw_user_tail",
            prompt: "calc.py's add() mishandles string inputs; investigate and fix.".to_string(),
        },
        vec![assistant_step, raw_user_step],
    );
}

#[test]
fn session_load_replays_tool_updates_in_order() {
    let case = SmokeCase {
        name: "session_load_tool_replay",
        prompt: "Read README.md and summarize it.".to_string(),
    };
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
        tool_call_sse_body_for("call_read", "read_file", r#"{"file_path":"README.md"}"#),
        text_sse_body("README contains a smoke heading."),
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

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": case.prompt }]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);
    let _ = client.take_update_kinds();

    let load = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": []
        }),
    );
    assert_response_ok(&case, "session/load", &load, &client);
    let updates = client.take_updates();
    let kinds: Vec<String> = updates
        .iter()
        .filter_map(|update| {
            update
                .get("sessionUpdate")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    let position = |kind: &str| {
        kinds
            .iter()
            .position(|observed| observed == kind)
            .unwrap_or_else(|| panic!("{}: missing {kind} in replay updates: {kinds:?}", case.name))
    };
    let user = position("user_message_chunk");
    let tool = position("tool_call");
    let tool_update = position("tool_call_update");
    let agent = position("agent_message_chunk");
    assert!(
        user < tool && tool < tool_update && tool_update < agent,
        "{}: replay updates were out of order: {kinds:?}",
        case.name
    );
    let tool_update_payload = updates
        .iter()
        .find(|update| {
            update.get("sessionUpdate").and_then(Value::as_str) == Some("tool_call_update")
        })
        .unwrap_or_else(|| panic!("{}: missing tool_call_update payload", case.name));
    assert_eq!(
        tool_update_payload["toolCallId"], "call_read",
        "{}: replayed update should keep the original tool call id: {tool_update_payload}",
        case.name
    );
    assert_eq!(
        tool_update_payload["status"], "completed",
        "{}: replayed successful tool should remain completed: {tool_update_payload}",
        case.name
    );
    assert!(
        tool_update_payload["rawOutput"]
            .as_str()
            .is_some_and(|output| output.contains("# smoke")),
        "{}: replayed update should include persisted raw output: {tool_update_payload}",
        case.name
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during session/load replay smoke test; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn additional_directories_scope_builtin_file_tools() {
    let case = SmokeCase {
        name: "additional_directories_tools",
        prompt: "Read files from the configured workspaces.".to_string(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");
    let additional = temp.path().join("additional");
    std::fs::create_dir_all(&additional).expect("create additional root");
    let allowed_file = additional.join("allowed.txt");
    std::fs::write(&allowed_file, "from additional root\n").expect("write allowed file");
    let outside = temp.path().join("outside.txt");
    std::fs::write(&outside, "outside\n").expect("write outside file");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");
    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let allowed_args = format!(
        r#"{{"file_path":{}}}"#,
        serde_json::to_string(&allowed_file.to_string_lossy()).expect("encode path")
    );
    let outside_args = format!(
        r#"{{"file_path":{}}}"#,
        serde_json::to_string(&outside.to_string_lossy()).expect("encode path")
    );
    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = start_openai_smoke_server(vec![
        tool_call_sse_body_for("call_allowed", "read_file", &allowed_args),
        text_sse_body("Read the additional root file."),
        tool_call_sse_body_for("call_outside", "read_file", &outside_args),
        text_sse_body("Outside read was rejected."),
    ]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        4,
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

    let new_session = client.request(
        "session/new",
        json!({ "cwd": cwd, "mcpServers": [], "additionalDirectories": [additional] }),
    );
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": case.prompt }]
        }),
    );
    assert_response_ok(&case, "session/prompt (allowed)", &prompt, &client);
    let allowed_update = client
        .take_updates()
        .into_iter()
        .find(|update| {
            update["sessionUpdate"] == "tool_call_update"
                && update["toolCallId"] == "call_allowed"
                && update["status"] == "completed"
        })
        .unwrap_or_else(|| panic!("{}: missing allowed tool update", case.name));
    assert_eq!(allowed_update["status"], "completed", "{allowed_update}");
    assert!(
        allowed_update["rawOutput"]
            .as_str()
            .is_some_and(|output| output.contains("from additional root")),
        "{}: expected additional-root content in update: {allowed_update}",
        case.name
    );

    let rejected = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": "Try the outside file." }]
        }),
    );
    assert_response_ok(&case, "session/prompt (outside)", &rejected, &client);
    let rejected_update = client
        .take_updates()
        .into_iter()
        .find(|update| {
            update["sessionUpdate"] == "tool_call_update"
                && update["toolCallId"] == "call_outside"
                && update["status"] == "failed"
        })
        .unwrap_or_else(|| panic!("{}: missing rejected tool update", case.name));
    assert_eq!(rejected_update["status"], "failed", "{rejected_update}");
    assert!(
        rejected_update["rawOutput"]
            .as_str()
            .is_some_and(|output| output.contains("escapes")),
        "{}: expected outside-root rejection in update: {rejected_update}",
        case.name
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during additionalDirectories tool checks; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
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
    let additional_root = temp.path().join("additional-root");
    std::fs::create_dir_all(&additional_root).expect("create additional root");

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

    let new_session = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [additional_root],
        }),
    );
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
    let listed_sessions_no_cwd = list_no_cwd["result"]["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("{}: missing sessions array: {list_no_cwd}", case.name));
    let listed_no_cwd = listed_sessions_no_cwd
        .iter()
        .find(|session| session["sessionId"].as_str() == Some(&session_id))
        .unwrap_or_else(|| panic!("{}: session missing from list: {list_no_cwd}", case.name));
    assert_eq!(
        listed_no_cwd["additionalDirectories"],
        json!([additional_root]),
        "{}: session/list did not include additionalDirectories: {list_no_cwd}",
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
    // exercise the cold-reload same-workspace path below.
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
    assert!(
        initialize["result"]["agentCapabilities"]["sessionCapabilities"]["additionalDirectories"]
            .is_object(),
        "{}: initialize did not advertise sessionCapabilities.additionalDirectories: {initialize}",
        case.name
    );

    let new_with_relative_dir = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": ["relative/repo"]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (relative additionalDirectories)",
        &new_with_relative_dir,
        "must be absolute",
        &client,
    );

    let new_with_empty_dir = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [""]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (empty additionalDirectories)",
        &new_with_empty_dir,
        "must be non-empty",
        &client,
    );

    let missing_dir = temp.path().join("missing-additional-root");
    let new_with_missing_dir = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [missing_dir]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (missing additionalDirectories)",
        &new_with_missing_dir,
        "must be an existing directory",
        &client,
    );

    let file_dir = temp.path().join("not-a-directory");
    std::fs::write(&file_dir, "not a directory").expect("write file root");
    let new_with_file_dir = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [file_dir]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/new (file additionalDirectories)",
        &new_with_file_dir,
        "must be a directory",
        &client,
    );

    // session/new accepts additionalDirectories and stores them on the session.
    let new_with_dirs = client.request(
        "session/new",
        json!({
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [other_cwd]
        }),
    );
    assert_response_ok(
        &case,
        "session/new (additionalDirectories)",
        &new_with_dirs,
        &client,
    );
    let session_id = new_with_dirs["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_with_dirs}", case.name))
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

    let load_with_relative_dir = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": ["relative/repo"]
        }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (relative additionalDirectories)",
        &load_with_relative_dir,
        "must be absolute",
        &client,
    );

    // load replaces the session's additionalDirectories with the supplied list.
    let load_with_dirs = client.request(
        "session/load",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": []
        }),
    );
    assert_response_ok(
        &case,
        "session/load (additionalDirectories)",
        &load_with_dirs,
        &client,
    );

    // resume accepts a replacement additionalDirectories list too.
    let resume_with_dirs = client.request(
        "session/resume",
        json!({
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": [],
            "additionalDirectories": [other_cwd]
        }),
    );
    assert_response_ok(
        &case,
        "session/resume (additionalDirectories)",
        &resume_with_dirs,
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
    // nested cwd that resolves to the same repo storage root. A nested checkout
    // (like a linked worktree) is the *same* workspace, so a cold reload must
    // reopen the session from there rather than reject it as a moved cwd --
    // proving the cwd check keys off the persisted workspace root and survives a
    // cold reload instead of being a mere warm in-memory comparison (#147, #241).
    let close = client.request("session/close", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/close", &close, &client);
    let nested_cwd = cwd.join("nested");
    std::fs::create_dir_all(&nested_cwd).expect("create nested cwd");
    let cold_same_workspace = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": nested_cwd, "mcpServers": [] }),
    );
    assert_response_ok(
        &case,
        "session/load (cold same-workspace nested cwd)",
        &cold_same_workspace,
        &client,
    );

    // A genuinely different repo root is a different workspace. After eviction
    // the foreign cwd has no session storage of its own, so a cold load there
    // reports an unknown session rather than silently adopting it.
    let close_again = client.request("session/close", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/close (again)", &close_again, &client);
    let cold_foreign = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": other_cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load (cold foreign workspace)",
        &cold_foreign,
        "unknown session",
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
fn session_fork_creates_independent_session() {
    let case = SmokeCase {
        name: "session_fork",
        prompt: "Summarize the repo.".to_string(),
    };
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");
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
    // One turn to name the source, one to exercise the fork.
    let provider = start_openai_smoke_server(vec![
        text_sse_body("Source summary."),
        text_sse_body("Fork summary."),
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
    assert!(
        initialize["result"]["agentCapabilities"]["sessionCapabilities"]["fork"].is_object(),
        "{}: initialize did not advertise sessionCapabilities.fork: {initialize}",
        case.name
    );

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let source_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": source_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt (source)", &prompt, &client);

    // Fork the source into a new, independent session.
    let fork = client.request(
        "session/fork",
        json!({ "sessionId": source_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/fork", &fork, &client);
    let fork_id = fork["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: fork missing sessionId: {fork}", case.name))
        .to_string();
    assert_ne!(
        fork_id, source_id,
        "{}: fork must have a fresh id",
        case.name
    );
    assert!(
        fork["result"]["modes"].is_object() && fork["result"]["configOptions"].is_array(),
        "{}: fork response missing modes/configOptions: {fork}",
        case.name
    );

    // Both the source and the fork are listed.
    let listed: Vec<String> =
        client.request("session/list", json!({ "cwd": cwd }))["result"]["sessions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
    assert!(
        listed.contains(&source_id) && listed.contains(&fork_id),
        "{}: expected both source and fork listed: {listed:?}",
        case.name
    );

    // The fork is independently promptable.
    let fork_prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": fork_id,
            "prompt": [ { "type": "text", "text": "continue" } ]
        }),
    );
    assert_response_ok(&case, "session/prompt (fork)", &fork_prompt, &client);

    // Forking an unknown source errors; a mismatched cwd errors.
    let fork_unknown = client.request(
        "session/fork",
        json!({ "sessionId": "never-existed", "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/fork (unknown)",
        &fork_unknown,
        "unknown session",
        &client,
    );
    let fork_mismatch = client.request(
        "session/fork",
        json!({ "sessionId": source_id, "cwd": other_cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/fork (cwd mismatch)",
        &fork_mismatch,
        "does not match",
        &client,
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during session/fork checks; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

#[test]
fn session_delete_removes_session_and_is_idempotent() {
    let case = SmokeCase {
        name: "session_delete",
        prompt: "Summarize the repo.".to_string(),
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
    let provider = start_openai_smoke_server(vec![text_sse_body("Summary.")]);
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
    assert!(
        initialize["result"]["agentCapabilities"]["sessionCapabilities"]["delete"].is_object(),
        "{}: initialize did not advertise sessionCapabilities.delete: {initialize}",
        case.name
    );

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    // Name the session (prompt) so it appears in session/list.
    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);

    let list_before = client.request("session/list", json!({ "cwd": cwd }));
    let listed_before: Vec<String> = list_before["result"]["sessions"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        listed_before.contains(&session_id),
        "{}: session not listed before delete: {listed_before:?}",
        case.name
    );

    let delete = client.request("session/delete", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/delete", &delete, &client);

    // Gone from session/list (#141).
    let list_after = client.request("session/list", json!({ "cwd": cwd }));
    let listed_after: Vec<String> = list_after["result"]["sessions"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !listed_after.contains(&session_id),
        "{}: session still listed after delete: {listed_after:?}",
        case.name
    );

    // Persisted state is gone: a subsequent load is an unknown-session error.
    let reload = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_invalid_params_contains(
        &case,
        "session/load after delete",
        &reload,
        "unknown session",
        &client,
    );

    // Deleting an already-deleted or nonexistent session still succeeds (#141).
    let delete_again = client.request("session/delete", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/delete (again)", &delete_again, &client);
    let delete_missing = client.request("session/delete", json!({ "sessionId": "never-existed" }));
    assert_response_ok(&case, "session/delete (missing)", &delete_missing, &client);

    assert!(
        !client.exited(),
        "{}: anvil exited during session/delete checks; stderr:\n{}\ntrace:\n{}",
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
        json!({ "sessionId": session_id, "configId": "behavior_mode", "value": "PLAN" }),
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
        Some("PLAN"),
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
        json!({ "sessionId": session_id, "modeId": "LUTZ" }),
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
    // The behavior-mode selector specifically must reflect the new mode (LUTZ),
    // not merely some unrelated field containing the substring.
    let behavior_option = config_update["configOptions"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|opt| opt.to_string().contains("behavior_mode"))
        .unwrap_or_else(|| {
            panic!(
                "{}: config_option_update missing the behavior_mode selector: {config_update}",
                case.name
            )
        });
    assert!(
        behavior_option.to_string().contains("LUTZ"),
        "{}: behavior_mode selector did not reflect the new mode after set_mode: {behavior_option}",
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

/// Drain captured `session/update` notifications and concatenate the text of
/// every `agent_message_chunk` -- the assistant-visible transcript output.
fn collect_agent_message_text(client: &mut JsonRpcClient) -> String {
    client
        .take_updates()
        .into_iter()
        .filter(|update| update["sessionUpdate"].as_str() == Some("agent_message_chunk"))
        .filter_map(|update| update["content"]["text"].as_str().map(str::to_string))
        .collect()
}

/// A user-initiated `session/cancel` while the LLM request is still in its
/// pre-stream HTTP send phase MUST resolve as a clean cancellation, not an LLM
/// failure. Regression: the loop's `Err` arm rendered the `http_retry`
/// "cancelled while sending request" bail as a
/// `**Error:** LLM request failed: ...` transcript line even though the user
/// simply cancelled. Driven model-free with a provider that records the request
/// then hangs, so the cancel lands squarely in `reqwest`'s `send().await`.
#[test]
fn llm_request_cancelled_mid_send_is_not_reported_as_error() {
    let case = SmokeCase {
        name: "llm_cancel_mid_send",
        prompt: "Do something that needs the model.".to_string(),
    };
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
    let provider = start_hanging_smoke_server();
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

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();
    let _ = client.take_updates();

    // Fire the prompt without blocking; the provider hangs on the resulting
    // chat-completion request, pinning anvil in `send().await`.
    let prompt_id = client.send_request_no_wait(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );

    // Cancel only once the provider has actually received the request, so the
    // cancel is guaranteed to land while anvil is blocked sending it.
    let deadline = Instant::now() + Duration::from_secs(20);
    while provider.request_count() == 0 {
        assert!(
            Instant::now() < deadline,
            "{}: provider never received the LLM request before cancel",
            case.name
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        provider.request_bodies()[0].contains("needs the model"),
        "{}: hung request was not the prompt's chat-completion: {:?}",
        case.name,
        provider.request_bodies()
    );
    client.send_notification("session/cancel", json!({ "sessionId": session_id }));

    let prompt = client.wait_for_response(prompt_id, "session/prompt");
    assert_response_ok(&case, "session/prompt", &prompt, &client);

    // A cancelled turn resolves as `cancelled`, never a normal end-turn...
    assert_eq!(
        prompt["result"]["stopReason"].as_str(),
        Some("cancelled"),
        "{}: cancel mid-send must report stopReason=cancelled: {prompt}",
        case.name
    );

    // ...and crucially, must NOT surface the cancellation as an LLM error in the
    // transcript. This is the regression under test.
    let agent_text = collect_agent_message_text(&mut client);
    assert!(
        !agent_text.contains("**Error:**") && !agent_text.contains("LLM request failed"),
        "{}: cancellation was rendered as an LLM error in the transcript: {agent_text:?}",
        case.name
    );

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

/// A turn that exhausts its `--max-turns` budget must NOT just stop silently:
/// the reason has to reach the transcript (a streamed `agent_message_chunk`)
/// AND the `PromptResponse.stopReason` must be `max_turn_requests`, not a
/// normal `end_turn`. Driven model-free: a single canned tool-call response
/// makes turn 0 a tool turn, so with `--max-turns 1` the loop runs out of its
/// budget before the model produces a final message.
#[test]
fn max_turns_exhaustion_is_reported_in_transcript_and_stop_reason() {
    let case = SmokeCase {
        name: "max_turns_exhaustion",
        prompt: "Read the README and keep going.".to_string(),
    };
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
    // Turn 0 returns a tool call (read_file is auto-allowed, so no permission
    // round-trip); with --max-turns 1 the loop then has no budget for a final
    // text turn and falls through to the turn-limit exit.
    //
    // Exactly one canned body == one LLM call: --max-turns 1 makes turn 0 the
    // only request. If loop semantics ever insert a second LLM call (e.g. a
    // retry/nudge), the mock runs out of bodies and the turn hangs until the
    // 1s idle timeout (set by spawn_anvil), surfacing here as a timeout-`Failed`
    // exit rather than the asserted max_turn_requests -- a signal to revisit
    // this fixture, not a flake to paper over.
    let provider = start_openai_smoke_server(vec![tool_call_sse_body_for(
        "call_read",
        "read_file",
        r#"{"file_path":"README.md"}"#,
    )]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        1,
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

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();
    let _ = client.take_updates();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);

    // (1) The machine-readable stop reason distinguishes a turn-limit exit from
    // a normal completion.
    assert_eq!(
        prompt["result"]["stopReason"].as_str(),
        Some("max_turn_requests"),
        "{}: exhausting --max-turns must report stopReason=max_turn_requests, not end_turn: {prompt}",
        case.name
    );

    // (2) The human-readable reason reached the transcript as agent text. This
    // is independent of the client rendering the stop reason -- it is ordinary
    // streamed assistant output.
    let agent_text = collect_agent_message_text(&mut client);
    assert!(
        agent_text.contains("reached the 1-turn limit"),
        "{}: turn-limit reason did not reach the transcript; agent text was: {agent_text:?}",
        case.name
    );

    // (3) The reason survives a cold reload. Evict the session from memory
    // (session/close), then load it back from disk: the persisted turn's
    // agent_response must still replay the closing notice. This proves the
    // notice was written durably, not just streamed to the live transcript.
    let close = client.request("session/close", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/close", &close, &client);
    let _ = client.take_updates();

    let load = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/load", &load, &client);
    let replayed_text = collect_agent_message_text(&mut client);
    assert!(
        replayed_text.contains("reached the 1-turn limit"),
        "{}: turn-limit reason did not survive a cold reload; replayed text was: {replayed_text:?}",
        case.name
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during max-turns smoke test; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

/// A turn whose final assistant message is empty must say so rather than
/// stopping silently, and the reason must survive a cold reload. This exercises
/// the empty-completion notice and the Case-A reload branch (no tool calls, so
/// `replay_events` is empty and reload replays `agent_response` directly) --
/// distinct from the tool-call/turn-limit path the max-turns test covers.
#[test]
fn empty_completion_is_reported_in_transcript_and_survives_reload() {
    let case = SmokeCase {
        name: "empty_completion",
        prompt: "Say nothing.".to_string(),
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
    // An empty completion is retried on the transient-failure budget
    // (`LLM_MAX_ATTEMPTS` = 4 total attempts), so the model must stay silent
    // across all four attempts for the `Completed { had_text: false }` notice to
    // surface. The retries happen inside a single turn, so `max_turns` stays 2.
    let provider = start_openai_smoke_server(vec![text_sse_body(""); 4]);
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

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();
    let _ = client.take_updates();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [ { "type": "text", "text": case.prompt } ]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);
    // An empty completion is still a finished turn, not a turn-limit exit.
    assert_eq!(
        prompt["result"]["stopReason"].as_str(),
        Some("end_turn"),
        "{}: empty completion should report stopReason=end_turn: {prompt}",
        case.name
    );
    let agent_text = collect_agent_message_text(&mut client);
    assert!(
        agent_text.contains("ended the turn without a final message"),
        "{}: empty-completion reason did not reach the transcript; agent text was: {agent_text:?}",
        case.name
    );

    // Survives a cold reload through the no-tool-calls (Case A) replay branch.
    let close = client.request("session/close", json!({ "sessionId": session_id }));
    assert_response_ok(&case, "session/close", &close, &client);
    let _ = client.take_updates();
    let load = client.request(
        "session/load",
        json!({ "sessionId": session_id, "cwd": cwd, "mcpServers": [] }),
    );
    assert_response_ok(&case, "session/load", &load, &client);
    let replayed_text = collect_agent_message_text(&mut client);
    assert!(
        replayed_text.contains("ended the turn without a final message"),
        "{}: empty-completion reason did not survive a cold reload; replayed text was: {replayed_text:?}",
        case.name
    );

    assert!(
        !client.exited(),
        "{}: anvil exited during empty-completion smoke test; stderr:\n{}\ntrace:\n{}",
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

fn run_p2t_prefix_tail_case(case: SmokeCase, prefix_steps: Vec<Value>) {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("calc.py"), "def add(a, b):\n    return a + b\n")
        .expect("write calc.py");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let p2t_dir = temp.path().join("p2t");
    std::fs::create_dir_all(&p2t_dir).expect("create p2t dir");
    let prefix_path = p2t_dir.join("prefix.jsonl");
    let prefix_jsonl = prefix_steps
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&prefix_path, format!("{prefix_jsonl}\n")).expect("write prefix");
    let step_trace_out = p2t_dir.join("steps.jsonl");
    let p2t_config = p2t_dir.join("p2t.json");
    std::fs::write(
        &p2t_config,
        json!({
            "prefix_steps": prefix_path,
            "forced_first_step": null,
            "max_steps": 4,
            "snapshot_dir": null,
            "temperature": 0.7,
            "step_trace_out": step_trace_out,
            "link_base": null
        })
        .to_string(),
    )
    .expect("write p2t config");

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = start_openai_smoke_server(vec![text_sse_body("Continuing after prefix.")]);
    let mut child = spawn_anvil_with_p2t(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        4,
        &p2t_config,
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

    let new_session = client.request("session/new", json!({ "cwd": cwd, "mcpServers": [] }));
    assert_response_ok(&case, "session/new", &new_session, &client);
    let session_id = new_session["result"]["sessionId"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: missing sessionId in {new_session}", case.name))
        .to_string();

    let prompt = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": case.prompt }]
        }),
    );
    assert_response_ok(&case, "session/prompt", &prompt, &client);
    assert_eq!(
        provider.request_count(),
        1,
        "{}: replayed P2T prefix should not satisfy the turn before the LLM call",
        case.name
    );
    assert!(
        prompt["result"]["usage"]["totalTokens"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "{}: prompt should include usage from the mock LLM call: {prompt}",
        case.name
    );

    let records = read_jsonl_values(&step_trace_out);
    assert!(
        records
            .iter()
            .any(|record| record["type"].as_str() == Some("window_start")),
        "{}: P2T trace missing window_start; records={records:?}",
        case.name
    );
    let step_count = records
        .iter()
        .filter(|record| record["type"].as_str() == Some("step"))
        .count();
    assert!(
        step_count > 0,
        "{}: P2T trace should contain an executed step after replayed prefix; records={records:?}",
        case.name
    );
    assert!(
        !client.exited(),
        "{}: anvil exited during P2T prefix-tail smoke test; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn run_direct_prompt_case(case: &SmokeCase) -> (OpenAiSmokeServer, usize) {
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
    let provider =
        start_openai_smoke_server(vec![text_sse_body("A trait defines shared behavior.")]);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        Some(provider.base_url.as_str()),
        1,
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
    let permission_requests = client.permission_request_count;
    assert!(
        !client.exited(),
        "{}: anvil exited during direct prompt smoke test; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
    (provider, permission_requests)
}

fn run_auto_classifier_denial_case(case: &SmokeCase) {
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
        text_sse_body(
            r#"{"allow":false,"sandbox":"normal","rationale":"outside the user request"}"#,
        ),
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
        "{}: anvil exited after auto-classifier denial; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    assert_eq!(
        client.permission_request_count, 0,
        "{}: auto mode must not send client permission prompts",
        case.name
    );
    assert_eq!(
        prompt["result"]["stopReason"].as_str(),
        Some("end_turn"),
        "{}: classifier denial should complete as a normal turn: {prompt}",
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
            .is_some_and(|body| body.contains("Tool use denied by auto permissions")),
        "{}: follow-up request did not include classifier denial result; requests: {:?}",
        case.name,
        provider.request_bodies()
    );

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn run_auto_classifier_escalation_case(case: &SmokeCase, escalation_case: AutoEscalationCase) {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("README.md"), "# smoke\n").expect("write readme");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");
    let Some(outside_marker_dir) = outside_sandbox_marker_dir_for_smoke(case.name) else {
        eprintln!("skipping outside-sandbox auto smoke: HOME is unavailable for outside marker");
        return;
    };
    std::fs::create_dir_all(&outside_marker_dir).expect("create outside marker dir");
    let _outside_marker_cleanup = RemoveDirOnDrop(outside_marker_dir.clone());
    let outside_marker = outside_marker_dir.join("outside-marker.txt");
    let outside_marker_arg = shell_single_quote(&outside_marker.to_string_lossy());
    let command =
        format!("printf 'auto-approved\\n' > {outside_marker_arg} && cat {outside_marker_arg}");
    let normal_shell_args = json!({ "command": command }).to_string();
    let shell_args = json!({
        "command": command,
        "sandbox_permissions": "require_escalated"
    })
    .to_string();

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    let bifrost_log = temp.path().join("bifrost-spawn.log");
    write_setup_with_fake_bifrost(&config_home, temp.path(), &bifrost_log);

    let classifier_body = match escalation_case {
        AutoEscalationCase::ApproveOutside => text_sse_body(
            r#"{"allow":true,"sandbox":"outside","rationale":"the user requested an outside-workspace marker file"}"#,
        ),
        AutoEscalationCase::ApproveNormalOnly => text_sse_body(
            r#"{"allow":true,"sandbox":"normal","rationale":"normal sandbox execution is enough"}"#,
        ),
    };
    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = start_openai_smoke_server(vec![
        tool_call_sse_body_for("call_shell_normal", "run_shell_command", &normal_shell_args),
        text_sse_body(
            r#"{"allow":true,"sandbox":"normal","rationale":"precondition should stay sandboxed"}"#,
        ),
        text_sse_body("Normal sandbox precondition was handled."),
        tool_call_sse_body_for("call_shell", "run_shell_command", &shell_args),
        classifier_body,
        text_sse_body("Escalation decision was handled."),
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
    assert_response_ok(
        case,
        "session/set_config_option (permission)",
        &config,
        &client,
    );

    let precondition = client.request(
        "session/prompt",
        json!({
            "sessionId": session_id,
            "prompt": [
                {
                    "type": "text",
                    "text": "Verify the normal sandbox blocks this outside marker write."
                }
            ]
        }),
    );
    assert_response_ok(
        case,
        "session/prompt (normal sandbox precondition)",
        &precondition,
        &client,
    );
    assert_eq!(
        precondition["result"]["stopReason"].as_str(),
        Some("end_turn"),
        "{}: normal-sandbox precondition should complete as a normal turn: {precondition}",
        case.name
    );
    assert!(
        !outside_marker.exists(),
        "{}: normal sandbox unexpectedly wrote outside marker at {}; the escalation smoke would not prove an outside-sandbox escape",
        case.name,
        outside_marker.display()
    );

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
        "{}: anvil exited after auto-classifier escalation case; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    assert_eq!(
        client.permission_request_count, 0,
        "{}: auto mode must not send client permission prompts",
        case.name
    );
    assert_eq!(
        prompt["result"]["stopReason"].as_str(),
        Some("end_turn"),
        "{}: classifier escalation case should complete as a normal turn: {prompt}",
        case.name
    );
    assert_eq!(
        provider.request_count(),
        6,
        "{}: expected provider to receive precondition and escalation tool/classifier/follow-up requests",
        case.name
    );

    let bodies = provider.request_bodies();
    assert!(
        bodies.get(1).is_some_and(
            |body| body.contains("run_shell_command") && body.contains("auto-approved")
        ) && !bodies
            .get(1)
            .is_some_and(|body| body.contains("sandbox_permissions")),
        "{}: precondition classifier request should be for a normal sandbox call; requests: {:?}",
        case.name,
        bodies
    );
    assert!(
        bodies.get(4).is_some_and(
            |body| body.contains("sandbox_permissions") && body.contains("require_escalated")
        ),
        "{}: classifier request did not include the escalation marker; requests: {:?}",
        case.name,
        bodies
    );
    let updates_json =
        serde_json::to_string(&client.take_updates()).expect("encode captured session updates");
    match escalation_case {
        AutoEscalationCase::ApproveOutside => {
            let marker = std::fs::read_to_string(&outside_marker).unwrap_or_else(|error| {
                panic!("{}: failed to read host marker: {error}", case.name)
            });
            assert_eq!(
                marker, "auto-approved\n",
                "{}: outside-sandbox approval did not create the host marker",
                case.name
            );
            assert!(
                updates_json.contains("approved outside-sandbox execution for this tool call"),
                "{}: update stream missing auto outside-sandbox approval notice: {updates_json}",
                case.name
            );
            assert!(
                bodies
                    .get(5)
                    .is_some_and(|body| body.contains("auto-approved")),
                "{}: follow-up request did not include successful shell output; requests: {:?}",
                case.name,
                bodies
            );
        }
        AutoEscalationCase::ApproveNormalOnly => {
            assert!(
                !outside_marker.exists(),
                "{}: classifier denial should not execute the outside write",
                case.name
            );
            assert!(
                updates_json.contains("did not approve outside-sandbox execution"),
                "{}: update stream missing auto outside-sandbox denial notice: {updates_json}",
                case.name
            );
            assert!(
                bodies
                    .get(5)
                    .is_some_and(|body| body
                        .contains("did not explicitly approve running outside the sandbox")),
                "{}: follow-up request did not include classifier escalation denial; requests: {:?}",
                case.name,
                bodies
            );
        }
    }
    assert!(
        !client
            .stderr_text()
            .contains("failed to prepare bundled bifrost"),
        "{}: fake managed Bifrost cache did not match production discovery path; stderr:\n{}",
        case.name,
        client.stderr_text()
    );

    client.shutdown();
    let _ = stdout_join.join();
    let _ = stderr_join.join();
}

fn outside_sandbox_marker_dir_for_smoke(case_name: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let safe_case: String = case_name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    Some(home.join(format!(
        ".anvil-acp-smoke-{safe_case}-{}",
        uuid::Uuid::new_v4()
    )))
}

struct RemoveDirOnDrop(PathBuf);

impl Drop for RemoveDirOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn os_shell_sandbox_available_for_smoke() -> bool {
    #[cfg(target_os = "macos")]
    {
        true
    }
    #[cfg(target_os = "linux")]
    {
        if Path::new("/usr/bin/bwrap").is_file() {
            return true;
        }
        std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).any(|dir| dir.join("bwrap").is_file()))
            .unwrap_or(false)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

fn spawn_anvil(
    home: &Path,
    config_home: &Path,
    trace_path: &Path,
    ollama_base_url: Option<&str>,
    max_turns: usize,
) -> Child {
    spawn_anvil_inner(
        home,
        config_home,
        trace_path,
        ollama_base_url,
        max_turns,
        None,
    )
}

fn spawn_anvil_with_p2t(
    home: &Path,
    config_home: &Path,
    trace_path: &Path,
    ollama_base_url: Option<&str>,
    max_turns: usize,
    p2t_config: &Path,
) -> Child {
    spawn_anvil_inner(
        home,
        config_home,
        trace_path,
        ollama_base_url,
        max_turns,
        Some(p2t_config),
    )
}

fn spawn_anvil_inner(
    home: &Path,
    config_home: &Path,
    trace_path: &Path,
    ollama_base_url: Option<&str>,
    max_turns: usize,
    p2t_config: Option<&Path>,
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
        // Recaps default on, but each enabled turn fires a recap-summary LLM
        // call that would consume a canned body from the deterministic mock
        // provider and desync multi-turn fixtures. No smoke test asserts recap
        // content, so force them off; the recap path has its own unit coverage.
        .env("ANVIL_TEST_DISABLE_TURN_RECAP", "1")
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("BEDROCK_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(url) = ollama_base_url {
        command.env("ANVIL_TEST_OLLAMA_BASE_URL", url);
    }
    if let Some(path) = p2t_config {
        command
            .env("BRK_PATCHES_TO_TRACES", "1")
            .env("BRK_P2T_CONFIG", path);
    }
    command.spawn().expect("spawn anvil")
}

fn read_jsonl_values(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("parse jsonl record"))
        .collect()
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

/// Read one HTTP request off `stream` and record its body in `request_bodies`,
/// returning once the full body has arrived. Shared by the responding provider
/// and the hanging provider (which records the request, then never replies).
fn read_provider_request(stream: &mut TcpStream, request_bodies: &Arc<Mutex<Vec<String>>>) {
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
}

/// A provider that accepts one connection, reads + records the chat-completion
/// request, then holds the socket open without ever sending response headers.
/// This pins the client in `reqwest`'s `send().await` (pre-stream phase), which
/// is the exact window in which a `session/cancel` exercises the
/// "cancelled while sending request" path in `http_retry`.
fn start_hanging_smoke_server() -> OpenAiSmokeServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind hanging provider");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let request_bodies = Arc::new(Mutex::new(Vec::new()));
    let bodies_for_thread = request_bodies.clone();
    std::thread::spawn(move || {
        if let Some(Ok(mut stream)) = listener.incoming().next() {
            read_provider_request(&mut stream, &bodies_for_thread);
            // Never write a response: hold the connection open past any
            // realistic test window so the client stays blocked in `send()`
            // until the turn is cancelled. Dropping `stream` afterwards closes
            // the socket; by then the test has already finished.
            std::thread::sleep(Duration::from_secs(30));
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
    read_provider_request(&mut stream, request_bodies);

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
    seed_fake_managed_bifrost(config_home, &fake_bifrost);
    let setup = json!({
        "mcp_servers": [
            {
                "name": "bifrost",
                "command": fake_bifrost,
                "args": ["--root", "{cwd}", "--mcp", "searchtools", "--no-line-numbers"],
                "framing": "line",
                "enabled": true
            }
        ]
    });
    std::fs::write(config_home.join("setup.json"), setup.to_string()).expect("write setup");
}

fn seed_fake_managed_bifrost(config_home: &Path, fake_bifrost: &str) {
    let cache_dir = config_home
        .join("bifrost")
        .join("0.7.4")
        .join(bifrost_target_triple_for_smoke());
    std::fs::create_dir_all(&cache_dir).expect("create fake managed bifrost cache");
    let target = cache_dir.join(bifrost_binary_name_for_smoke());
    std::fs::copy(fake_bifrost, &target).expect("seed fake managed bifrost");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut perms = std::fs::metadata(&target)
            .expect("stat fake managed bifrost")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&target, perms).expect("chmod fake managed bifrost");
    }
}

fn bifrost_binary_name_for_smoke() -> &'static str {
    #[cfg(windows)]
    {
        "bifrost.exe"
    }
    #[cfg(not(windows))]
    {
        "bifrost"
    }
}

fn bifrost_target_triple_for_smoke() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "universal-apple-darwin"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-msvc"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "aarch64-pc-windows-msvc"
    }
    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    {
        "aarch64-linux-android"
    }
    #[cfg(not(any(
        target_os = "macos",
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "aarch64"),
    )))]
    {
        "unsupported"
    }
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
    permission_request_count: usize,
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
            permission_request_count: 0,
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

    fn take_updates(&mut self) -> Vec<Value> {
        self.session_updates
            .drain(..)
            .filter_map(|params| params.get("update").cloned())
            .collect()
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.send_request_no_wait(method, params);
        self.wait_for_response(id, method)
    }

    /// Write a request without blocking on its response, returning its id so the
    /// caller can `wait_for_response` later. Lets a test interleave other
    /// traffic (e.g. a `session/cancel`) while a long-running request is in
    /// flight.
    fn send_request_no_wait(&mut self, method: &str, params: Value) -> u64 {
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
        id
    }

    fn send_notification(&mut self, method: &str, params: Value) {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{notification}").expect("write notification");
        self.stdin.flush().expect("flush notification");
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
                        if value.get("method").and_then(Value::as_str)
                            == Some("session/request_permission")
                        {
                            self.permission_request_count += 1;
                        }
                        self.respond_error(&value);
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
