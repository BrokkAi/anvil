//! End-to-end smoke tests for the headless `anvil --print` client (#356).
//!
//! Each test spawns the real binary with a deterministic mock OpenAI-style
//! provider (the same approach as `acp_smoke.rs`): canned SSE bodies are
//! served to chat-completion requests in order, so the agent's behavior --
//! and therefore the headless client's stdout contract -- is fully scripted.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

const SMOKE_BUNDLED_BIFROST_VERSION: &str = "0.8.21";

// ---------------------------------------------------------------------------
// Mock provider (canned SSE bodies served per connection, in order)
// ---------------------------------------------------------------------------

fn start_provider(response_bodies: Vec<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind smoke provider");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    std::thread::spawn(move || {
        // Keep the listener (and therefore the port) alive for the whole test
        // process: dropping it after the last body would let the OS hand the
        // port to a parallel test's provider, and a late retry from this test
        // could then consume that provider's canned bodies. Connections past
        // the scripted set are dropped without a response.
        for (idx, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else {
                break;
            };
            let Some(response_body) = response_bodies.get(idx) else {
                continue;
            };
            read_provider_request(&mut stream);
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
    });
    base_url
}

fn read_provider_request(stream: &mut TcpStream) {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 1024];
    loop {
        let read = stream.read(&mut buf).expect("read provider request");
        if read == 0 {
            return;
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

// ---------------------------------------------------------------------------
// Fake managed Bifrost (so no MCP download/spawn escapes the sandbox)
// ---------------------------------------------------------------------------

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
        "x86_64-unknown-linux-musl"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-msvc"
    }
    #[cfg(not(any(
        target_os = "macos",
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    {
        "unknown"
    }
}

fn install_fake_managed_bifrost(config_home: &Path, temp: &Path) {
    let bifrost_log = temp.join("bifrost-spawn.log");
    let fake_bifrost = make_fake_bifrost_binary(temp, &bifrost_log);
    let cache_dir = config_home
        .join("bifrost")
        .join(SMOKE_BUNDLED_BIFROST_VERSION)
        .join(bifrost_target_triple_for_smoke());
    std::fs::create_dir_all(&cache_dir).expect("create fake managed bifrost cache");
    let target = cache_dir.join(bifrost_binary_name_for_smoke());
    std::fs::copy(&fake_bifrost, &target).expect("seed fake managed bifrost");

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

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct SmokeEnv {
    _temp: tempfile::TempDir,
    cwd: PathBuf,
    home: PathBuf,
    config_home: PathBuf,
}

fn smoke_env() -> SmokeEnv {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("README.md"), "# smoke\n").expect("write readme");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");
    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    install_fake_managed_bifrost(&config_home, temp.path());

    SmokeEnv {
        cwd,
        home,
        config_home,
        _temp: temp,
    }
}

fn anvil_binary() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_anvil")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_anvil").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/anvil"))
}

fn run_print(
    env: &SmokeEnv,
    provider_url: &str,
    extra_args: &[&str],
    stdin_text: Option<&str>,
) -> Output {
    let mut command = Command::new(anvil_binary());
    command
        .args([
            "--no-wasm-sandbox",
            "--transient-setup",
            "--default-model",
            "ollama::smoke",
            "--max-turns",
            "3",
        ])
        .args(extra_args)
        .env("HOME", &env.home)
        .env("CODEX_HOME", env.home.join(".codex"))
        .env("BROKK_CONFIG_HOME", &env.config_home)
        .env("ANVIL_TEST_OLLAMA_BASE_URL", provider_url)
        // Deliberately NOT setting ANVIL_TEST_DISABLE_TURN_RECAP: headless
        // mode itself must suppress turn recaps. If that regresses, the
        // yolo test's file-changing turn emits a recap that both consumes an
        // extra canned LLM body and pollutes the result text.
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("BEDROCK_API_KEY")
        .env_remove("DEEPSEEK_API_KEY")
        .env_remove("KIMI_API_KEY")
        .env_remove("ANVIL_TRACE_JSONL")
        .env_remove("BROKK_SESSION_STORAGE_ROOT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().expect("spawn anvil --print");
    {
        let mut stdin = child.stdin.take().expect("stdin");
        if let Some(text) = stdin_text {
            stdin.write_all(text.as_bytes()).expect("write stdin");
        }
        // Dropping stdin closes it, which is what unblocks a `-p -` read.
    }

    let deadline = Instant::now() + Duration::from_secs(120);
    let (stdout, stdout_reader) = capture_pipe(child.stdout.take().expect("stdout"));
    let (stderr, stderr_reader) = capture_pipe(child.stderr.take().expect("stderr"));
    let status = loop {
        match child.try_wait().expect("wait on anvil") {
            Some(status) => break status,
            None if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "anvil --print did not exit within the deadline;\nstdout:\n{}\nstderr:\n{}",
                    pipe_text(&stdout),
                    pipe_text(&stderr)
                );
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    };
    // Join the reader threads before snapshotting: bytes the child wrote just
    // before exiting can still be in the pipe when try_wait reports exit.
    stdout_reader.join().expect("join stdout reader");
    stderr_reader.join().expect("join stderr reader");
    Output {
        status,
        stdout: pipe_text(&stdout).into_bytes(),
        stderr: pipe_text(&stderr).into_bytes(),
    }
}

fn capture_pipe<R: Read + Send + 'static>(
    mut pipe: R,
) -> (Arc<Mutex<Vec<u8>>>, std::thread::JoinHandle<()>) {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let writer = buffer.clone();
    let reader = std::thread::spawn(move || {
        let mut chunk = [0_u8; 4096];
        while let Ok(read) = pipe.read(&mut chunk) {
            if read == 0 {
                break;
            }
            writer.lock().unwrap().extend_from_slice(&chunk[..read]);
        }
    });
    (buffer, reader)
}

fn pipe_text(buffer: &Arc<Mutex<Vec<u8>>>) -> String {
    String::from_utf8_lossy(&buffer.lock().unwrap()).to_string()
}

fn stdout_str(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr_str(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

fn parse_json_stdout(output: &Output) -> Value {
    let stdout = stdout_str(output);
    serde_json::from_str(stdout.trim()).unwrap_or_else(|err| {
        panic!(
            "stdout is not a single JSON object ({err});\nstdout:\n{stdout}\nstderr:\n{}",
            stderr_str(output)
        )
    })
}

fn parse_stream_stdout(output: &Output) -> Vec<Value> {
    let stdout = stdout_str(output);
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|err| {
                panic!(
                    "stream-json line is not JSON ({err}): {line}\nstderr:\n{}",
                    stderr_str(output)
                )
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn print_text_outputs_final_message_only() {
    let env = smoke_env();
    let provider = start_provider(vec![text_sse_body("Hello from the smoke model.")]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &["--print", "Say hello.", "--cwd", &cwd],
        None,
    );
    assert!(
        output.status.success(),
        "text run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    assert_eq!(stdout_str(&output), "Hello from the smoke model.\n");
}

#[test]
fn print_json_reports_result_and_stop_reason() {
    let env = smoke_env();
    let provider = start_provider(vec![text_sse_body("The answer is 4.")]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &[
            "-p",
            "What is 2+2?",
            "--output-format",
            "json",
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        output.status.success(),
        "json run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    let payload = parse_json_stdout(&output);
    assert_eq!(payload["result"], "The answer is 4.");
    assert_eq!(payload["stop_reason"], "end_turn");
    assert_eq!(payload["resumed"], false);
    assert_eq!(payload["error"], Value::Null);
    assert!(
        payload["session_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()),
        "missing session_id in {payload}"
    );
    assert!(
        payload["usage"]["totalTokens"].is_u64(),
        "usage should carry token totals: {payload}"
    );
}

#[test]
fn print_structured_output_retries_with_validation_feedback() {
    let env = smoke_env();
    let provider = start_provider(vec![
        text_sse_body(r#"{"rank":"six"}"#),
        text_sse_body(r#"{"rank":6}"#),
    ]);
    let schema_path = env._temp.path().join("evaluation.schema.json");
    std::fs::write(
        &schema_path,
        serde_json::json!({
            "type": "object",
            "properties": {"rank": {"type": "integer"}},
            "required": ["rank"],
            "additionalProperties": false
        })
        .to_string(),
    )
    .expect("write response schema");
    let cwd = env.cwd.display().to_string();
    let schema = schema_path.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &[
            "--print",
            "Evaluate the task.",
            "--output-format",
            "json",
            "--cwd",
            &cwd,
            "--response-schema",
            &schema,
            "--response-schema-name",
            "evaluation",
        ],
        None,
    );
    assert!(
        output.status.success(),
        "structured-output repair failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    let payload = parse_json_stdout(&output);
    assert_eq!(payload["structured_output"], serde_json::json!({"rank": 6}));
    assert_eq!(payload["result"], r#"{"rank":6}"#);
    assert_eq!(payload["error"], Value::Null);
}

#[test]
fn print_reads_prompt_from_stdin() {
    let env = smoke_env();
    let provider = start_provider(vec![text_sse_body("Read you loud and clear.")]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &["--print", "--output-format", "json", "--cwd", &cwd],
        Some("Repeat after me."),
    );
    assert!(
        output.status.success(),
        "stdin run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    let payload = parse_json_stdout(&output);
    assert_eq!(payload["result"], "Read you loud and clear.");
}

#[test]
fn print_stream_json_emits_records_with_result_last() {
    let env = smoke_env();
    // Turn 1 requests a shell command; default (manual) permission mode must
    // reject it without hanging, and turn 2 answers with text.
    let provider = start_provider(vec![
        tool_call_sse_body_for(
            "call_shell",
            "run_shell_command",
            r#"{"command":"touch should-not-exist.txt"}"#,
        ),
        text_sse_body("Done without running the command."),
    ]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &[
            "-p",
            "Touch a file.",
            "--output-format",
            "stream-json",
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        output.status.success(),
        "stream-json run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    let records = parse_stream_stdout(&output);
    let types: Vec<&str> = records
        .iter()
        .map(|record| record["type"].as_str().unwrap_or("?"))
        .collect();
    assert_eq!(
        types.first(),
        Some(&"connected"),
        "expected connected first: {types:?}"
    );
    assert_eq!(
        types.get(1),
        Some(&"session_started"),
        "expected session_started second: {types:?}"
    );
    assert_eq!(
        types.last(),
        Some(&"result"),
        "expected result last: {types:?}"
    );
    assert!(
        types.contains(&"tool_call"),
        "expected a tool_call record: {types:?}"
    );
    assert!(
        types.contains(&"agent_message"),
        "expected agent_message records: {types:?}"
    );

    let permission = records
        .iter()
        .find(|record| record["type"] == "permission")
        .unwrap_or_else(|| panic!("expected a permission record in {types:?}"));
    assert_eq!(
        permission["decision"], "reject",
        "manual mode must reject the shell request: {permission}"
    );
    assert!(
        !env.cwd.join("should-not-exist.txt").exists(),
        "rejected shell command must not run"
    );

    let result = records.last().expect("result record");
    assert_eq!(result["result"], "Done without running the command.");
    assert_eq!(result["stop_reason"], "end_turn");
    assert_eq!(result["error"], Value::Null);
}

#[test]
fn print_auto_mode_allows_first_class_move_and_delete() {
    let env = smoke_env();
    std::fs::write(env.cwd.join("source.txt"), "move me").expect("write source file");
    let provider = start_provider(vec![
        tool_call_sse_body_for(
            "call_move",
            "move_file",
            r#"{"source_path":"source.txt","destination_path":"moved.txt"}"#,
        ),
        tool_call_sse_body_for("call_delete", "delete_file", r#"{"file_path":"moved.txt"}"#),
        text_sse_body("File moved and deleted."),
    ]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &[
            "-p",
            "Move source.txt to moved.txt, then delete it.",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "auto",
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        output.status.success(),
        "auto run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    assert!(!env.cwd.join("source.txt").exists());
    assert!(!env.cwd.join("moved.txt").exists());

    let records = parse_stream_stdout(&output);
    let permissions: Vec<(&str, &str)> = records
        .iter()
        .filter(|record| record["type"] == "permission")
        .map(|record| {
            (
                record["tool_call_id"].as_str().expect("permission call id"),
                record["decision"].as_str().expect("permission decision"),
            )
        })
        .collect();
    assert_eq!(
        permissions,
        vec![("call_move", "allow"), ("call_delete", "allow")]
    );
    let result = records.last().expect("result record");
    assert_eq!(result["type"], "result");
    assert_eq!(result["result"], "File moved and deleted.");
}

#[test]
fn print_yolo_mode_allows_edits() {
    let env = smoke_env();
    let provider = start_provider(vec![
        tool_call_sse_body_for(
            "call_write",
            "write_file",
            r#"{"file_path":"created.txt","content":"made it"}"#,
        ),
        text_sse_body("File written."),
    ]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &[
            "-p",
            "Write created.txt.",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "yolo",
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        output.status.success(),
        "yolo run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&output),
        stderr_str(&output)
    );
    // Agent-side bypassPermissions means the write happens without any
    // client-side permission round-trip.
    assert_eq!(
        std::fs::read_to_string(env.cwd.join("created.txt"))
            .ok()
            .as_deref(),
        Some("made it"),
        "yolo mode should allow the write"
    );
    let records = parse_stream_stdout(&output);
    let result = records.last().expect("result record");
    assert_eq!(result["type"], "result");
    assert_eq!(result["result"], "File written.");
}

#[test]
fn print_resume_continues_session() {
    let env = smoke_env();
    let cwd = env.cwd.display().to_string();

    let provider = start_provider(vec![text_sse_body("First answer.")]);
    let first = run_print(
        &env,
        &provider,
        &[
            "-p",
            "First question.",
            "--output-format",
            "json",
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        first.status.success(),
        "first run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&first),
        stderr_str(&first)
    );
    let first_payload = parse_json_stdout(&first);
    let session_id = first_payload["session_id"]
        .as_str()
        .expect("first run session id")
        .to_string();

    let provider = start_provider(vec![text_sse_body("Second answer.")]);
    let second = run_print(
        &env,
        &provider,
        &[
            "-p",
            "Follow-up question.",
            "--output-format",
            "json",
            "--resume",
            &session_id,
            "--cwd",
            &cwd,
        ],
        None,
    );
    assert!(
        second.status.success(),
        "resume run failed;\nstdout:\n{}\nstderr:\n{}",
        stdout_str(&second),
        stderr_str(&second)
    );
    let second_payload = parse_json_stdout(&second);
    assert_eq!(second_payload["session_id"], session_id.as_str());
    assert_eq!(second_payload["resumed"], true);
    assert_eq!(second_payload["result"], "Second answer.");
}

#[test]
fn print_reports_failure_with_nonzero_exit() {
    let env = smoke_env();
    // No canned bodies: the provider connection closes immediately, the LLM
    // call fails, and the turn ends with a turn failure.
    let provider = start_provider(vec![]);
    let cwd = env.cwd.display().to_string();
    let output = run_print(
        &env,
        &provider,
        &["-p", "Anything.", "--output-format", "json", "--cwd", &cwd],
        None,
    );
    assert!(
        !output.status.success(),
        "run against a dead provider must fail;\nstdout:\n{}",
        stdout_str(&output)
    );
    let payload = parse_json_stdout(&output);
    assert_eq!(payload["stop_reason"], "error");
    assert!(
        payload["error"].as_str().is_some_and(|e| !e.is_empty()),
        "error field must carry the reason: {payload}"
    );
}

#[test]
fn print_rejects_empty_prompt() {
    let env = smoke_env();
    let provider = start_provider(vec![]);
    let output = run_print(&env, &provider, &["-p", "   "], None);
    assert!(!output.status.success(), "empty prompt must fail");
    assert!(
        stderr_str(&output).contains("empty prompt"),
        "stderr should name the failure: {}",
        stderr_str(&output)
    );
}
