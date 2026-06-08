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
    duplicate_global_skills: bool,
    read_only: bool,
    structured_output: bool,
    tool_follow_up: bool,
    prompt: String,
}

#[test]
fn slopcop_shaped_acp_path_does_not_abort() {
    let cases = [
        SmokeCase {
            name: "plain_default_no_skills",
            duplicate_global_skills: false,
            read_only: false,
            structured_output: false,
            tool_follow_up: false,
            prompt: "List the files you would inspect first. Do not call tools.".to_string(),
        },
        SmokeCase {
            name: "structured_readonly_duplicate_global_skills",
            duplicate_global_skills: true,
            read_only: true,
            structured_output: true,
            tool_follow_up: false,
            prompt: slopcop_sized_prompt(),
        },
        SmokeCase {
            name: "structured_readonly_tool_followup_duplicate_global_skills",
            duplicate_global_skills: true,
            read_only: true,
            structured_output: true,
            tool_follow_up: true,
            prompt: slopcop_sized_prompt(),
        },
    ];

    for case in cases {
        run_smoke_case(&case);
    }
}

fn run_smoke_case(case: &SmokeCase) {
    let temp = tempfile::tempdir().expect("tempdir");
    let cwd = temp.path().join("repo");
    std::fs::create_dir_all(&cwd).expect("create cwd");
    std::fs::write(cwd.join("README.md"), "# smoke\n").expect("write readme");
    std::fs::create_dir_all(cwd.join(".git")).expect("create git marker");

    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).expect("create home");
    if case.duplicate_global_skills {
        write_skill(
            &home.join(".claude/skills/use-railway/SKILL.md"),
            "use-railway",
            "Shadowed lower-priority test skill",
        );
        write_skill(
            &home.join(".agents/skills/use-railway/SKILL.md"),
            "use-railway",
            "Winning higher-priority test skill",
        );
    }

    let config_home = temp.path().join("config");
    std::fs::create_dir_all(&config_home).expect("create config home");
    std::fs::write(config_home.join("setup.json"), r#"{"mcp_servers":[]}"#).expect("write setup");

    let trace_path = temp.path().join(format!("{}.trace.jsonl", case.name));
    let provider = case.tool_follow_up.then(start_openai_smoke_server);
    let mut child = spawn_anvil(
        &home,
        &config_home,
        &trace_path,
        provider.as_ref().map(|server| server.base_url.as_str()),
        if case.tool_follow_up { 2 } else { 1 },
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

    if case.read_only {
        let config = client.request(
            "session/set_config_option",
            json!({
                "sessionId": session_id,
                "configId": "permission_mode",
                "value": "readOnly"
            }),
        );
        assert_response_ok(case, "session/set_config_option", &config, &client);
    }

    let mut prompt_params = json!({
        "sessionId": session_id,
        "prompt": [
            {
                "type": "text",
                "text": case.prompt
            }
        ]
    });
    if case.structured_output {
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
    }

    let prompt = client.request("session/prompt", prompt_params);
    assert_response_ok(case, "session/prompt", &prompt, &client);
    assert!(
        !client.exited(),
        "{}: anvil exited after prompt; stderr:\n{}\ntrace:\n{}",
        case.name,
        client.stderr_text(),
        client.trace_text()
    );
    let trace = client.trace_text();
    for checkpoint in [
        "acp_initialize_end",
        "acp_session_new_end",
        "session_context_discovery_end",
        "skills_discovery_end",
        "subagents_discovery_end",
        "mcp_server_resolution_end",
        "tool_registry_construction_end",
        "structured_output_metadata_parse_end",
        "prompt_messages_construction_end",
        "token_counting_end",
        "final_llm_request_dispatch_begin",
        "final_llm_request_dispatch_end",
    ] {
        assert!(
            trace.contains(&format!(r#""checkpoint":"{checkpoint}""#)),
            "{}: trace missing checkpoint {checkpoint}\ntrace:\n{}\nstderr:\n{}",
            case.name,
            trace,
            client.stderr_text()
        );
    }
    if case.read_only {
        assert!(
            trace.contains(r#""checkpoint":"acp_set_config_option_end""#),
            "{}: trace missing readOnly config checkpoint\ntrace:\n{}\nstderr:\n{}",
            case.name,
            trace,
            client.stderr_text()
        );
    }

    if case.tool_follow_up {
        let trace = client.trace_text();
        for checkpoint in [
            "turn_provider_dispatch_begin",
            "turn_provider_dispatch_end",
            "provider_stream_dispatch_begin",
            "provider_stream_request_body_ready",
            "provider_http_send_begin",
            "provider_http_send_end",
            "provider_sse_stream_begin",
            "provider_sse_tool_delta",
            "provider_sse_done_tool_calls",
            "provider_sse_text_delta",
            "provider_sse_done_text",
        ] {
            assert!(
                trace.contains(&format!(r#""checkpoint":"{checkpoint}""#)),
                "{}: trace missing provider/tool-follow-up checkpoint {checkpoint}\ntrace:\n{}\nstderr:\n{}",
                case.name,
                trace,
                client.stderr_text()
            );
        }
        assert!(
            trace_has_event_for_turn(&trace, "llm_request", 1),
            "{}: trace missing turn-1 llm_request after tool result\ntrace:\n{}\nstderr:\n{}",
            case.name,
            trace,
            client.stderr_text()
        );
        assert!(
            trace_has_event_for_turn(&trace, "llm_response", 1),
            "{}: trace missing turn-1 llm_response after tool result\ntrace:\n{}\nstderr:\n{}",
            case.name,
            trace,
            client.stderr_text()
        );
        let provider = provider.as_ref().expect("provider server");
        assert_eq!(
            provider.request_count(),
            2,
            "{}: expected provider to receive turn 0 and turn 1 requests",
            case.name
        );
        assert!(
            provider
                .request_bodies()
                .get(1)
                .is_some_and(|body| body.contains(r#""role":"tool""#)
                    && body.contains("README.md")),
            "{}: turn-1 provider request did not include the tool result body; requests: {:?}",
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
        .env("RUST_MIN_STACK", "33554432")
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

fn start_openai_smoke_server() -> OpenAiSmokeServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind smoke provider");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let request_bodies = Arc::new(Mutex::new(Vec::new()));
    let bodies_for_thread = request_bodies.clone();
    std::thread::spawn(move || {
        for (idx, stream) in listener.incoming().enumerate() {
            let Ok(stream) = stream else {
                break;
            };
            handle_provider_connection(stream, idx, &bodies_for_thread);
            if idx >= 1 {
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
    request_index: usize,
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

    let response_body = if request_index == 0 {
        tool_call_sse_body()
    } else {
        text_sse_body()
    };
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
    let args = serde_json::to_string(r#"{"file_path":"README.md"}"#).expect("encode args");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_readme\",\"function\":{{\"name\":\"read_file\",\"arguments\":{args}}}}}]}}}}]}}\n\
         \n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":4}}}}\n\
         \n\
         data: [DONE]\n\n"
    )
}

fn text_sse_body() -> String {
    "data: {\"choices\":[{\"delta\":{\"content\":\"README inspected.\"}}]}\n\
     \n\
     data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":5}}\n\
     \n\
     data: [DONE]\n\n"
        .to_string()
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
        }
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
        let deadline = Instant::now() + Duration::from_secs(20);
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
                    if value.get("id").is_some() && value.get("method").is_some() {
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

fn write_skill(path: &Path, name: &str, description: &str) {
    std::fs::create_dir_all(path.parent().expect("skill parent")).expect("create skill parent");
    std::fs::write(
        path,
        format!("---\nname: {name}\ndescription: {description}\n---\n\nBody"),
    )
    .expect("write skill");
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
