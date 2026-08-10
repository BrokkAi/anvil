use std::io::Write;
use std::time::Duration;
const TRACE_JSONL_ENV: &str = "ANVIL_TRACE_JSONL";

tokio::task_local! {
    /// Trace destination for the current task, overriding the env var. Set by
    /// `with_trace_path` so a test can read its own records back without
    /// touching the process-global environment.
    static TRACE_PATH: String;
}

/// The one `tool_timing` shape every model-visible tool call must leave behind.
/// Trace consumers count calls per tool by `tool` and sum `duration_ms`, so a
/// dispatch path that skips this record makes its tool look unused.
pub(crate) fn tool_timing_record(
    tool_name: &str,
    shell_command: Option<&str>,
    duration: Duration,
    success: bool,
) -> serde_json::Value {
    let mut record = serde_json::Map::new();
    record.insert("type".to_string(), serde_json::json!("tool_timing"));
    record.insert("tool".to_string(), serde_json::json!(tool_name));
    if let Some(command) = shell_command {
        record.insert("command".to_string(), serde_json::json!(command));
    }
    record.insert(
        "duration_ms".to_string(),
        serde_json::json!(duration.as_millis()),
    );
    record.insert("success".to_string(), serde_json::json!(success));
    serde_json::Value::Object(record)
}

/// Route the records written by `future` to `path` for this task only.
#[cfg(test)]
pub(crate) async fn with_trace_path<F: std::future::Future>(
    path: &std::path::Path,
    future: F,
) -> F::Output {
    TRACE_PATH
        .scope(path.to_string_lossy().into_owned(), future)
        .await
}

pub fn append_trace_record(record: serde_json::Value) {
    let path = match TRACE_PATH.try_with(Clone::clone) {
        Ok(path) => path,
        Err(_) => match std::env::var(TRACE_JSONL_ENV) {
            Ok(path) => path,
            Err(_) => return,
        },
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }

    let mut record = record;
    if let Some(obj) = record.as_object_mut() {
        obj.insert(
            "timestamp".to_string(),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );
    }

    let line = match serde_json::to_string(&record) {
        Ok(line) => line,
        Err(e) => {
            tracing::warn!("failed to serialize LLM trace record: {e:#}");
            return;
        }
    };

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{line}")?;
        Ok(())
    })();

    if let Err(e) = result {
        tracing::warn!("failed to append LLM trace record to {path}: {e:#}");
    }
}
