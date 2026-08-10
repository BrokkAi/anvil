use std::io::Write;
use std::sync::Mutex;
use std::time::Duration;
const TRACE_JSONL_ENV: &str = "ANVIL_TRACE_JSONL";
static TRACE_WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone)]
struct TraceContext {
    path: String,
    run_id: Option<String>,
}

tokio::task_local! {
    static TRACE_CONTEXT: TraceContext;
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

/// Route the records written by `future` to `path` for this task only, so a
/// test can read them back without setting the process-global trace env var.
#[cfg(test)]
pub(crate) async fn with_trace_path<F: std::future::Future>(
    path: &std::path::Path,
    future: F,
) -> F::Output {
    TRACE_CONTEXT
        .scope(
            TraceContext {
                path: path.to_string_lossy().into_owned(),
                run_id: None,
            },
            future,
        )
        .await
}

pub fn append_trace_record(record: serde_json::Value) {
    let context = TRACE_CONTEXT.try_with(Clone::clone).ok().or_else(|| {
        let path = std::env::var(TRACE_JSONL_ENV).ok()?;
        let path = path.trim().to_string();
        (!path.is_empty()).then_some(TraceContext { path, run_id: None })
    });
    let Some(context) = context else { return };

    let mut record = record;
    if let Some(obj) = record.as_object_mut() {
        if let Some(run_id) = context.run_id {
            obj.insert(
                "trace_run_id".to_string(),
                serde_json::Value::String(run_id),
            );
        }
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
        let _guard = TRACE_WRITE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&context.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    })();

    if let Err(e) = result {
        tracing::warn!(
            "failed to append LLM trace record to {}: {e:#}",
            context.path
        );
    }
}
