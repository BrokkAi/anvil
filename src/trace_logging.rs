use std::io::Write;
use std::sync::OnceLock;

const TRACE_JSONL_ENV: &str = "ANVIL_TRACE_JSONL";

pub fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(TRACE_JSONL_ENV)
            .map(|path| !path.trim().is_empty())
            .unwrap_or(false)
    })
}

pub fn append_trace_record(record: serde_json::Value) {
    let Ok(path) = std::env::var(TRACE_JSONL_ENV) else {
        return;
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

pub fn trace_checkpoint(name: &str, fields: serde_json::Value) {
    if !trace_enabled() {
        return;
    }
    append_trace_record(serde_json::json!({
        "type": "checkpoint",
        "checkpoint": name,
        "fields": fields,
    }));
}

#[macro_export]
macro_rules! trace_checkpoint {
    ($name:expr, $fields:expr $(,)?) => {{
        if $crate::trace_logging::trace_enabled() {
            $crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "checkpoint",
                "checkpoint": $name,
                "fields": $fields,
            }));
        }
    }};
}
