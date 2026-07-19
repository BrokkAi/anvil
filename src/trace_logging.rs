use std::io::Write;
use std::sync::Mutex;
const TRACE_JSONL_ENV: &str = "ANVIL_TRACE_JSONL";
static TRACE_WRITE_LOCK: Mutex<()> = Mutex::new(());

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
        let _guard = TRACE_WRITE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
