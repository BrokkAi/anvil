//! Thin async Python boundary. Provider behavior lives in anvil-client.
use std::{sync::Arc, time::Duration};

use anvil_client::infer::{HostedClient, InferOptions, StructuredInferRequest};
use anvil_client::llm_client::IdleTimeouts;
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

create_exception!(_native, NativeError, PyException);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: String,
    request: StructuredInferRequest,
    reasoning_effort: Option<String>,
    service_tier: Option<String>,
    timeout: f64,
    idle_timeout: f64,
    stall_timeout: f64,
    validation_retries: usize,
}

#[pyclass(module = "anvil_client._native")]
struct NativeClient {
    client: Arc<HostedClient>,
    cancel: CancellationToken,
}

#[pymethods]
impl NativeClient {
    #[new]
    fn new() -> Self {
        Self {
            client: Arc::new(HostedClient::default()),
            cancel: CancellationToken::new(),
        }
    }

    fn close(&self) {
        self.cancel.cancel();
    }

    fn infer<'py>(&self, py: Python<'py>, raw: &str) -> PyResult<Bound<'py, PyAny>> {
        let request: Request = serde_json::from_str(raw)
            .map_err(|e| NativeError::new_err(("InvalidRequest", e.to_string())))?;
        let timeout = duration(request.timeout)?;
        let idle = duration(request.idle_timeout)?;
        let stall = duration(request.stall_timeout)?;
        let client = self.client.clone();
        let cancel = self.cancel.child_token();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Python Future cancellation drops this guard and cancels provider work.
            let _guard = cancel.clone().drop_guard();
            let result = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(NativeError::new_err(("Cancelled", "client closed"))),
                result = tokio::time::timeout(timeout, client.infer(
                    &request.model, request.request,
                    InferOptions {
                        reasoning_effort: request.reasoning_effort,
                        service_tier: request.service_tier,
                        idle_timeouts: IdleTimeouts { first_progress: idle, inter_chunk: stall },
                        validation_retries: request.validation_retries,
                    }, cancel.clone(),
                )) => result.map_err(|_| NativeError::new_err(("Timeout", "inference deadline exceeded")))?,
            }.map_err(|e| NativeError::new_err((format!("{:?}", e.kind()), e.to_string())))?;
            serde_json::to_string(&result)
                .map_err(|e| NativeError::new_err(("Provider", e.to_string())))
        })
    }
}

fn duration(value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| {
            NativeError::new_err(("InvalidRequest", "timeouts must be finite and positive"))
        })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeClient>()?;
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
