//! Request and response types for ChatGPT-subscription audio transcription.
//!
//! [`CodexClient`](crate::codex_client::CodexClient) sends these requests to
//! ChatGPT's subscription-backed transcription endpoint. The audio bytes are
//! retained as [`bytes::Bytes`] so a retry can cheaply rebuild the multipart
//! form without copying the caller's input up front.

use std::time::Duration;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Model name reported by Codex for subscription-backed transcription.
pub const DEFAULT_CODEX_TRANSCRIBE_MODEL: &str = "gpt-4o-mini-transcribe";

/// A transcription request sent through a [`CodexClient`](crate::codex_client::CodexClient).
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    /// Complete audio file contents. The backend inspects the container bytes.
    pub audio: Bytes,
    /// Filename supplied in the multipart `file` part (for example,
    /// `audio.wav`).
    pub file_name: String,
    /// MIME type supplied in the multipart `file` part (for example,
    /// `audio/wav`).
    pub mime_type: String,
    /// Optional model override. `None` intentionally omits the multipart part,
    /// matching the request emitted by Codex Desktop.
    pub model: Option<String>,
    /// Optional ISO-639-1 language hint.
    pub language: Option<String>,
    /// Optional transcription prompt.
    pub prompt: Option<String>,
    /// Optional sampling temperature.
    pub temperature: Option<f32>,
    /// Cancels the request, including while waiting for its response body.
    pub cancel: CancellationToken,
    /// Whole-request deadline, including retries and response-body reading.
    pub timeout: Duration,
}

impl TranscribeRequest {
    /// Construct a request with no optional form parts, a fresh cancellation
    /// token, and a two-minute whole-request timeout.
    pub fn new(audio: Bytes, file_name: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            audio,
            file_name: file_name.into(),
            mime_type: mime_type.into(),
            model: None,
            language: None,
            prompt: None,
            temperature: None,
            cancel: CancellationToken::new(),
            timeout: Duration::from_secs(120),
        }
    }
}

/// Successful response from ChatGPT's transcription endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    pub text: String,
}
