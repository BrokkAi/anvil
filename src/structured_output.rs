use anyhow::{Result, anyhow};
use jsonschema::JSONSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const ACP_META_NAMESPACE: &str = "anvil";
const ACP_META_STRUCTURED_OUTPUT_KEY: &str = "structuredOutput";
const MAX_INVALID_EXCERPT_CHARS: usize = 400;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputRequest {
    pub schema_name: String,
    pub schema: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputSchemaError {
    pub instance_location: String,
    pub schema_location: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputSuccess {
    pub schema_name: String,
    pub validated_output: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputValidationError {
    pub schema_name: String,
    pub errors: Vec<StructuredOutputSchemaError>,
    pub invalid_excerpt: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StructuredOutputResult {
    Success(StructuredOutputSuccess),
    ValidationError(StructuredOutputValidationError),
}

pub fn parse_structured_output_request(
    meta: Option<&Map<String, Value>>,
) -> Result<Option<StructuredOutputRequest>> {
    let Some(meta) = meta else {
        return Ok(None);
    };
    let Some(namespace) = meta.get(ACP_META_NAMESPACE) else {
        return Ok(None);
    };
    let namespace = namespace
        .as_object()
        .ok_or_else(|| anyhow!("`_meta.{ACP_META_NAMESPACE}` must be an object"))?;
    let Some(payload) = namespace.get(ACP_META_STRUCTURED_OUTPUT_KEY) else {
        return Ok(None);
    };
    let payload = payload.as_object().ok_or_else(|| {
        anyhow!("`_meta.{ACP_META_NAMESPACE}.structuredOutput` must be an object")
    })?;
    let schema_name = payload
        .get("schemaName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow!("`schemaName` must be a non-empty string"))?
        .to_string();
    let schema = payload
        .get("schema")
        .cloned()
        .ok_or_else(|| anyhow!("`schema` is required"))?;
    if !schema.is_object() {
        anyhow::bail!("`schema` must be a JSON object");
    }
    let schema_for_compile = schema.clone();
    JSONSchema::compile(&schema_for_compile)
        .map_err(|err| anyhow!("invalid structured-output schema: {err}"))?;
    Ok(Some(StructuredOutputRequest {
        schema_name,
        schema,
    }))
}

pub fn build_structured_output_meta(
    result: Option<&StructuredOutputResult>,
) -> Option<Map<String, Value>> {
    let result = result?;
    let payload = serde_json::to_value(result).expect("structured output result serializes");
    let mut namespace = Map::new();
    namespace.insert(ACP_META_STRUCTURED_OUTPUT_KEY.to_string(), payload);

    let mut meta = Map::new();
    meta.insert(ACP_META_NAMESPACE.to_string(), Value::Object(namespace));
    Some(meta)
}

pub fn validate_response(
    request: &StructuredOutputRequest,
    response_text: &str,
) -> StructuredOutputResult {
    let parsed = match serde_json::from_str::<Value>(response_text) {
        Ok(value) => value,
        Err(err) => {
            return StructuredOutputResult::ValidationError(StructuredOutputValidationError {
                schema_name: request.schema_name.clone(),
                errors: vec![StructuredOutputSchemaError {
                    instance_location: String::new(),
                    schema_location: String::new(),
                    message: format!("response is not valid JSON: {err}"),
                }],
                invalid_excerpt: truncate_excerpt(response_text),
            });
        }
    };

    let schema_for_compile = request.schema.clone();
    let compiled = match JSONSchema::compile(&schema_for_compile) {
        Ok(compiled) => compiled,
        Err(err) => {
            return StructuredOutputResult::ValidationError(StructuredOutputValidationError {
                schema_name: request.schema_name.clone(),
                errors: vec![StructuredOutputSchemaError {
                    instance_location: String::new(),
                    schema_location: String::new(),
                    message: format!("schema compilation failed: {err:#}"),
                }],
                invalid_excerpt: truncate_excerpt(response_text),
            });
        }
    };

    if compiled.is_valid(&parsed) {
        StructuredOutputResult::Success(StructuredOutputSuccess {
            schema_name: request.schema_name.clone(),
            validated_output: parsed,
        })
    } else {
        let errors = compiled
            .validate(&parsed)
            .expect_err("is_valid false must produce validation errors");
        StructuredOutputResult::ValidationError(StructuredOutputValidationError {
            schema_name: request.schema_name.clone(),
            errors: errors
                .map(|error| StructuredOutputSchemaError {
                    instance_location: error.instance_path.to_string(),
                    schema_location: error.schema_path.to_string(),
                    message: error.to_string(),
                })
                .collect(),
            invalid_excerpt: truncate_excerpt(response_text),
        })
    }
}

pub fn native_response_format(request: &StructuredOutputRequest) -> NativeResponseFormat {
    NativeResponseFormat {
        name: request.schema_name.clone(),
        schema: request.schema.clone(),
        strict: true,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeResponseFormat {
    pub name: String,
    pub schema: Value,
    pub strict: bool,
}

fn truncate_excerpt(raw: &str) -> String {
    let mut excerpt: String = raw.chars().take(MAX_INVALID_EXCERPT_CHARS).collect();
    if raw.chars().count() > MAX_INVALID_EXCERPT_CHARS {
        excerpt.push_str("...");
    }
    excerpt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
            "additionalProperties": false
        })
    }

    #[test]
    fn parses_valid_request_meta() {
        let meta = serde_json::json!({
            "anvil": {
                "structuredOutput": {
                    "schemaName": "audit_result",
                    "schema": sample_schema()
                }
            }
        });
        let parsed = parse_structured_output_request(meta.as_object()).unwrap();
        assert_eq!(
            parsed,
            Some(StructuredOutputRequest {
                schema_name: "audit_result".to_string(),
                schema: sample_schema(),
            })
        );
    }

    #[test]
    fn rejects_missing_schema_fields() {
        let meta = serde_json::json!({
            "anvil": {
                "structuredOutput": {
                    "schema": sample_schema()
                }
            }
        });
        let err = parse_structured_output_request(meta.as_object()).unwrap_err();
        assert!(err.to_string().contains("schemaName"));
    }

    #[test]
    fn rejects_malformed_namespace() {
        let meta = serde_json::json!({
            "anvil": "not-an-object"
        });
        let err = parse_structured_output_request(meta.as_object()).unwrap_err();
        assert!(err.to_string().contains("_meta.anvil"));
    }

    #[test]
    fn validates_successful_json_payload() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
        };
        let result = validate_response(&request, r#"{"answer":"ok"}"#);
        match result {
            StructuredOutputResult::Success(success) => {
                assert_eq!(success.schema_name, "audit_result");
                assert_eq!(success.validated_output["answer"], "ok");
            }
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[test]
    fn invalid_json_returns_structured_diagnostics() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
        };
        let result = validate_response(&request, r#"{"answer":"ok""#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert_eq!(error.schema_name, "audit_result");
                assert_eq!(error.errors.len(), 1);
                assert!(error.errors[0].message.contains("not valid JSON"));
                assert!(error.invalid_excerpt.contains("\"answer\""));
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn schema_mismatch_returns_machine_readable_errors() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
        };
        let result = validate_response(&request, r#"{"answer":12}"#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert_eq!(error.schema_name, "audit_result");
                assert!(!error.errors.is_empty());
                assert!(error.errors.iter().any(|entry| !entry.message.is_empty()));
                assert_eq!(error.invalid_excerpt, r#"{"answer":12}"#);
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }
}
