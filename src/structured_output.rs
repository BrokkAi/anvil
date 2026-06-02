use anyhow::{Result, anyhow};
use jsonschema::{ErrorIterator, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const ACP_META_NAMESPACE: &str = "anvil";
const ACP_META_STRUCTURED_OUTPUT_KEY: &str = "structuredOutput";
const MAX_INVALID_EXCERPT_CHARS: usize = 400;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputRequest {
    pub schema_name: String,
    pub schema: Value,
    pub allow_coercion: bool,
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
    pub coercion_requested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputCoercedSuccess {
    pub schema_name: String,
    pub validated_output: Value,
    pub coercions: Vec<String>,
    pub coercion_requested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputValidationError {
    pub schema_name: String,
    pub errors: Vec<StructuredOutputSchemaError>,
    pub invalid_excerpt: String,
    pub coercion_requested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StructuredOutputResult {
    Success(StructuredOutputSuccess),
    CoercedSuccess(StructuredOutputCoercedSuccess),
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
    let allow_coercion = match payload.get("allowCoercion") {
        Some(Value::Bool(value)) => *value,
        Some(_) => anyhow::bail!("`allowCoercion` must be a boolean"),
        None => false,
    };
    if !schema.is_object() {
        anyhow::bail!("`schema` must be a JSON object");
    }
    let schema_for_compile = schema.clone();
    JSONSchema::compile(&schema_for_compile)
        .map_err(|err| anyhow!("invalid structured-output schema: {err}"))?;
    Ok(Some(StructuredOutputRequest {
        schema_name,
        schema,
        allow_coercion,
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
            return validation_error_result(
                request,
                vec![StructuredOutputSchemaError {
                    instance_location: String::new(),
                    schema_location: String::new(),
                    message: format!("response is not valid JSON: {err}"),
                }],
                response_text,
            );
        }
    };

    let schema_for_compile = request.schema.clone();
    let compiled = match JSONSchema::compile(&schema_for_compile) {
        Ok(compiled) => compiled,
        Err(err) => {
            return validation_error_result(
                request,
                vec![StructuredOutputSchemaError {
                    instance_location: String::new(),
                    schema_location: String::new(),
                    message: format!("schema compilation failed: {err:#}"),
                }],
                response_text,
            );
        }
    };

    if compiled.is_valid(&parsed) {
        StructuredOutputResult::Success(StructuredOutputSuccess {
            schema_name: request.schema_name.clone(),
            validated_output: parsed,
            coercion_requested: request.allow_coercion,
        })
    } else {
        let errors = compiled
            .validate(&parsed)
            .expect_err("is_valid false must produce validation errors");
        let original_errors = collect_schema_errors(errors);
        if !request.allow_coercion {
            return validation_error_result(request, original_errors, response_text);
        }

        let mut coercions = Vec::new();
        let coerced = coerce_output_node(&parsed, &request.schema, "response", &mut coercions);
        if coercions.is_empty() {
            return validation_error_result(request, original_errors, response_text);
        }

        if compiled.is_valid(&coerced) {
            StructuredOutputResult::CoercedSuccess(StructuredOutputCoercedSuccess {
                schema_name: request.schema_name.clone(),
                validated_output: coerced,
                coercions,
                coercion_requested: true,
            })
        } else {
            let errors = compiled
                .validate(&coerced)
                .expect_err("is_valid false must produce validation errors");
            validation_error_result(request, collect_schema_errors(errors), response_text)
        }
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

fn validation_error_result(
    request: &StructuredOutputRequest,
    errors: Vec<StructuredOutputSchemaError>,
    response_text: &str,
) -> StructuredOutputResult {
    StructuredOutputResult::ValidationError(StructuredOutputValidationError {
        schema_name: request.schema_name.clone(),
        errors,
        invalid_excerpt: truncate_excerpt(response_text),
        coercion_requested: request.allow_coercion,
    })
}

fn collect_schema_errors(errors: ErrorIterator<'_>) -> Vec<StructuredOutputSchemaError> {
    errors
        .map(|error| StructuredOutputSchemaError {
            instance_location: error.instance_path.to_string(),
            schema_location: error.schema_path.to_string(),
            message: error.to_string(),
        })
        .collect()
}

fn coerce_output_node(
    value: &Value,
    schema: &Value,
    path: &str,
    changes: &mut Vec<String>,
) -> Value {
    match schema_type(schema) {
        Some("object") => coerce_output_object(value, schema, path, changes),
        Some("array") => coerce_output_array(value, schema, path, changes),
        Some("string") => coerce_output_string(value, schema, path, changes),
        Some("integer") => coerce_output_integer(value, path, changes),
        Some("number") => coerce_output_number(value, path, changes),
        Some("boolean") => coerce_output_boolean(value, path, changes),
        _ => value.clone(),
    }
}

fn coerce_output_object(
    value: &Value,
    schema: &Value,
    path: &str,
    changes: &mut Vec<String>,
) -> Value {
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return value.clone();
    };

    let mut out = object.clone();
    for (key, property_schema) in properties {
        if let Some(child) = out.get(key).cloned() {
            let child_path = format!("{path}.{key}");
            out.insert(
                key.clone(),
                coerce_output_node(&child, property_schema, &child_path, changes),
            );
        }
    }
    Value::Object(out)
}

fn coerce_output_array(
    value: &Value,
    schema: &Value,
    path: &str,
    changes: &mut Vec<String>,
) -> Value {
    let Some(array) = value.as_array() else {
        return value.clone();
    };
    let Some(items) = schema.get("items") else {
        return value.clone();
    };

    Value::Array(
        array
            .iter()
            .enumerate()
            .map(|(index, element)| {
                coerce_output_node(element, items, &format!("{path}[{index}]"), changes)
            })
            .collect(),
    )
}

fn coerce_output_string(
    value: &Value,
    schema: &Value,
    path: &str,
    changes: &mut Vec<String>,
) -> Value {
    if schema.get("enum").is_some() || value.is_string() || value.is_null() {
        return value.clone();
    }

    if let Some(array) = value.as_array() {
        if let Some(joined) = array_to_string(array) {
            changes.push(format!("{path} array -> string"));
            return Value::String(joined);
        }
    }

    if value.is_number() || value.is_boolean() {
        changes.push(format!("{path} {} -> string", output_type_of(value)));
        return Value::String(match value {
            Value::Bool(boolean) => boolean.to_string(),
            _ => value.to_string(),
        });
    }

    value.clone()
}

fn coerce_output_integer(value: &Value, path: &str, changes: &mut Vec<String>) -> Value {
    let Some(text) = value.as_str() else {
        return value.clone();
    };

    if let Ok(parsed) = text.parse::<i64>() {
        changes.push(format!("{path} string -> integer"));
        return Value::Number(parsed.into());
    }

    if let Ok(parsed) = text.parse::<u64>() {
        changes.push(format!("{path} string -> integer"));
        return Value::Number(parsed.into());
    }

    value.clone()
}

fn coerce_output_number(value: &Value, path: &str, changes: &mut Vec<String>) -> Value {
    let Some(text) = value.as_str() else {
        return value.clone();
    };

    let Ok(parsed) = text.parse::<f64>() else {
        return value.clone();
    };
    let Some(number) = serde_json::Number::from_f64(parsed) else {
        return value.clone();
    };

    changes.push(format!("{path} string -> number"));
    Value::Number(number)
}

fn coerce_output_boolean(value: &Value, path: &str, changes: &mut Vec<String>) -> Value {
    let Some(text) = value.as_str() else {
        return value.clone();
    };

    let parsed = match text {
        "true" => true,
        "false" => false,
        _ => return value.clone(),
    };

    changes.push(format!("{path} string -> boolean"));
    Value::Bool(parsed)
}

fn array_to_string(array: &[Value]) -> Option<String> {
    array
        .iter()
        .map(node_to_string)
        .collect::<Option<Vec<_>>>()
        .map(|lines| lines.join("\n"))
}

fn node_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(boolean) => Some(boolean.to_string()),
        _ => None,
    }
}

fn schema_type(schema: &Value) -> Option<&str> {
    schema.get("type").and_then(Value::as_str)
}

fn output_type_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
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
                    "schema": sample_schema(),
                    "allowCoercion": true
                }
            }
        });
        let parsed = parse_structured_output_request(meta.as_object()).unwrap();
        assert_eq!(
            parsed,
            Some(StructuredOutputRequest {
                schema_name: "audit_result".to_string(),
                schema: sample_schema(),
                allow_coercion: true,
            })
        );
    }

    #[test]
    fn parses_missing_allow_coercion_as_false() {
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
                allow_coercion: false,
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
    fn rejects_non_boolean_allow_coercion() {
        let meta = serde_json::json!({
            "anvil": {
                "structuredOutput": {
                    "schemaName": "audit_result",
                    "schema": sample_schema(),
                    "allowCoercion": "yes"
                }
            }
        });
        let err = parse_structured_output_request(meta.as_object()).unwrap_err();
        assert!(err.to_string().contains("allowCoercion"));
    }

    #[test]
    fn validates_successful_json_payload() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: false,
        };
        let result = validate_response(&request, r#"{"answer":"ok"}"#);
        match result {
            StructuredOutputResult::Success(success) => {
                assert_eq!(success.schema_name, "audit_result");
                assert_eq!(success.validated_output["answer"], "ok");
                assert!(!success.coercion_requested);
            }
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[test]
    fn invalid_json_returns_structured_diagnostics() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: false,
        };
        let result = validate_response(&request, r#"{"answer":"ok""#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert_eq!(error.schema_name, "audit_result");
                assert_eq!(error.errors.len(), 1);
                assert!(error.errors[0].message.contains("not valid JSON"));
                assert!(error.invalid_excerpt.contains("\"answer\""));
                assert!(!error.coercion_requested);
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn schema_mismatch_returns_machine_readable_errors() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: false,
        };
        let result = validate_response(&request, r#"{"answer":12}"#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert_eq!(error.schema_name, "audit_result");
                assert!(!error.errors.is_empty());
                assert!(error.errors.iter().any(|entry| !entry.message.is_empty()));
                assert_eq!(error.invalid_excerpt, r#"{"answer":12}"#);
                assert!(!error.coercion_requested);
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn coercion_disabled_leaves_validation_failure_unchanged() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: false,
        };
        let result = validate_response(&request, r#"{"answer":["one","two"]}"#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert!(error.errors.iter().any(|entry| !entry.message.is_empty()));
                assert_eq!(error.invalid_excerpt, r#"{"answer":["one","two"]}"#);
                assert!(!error.coercion_requested);
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn coerces_array_to_string_for_non_enum_string_field() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"answer":["one",true]}"#);
        match result {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.validated_output["answer"], "one\ntrue");
                assert_eq!(success.coercions, vec!["response.answer array -> string"]);
                assert!(success.coercion_requested);
            }
            other => panic!("expected coerced success, got {other:?}"),
        }
    }

    #[test]
    fn coerces_scalar_to_string_for_non_enum_string_field() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"answer":12}"#);
        match result {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.validated_output["answer"], "12");
                assert_eq!(success.coercions, vec!["response.answer integer -> string"]);
                assert!(success.coercion_requested);
            }
            other => panic!("expected coerced success, got {other:?}"),
        }
    }

    #[test]
    fn does_not_coerce_enum_string_fields() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string", "enum": ["low", "high"] }
                },
                "required": ["answer"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"answer":["high"]}"#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert!(error.coercion_requested);
                assert_eq!(error.invalid_excerpt, r#"{"answer":["high"]}"#);
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn does_not_coerce_object_or_null_or_invalid_array_members() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: sample_schema(),
            allow_coercion: true,
        };

        for raw in [
            r#"{"answer":{"text":"one"}}"#,
            r#"{"answer":null}"#,
            r#"{"answer":[{"text":"one"}]}"#,
            r#"{"answer":[null]}"#,
        ] {
            let result = validate_response(&request, raw);
            match result {
                StructuredOutputResult::ValidationError(error) => {
                    assert!(error.coercion_requested);
                    assert_eq!(error.invalid_excerpt, raw);
                }
                other => panic!("expected validation error, got {other:?}"),
            }
        }
    }

    #[test]
    fn coercion_revalidates_and_falls_back_to_validation_error() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" },
                    "count": { "type": "integer" }
                },
                "required": ["answer", "count"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"answer":7,"count":"bad"}"#);
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert!(error.coercion_requested);
                assert!(
                    error
                        .errors
                        .iter()
                        .any(|entry| entry.message.contains("integer"))
                );
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    #[test]
    fn coerces_string_to_integer_for_integer_field() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "score": { "type": "integer" }
                },
                "required": ["score"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"score":"1"}"#);
        match result {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.validated_output["score"], 1);
                assert_eq!(success.coercions, vec!["response.score string -> integer"]);
            }
            other => panic!("expected coerced success, got {other:?}"),
        }
    }

    #[test]
    fn coerces_string_to_number_for_number_field() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "score": { "type": "number" }
                },
                "required": ["score"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"score":"1.5"}"#);
        match result {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.validated_output["score"], 1.5);
                assert_eq!(success.coercions, vec!["response.score string -> number"]);
            }
            other => panic!("expected coerced success, got {other:?}"),
        }
    }

    #[test]
    fn coerces_string_to_boolean_for_boolean_field() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "enabled": { "type": "boolean" }
                },
                "required": ["enabled"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(&request, r#"{"enabled":"true"}"#);
        match result {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.validated_output["enabled"], true);
                assert_eq!(
                    success.coercions,
                    vec!["response.enabled string -> boolean"]
                );
            }
            other => panic!("expected coerced success, got {other:?}"),
        }
    }

    #[test]
    fn invalid_scalar_strings_do_not_coerce() {
        let request = StructuredOutputRequest {
            schema_name: "audit_result".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "score": { "type": "integer" },
                    "ratio": { "type": "number" },
                    "enabled": { "type": "boolean" }
                },
                "required": ["score", "ratio", "enabled"],
                "additionalProperties": false
            }),
            allow_coercion: true,
        };
        let result = validate_response(
            &request,
            r#"{"score":"1.5","ratio":"abc","enabled":"TRUE"}"#,
        );
        match result {
            StructuredOutputResult::ValidationError(error) => {
                assert!(error.coercion_requested);
                assert_eq!(
                    error.invalid_excerpt,
                    r#"{"score":"1.5","ratio":"abc","enabled":"TRUE"}"#
                );
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }
}
