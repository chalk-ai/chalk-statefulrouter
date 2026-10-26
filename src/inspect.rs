//! Request inspection: parse the prompt out of inference request bodies.
//!
//! Recognized shapes, detected from the body (and the path, where shapes are
//! ambiguous):
//!
//! | API | prompt fields |
//! |---|---|
//! | OpenAI chat completions | `messages` |
//! | Anthropic messages | `system` + `messages` |
//! | OpenAI responses (`…/responses`) | `instructions` + `input` |
//! | OpenAI / vLLM completions | `prompt` (text or token ids), `prompt_token_ids` |
//! | SGLang `/generate` | `text` or `input_ids` |
//!
//! Message content is flattened to text: text parts verbatim, anything else
//! (images, tool calls, tool results) as canonical JSON, so content that
//! differs always flattens differently.

use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InspectedRequest {
    pub model: Option<String>,
    /// vLLM `cache_salt`: requests with different salts never share cache.
    pub cache_salt: Option<String>,
    pub prompt: Option<Prompt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat(Vec<Message>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl InspectedRequest {
    pub fn token_ids(&self) -> Option<&[u32]> {
        match &self.prompt {
            Some(Prompt::Tokens(t)) => Some(t),
            _ => None,
        }
    }
}

pub fn inspect(path: &str, body: &Value) -> InspectedRequest {
    let string = |field: &str| body.get(field).and_then(Value::as_str).map(str::to_string);
    InspectedRequest {
        model: string("model"),
        cache_salt: string("cache_salt"),
        prompt: prompt(path, body),
    }
}

fn prompt(path: &str, body: &Value) -> Option<Prompt> {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        // Anthropic puts the system prompt beside the messages, not in them.
        let system = body
            .get("system")
            .map(|s| message("system", flatten(s)))
            .into_iter();
        return Some(Prompt::Chat(
            system.chain(messages.iter().map(chat_message)).collect(),
        ));
    }
    // Embeddings also use `input`, so only read it on the responses API.
    if path.trim_end_matches('/').ends_with("/responses") {
        if let Some(input) = body.get("input") {
            let instructions = body
                .get("instructions")
                .map(|i| message("system", flatten(i)))
                .into_iter();
            let items: Vec<Message> = match input {
                Value::String(s) => vec![message("user", s.clone())],
                Value::Array(items) => items.iter().map(chat_message).collect(),
                _ => return None,
            };
            return Some(Prompt::Chat(instructions.chain(items).collect()));
        }
    }
    for field in ["prompt", "text", "input_ids", "prompt_token_ids"] {
        match body.get(field) {
            Some(Value::String(s)) => return Some(Prompt::Text(s.clone())),
            Some(v @ Value::Array(_)) => return token_ids(v).map(Prompt::Tokens),
            _ => {}
        }
    }
    None
}

fn message(role: &str, content: String) -> Message {
    Message {
        role: role.to_string(),
        content,
    }
}

/// A chat message, or a responses-API input item. Fields beyond role and
/// content (`name`, `tool_calls`, `tool_call_id`, …) are folded into the
/// content so they distinguish messages too.
fn chat_message(value: &Value) -> Message {
    let Some(obj) = value.as_object() else {
        return message("user", flatten(value));
    };
    let role = obj
        .get("role")
        .or_else(|| obj.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let mut content = obj.get("content").map(flatten).unwrap_or_default();
    let extras: serde_json::Map<String, Value> = obj
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "role" | "content"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if !extras.is_empty() {
        content.push_str(&Value::Object(extras).to_string());
    }
    message(role, content)
}

/// Content as text: strings verbatim, text parts concatenated, other parts as
/// canonical JSON (serde_json orders object keys).
fn flatten(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match text_part(part) {
                Some(text) => text.to_string(),
                None => part.to_string(),
            })
            .collect(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn text_part(part: &Value) -> Option<&str> {
    let kind = part.get("type")?.as_str()?;
    matches!(kind, "text" | "input_text" | "output_text")
        .then(|| part.get("text")?.as_str())
        .flatten()
}

fn token_ids(value: &Value) -> Option<Vec<u32>> {
    let items = value.as_array()?;
    if items.is_empty() {
        return None;
    }
    // Batched prompts (arrays of strings or of token arrays) have no single
    // prefix to route by, and fail here.
    items
        .iter()
        .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn chat(path: &str, body: Value) -> Vec<(String, String)> {
        match inspect(path, &body).prompt {
            Some(Prompt::Chat(m)) => m.into_iter().map(|m| (m.role, m.content)).collect(),
            other => panic!("expected chat, got {other:?}"),
        }
    }

    fn pair(role: &str, content: &str) -> (String, String) {
        (role.to_string(), content.to_string())
    }

    #[test]
    fn openai_chat() {
        let req = json!({
            "model": "qwen",
            "cache_salt": "tenant-1",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is "},
                    {"type": "text", "text": "this?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "1"}]}
            ]
        });
        let inspected = inspect("/v1/chat/completions", &req);
        assert_eq!(inspected.model.as_deref(), Some("qwen"));
        assert_eq!(inspected.cache_salt.as_deref(), Some("tenant-1"));
        assert_eq!(
            chat("/v1/chat/completions", req),
            vec![
                pair("system", "be brief"),
                pair(
                    "user",
                    r#"what is this?{"image_url":{"url":"data:..."},"type":"image_url"}"#
                ),
                pair("assistant", r#"{"tool_calls":[{"id":"1"}]}"#),
            ]
        );
    }

    #[test]
    fn anthropic_messages() {
        let req = json!({
            "system": [{"type": "text", "text": "sys"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(
            chat("/v1/messages", req),
            vec![pair("system", "sys"), pair("user", "hi")]
        );
    }

    #[test]
    fn responses_api() {
        let req = json!({
            "instructions": "sys",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call_output", "call_id": "c", "output": "42"}
            ]
        });
        assert_eq!(
            chat("/v1/responses", req),
            vec![
                pair("system", "sys"),
                pair("user", "hi"),
                pair(
                    "function_call_output",
                    r#"{"call_id":"c","output":"42","type":"function_call_output"}"#
                ),
            ]
        );
        assert_eq!(
            chat("/v1/responses", json!({"input": "hello"})),
            vec![pair("user", "hello")]
        );
        // Embeddings `input` is not a prompt.
        assert_eq!(
            inspect("/v1/embeddings", &json!({"input": "x"})).prompt,
            None
        );
    }

    #[test]
    fn completions_and_generate() {
        let p = |body: Value| inspect("/v1/completions", &body).prompt;
        assert_eq!(
            p(json!({"prompt": "abc"})),
            Some(Prompt::Text("abc".into()))
        );
        assert_eq!(
            p(json!({"prompt": [1, 2]})),
            Some(Prompt::Tokens(vec![1, 2]))
        );
        assert_eq!(p(json!({"prompt": ["a", "b"]})), None);
        assert_eq!(p(json!({"prompt": [[1], [2]]})), None);
        assert_eq!(p(json!({"text": "abc"})), Some(Prompt::Text("abc".into())));
        assert_eq!(p(json!({"input_ids": [4]})), Some(Prompt::Tokens(vec![4])));
        assert_eq!(
            p(json!({"prompt_token_ids": [5]})),
            Some(Prompt::Tokens(vec![5]))
        );
        assert_eq!(p(json!({"foo": 1})), None);
    }
}
