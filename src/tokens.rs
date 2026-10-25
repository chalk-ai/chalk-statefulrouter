//! Extracting prompt token ids from a request, for KV-aware scoring.
//!
//! The router can only match a request against the KV index if it knows the
//! request's tokens exactly as the engine will see them. Requests that already
//! carry token ids give that for free; text prompts need the model's tokenizer
//! (and, for chat, its chat template) — plug one in via [`TokenSource`].

use serde_json::Value;

pub trait TokenSource: Send + Sync + 'static {
    /// Token ids of the request's prompt, if they can be determined.
    fn tokens(&self, path: &str, body: &Value) -> Option<Vec<u32>>;
}

/// Reads token ids requests carry themselves:
/// - OpenAI `/v1/completions` with `prompt` as an array of token ids
/// - SGLang `/generate` with `input_ids`
/// - vLLM `prompt_token_ids`
pub struct RequestTokenIds;

impl TokenSource for RequestTokenIds {
    fn tokens(&self, _path: &str, body: &Value) -> Option<Vec<u32>> {
        ["prompt", "input_ids", "prompt_token_ids"]
            .iter()
            .find_map(|field| token_array(body.get(field)?))
    }
}

fn token_array(value: &Value) -> Option<Vec<u32>> {
    let items = value.as_array()?;
    if items.is_empty() {
        return None;
    }
    items
        .iter()
        .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn extracts_token_prompts() {
        let t = RequestTokenIds;
        assert_eq!(
            t.tokens("/v1/completions", &json!({"prompt": [1, 2, 3]})),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            t.tokens("/generate", &json!({"input_ids": [4]})),
            Some(vec![4])
        );
        assert_eq!(
            t.tokens("/v1/completions", &json!({"prompt": "hello"})),
            None
        );
        // Batched prompts aren't routed by prefix.
        assert_eq!(
            t.tokens("/v1/completions", &json!({"prompt": [[1], [2]]})),
            None
        );
        assert_eq!(
            t.tokens("/v1/chat/completions", &json!({"messages": []})),
            None
        );
    }
}
