//! Prompt token ids for KV-aware scoring.
//!
//! The router can only match a request against the KV index if it knows the
//! request's tokens exactly as the engine will see them. Requests that already
//! carry token ids give that for free; text prompts need the model's tokenizer
//! (and, for chat, its chat template) — plug one in via [`TokenSource`].

use crate::inspect::InspectedRequest;

pub trait TokenSource: Send + Sync + 'static {
    /// Token ids of the request's prompt, if they can be determined.
    fn tokens(&self, request: &InspectedRequest) -> Option<Vec<u32>>;
}

/// Uses token ids requests carry themselves (`prompt` as token ids,
/// `input_ids`, `prompt_token_ids`).
pub struct RequestTokenIds;

impl TokenSource for RequestTokenIds {
    fn tokens(&self, request: &InspectedRequest) -> Option<Vec<u32>> {
        request.token_ids().map(<[u32]>::to_vec)
    }
}
