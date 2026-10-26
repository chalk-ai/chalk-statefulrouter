//! Working out a request's session key.
//!
//! In order:
//! 1. The sticky-session header (`x-chalk-sticky-session-id`), if present.
//! 2. Each configured [`DeriveRule`], first match wins: another header, a body
//!    field (e.g. OpenAI's `prompt_cache_key`), or a hash of a prefix of the
//!    prompt.
//! 3. A freshly minted id, if `generate_if_missing`.
//!
//! Hashing a prompt prefix gives clients that send no session id stickiness
//! anyway: a multi-turn conversation resends its history every turn, so a
//! window over its opening messages hashes identically on every turn. Two
//! rules of thumb for choosing the window:
//!
//! - **Stable:** turn 1 must already fill the window, or the key changes when
//!   later turns extend it. Windows counted in messages (`user_turns`,
//!   `messages`) are stable by construction; for raw text or token prompts set
//!   `min_*` equal to `max_*` so short prompts get no key rather than a
//!   shifting one.
//! - **Distinguishing:** the window must reach past content every session
//!   shares. A fixed window of the first N tokens maps every session to the
//!   same key when they share a system prompt longer than N; ending the window
//!   at the first user message (the default) always includes what differs.

use axum::http::{HeaderMap, HeaderName};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use xxhash_rust::xxh3::xxh3_128;

use super::{new_session_id, valid_session_id};
use crate::{
    config::SessionConfig,
    inspect::{InspectedRequest, Prompt},
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeriveRule {
    /// The value of another request header (e.g. `modal-session-id`).
    Header { name: String },
    /// A JSON pointer into the request body (e.g. `/prompt_cache_key`,
    /// `/metadata/conversation_id`, `/user`).
    BodyField { pointer: String },
    /// A hash of a window over the prompt.
    Prefix(PrefixRule),
}

/// Selects the window of the prompt that is hashed. Message-level options
/// apply to chat-shaped requests, `*_chars` to the selected text (chat
/// messages are rendered as `role␟content␞`), `*_tokens` to token-id prompts.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PrefixRule {
    /// Drop messages with these roles first, e.g. `[system]` when the system
    /// prompt is shared and adds nothing distinguishing.
    pub skip_roles: Vec<String>,
    /// Then drop this many leading messages.
    pub skip_messages: usize,
    /// End the window at the Nth user message (inclusive). Requests with fewer
    /// user messages get no key.
    pub user_turns: Option<usize>,
    /// Cap the window at this many messages. Requests with fewer get no key.
    pub messages: Option<usize>,
    pub skip_chars: usize,
    pub max_chars: Option<usize>,
    /// Selections shorter than this get no key.
    pub min_chars: usize,
    pub skip_tokens: usize,
    pub max_tokens: Option<usize>,
    pub min_tokens: usize,
}

impl Default for PrefixRule {
    fn default() -> Self {
        Self {
            skip_roles: Vec::new(),
            skip_messages: 0,
            user_turns: Some(1),
            messages: None,
            skip_chars: 0,
            max_chars: None,
            min_chars: 1,
            skip_tokens: 0,
            max_tokens: None,
            min_tokens: 1,
        }
    }
}

impl DeriveRule {
    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            DeriveRule::Header { name } => {
                HeaderName::try_from(name.as_str())?;
            }
            DeriveRule::BodyField { pointer } => anyhow::ensure!(
                pointer.starts_with('/'),
                "body_field pointer {pointer:?} must start with '/'"
            ),
            DeriveRule::Prefix(rule) => anyhow::ensure!(
                rule.user_turns != Some(0) && rule.messages != Some(0),
                "prefix user_turns / messages must be > 0"
            ),
        }
        Ok(())
    }

    fn tag(&self) -> &'static str {
        match self {
            DeriveRule::Header { .. } => "hdr",
            DeriveRule::BodyField { .. } => "field",
            DeriveRule::Prefix(_) => "prefix",
        }
    }
}

/// Where a session key came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionSource {
    /// The sticky-session header.
    Explicit,
    /// A header or body field rule: the client named the session.
    Declared,
    /// A prefix rule: the session was inferred from content.
    Inferred,
    /// Minted by the router.
    Generated,
}

impl SessionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionSource::Explicit => "explicit",
            SessionSource::Declared => "declared",
            SessionSource::Inferred => "inferred",
            SessionSource::Generated => "generated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SessionKey {
    pub id: String,
    pub source: SessionSource,
}

impl SessionKey {
    pub fn explicit(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source: SessionSource::Explicit,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid {0}: expected 1-256 visible ASCII characters")]
pub struct InvalidSessionId(pub HeaderName);

/// The session key for a request, per the order in the module docs.
pub fn resolve(
    config: &SessionConfig,
    header: &HeaderName,
    headers: &HeaderMap,
    body: Option<&Value>,
    request: &InspectedRequest,
) -> Result<Option<SessionKey>, InvalidSessionId> {
    if let Some(value) = headers.get(header) {
        return match value.to_str() {
            Ok(id) if valid_session_id(id) => Ok(Some(SessionKey::explicit(id))),
            _ => Err(InvalidSessionId(header.clone())),
        };
    }
    if let Some(key) = derive(config, headers, body, request) {
        return Ok(Some(key));
    }
    Ok(config.generate_if_missing.then(|| SessionKey {
        id: new_session_id(),
        source: SessionSource::Generated,
    }))
}

/// The first configured derive rule that yields a key.
pub fn derive(
    config: &SessionConfig,
    headers: &HeaderMap,
    body: Option<&Value>,
    request: &InspectedRequest,
) -> Option<SessionKey> {
    config.derive.iter().find_map(|rule| {
        let (source, selector, material) = match rule {
            DeriveRule::Header { name } => {
                let value = headers.get(name.as_str())?.as_bytes();
                (
                    SessionSource::Declared,
                    name.as_bytes(),
                    nonempty(value.to_vec())?,
                )
            }
            DeriveRule::BodyField { pointer } => {
                let value = match body?.pointer(pointer)? {
                    Value::String(s) => s.clone(),
                    Value::Null => return None,
                    other => other.to_string(),
                };
                (
                    SessionSource::Declared,
                    pointer.as_bytes(),
                    nonempty(value.into_bytes())?,
                )
            }
            DeriveRule::Prefix(prefix) => (
                SessionSource::Inferred,
                &[][..],
                select(prefix, request.prompt.as_ref()?)?,
            ),
        };
        let model = config
            .derive_include_model
            .then_some(request.model.as_deref())
            .flatten();
        Some(SessionKey {
            id: hashed_id(
                rule.tag(),
                model,
                request.cache_salt.as_deref(),
                selector,
                &material,
            ),
            source,
        })
    })
}

fn nonempty(bytes: Vec<u8>) -> Option<Vec<u8>> {
    (!bytes.is_empty()).then_some(bytes)
}

/// Derived ids are hashed so they're bounded, store-safe, and can't collide
/// with explicit ids or with each other across rules. Fields are
/// length-prefixed so no two field sequences encode the same bytes.
fn hashed_id(
    tag: &str,
    model: Option<&str>,
    cache_salt: Option<&str>,
    selector: &[u8],
    material: &[u8],
) -> String {
    let mut buf = Vec::with_capacity(material.len() + 64);
    for field in [
        tag.as_bytes(),
        model.unwrap_or_default().as_bytes(),
        cache_salt.unwrap_or_default().as_bytes(),
        selector,
        material,
    ] {
        buf.extend_from_slice(&(field.len() as u64).to_le_bytes());
        buf.extend_from_slice(field);
    }
    format!("{tag}-{:032x}", xxh3_128(&buf))
}

/// The bytes of the prompt window `rule` selects, or `None` if the prompt
/// doesn't fill it.
fn select(rule: &PrefixRule, prompt: &Prompt) -> Option<Vec<u8>> {
    match prompt {
        Prompt::Text(text) => select_chars(rule, text),
        Prompt::Tokens(tokens) => {
            let tokens = tokens.get(rule.skip_tokens..)?;
            let tokens = &tokens[..rule
                .max_tokens
                .map_or(tokens.len(), |m| m.min(tokens.len()))];
            if tokens.len() < rule.min_tokens.max(1) {
                return None;
            }
            Some(tokens.iter().flat_map(|t| t.to_le_bytes()).collect())
        }
        Prompt::Chat(messages) => {
            let mut window: Vec<_> = messages
                .iter()
                .filter(|m| !rule.skip_roles.contains(&m.role))
                .skip(rule.skip_messages)
                .collect();
            if let Some(n) = rule.user_turns {
                let end = window
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.role == "user")
                    .nth(n - 1)?
                    .0;
                window.truncate(end + 1);
            }
            if let Some(n) = rule.messages {
                if window.len() < n {
                    return None;
                }
                window.truncate(n);
            }
            let text: String = window
                .iter()
                .map(|m| format!("{}\u{1f}{}\u{1e}", m.role, m.content))
                .collect();
            select_chars(rule, &text)
        }
    }
}

fn select_chars(rule: &PrefixRule, text: &str) -> Option<Vec<u8>> {
    let byte_at = |s: &str, chars: usize| s.char_indices().nth(chars).map_or(s.len(), |(i, _)| i);
    let rest = &text[byte_at(text, rule.skip_chars)..];
    let window = match rule.max_chars {
        Some(max) => &rest[..byte_at(rest, max)],
        None => rest,
    };
    if window.chars().count() < rule.min_chars.max(1) {
        return None;
    }
    Some(window.as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::inspect::inspect;

    fn config(derive: Vec<DeriveRule>) -> SessionConfig {
        SessionConfig {
            derive,
            generate_if_missing: false,
            ..Default::default()
        }
    }

    fn prefix(rule: PrefixRule) -> SessionConfig {
        config(vec![DeriveRule::Prefix(rule)])
    }

    fn key(config: &SessionConfig, path: &str, body: Value) -> Option<String> {
        key_with_headers(config, path, body, HeaderMap::new())
    }

    fn key_with_headers(
        config: &SessionConfig,
        path: &str,
        body: Value,
        headers: HeaderMap,
    ) -> Option<String> {
        let request = inspect(path, &body);
        let header = HeaderName::from_static(crate::STICKY_SESSION_HEADER);
        resolve(config, &header, &headers, Some(&body), &request)
            .unwrap()
            .map(|k| k.id)
    }

    fn chat(messages: &[(&str, &str)]) -> Value {
        let messages: Vec<Value> = messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        json!({"model": "m", "messages": messages})
    }

    const SYSTEM: (&str, &str) = ("system", "You are a coding agent. <30k tokens of tools>");

    #[test]
    fn conversation_keeps_its_key_across_turns() {
        let cfg = prefix(PrefixRule::default());
        let turn1 = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "fix bug 1")]),
        );
        let turn2 = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[
                SYSTEM,
                ("user", "fix bug 1"),
                ("assistant", "done"),
                ("user", "thanks"),
            ]),
        );
        assert!(turn1.as_deref().unwrap().starts_with("prefix-"));
        assert_eq!(turn1, turn2);
    }

    #[test]
    fn shared_system_prompt_does_not_collapse_sessions() {
        let cfg = prefix(PrefixRule::default());
        let a = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "fix bug 1")]),
        );
        let b = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "fix bug 2")]),
        );
        assert_ne!(a, b);

        // The failure mode of a fixed window shorter than the shared prefix.
        let first_n = prefix(PrefixRule {
            user_turns: None,
            max_chars: Some(20),
            ..Default::default()
        });
        let a = key(
            &first_n,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "fix bug 1")]),
        );
        let b = key(
            &first_n,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "fix bug 2")]),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn skip_roles_and_message_windows() {
        let cfg = prefix(PrefixRule {
            skip_roles: vec!["system".into()],
            ..Default::default()
        });
        let a = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[("system", "x"), ("user", "q")]),
        );
        let b = key(
            &cfg,
            "/v1/chat/completions",
            chat(&[("system", "y"), ("user", "q")]),
        );
        assert_eq!(a, b);

        let two_turns = prefix(PrefixRule {
            user_turns: Some(2),
            ..Default::default()
        });
        assert_eq!(
            key(&two_turns, "/v1/chat/completions", chat(&[("user", "q")])),
            None
        );

        let fixed = prefix(PrefixRule {
            user_turns: None,
            messages: Some(3),
            ..Default::default()
        });
        assert_eq!(
            key(
                &fixed,
                "/v1/chat/completions",
                chat(&[SYSTEM, ("user", "q")])
            ),
            None
        );
        assert!(key(
            &fixed,
            "/v1/chat/completions",
            chat(&[SYSTEM, ("user", "q"), ("assistant", "a")])
        )
        .is_some());
    }

    #[test]
    fn text_and_token_windows() {
        let cfg = prefix(PrefixRule {
            skip_chars: 2,
            max_chars: Some(4),
            min_chars: 4,
            ..Default::default()
        });
        let k = |prompt: &str| key(&cfg, "/v1/completions", json!({"prompt": prompt}));
        assert_eq!(k("xxabcdTAIL"), k("yyabcdOTHER"));
        assert_ne!(k("xxabcd"), k("xxabce"));
        assert_eq!(k("xxabc"), None, "shorter than min_chars");
        // Character, not byte, offsets.
        assert_eq!(k("ééabcd"), k("xxabcd"));

        let cfg = prefix(PrefixRule {
            skip_tokens: 1,
            max_tokens: Some(2),
            ..Default::default()
        });
        let k = |prompt: Value| key(&cfg, "/v1/completions", json!({"prompt": prompt}));
        assert_eq!(k(json!([9, 1, 2, 3])), k(json!([8, 1, 2, 4])));
        assert_eq!(k(json!([9])), None);
    }

    #[test]
    fn model_and_cache_salt_namespace_keys() {
        let cfg = prefix(PrefixRule::default());
        let mut body = chat(&[("user", "q")]);
        let base = key(&cfg, "/v1/chat/completions", body.clone());
        body["model"] = json!("other");
        assert_ne!(key(&cfg, "/v1/chat/completions", body.clone()), base);
        body["model"] = json!("m");
        body["cache_salt"] = json!("tenant");
        assert_ne!(key(&cfg, "/v1/chat/completions", body), base);

        let no_model = SessionConfig {
            derive_include_model: false,
            ..prefix(PrefixRule::default())
        };
        let mut body = chat(&[("user", "q")]);
        let a = key(&no_model, "/v1/chat/completions", body.clone());
        body["model"] = json!("other");
        assert_eq!(key(&no_model, "/v1/chat/completions", body), a);
    }

    #[test]
    fn rule_order_and_sources() {
        let cfg = config(vec![
            DeriveRule::Header {
                name: "modal-session-id".into(),
            },
            DeriveRule::BodyField {
                pointer: "/prompt_cache_key".into(),
            },
            DeriveRule::Prefix(PrefixRule::default()),
        ]);
        let body =
            json!({"prompt_cache_key": "conv-7", "messages": [{"role": "user", "content": "q"}]});
        let request = inspect("/v1/chat/completions", &body);
        let header = HeaderName::from_static(crate::STICKY_SESSION_HEADER);

        let mut headers = HeaderMap::new();
        headers.insert("modal-session-id", "abc".parse().unwrap());
        let k = resolve(&cfg, &header, &headers, Some(&body), &request)
            .unwrap()
            .unwrap();
        assert!(k.id.starts_with("hdr-") && k.source == SessionSource::Declared);

        let k = resolve(&cfg, &header, &HeaderMap::new(), Some(&body), &request)
            .unwrap()
            .unwrap();
        assert!(k.id.starts_with("field-") && k.source == SessionSource::Declared);

        let body = json!({"messages": [{"role": "user", "content": "q"}]});
        let request = inspect("/v1/chat/completions", &body);
        let k = resolve(&cfg, &header, &HeaderMap::new(), Some(&body), &request)
            .unwrap()
            .unwrap();
        assert!(k.id.starts_with("prefix-") && k.source == SessionSource::Inferred);

        // The explicit header beats every rule.
        headers.insert(header.clone(), "mine".parse().unwrap());
        let k = resolve(&cfg, &header, &headers, Some(&body), &request)
            .unwrap()
            .unwrap();
        assert_eq!(k, SessionKey::explicit("mine"));

        headers.insert(header.clone(), "not valid".parse().unwrap());
        assert!(resolve(&cfg, &header, &headers, Some(&body), &request).is_err());
    }

    #[test]
    fn falls_back_to_generated() {
        let cfg = SessionConfig {
            derive: vec![DeriveRule::Prefix(PrefixRule::default())],
            generate_if_missing: true,
            ..Default::default()
        };
        let body = json!({"foo": 1});
        let request = inspect("/v1/completions", &body);
        let header = HeaderName::from_static(crate::STICKY_SESSION_HEADER);
        let k = resolve(&cfg, &header, &HeaderMap::new(), Some(&body), &request)
            .unwrap()
            .unwrap();
        assert_eq!(k.source, SessionSource::Generated);
    }

    #[test]
    fn parses_rules_from_yaml() {
        let rules: Vec<DeriveRule> = serde_yaml::from_str(
            "- {kind: header, name: modal-session-id}\n- {kind: body_field, pointer: /prompt_cache_key}\n- {kind: prefix, skip_roles: [system], user_turns: 1, max_chars: 8192}\n- {kind: prefix}\n",
        )
        .unwrap();
        assert_eq!(rules.len(), 4);
        assert_eq!(rules[3], DeriveRule::Prefix(PrefixRule::default()));
        assert!(matches!(&rules[2], DeriveRule::Prefix(p) if p.max_chars == Some(8192)));
        assert!(serde_yaml::from_str::<Vec<DeriveRule>>("- {kind: prefix, bogus: 1}").is_err());
    }
}
