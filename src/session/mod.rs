//! Session -> backend bindings.
//!
//! Every router replica ("frontend") that sees a given
//! `x-chalk-sticky-session-id` must send it to the same backend. With a shared
//! store that is enforced by atomic operations: a new session is bound with
//! set-if-absent, so when two replicas race, both adopt whichever write won;
//! a binding to a dead backend is moved with compare-and-swap, so replicas
//! that notice the failure simultaneously still converge on one replacement.

pub mod derive;
mod memory;
mod redis;

use std::time::Duration;

pub use derive::{SessionKey, SessionSource};
pub use memory::MemoryStore;
pub use redis::RedisStore;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("redis: {0}")]
    Redis(#[from] ::redis::RedisError),
}

#[async_trait::async_trait]
pub trait SessionStore: Send + Sync + 'static {
    /// The backend `session` is bound to, refreshing the binding's TTL.
    async fn get(&self, session: &str, ttl: Duration) -> Result<Option<String>, StoreError>;

    /// Bind `session` to `backend` unless it is already bound. Returns the
    /// binding in effect afterwards, which may be another replica's choice.
    async fn bind_if_absent(
        &self,
        session: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError>;

    /// Rebind `session` from `expected` to `backend` if it is still bound to
    /// `expected` (or no longer bound at all). Returns the binding in effect
    /// afterwards.
    async fn compare_and_swap(
        &self,
        session: &str,
        expected: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError>;

    async fn remove(&self, session: &str) -> Result<bool, StoreError>;
}

/// Longest session id accepted, to bound store key size.
pub const MAX_SESSION_ID_LEN: usize = 256;

/// Session ids end up in store keys and logs: require short, visible ASCII.
pub fn valid_session_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_SESSION_ID_LEN && id.bytes().all(|b| b.is_ascii_graphic())
}

pub fn new_session_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[cfg(test)]
pub(crate) mod conformance {
    //! Behavior every store must share.
    use super::*;

    const TTL: Duration = Duration::from_secs(60);

    pub async fn run(store: &dyn SessionStore, prefix: &str) {
        let s = format!("{prefix}-s1");
        assert_eq!(store.get(&s, TTL).await.unwrap(), None);
        assert_eq!(store.bind_if_absent(&s, "a", TTL).await.unwrap(), "a");
        // Second binder loses and adopts the first binding.
        assert_eq!(store.bind_if_absent(&s, "b", TTL).await.unwrap(), "a");
        assert_eq!(store.get(&s, TTL).await.unwrap().as_deref(), Some("a"));
        // CAS from a stale expectation fails.
        assert_eq!(
            store.compare_and_swap(&s, "x", "b", TTL).await.unwrap(),
            "a"
        );
        assert_eq!(
            store.compare_and_swap(&s, "a", "b", TTL).await.unwrap(),
            "b"
        );
        // A second replica that also saw "a" die adopts the first rebind.
        assert_eq!(
            store.compare_and_swap(&s, "a", "c", TTL).await.unwrap(),
            "b"
        );
        assert!(store.remove(&s).await.unwrap());
        assert!(!store.remove(&s).await.unwrap());
        // CAS on an expired/removed binding binds.
        assert_eq!(
            store.compare_and_swap(&s, "a", "c", TTL).await.unwrap(),
            "c"
        );
        store.remove(&s).await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_validation() {
        assert!(valid_session_id("abc-123_x.y"));
        assert!(!valid_session_id(""));
        assert!(!valid_session_id("has space"));
        assert!(!valid_session_id(&"x".repeat(MAX_SESSION_ID_LEN + 1)));
        assert!(valid_session_id(&new_session_id()));
    }
}
