use std::time::{Duration, Instant};

use dashmap::{DashMap, Entry};

use super::{SessionStore, StoreError};

/// In-process bindings. Correct only for a single router replica.
#[derive(Default)]
pub struct MemoryStore {
    bindings: DashMap<String, Binding>,
}

struct Binding {
    backend: String,
    expires_at: Instant,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop expired bindings. Expiry is also checked on access, so this only
    /// bounds memory held by abandoned sessions.
    pub fn sweep(&self) {
        let now = Instant::now();
        self.bindings.retain(|_, b| b.expires_at > now);
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Run `f` on the live binding for `session` (expired ones are treated as
    /// absent), holding the entry lock so check-and-set is atomic.
    fn with_entry<T>(&self, session: &str, f: impl FnOnce(Entry<'_, String, Binding>) -> T) -> T {
        let mut entry = self.bindings.entry(session.to_string());
        if let Entry::Occupied(occupied) = entry {
            entry = if occupied.get().expires_at <= Instant::now() {
                let (key, _) = occupied.remove_entry();
                self.bindings.entry(key)
            } else {
                Entry::Occupied(occupied)
            };
        }
        f(entry)
    }
}

#[async_trait::async_trait]
impl SessionStore for MemoryStore {
    async fn get(&self, session: &str, ttl: Duration) -> Result<Option<String>, StoreError> {
        Ok(self.with_entry(session, |entry| match entry {
            Entry::Occupied(mut b) => {
                b.get_mut().expires_at = Instant::now() + ttl;
                Some(b.get().backend.clone())
            }
            Entry::Vacant(_) => None,
        }))
    }

    async fn bind_if_absent(
        &self,
        session: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError> {
        Ok(self.with_entry(session, |entry| match entry {
            Entry::Occupied(b) => b.get().backend.clone(),
            Entry::Vacant(v) => {
                v.insert(Binding {
                    backend: backend.to_string(),
                    expires_at: Instant::now() + ttl,
                });
                backend.to_string()
            }
        }))
    }

    async fn compare_and_swap(
        &self,
        session: &str,
        expected: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError> {
        let binding = Binding {
            backend: backend.to_string(),
            expires_at: Instant::now() + ttl,
        };
        Ok(self.with_entry(session, |entry| match entry {
            Entry::Occupied(mut b) if b.get().backend == expected => {
                b.insert(binding);
                backend.to_string()
            }
            Entry::Occupied(b) => b.get().backend.clone(),
            Entry::Vacant(v) => {
                v.insert(binding);
                backend.to_string()
            }
        }))
    }

    async fn remove(&self, session: &str) -> Result<bool, StoreError> {
        Ok(self.with_entry(session, |entry| match entry {
            Entry::Occupied(b) => {
                b.remove();
                true
            }
            Entry::Vacant(_) => false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn conformance() {
        super::super::conformance::run(&MemoryStore::new(), "mem").await;
    }

    #[tokio::test]
    async fn expiry() {
        let store = MemoryStore::new();
        let ttl = Duration::from_millis(20);
        store.bind_if_absent("s", "a", ttl).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(store.get("s", ttl).await.unwrap(), None);
        assert_eq!(store.bind_if_absent("s", "b", ttl).await.unwrap(), "b");
        tokio::time::sleep(Duration::from_millis(30)).await;
        store.sweep();
        assert!(store.is_empty());
    }
}
