use std::time::Duration;

use redis::{
    aio::{ConnectionManager, ConnectionManagerConfig},
    Script,
};

use super::{SessionStore, StoreError};

/// Bindings shared by all replicas through Redis / Valkey. Each operation is a
/// single round trip; check-and-set steps run as Lua so they are atomic.
pub struct RedisStore {
    conn: ConnectionManager,
    key_prefix: String,
    bind_if_absent: Script,
    compare_and_swap: Script,
}

const BIND_IF_ABSENT: &str = r#"
local current = redis.call('GET', KEYS[1])
if current then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return current
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return ARGV[1]
"#;

const COMPARE_AND_SWAP: &str = r#"
local current = redis.call('GET', KEYS[1])
if current and current ~= ARGV[1] then
  return current
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3])
return ARGV[2]
"#;

impl RedisStore {
    pub async fn connect(url: &str, key_prefix: &str) -> Result<Self, StoreError> {
        let client = redis::Client::open(url)?;
        // Every routed request makes a store round trip, so a slow store must
        // fail fast: the router then degrades to rendezvous hashing rather than
        // stalling requests behind it.
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Duration::from_secs(2))
            .set_response_timeout(Duration::from_millis(250))
            .set_number_of_retries(2);
        let conn = ConnectionManager::new_with_config(client, config).await?;
        Ok(Self {
            conn,
            key_prefix: key_prefix.to_string(),
            bind_if_absent: Script::new(BIND_IF_ABSENT),
            compare_and_swap: Script::new(COMPARE_AND_SWAP),
        })
    }

    fn key(&self, session: &str) -> String {
        format!("{}session:{}", self.key_prefix, session)
    }
}

fn millis(ttl: Duration) -> u64 {
    ttl.as_millis().max(1) as u64
}

#[async_trait::async_trait]
impl SessionStore for RedisStore {
    async fn get(&self, session: &str, ttl: Duration) -> Result<Option<String>, StoreError> {
        let mut conn = self.conn.clone();
        Ok(redis::cmd("GETEX")
            .arg(self.key(session))
            .arg("PX")
            .arg(millis(ttl))
            .query_async(&mut conn)
            .await?)
    }

    async fn bind_if_absent(
        &self,
        session: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError> {
        let mut conn = self.conn.clone();
        Ok(self
            .bind_if_absent
            .key(self.key(session))
            .arg(backend)
            .arg(millis(ttl))
            .invoke_async(&mut conn)
            .await?)
    }

    async fn compare_and_swap(
        &self,
        session: &str,
        expected: &str,
        backend: &str,
        ttl: Duration,
    ) -> Result<String, StoreError> {
        let mut conn = self.conn.clone();
        Ok(self
            .compare_and_swap
            .key(self.key(session))
            .arg(expected)
            .arg(backend)
            .arg(millis(ttl))
            .invoke_async(&mut conn)
            .await?)
    }

    async fn remove(&self, session: &str) -> Result<bool, StoreError> {
        let mut conn = self.conn.clone();
        let removed: u64 = redis::cmd("DEL")
            .arg(self.key(session))
            .query_async(&mut conn)
            .await?;
        Ok(removed > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs against a real server when `REDIS_URL` is set.
    #[tokio::test]
    async fn conformance() {
        let Ok(url) = std::env::var("REDIS_URL") else {
            eprintln!("REDIS_URL unset; skipping redis conformance test");
            return;
        };
        let store = RedisStore::connect(&url, "chalk-statefulrouter-test:")
            .await
            .unwrap();
        let prefix = super::super::new_session_id();
        super::super::conformance::run(&store, &prefix).await;
    }
}
