//! Backend registry and per-backend runtime state (health, load, draining).

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::{Mutex, RwLock};
use serde::Serialize;

use crate::config::{BackendConfig, HealthConfig};

pub type BackendId = Arc<str>;

pub struct Backend {
    pub id: BackendId,
    /// Base URL without a trailing slash.
    pub url: String,
    pub config: BackendConfig,
    state: BackendState,
}

#[derive(Default)]
struct BackendState {
    /// Verdict of the active health checker. Starts false: a backend takes no
    /// traffic until it has passed a probe.
    active_healthy: AtomicBool,
    consecutive_probe_failures: AtomicU32,
    consecutive_probe_successes: AtomicU32,
    consecutive_request_failures: AtomicU32,
    ejected_until: Mutex<Option<Instant>>,
    /// Draining backends keep their existing sessions but take no new ones.
    draining: AtomicBool,
    inflight: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl Backend {
    pub fn new(config: BackendConfig) -> Self {
        Self {
            id: Arc::from(config.id.as_str()),
            url: config.url.trim_end_matches('/').to_string(),
            config,
            state: BackendState::default(),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.state.active_healthy.load(Ordering::Acquire) && !self.is_ejected()
    }

    /// May serve requests for sessions already bound to it.
    pub fn accepts_existing(&self) -> bool {
        self.is_healthy()
    }

    /// May be picked for new sessions and sessionless requests.
    pub fn accepts_new(&self) -> bool {
        self.is_healthy() && !self.is_draining()
    }

    pub fn is_draining(&self) -> bool {
        self.state.draining.load(Ordering::Acquire)
    }

    pub fn set_draining(&self, draining: bool) {
        self.state.draining.store(draining, Ordering::Release);
    }

    fn is_ejected(&self) -> bool {
        let mut ejected = self.state.ejected_until.lock();
        match *ejected {
            Some(until) if Instant::now() < until => true,
            Some(_) => {
                *ejected = None;
                false
            }
            None => false,
        }
    }

    pub fn inflight(&self) -> u64 {
        self.state.inflight.load(Ordering::Relaxed)
    }

    /// Count a request against this backend until the guard drops, which for
    /// streamed responses is when the body stream finishes or is abandoned.
    pub fn inflight_guard(self: &Arc<Self>) -> InflightGuard {
        self.state.inflight.fetch_add(1, Ordering::Relaxed);
        InflightGuard(self.clone())
    }

    /// Record an active probe result, applying the configured thresholds.
    pub fn record_probe(&self, result: Result<(), String>, cfg: &HealthConfig) {
        let s = &self.state;
        match result {
            Ok(()) => {
                s.consecutive_probe_failures.store(0, Ordering::Relaxed);
                let ok = s
                    .consecutive_probe_successes
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;
                if ok >= cfg.healthy_threshold && !s.active_healthy.swap(true, Ordering::AcqRel) {
                    tracing::info!(backend = %self.id, "backend healthy");
                }
            }
            Err(err) => {
                s.consecutive_probe_successes.store(0, Ordering::Relaxed);
                let failed = s.consecutive_probe_failures.fetch_add(1, Ordering::Relaxed) + 1;
                if failed >= cfg.unhealthy_threshold
                    && s.active_healthy.swap(false, Ordering::AcqRel)
                {
                    tracing::warn!(backend = %self.id, error = %err, "backend unhealthy");
                }
                *s.last_error.lock() = Some(err);
            }
        }
    }

    /// Record the outcome of a proxied request (passive health checking).
    pub fn record_request(&self, result: Result<(), String>, cfg: &HealthConfig) {
        let s = &self.state;
        match result {
            Ok(()) => s.consecutive_request_failures.store(0, Ordering::Relaxed),
            Err(err) => {
                let failed = s
                    .consecutive_request_failures
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;
                if failed >= cfg.passive_failure_threshold {
                    s.consecutive_request_failures.store(0, Ordering::Relaxed);
                    *s.ejected_until.lock() = Some(Instant::now() + cfg.ejection_duration);
                    tracing::warn!(
                        backend = %self.id,
                        error = %err,
                        for_ = ?cfg.ejection_duration,
                        "ejecting backend after consecutive request failures"
                    );
                }
                *s.last_error.lock() = Some(err);
            }
        }
    }

    pub fn status(&self) -> BackendStatus {
        BackendStatus {
            id: self.id.to_string(),
            url: self.url.clone(),
            healthy: self.state.active_healthy.load(Ordering::Acquire),
            ejected: self.is_ejected(),
            draining: self.is_draining(),
            inflight: self.inflight(),
            last_error: self.state.last_error.lock().clone(),
            kv_blocks: 0,
        }
    }
}

pub struct InflightGuard(Arc<Backend>);

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.state.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendStatus {
    pub id: String,
    pub url: String,
    pub healthy: bool,
    pub ejected: bool,
    pub draining: bool,
    pub inflight: u64,
    pub last_error: Option<String>,
    /// Blocks the KV index currently attributes to this backend.
    pub kv_blocks: usize,
}

/// The set of known backends. Backends can be added and removed at runtime.
#[derive(Default)]
pub struct BackendRegistry {
    backends: RwLock<BTreeMap<BackendId, Arc<Backend>>>,
}

impl BackendRegistry {
    pub fn new(configs: impl IntoIterator<Item = BackendConfig>) -> Self {
        let reg = Self::default();
        for c in configs {
            reg.upsert(c);
        }
        reg
    }

    /// Add a backend, or replace one whose config changed. Re-adding an
    /// identical config keeps its runtime state.
    pub fn upsert(&self, config: BackendConfig) -> Arc<Backend> {
        let mut map = self.backends.write();
        if let Some(existing) = map.get(config.id.as_str()) {
            if existing.config == config {
                return existing.clone();
            }
        }
        let backend = Arc::new(Backend::new(config));
        map.insert(backend.id.clone(), backend.clone());
        backend
    }

    pub fn remove(&self, id: &str) -> Option<Arc<Backend>> {
        self.backends.write().remove(id)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Backend>> {
        self.backends.read().get(id).cloned()
    }

    pub fn all(&self) -> Vec<Arc<Backend>> {
        self.backends.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn backend() -> Backend {
        Backend::new(BackendConfig {
            id: "b".into(),
            url: "http://b/".into(),
            kv_events: None,
        })
    }

    #[test]
    fn probe_thresholds() {
        let cfg = HealthConfig {
            unhealthy_threshold: 2,
            healthy_threshold: 2,
            ..Default::default()
        };
        let b = backend();
        assert_eq!(b.url, "http://b");
        assert!(!b.is_healthy());
        b.record_probe(Ok(()), &cfg);
        assert!(!b.is_healthy());
        b.record_probe(Ok(()), &cfg);
        assert!(b.is_healthy());
        b.record_probe(Err("x".into()), &cfg);
        assert!(b.is_healthy());
        b.record_probe(Err("x".into()), &cfg);
        assert!(!b.is_healthy());
    }

    #[test]
    fn passive_ejection_expires() {
        let cfg = HealthConfig {
            healthy_threshold: 1,
            passive_failure_threshold: 2,
            ejection_duration: Duration::from_millis(20),
            ..Default::default()
        };
        let b = backend();
        b.record_probe(Ok(()), &cfg);
        b.record_request(Err("refused".into()), &cfg);
        assert!(b.is_healthy());
        b.record_request(Err("refused".into()), &cfg);
        assert!(!b.is_healthy());
        std::thread::sleep(Duration::from_millis(30));
        assert!(b.is_healthy());
    }

    #[test]
    fn draining_keeps_existing_sessions() {
        let b = backend();
        b.record_probe(Ok(()), &HealthConfig::default());
        b.set_draining(true);
        assert!(b.accepts_existing());
        assert!(!b.accepts_new());
    }
}
