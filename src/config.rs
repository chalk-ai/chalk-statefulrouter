//! Router configuration, loaded from YAML.

use std::{net::SocketAddr, path::Path, time::Duration};

use serde::{Deserialize, Serialize};

use crate::session::derive::{DeriveRule, SessionSource};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Address the proxied inference API listens on.
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// Address the admin / control API listens on. Kept off the proxy port so
    /// admin routes can never shadow a backend path.
    #[serde(default = "default_admin_listen")]
    pub admin_listen: SocketAddr,
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
    #[serde(default)]
    pub health: HealthConfig,
    #[serde(default)]
    pub sessions: SessionConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub kv: KvConfig,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let cfg: Config = serde_yaml::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let mut seen = std::collections::HashSet::new();
        for b in &self.backends {
            anyhow::ensure!(!b.id.is_empty(), "backend id must not be empty");
            anyhow::ensure!(seen.insert(&b.id), "duplicate backend id {:?}", b.id);
        }
        anyhow::ensure!(
            self.health.unhealthy_threshold > 0 && self.health.healthy_threshold > 0,
            "health thresholds must be > 0"
        );
        for rule in &self.sessions.derive {
            rule.validate()?;
        }
        if let Some(factor) = self.routing.bounded_load.factor {
            anyhow::ensure!(factor >= 1.0, "bounded_load.factor must be >= 1.0");
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            admin_listen: default_admin_listen(),
            backends: Vec::new(),
            health: HealthConfig::default(),
            sessions: SessionConfig::default(),
            routing: RoutingConfig::default(),
            kv: KvConfig::default(),
        }
    }
}

fn default_listen() -> SocketAddr {
    "0.0.0.0:8080".parse().unwrap()
}

fn default_admin_listen() -> SocketAddr {
    "0.0.0.0:8081".parse().unwrap()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    /// Stable identifier. Must be identical across router replicas, since it is
    /// what gets written to the shared session store and what KV events carry.
    pub id: String,
    /// Base URL, e.g. `http://vllm-0.vllm:8000`.
    pub url: String,
    /// Where this backend publishes KV cache events, if it does.
    #[serde(default)]
    pub kv_events: Option<KvEventsEndpoint>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KvEventsEndpoint {
    /// ZMQ PUB endpoint, e.g. `tcp://vllm-0.vllm:5557` (vLLM `--kv-events-config`
    /// / SGLang `--kv-events-config` with `"publisher": "zmq"`).
    pub zmq_endpoint: String,
    #[serde(default = "default_kv_topic")]
    pub topic: String,
}

fn default_kv_topic() -> String {
    "kv-events".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct HealthConfig {
    /// Path probed on each backend. vLLM and SGLang both serve `/health`.
    pub path: String,
    #[serde(with = "humantime_serde")]
    pub interval: Duration,
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
    /// Consecutive failed probes before a healthy backend is marked unhealthy.
    pub unhealthy_threshold: u32,
    /// Consecutive successful probes before an unhealthy backend is restored.
    pub healthy_threshold: u32,
    /// Consecutive proxied-request failures (connect errors, 502/503/504) before
    /// the backend is ejected without waiting for the active checker.
    pub passive_failure_threshold: u32,
    #[serde(with = "humantime_serde")]
    pub ejection_duration: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            path: "/health".to_string(),
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(2),
            unhealthy_threshold: 2,
            healthy_threshold: 1,
            passive_failure_threshold: 3,
            ejection_duration: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct SessionConfig {
    pub header: String,
    /// Mint a session id when a request arrives without one, and return it on
    /// the response so the client can reuse it.
    pub generate_if_missing: bool,
    /// Sliding TTL of a session -> backend binding.
    #[serde(with = "humantime_serde")]
    pub ttl: Duration,
    pub store: SessionStoreConfig,
    /// How to find a session key for requests without the sticky-session
    /// header, tried in order. See [`crate::session::derive`].
    pub derive: Vec<DeriveRule>,
    /// Fold the request's `model` into derived keys, so one conversation
    /// sent to two models is two sessions (two caches).
    pub derive_include_model: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            header: crate::STICKY_SESSION_HEADER.to_string(),
            generate_if_missing: true,
            ttl: Duration::from_secs(60 * 60),
            store: SessionStoreConfig::Memory,
            // OpenAI's own cache-affinity hint; only applies when sent.
            derive: vec![DeriveRule::BodyField {
                pointer: "/prompt_cache_key".to_string(),
            }],
            derive_include_model: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionStoreConfig {
    /// Bindings live in this process only. Correct for a single replica.
    Memory,
    /// Bindings live in Redis/Valkey and are shared by every replica.
    Redis {
        url: String,
        #[serde(default = "default_redis_prefix")]
        key_prefix: String,
    },
    /// No stored bindings: every replica maps a session to a backend by
    /// rendezvous hashing over the healthy set. Replicas agree whenever their
    /// health views agree, with no coordination, but bindings move when the
    /// backend set changes.
    Rendezvous,
}

fn default_redis_prefix() -> String {
    "chalk-statefulrouter:".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct RoutingConfig {
    /// Weight of the fraction of the prompt already cached on a backend.
    pub overlap_weight: f64,
    /// Weight of a backend's relative in-flight load.
    pub load_weight: f64,
    /// Retries on a different backend when connecting to the chosen one fails.
    /// Only connection failures are retried: once a backend has accepted the
    /// request, it may have started generating.
    pub max_retries: u32,
    /// Echo the chosen backend id on responses as `x-chalk-backend`.
    pub expose_backend_header: bool,
    pub max_body_bytes: usize,
    #[serde(with = "humantime_serde")]
    pub connect_timeout: Duration,
    pub bounded_load: BoundedLoadConfig,
}

/// Lets a request leave its session's backend when that backend is much
/// busier than average, without moving the session's binding.
///
/// Content-inferred sessions need this most: unrelated conversations with
/// identical openings share a key, and one busy agent session can fan out
/// many concurrent requests. Breaking affinity for one request costs a
/// partial prefill elsewhere; queueing behind a hot backend usually costs more.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct BoundedLoadConfig {
    /// A sticky backend is skipped when its in-flight requests (counting this
    /// one) would exceed `ceil(factor × average)` across healthy backends.
    /// With N backends a factor >= N never spills. `null` disables spilling.
    pub factor: Option<f64>,
    /// Which session sources may spill.
    pub applies_to: Vec<SessionSource>,
}

impl Default for BoundedLoadConfig {
    fn default() -> Self {
        Self {
            factor: Some(1.5),
            applies_to: vec![SessionSource::Inferred],
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            overlap_weight: 1.0,
            load_weight: 0.5,
            max_retries: 1,
            expose_backend_header: true,
            max_body_bytes: 64 * 1024 * 1024,
            connect_timeout: Duration::from_secs(2),
            bounded_load: BoundedLoadConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct KvConfig {
    /// Use the KV index when scoring backends.
    pub enabled: bool,
    /// What to do when an event batch sequence number skips ahead.
    pub on_gap: GapPolicy,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            on_gap: GapPolicy::Ignore,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GapPolicy {
    /// Keep the index. Missed removals leave stale entries (we may predict a hit
    /// that isn't there); missed stores leave holes. Both only cost efficiency.
    Ignore,
    /// Drop everything known about the backend and rebuild from new events.
    Clear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        let raw = include_str!("../config.example.yaml");
        let cfg: Config = serde_yaml::from_str(raw).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.backends.len(), 2);
        assert!(matches!(
            cfg.sessions.store,
            SessionStoreConfig::Redis { .. }
        ));
        assert_eq!(cfg.health.interval, Duration::from_secs(5));
        assert_eq!(cfg.sessions.derive.len(), 3);
        assert_eq!(cfg.routing.bounded_load.factor, Some(1.5));
    }

    #[test]
    fn rejects_duplicate_backends() {
        let cfg: Config = serde_yaml::from_str(
            "backends:\n  - {id: a, url: 'http://x'}\n  - {id: a, url: 'http://y'}\n",
        )
        .unwrap();
        assert!(cfg.validate().is_err());
    }
}
