//! Active health checking. Passive checking lives in the proxy, which reports
//! request outcomes through [`crate::backend::Backend::record_request`].

use std::sync::Arc;

use futures::future::join_all;

use crate::{
    backend::{Backend, BackendRegistry},
    config::HealthConfig,
};

pub async fn probe(backend: &Backend, client: &reqwest::Client, cfg: &HealthConfig) {
    let url = format!("{}{}", backend.url, cfg.path);
    let result = match client.get(&url).timeout(cfg.timeout).send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(format!("health probe returned {}", resp.status())),
        Err(err) => Err(format!("health probe failed: {err}")),
    };
    backend.record_probe(result, cfg);
}

/// Probe every registered backend once.
pub async fn probe_all(registry: &BackendRegistry, client: &reqwest::Client, cfg: &HealthConfig) {
    join_all(registry.all().iter().map(|b| probe(b, client, cfg))).await;
}

/// Probe immediately, then every `cfg.interval`. Backends added at runtime are
/// picked up on the next tick.
pub async fn run(registry: Arc<BackendRegistry>, client: reqwest::Client, cfg: HealthConfig) {
    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        probe_all(&registry, &client, &cfg).await;
    }
}
