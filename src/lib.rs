//! A stateful router for LLM inference backends.
//!
//! Requests carrying the same [`STICKY_SESSION_HEADER`] go to the same backend,
//! across every router replica, for as long as that backend is healthy. New
//! sessions and sessionless requests go to the backend with the most of the
//! prompt already in its KV cache (learned from engine KV events), weighed
//! against load.

pub mod admin;
pub mod backend;
pub mod config;
pub mod health;
pub mod kv;
pub mod proxy;
pub mod routing;
pub mod session;
pub mod tokens;

use std::{sync::Arc, time::Duration};

use axum::http::HeaderName;
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    backend::BackendRegistry,
    config::{Config, SessionStoreConfig},
    kv::{KvIndex, Subscriptions},
    routing::Router,
    session::{MemoryStore, RedisStore, SessionStore},
    tokens::{RequestTokenIds, TokenSource},
};

pub const STICKY_SESSION_HEADER: &str = "x-chalk-sticky-session-id";

pub struct AppState {
    pub router: Router,
    pub client: reqwest::Client,
    pub tokens: Box<dyn TokenSource>,
    pub subscriptions: Subscriptions,
    pub session_header: HeaderName,
}

/// A configured router plus the background tasks it needs (health checking,
/// KV event consumption and subscriptions, session sweeping).
pub struct Runtime {
    pub state: Arc<AppState>,
    pub tasks: JoinSet<()>,
}

impl Runtime {
    pub async fn new(config: Config) -> anyhow::Result<Self> {
        match &config.sessions.store {
            SessionStoreConfig::Memory => {
                let store = Arc::new(MemoryStore::new());
                let mut runtime = Self::with_store(config, Some(store.clone()))?;
                runtime.sweep_memory_store(store, Duration::from_secs(60));
                Ok(runtime)
            }
            SessionStoreConfig::Redis { url, key_prefix } => {
                let store = Arc::new(RedisStore::connect(url, key_prefix).await?);
                Self::with_store(config, Some(store))
            }
            SessionStoreConfig::Rendezvous => Self::with_store(config, None),
        }
    }

    /// Like [`Runtime::new`] with an explicit store, e.g. one shared by several
    /// in-process routers in tests.
    pub fn with_store(
        config: Config,
        store: Option<Arc<dyn SessionStore>>,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let session_header = HeaderName::try_from(config.sessions.header.as_str())?;
        let config = Arc::new(config);
        let registry = Arc::new(BackendRegistry::new(config.backends.clone()));
        let index = Arc::new(KvIndex::new(config.kv.on_gap));
        let client = reqwest::Client::builder()
            .connect_timeout(config.routing.connect_timeout)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()?;

        let mut tasks = JoinSet::new();
        let (kv_tx, kv_rx) = mpsc::channel(4096);
        tasks.spawn(kv::run_consumer(kv_rx, index.clone(), registry.clone()));
        tasks.spawn(health::run(
            registry.clone(),
            client.clone(),
            config.health.clone(),
        ));

        let subscriptions = Subscriptions::new(kv_tx);
        for backend in registry.all() {
            subscriptions.sync(&backend);
        }

        let state = Arc::new(AppState {
            router: Router {
                registry,
                sessions: store,
                index,
                config,
            },
            client,
            tokens: Box::new(RequestTokenIds),
            subscriptions,
            session_header,
        });
        Ok(Self { state, tasks })
    }

    /// Bindings in a [`MemoryStore`] expire lazily; sweep abandoned ones.
    fn sweep_memory_store(&mut self, store: Arc<MemoryStore>, every: Duration) {
        self.tasks.spawn(async move {
            let mut ticker = tokio::time::interval(every);
            loop {
                ticker.tick().await;
                store.sweep();
            }
        });
    }

    pub fn proxy_app(&self) -> axum::Router {
        axum::Router::new()
            .fallback(proxy::proxy)
            .with_state(self.state.clone())
    }

    pub fn admin_app(&self) -> axum::Router {
        admin::router(self.state.clone())
    }
}
