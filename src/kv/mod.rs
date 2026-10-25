//! KV cache event subscription.
//!
//! Events reach the router through one or more [`KvEventSource`]s, all feeding a
//! single channel drained into the [`index::KvIndex`]. Sources so far:
//!
//! - [`zmq::ZmqSource`]: subscribe directly to a vLLM / SGLang ZMQ publisher.
//! - The admin API's `POST /kv-events` (JSON envelopes) and
//!   `POST /kv-events/{backend}` (raw engine msgpack), for a bus bridge to push
//!   into.
//!
//! A shared bus (NATS, Kafka, Redis Streams, Pub/Sub) is another source:
//! implement [`KvEventSource`] to read from it and send [`KvEventEnvelope`]s.

pub mod event;
pub mod hash;
pub mod index;
#[cfg(feature = "zmq")]
pub mod zmq;

use std::sync::Arc;

use dashmap::DashMap;
use tokio::{sync::mpsc, task::JoinHandle};

pub use event::{KvEvent, KvEventEnvelope};
pub use index::KvIndex;

use crate::backend::{Backend, BackendId, BackendRegistry};

pub type KvEventSink = mpsc::Sender<KvEventEnvelope>;

#[async_trait::async_trait]
pub trait KvEventSource: Send + 'static {
    /// Deliver events to `sink` until the source is exhausted or the sink
    /// closes. Implementations own their reconnect / backoff.
    async fn run(self: Box<Self>, sink: KvEventSink);
}

/// Drain the event channel into the index. A single consumer keeps each
/// backend's events in order.
pub async fn run_consumer(
    mut rx: mpsc::Receiver<KvEventEnvelope>,
    index: Arc<KvIndex>,
    registry: Arc<BackendRegistry>,
) {
    while let Some(envelope) = rx.recv().await {
        if registry.get(&envelope.backend).is_none() {
            tracing::debug!(backend = %envelope.backend, "dropping kv events for unknown backend");
            continue;
        }
        let stats = index.apply(&envelope);
        tracing::trace!(backend = %envelope.backend, ?stats, "applied kv events");
    }
}

/// Per-backend event subscriptions, started and stopped as backends come and go.
pub struct Subscriptions {
    sink: KvEventSink,
    tasks: DashMap<BackendId, JoinHandle<()>>,
}

impl Subscriptions {
    pub fn new(sink: KvEventSink) -> Self {
        Self {
            sink,
            tasks: DashMap::new(),
        }
    }

    pub fn sink(&self) -> KvEventSink {
        self.sink.clone()
    }

    /// (Re)start the subscription for `backend` per its config.
    pub fn sync(&self, backend: &Backend) {
        self.stop(&backend.id);
        let Some(endpoint) = &backend.config.kv_events else {
            return;
        };
        #[cfg(feature = "zmq")]
        {
            let source = Box::new(zmq::ZmqSource {
                backend: backend.id.to_string(),
                endpoint: endpoint.zmq_endpoint.clone(),
                topic: endpoint.topic.clone(),
            });
            self.tasks.insert(
                backend.id.clone(),
                tokio::spawn(source.run(self.sink.clone())),
            );
        }
        #[cfg(not(feature = "zmq"))]
        tracing::warn!(
            backend = %backend.id,
            endpoint = %endpoint.zmq_endpoint,
            "kv_events configured but built without the `zmq` feature"
        );
    }

    pub fn stop(&self, backend: &str) {
        if let Some((_, task)) = self.tasks.remove(backend) {
            task.abort();
        }
    }
}
