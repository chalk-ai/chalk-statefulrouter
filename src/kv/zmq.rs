//! Direct subscription to an engine's ZMQ KV event publisher.
//!
//! vLLM's `ZmqEventPublisher` (and SGLang's port of it) sends multipart
//! messages `[topic, seq: u64 big-endian, payload: msgpack KVEventBatch]`.

use std::time::Duration;

use zeromq::{Socket, SocketRecv, SubSocket};

use super::{event::decode_msgpack_batch, KvEventEnvelope, KvEventSink, KvEventSource};

pub struct ZmqSource {
    pub backend: String,
    pub endpoint: String,
    pub topic: String,
}

#[async_trait::async_trait]
impl KvEventSource for ZmqSource {
    async fn run(self: Box<Self>, sink: KvEventSink) {
        let mut backoff = Duration::from_millis(250);
        loop {
            match self.subscribe_once(&sink).await {
                Ok(()) => return, // sink closed
                Err(err) => {
                    tracing::warn!(
                        backend = %self.backend,
                        endpoint = %self.endpoint,
                        error = %err,
                        "kv event subscription failed; retrying in {backoff:?}"
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(10));
                }
            }
        }
    }
}

impl ZmqSource {
    async fn subscribe_once(&self, sink: &KvEventSink) -> anyhow::Result<()> {
        let mut socket = SubSocket::new();
        socket.connect(&self.endpoint).await?;
        socket.subscribe(&self.topic).await?;
        tracing::info!(backend = %self.backend, endpoint = %self.endpoint, "subscribed to kv events");
        loop {
            let frames = socket.recv().await?.into_vec();
            let (seq, payload) = match frames.as_slice() {
                [_topic, seq, payload] => (parse_seq(seq), payload),
                [seq, payload] => (parse_seq(seq), payload),
                _ => {
                    tracing::warn!(backend = %self.backend, frames = frames.len(), "unexpected kv event frame count");
                    continue;
                }
            };
            let events = match decode_msgpack_batch(payload) {
                Ok(events) => events,
                Err(err) => {
                    tracing::warn!(backend = %self.backend, error = %err, "undecodable kv event batch");
                    continue;
                }
            };
            let envelope = KvEventEnvelope {
                backend: self.backend.clone(),
                seq,
                events,
            };
            if sink.send(envelope).await.is_err() {
                return Ok(());
            }
        }
    }
}

fn parse_seq(frame: &[u8]) -> Option<u64> {
    Some(u64::from_be_bytes(frame.try_into().ok()?))
}
