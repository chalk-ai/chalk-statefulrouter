//! KV cache events, normalized across engines, plus decoders for the wire
//! formats engines emit.
//!
//! vLLM (`vllm/distributed/kv_events.py`) and SGLang
//! (`sglang/srt/disaggregation/kv_events.py`) publish the same msgspec
//! structures, encoded as msgpack with `array_like=True, tag=True`:
//!
//! ```text
//! KVEventBatch     = [ts: float, events: [Event...], data_parallel_rank?: int]
//! BlockStored      = ["BlockStored", block_hashes: [Hash], parent_block_hash: Hash|nil,
//!                     token_ids: [int], block_size: int, lora_id?: int|nil, medium?: str|nil, ...]
//! BlockRemoved     = ["BlockRemoved", block_hashes: [Hash], medium?: str|nil]
//! AllBlocksCleared = ["AllBlocksCleared"]
//! Hash             = int | bytes          (older vLLM: int; newer: sha256 bytes)
//! ```
//!
//! Trailing fields may be omitted (`omit_defaults`) and newer engine versions
//! append fields, so decoding is positional and ignores anything extra.

use rmpv::Value;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::backend::BackendId;

/// An engine's own block hash, folded to 64 bits. Only used to correlate
/// `BlockStored` with later `BlockRemoved` / child `parent_block_hash` from the
/// same backend, so a lossy fold is fine.
pub type EngineHash = u64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KvEvent {
    BlockStored {
        #[serde(deserialize_with = "de_engine_hashes")]
        block_hashes: Vec<EngineHash>,
        #[serde(default, deserialize_with = "de_opt_engine_hash")]
        parent_block_hash: Option<EngineHash>,
        token_ids: Vec<u32>,
        block_size: usize,
        #[serde(default)]
        lora_id: Option<u64>,
    },
    BlockRemoved {
        #[serde(deserialize_with = "de_engine_hashes")]
        block_hashes: Vec<EngineHash>,
    },
    AllBlocksCleared,
}

/// One batch of events from one backend, as it arrives off the bus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEventEnvelope {
    pub backend: String,
    /// Publisher sequence number, used to detect gaps and engine restarts.
    #[serde(default)]
    pub seq: Option<u64>,
    pub events: Vec<KvEvent>,
}

impl KvEventEnvelope {
    pub fn backend_id(&self) -> BackendId {
        BackendId::from(self.backend.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("msgpack: {0}")]
    Msgpack(#[from] rmpv::decode::Error),
    #[error("malformed event batch: {0}")]
    Malformed(&'static str),
}

/// Decode a vLLM/SGLang msgpack `KVEventBatch`. Unknown event tags are skipped.
pub fn decode_msgpack_batch(mut payload: &[u8]) -> Result<Vec<KvEvent>, DecodeError> {
    let value = rmpv::decode::read_value(&mut payload)?;
    let batch = value
        .as_array()
        .ok_or(DecodeError::Malformed("batch is not an array"))?;
    let events = batch
        .get(1)
        .and_then(Value::as_array)
        .ok_or(DecodeError::Malformed("batch has no events array"))?;
    let mut out = Vec::with_capacity(events.len());
    for event in events {
        if let Some(e) = decode_event(event)? {
            out.push(e);
        }
    }
    Ok(out)
}

fn decode_event(value: &Value) -> Result<Option<KvEvent>, DecodeError> {
    let fields = value
        .as_array()
        .ok_or(DecodeError::Malformed("event is not an array"))?;
    let tag = fields
        .first()
        .and_then(Value::as_str)
        .ok_or(DecodeError::Malformed("event has no tag"))?;
    let field = |i: usize| fields.get(i).unwrap_or(&Value::Nil);
    Ok(Some(match tag {
        "BlockStored" => KvEvent::BlockStored {
            block_hashes: engine_hashes(field(1))?,
            parent_block_hash: opt_engine_hash(field(2))?,
            token_ids: field(3)
                .as_array()
                .ok_or(DecodeError::Malformed("token_ids is not an array"))?
                .iter()
                .map(|t| t.as_u64().map(|t| t as u32))
                .collect::<Option<_>>()
                .ok_or(DecodeError::Malformed("token id is not an integer"))?,
            block_size: field(4)
                .as_u64()
                .filter(|&b| b > 0)
                .ok_or(DecodeError::Malformed(
                    "block_size is not a positive integer",
                ))? as usize,
            lora_id: field(5).as_u64(),
        },
        "BlockRemoved" => KvEvent::BlockRemoved {
            block_hashes: engine_hashes(field(1))?,
        },
        "AllBlocksCleared" => KvEvent::AllBlocksCleared,
        other => {
            tracing::debug!(tag = other, "skipping unknown kv event");
            return Ok(None);
        }
    }))
}

fn engine_hashes(value: &Value) -> Result<Vec<EngineHash>, DecodeError> {
    value
        .as_array()
        .ok_or(DecodeError::Malformed("block_hashes is not an array"))?
        .iter()
        .map(|v| opt_engine_hash(v)?.ok_or(DecodeError::Malformed("null block hash")))
        .collect()
}

fn opt_engine_hash(value: &Value) -> Result<Option<EngineHash>, DecodeError> {
    match value {
        Value::Nil => Ok(None),
        // Python's hash() is signed; reinterpret rather than reject negatives.
        Value::Integer(i) => Ok(i.as_u64().or_else(|| i.as_i64().map(|i| i as u64))),
        Value::Binary(b) => Ok(Some(xxh3_64(b))),
        Value::String(s) => Ok(Some(xxh3_64(s.as_bytes()))),
        _ => Err(DecodeError::Malformed("block hash is not int or bytes")),
    }
}

/// JSON form of an engine hash: an integer, or a string (e.g. hex of the
/// engine's bytes hash) that is folded the same way msgpack bytes are.
#[derive(Deserialize)]
#[serde(untagged)]
enum JsonHash {
    UInt(u64),
    Int(i64),
    Str(String),
}

impl JsonHash {
    fn fold(self) -> EngineHash {
        match self {
            JsonHash::UInt(u) => u,
            JsonHash::Int(i) => i as u64,
            JsonHash::Str(s) => xxh3_64(s.as_bytes()),
        }
    }
}

fn de_engine_hashes<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<EngineHash>, D::Error> {
    Ok(Vec::<JsonHash>::deserialize(d)?
        .into_iter()
        .map(JsonHash::fold)
        .collect())
}

fn de_opt_engine_hash<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<EngineHash>, D::Error> {
    Ok(Option::<JsonHash>::deserialize(d)?.map(JsonHash::fold))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Encode a batch exactly the way msgspec does for vLLM's structs.
    pub fn vllm_batch(events: Vec<Value>) -> Vec<u8> {
        let batch = Value::Array(vec![Value::F64(1.5), Value::Array(events), Value::Nil]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &batch).unwrap();
        buf
    }

    pub fn stored(hashes: Vec<Value>, parent: Value, tokens: &[u32], block_size: u64) -> Value {
        Value::Array(vec![
            "BlockStored".into(),
            Value::Array(hashes),
            parent,
            Value::Array(tokens.iter().map(|&t| Value::from(t)).collect()),
            block_size.into(),
            Value::Nil,
            "GPU".into(),
        ])
    }

    #[test]
    fn decodes_vllm_msgpack() {
        let payload = vllm_batch(vec![
            stored(
                vec![1.into(), Value::from(-2i64)],
                Value::Nil,
                &[1, 2, 3, 4],
                2,
            ),
            Value::Array(vec![
                "BlockRemoved".into(),
                Value::Array(vec![Value::Binary(vec![0xab; 32])]),
            ]),
            Value::Array(vec!["AllBlocksCleared".into()]),
            Value::Array(vec!["SomethingNew".into(), 7.into()]),
        ]);
        let events = decode_msgpack_batch(&payload).unwrap();
        assert_eq!(
            events,
            vec![
                KvEvent::BlockStored {
                    block_hashes: vec![1, (-2i64) as u64],
                    parent_block_hash: None,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 2,
                    lora_id: None,
                },
                KvEvent::BlockRemoved {
                    block_hashes: vec![xxh3_64(&[0xab; 32])],
                },
                KvEvent::AllBlocksCleared,
            ]
        );
    }

    #[test]
    fn decodes_json_envelope() {
        let env: KvEventEnvelope = serde_json::from_str(
            r#"{"backend":"b","seq":3,"events":[
                {"type":"block_stored","block_hashes":[1,"abcd"],"parent_block_hash":null,
                 "token_ids":[1,2],"block_size":1},
                {"type":"block_removed","block_hashes":[1]},
                {"type":"all_blocks_cleared"}]}"#,
        )
        .unwrap();
        assert_eq!(env.seq, Some(3));
        assert_eq!(env.events.len(), 3);
        assert!(matches!(
            &env.events[0],
            KvEvent::BlockStored { block_hashes, .. } if block_hashes[1] == xxh3_64(b"abcd")
        ));
    }
}
