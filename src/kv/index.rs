//! Index of which backends hold which prefix blocks, built from KV events.
//!
//! Each router replica consumes the event bus independently and builds its own
//! copy; the index is deterministic given the same events, so it needs no
//! cross-replica coordination.

use std::collections::{BTreeSet, HashMap, HashSet};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::Serialize;

use super::{
    event::{EngineHash, KvEvent, KvEventEnvelope},
    hash::{self, RouterHash},
};
use crate::{backend::BackendId, config::GapPolicy};

pub struct KvIndex {
    /// Block -> backends holding it, with a refcount because two engine blocks
    /// on one backend can have equal token content (e.g. differing only in a
    /// cache salt the router can't see).
    blocks: DashMap<RouterHash, HashMap<BackendId, u32>>,
    backends: DashMap<BackendId, BackendBlocks>,
    /// Block sizes seen in events. Requests are hashed once per size.
    block_sizes: RwLock<BTreeSet<usize>>,
    on_gap: GapPolicy,
}

#[derive(Default)]
struct BackendBlocks {
    engine_to_router: HashMap<EngineHash, RouterHash>,
    last_seq: Option<u64>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct ApplyStats {
    pub stored: usize,
    pub removed: usize,
    /// Blocks dropped because their parent was never seen (e.g. it was stored
    /// before this router subscribed).
    pub orphaned: usize,
    pub cleared: bool,
    pub gap: bool,
}

/// How much of a request's prompt each backend already has cached.
#[derive(Debug, Default, Clone)]
pub struct Overlap {
    pub request_tokens: usize,
    pub matched_tokens: HashMap<BackendId, usize>,
}

impl Overlap {
    pub fn ratio(&self, backend: &BackendId) -> f64 {
        if self.request_tokens == 0 {
            return 0.0;
        }
        self.matched_tokens.get(backend).copied().unwrap_or(0) as f64 / self.request_tokens as f64
    }
}

impl KvIndex {
    pub fn new(on_gap: GapPolicy) -> Self {
        Self {
            blocks: DashMap::new(),
            backends: DashMap::new(),
            block_sizes: RwLock::new(BTreeSet::new()),
            on_gap,
        }
    }

    pub fn apply(&self, envelope: &KvEventEnvelope) -> ApplyStats {
        let backend = envelope.backend_id();
        let mut stats = ApplyStats::default();
        let mut state = self.backends.entry(backend.clone()).or_default();

        if let Some(seq) = envelope.seq {
            match state.last_seq {
                // Sequence went backwards: the publisher restarted, and with it
                // the engine and its cache.
                Some(last) if seq <= last => {
                    tracing::info!(%backend, last, seq, "kv event sequence reset; clearing backend");
                    self.clear_locked(&backend, &mut state);
                    stats.cleared = true;
                }
                Some(last) if seq > last + 1 => {
                    tracing::warn!(%backend, last, seq, "kv event gap");
                    stats.gap = true;
                    if self.on_gap == GapPolicy::Clear {
                        self.clear_locked(&backend, &mut state);
                        stats.cleared = true;
                    }
                }
                _ => {}
            }
            state.last_seq = Some(seq);
        }

        for event in &envelope.events {
            match event {
                KvEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    lora_id,
                } => {
                    let mut parent = match parent_block_hash {
                        None => hash::root(*lora_id),
                        Some(p) => match state.engine_to_router.get(p) {
                            Some(&h) => h,
                            None => {
                                stats.orphaned += block_hashes.len();
                                continue;
                            }
                        },
                    };
                    if !self.block_sizes.read().contains(block_size) {
                        self.block_sizes.write().insert(*block_size);
                    }
                    for (engine_hash, tokens) in
                        block_hashes.iter().zip(token_ids.chunks_exact(*block_size))
                    {
                        let router_hash = hash::chain(parent, tokens);
                        parent = router_hash;
                        match state.engine_to_router.insert(*engine_hash, router_hash) {
                            Some(old) if old == router_hash => continue,
                            Some(old) => self.release(&backend, old),
                            None => {}
                        }
                        *self
                            .blocks
                            .entry(router_hash)
                            .or_default()
                            .entry(backend.clone())
                            .or_insert(0) += 1;
                        stats.stored += 1;
                    }
                }
                KvEvent::BlockRemoved { block_hashes } => {
                    for engine_hash in block_hashes {
                        if let Some(router_hash) = state.engine_to_router.remove(engine_hash) {
                            self.release(&backend, router_hash);
                            stats.removed += 1;
                        }
                    }
                }
                KvEvent::AllBlocksCleared => {
                    self.clear_locked(&backend, &mut state);
                    stats.cleared = true;
                }
            }
        }
        stats
    }

    fn release(&self, backend: &BackendId, router_hash: RouterHash) {
        if let dashmap::Entry::Occupied(mut holders) = self.blocks.entry(router_hash) {
            if let Some(count) = holders.get_mut().get_mut(backend) {
                *count -= 1;
                if *count == 0 {
                    holders.get_mut().remove(backend);
                }
            }
            if holders.get().is_empty() {
                holders.remove();
            }
        }
    }

    fn clear_locked(&self, backend: &BackendId, state: &mut BackendBlocks) {
        for (_, router_hash) in state.engine_to_router.drain() {
            self.release(backend, router_hash);
        }
    }

    /// Forget everything about a backend (it was removed from the registry).
    pub fn remove_backend(&self, backend: &str) {
        if let Some((id, mut state)) = self.backends.remove(backend) {
            self.clear_locked(&id, &mut state);
        }
    }

    pub fn block_count(&self, backend: &str) -> usize {
        self.backends
            .get(backend)
            .map(|s| s.engine_to_router.len())
            .unwrap_or(0)
    }

    /// Longest cached prefix of `tokens` on each backend.
    pub fn overlap(&self, tokens: &[u32]) -> Overlap {
        let mut overlap = Overlap {
            request_tokens: tokens.len(),
            matched_tokens: HashMap::new(),
        };
        let sizes: Vec<usize> = self.block_sizes.read().iter().copied().collect();
        for block_size in sizes {
            let mut alive: Option<HashSet<BackendId>> = None;
            for (depth, h) in hash::sequence(tokens, block_size, None).iter().enumerate() {
                let Some(holders) = self.blocks.get(h) else {
                    break;
                };
                let next: HashSet<BackendId> = match alive {
                    None => holders.keys().cloned().collect(),
                    Some(prev) => prev
                        .into_iter()
                        .filter(|b| holders.contains_key(b))
                        .collect(),
                };
                if next.is_empty() {
                    break;
                }
                let matched = (depth + 1) * block_size;
                for b in &next {
                    let best = overlap.matched_tokens.entry(b.clone()).or_insert(0);
                    *best = (*best).max(matched);
                }
                alive = Some(next);
            }
        }
        overlap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(backend: &str, seq: Option<u64>, events: Vec<KvEvent>) -> KvEventEnvelope {
        KvEventEnvelope {
            backend: backend.into(),
            seq,
            events,
        }
    }

    fn stored(hashes: &[u64], parent: Option<u64>, tokens: &[u32], block_size: usize) -> KvEvent {
        KvEvent::BlockStored {
            block_hashes: hashes.to_vec(),
            parent_block_hash: parent,
            token_ids: tokens.to_vec(),
            block_size,
            lora_id: None,
        }
    }

    fn matched(idx: &KvIndex, tokens: &[u32], backend: &str) -> usize {
        idx.overlap(tokens)
            .matched_tokens
            .get(backend)
            .copied()
            .unwrap_or(0)
    }

    #[test]
    fn store_chain_and_lookup() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        idx.apply(&env(
            "a",
            None,
            vec![stored(&[10, 11], None, &[1, 2, 3, 4], 2)],
        ));
        // Child stored in a later event, linked by engine parent hash.
        idx.apply(&env("a", None, vec![stored(&[12], Some(11), &[5, 6], 2)]));
        idx.apply(&env("b", None, vec![stored(&[99], None, &[1, 2], 2)]));

        let tokens = [1, 2, 3, 4, 5, 6, 7];
        assert_eq!(matched(&idx, &tokens, "a"), 6);
        assert_eq!(matched(&idx, &tokens, "b"), 2);
        assert_eq!(matched(&idx, &[9, 9, 9], "a"), 0);
        assert!((idx.overlap(&tokens).ratio(&"a".into()) - 6.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn removal_truncates_prefix() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        idx.apply(&env(
            "a",
            None,
            vec![stored(&[10, 11, 12], None, &[1, 2, 3, 4, 5, 6], 2)],
        ));
        idx.apply(&env(
            "a",
            None,
            vec![KvEvent::BlockRemoved {
                block_hashes: vec![11],
            }],
        ));
        assert_eq!(matched(&idx, &[1, 2, 3, 4, 5, 6], "a"), 2);
        assert_eq!(idx.block_count("a"), 2);
    }

    #[test]
    fn orphans_are_dropped() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        let stats = idx.apply(&env("a", None, vec![stored(&[12], Some(11), &[5, 6], 2)]));
        assert_eq!(stats.orphaned, 1);
        assert_eq!(idx.block_count("a"), 0);
    }

    #[test]
    fn refcounted_duplicate_content() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        // Two engine blocks with identical tokens (different salt engine-side).
        idx.apply(&env(
            "a",
            None,
            vec![
                stored(&[1], None, &[7, 7], 2),
                stored(&[2], None, &[7, 7], 2),
            ],
        ));
        idx.apply(&env(
            "a",
            None,
            vec![KvEvent::BlockRemoved {
                block_hashes: vec![1],
            }],
        ));
        assert_eq!(matched(&idx, &[7, 7], "a"), 2);
        idx.apply(&env(
            "a",
            None,
            vec![KvEvent::BlockRemoved {
                block_hashes: vec![2],
            }],
        ));
        assert_eq!(matched(&idx, &[7, 7], "a"), 0);
        assert!(idx.blocks.is_empty());
    }

    #[test]
    fn clear_and_sequence_reset() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        idx.apply(&env("a", Some(5), vec![stored(&[1], None, &[1, 2], 2)]));
        let stats = idx.apply(&env("a", Some(7), vec![]));
        assert!(stats.gap && !stats.cleared);
        assert_eq!(idx.block_count("a"), 1);
        let stats = idx.apply(&env("a", Some(0), vec![]));
        assert!(stats.cleared);
        assert_eq!(idx.block_count("a"), 0);

        idx.apply(&env("a", Some(1), vec![stored(&[1], None, &[1, 2], 2)]));
        idx.apply(&env("a", Some(2), vec![KvEvent::AllBlocksCleared]));
        assert_eq!(matched(&idx, &[1, 2], "a"), 0);
    }

    #[test]
    fn gap_clear_policy() {
        let idx = KvIndex::new(GapPolicy::Clear);
        idx.apply(&env("a", Some(1), vec![stored(&[1], None, &[1, 2], 2)]));
        let stats = idx.apply(&env("a", Some(3), vec![]));
        assert!(stats.cleared);
        assert_eq!(idx.block_count("a"), 0);
    }

    #[test]
    fn mixed_block_sizes() {
        let idx = KvIndex::new(GapPolicy::Ignore);
        idx.apply(&env(
            "vllm",
            None,
            vec![stored(&[1, 2], None, &[1, 2, 3, 4], 2)],
        ));
        idx.apply(&env(
            "sglang",
            None,
            vec![stored(&[1, 2, 3], None, &[1, 2, 3], 1)],
        ));
        let tokens = [1, 2, 3, 4];
        assert_eq!(matched(&idx, &tokens, "vllm"), 4);
        assert_eq!(matched(&idx, &tokens, "sglang"), 3);
    }
}
