//! The router's own block hashing.
//!
//! Engine block hashes can't be reproduced by the router (vLLM seeds them per
//! process and may fold in multimodal/salt keys), so the index is keyed by a
//! hash the router computes itself from token ids: both from the `token_ids`
//! carried in `BlockStored` events and from incoming request tokens. A block's
//! hash chains in its parent's, so equal hashes mean equal prefixes.

use xxhash_rust::xxh3::xxh3_64_with_seed;

pub type RouterHash = u64;

const ROOT_SEED: u64 = 0x6368_616c_6b6b_7631; // "chalkkv1"

/// Seed for the first block of a sequence. Blocks computed under a LoRA
/// adapter are a different cache from base-model blocks.
pub fn root(lora_id: Option<u64>) -> RouterHash {
    match lora_id {
        None => ROOT_SEED,
        Some(id) => xxh3_64_with_seed(&id.to_le_bytes(), ROOT_SEED),
    }
}

pub fn chain(parent: RouterHash, tokens: &[u32]) -> RouterHash {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    xxh3_64_with_seed(&bytes, parent)
}

/// Hashes of every full block of `tokens`. A trailing partial block is never
/// cached by the engine, so it is not hashed.
pub fn sequence(tokens: &[u32], block_size: usize, lora_id: Option<u64>) -> Vec<RouterHash> {
    let mut parent = root(lora_id);
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            parent = chain(parent, block);
            parent
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_shares_hashes() {
        let a = sequence(&[1, 2, 3, 4, 5, 6, 7], 2, None);
        let b = sequence(&[1, 2, 3, 4, 9, 9], 2, None);
        assert_eq!(a.len(), 3);
        assert_eq!(a[..2], b[..2]);
        assert_ne!(a[2], b[2]);
    }

    #[test]
    fn position_matters() {
        // Same block contents at a different depth must hash differently.
        let a = sequence(&[1, 2, 1, 2], 2, None);
        assert_ne!(a[0], a[1]);
    }

    #[test]
    fn lora_namespaces() {
        assert_ne!(sequence(&[1, 2], 2, None), sequence(&[1, 2], 2, Some(1)));
    }
}
