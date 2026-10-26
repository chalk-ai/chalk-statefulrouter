# chalk-statefulrouter

Session- and KV-cache-aware HTTP router for vLLM / SGLang backends. Read the
README's Design section before changing routing, session-store, or KV-index
behavior: the cross-replica guarantees depend on store operations staying
atomic (set-if-absent for new bindings, compare-and-swap for rebinding).

## Layout

- `src/routing.rs` — backend selection (sticky → score → bind)
- `src/session/` — `SessionStore` trait; memory, Redis (Lua CAS) impls;
  `derive.rs` resolves a request's session key (header → derive rules → mint)
- `src/inspect.rs` — parses prompts out of chat / messages / responses /
  completions / generate bodies
- `src/kv/` — event types + vLLM/SGLang msgpack decoding, router-side block
  hashing, the prefix index, ZMQ source
- `src/backend.rs`, `src/health.rs` — registry, active + passive health
- `src/proxy.rs` — the forwarded API; `src/admin.rs` — admin port

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -Dwarnings
cargo test                                  # REDIS_URL=... to include Redis tests
cargo test --no-default-features            # without the zmq feature
```

New store implementations must pass `session::conformance::run`.
