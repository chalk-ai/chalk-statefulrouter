# chalk-statefulrouter

A stateful HTTP router for LLM inference backends (vLLM, SGLang). It does two
jobs:

1. **Session affinity.** Requests that share an `x-chalk-sticky-session-id` go
   to the same backend for as long as that backend is healthy. This holds even
   when the requests reach *different router replicas*.
2. **Cache-aware placement.** New sessions and sessionless requests go to the
   backend that already holds the longest prefix of the prompt in its KV
   cache, weighed against load. The router learns cache contents from the
   engines' KV cache event streams.

Each request is buffered just long enough to choose a backend. The response,
including SSE token streams, is passed through as it arrives.

```
                 ┌──────────────── router replica ────────────────┐
 client ──HTTP──▶│ session header ─▶ SessionStore ──(bound+healthy)─┼──▶ backend
                 │        │             ▲  (Redis: SET-NX / CAS)    │
                 │        ▼             │                           │
                 │  score candidates: KV overlap − load ────────────┤
                 │        ▲                                         │
                 │   KvIndex ◀── KV events (ZMQ / bus / HTTP)       │
                 │   health: active probes + passive ejection       │
                 └──────────────────────────────────────────────────┘
```

## Design

### Sticky sessions and `x-chalk-sticky-session-id`

- If a request carries `x-chalk-sticky-session-id`, that header is the session
  key. It must be 1–256 visible ASCII characters, or the router returns 400.
- If the header is missing and `sessions.generate_if_missing` is set (the
  default), the router mints a UUIDv7.
- The router forwards the session id to the backend and echoes it on the
  response, so a client can adopt a minted id for its next turn. It also
  returns `x-chalk-backend` (the chosen backend's id) unless
  `routing.expose_backend_header` is false.
- A binding has a sliding TTL (`sessions.ttl`). Every sticky hit refreshes it.

### Multiple frontends sharing one session

"Frontends" here means router replicas. Several of them can see requests for
the same session, and they must all agree on the backend. The session store
decides this, and there are three modes (`sessions.store.kind`):

| mode | how replicas agree | trade-off |
|---|---|---|
| `redis` (production) | Bindings live in Redis/Valkey. A new session is bound with an atomic **set-if-absent**: when two replicas race, both adopt the write that won. Moving a binding off a dead backend is an atomic **compare-and-swap** keyed on the old backend, so replicas that notice the failure at the same moment still converge on one replacement. | One Redis round trip per request (sub-ms in-cluster). The store fails fast (250ms response timeout), and on failure the router falls back to rendezvous for that request instead of failing it. |
| `rendezvous` | No stored state. Each replica runs highest-random-weight hashing over the backends it considers healthy. Replicas agree whenever their health views agree. Losing a backend moves only that backend's sessions. | Bindings ignore KV overlap and load. They briefly diverge while replicas disagree about health, and sessions move back when a backend recovers. |
| `memory` | Bindings live in the process. | Correct for a single replica only. |

Choices that keep sessions where their cache is:

- A session stays bound when its backend recovers from an outage. Its KV cache
  is now on the new backend.
- **Draining** a backend (`POST /backends/{id}/drain`) keeps its existing
  sessions but gives it no new ones. Use it for rolling restarts.
- A backend's identity is its configured `id`. The id is what gets stored and
  what KV events are keyed by, so it must be identical across replicas.

### KV cache event subscription

vLLM and SGLang both publish the same msgpack-encoded `KVEventBatch`
(`BlockStored`, `BlockRemoved`, `AllBlocksCleared`) over ZMQ, framed as
`[topic, seq, payload]` (vLLM `--kv-events-config`, SGLang
`--kv-events-config`). `src/kv/event.rs` decodes that format positionally and
tolerantly: it accepts omitted trailing fields, appended newer fields, and
int or bytes block hashes.

Events get into the router through `KvEventSource`s that feed one channel:

- `kv_events.zmq_endpoint` on a backend: the router subscribes directly to that
  engine's publisher (`zmq` feature, on by default, pure Rust).
- `POST /kv-events` on the admin port takes JSON envelopes, and
  `POST /kv-events/{backend}` takes a raw engine msgpack batch with an optional
  `x-chalk-kv-seq` header. A bridge from a shared bus pushes through these.
- To consume a shared bus (NATS, Kafka, Redis Streams, Pub/Sub) directly,
  implement `KvEventSource`. The trait is a single `run(sink)` method.

**The index** (`src/kv/index.rs`) maps blocks to the backends that hold them.
The router can't reproduce engine block hashes: vLLM seeds them per process and
can fold in multimodal or salt keys. So the index is keyed by the router's own
chained hash over each block's `token_ids` (which `BlockStored` carries). The
router keeps an engine-hash → router-hash map per backend, so it can apply
`BlockRemoved` and link a child block to its `parent_block_hash`.

For a request, the router hashes the prompt's full blocks once per block size
seen in events. vLLM and SGLang page sizes can differ. It then walks the chain
to find each backend's longest cached prefix. Each router replica builds its
own index from the same events, so the index needs no coordination.

When the stream is disrupted:

- **Sequence goes backwards:** the publisher restarted, so the router clears
  that backend.
- **Sequence gap:** by default the router keeps the index (`kv.on_gap: ignore`)
  and logs a warning. A missed removal can make the router predict a hit that
  isn't there; a missed store leaves a hole. Both only cost efficiency.
  `kv.on_gap: clear` drops the backend's index instead.
- **Block whose parent was never seen:** it is counted as orphaned and dropped.
  This happens when it was stored before the router subscribed.
- **LoRA blocks:** these go in their own hash namespace, so base-model requests
  never match them.

### Placement scoring

The router scores a backend when it binds a new session, rebinds one, or routes
a request with no session. Only backends that are healthy, not ejected, not
draining, and not already failed for this request are candidates:

```
score = overlap_weight × (cached prefix tokens / prompt tokens)
      − load_weight    × (in-flight on backend / max in-flight across candidates)
```

Ties break by rendezvous hash of the session id, so replicas with the same view
choose the same backend. `POST /route/explain` on the admin port shows the
scores for a request body without routing it.

**Prompt tokens.** The router matches against the index only when it knows the
request's exact token ids. Today those come from requests that carry them:
`prompt` as a token array, `input_ids`, or `prompt_token_ids`. Text and chat
requests still get affinity and load balancing, but no KV scoring. That needs a
`TokenSource` that runs the model's tokenizer and chat template. See next steps.

### Health checking

- **Active:** the router probes `health.path` (`/health` on both engines) on
  every backend every `health.interval`, with `unhealthy_threshold` and
  `healthy_threshold` hysteresis. A backend takes no traffic until its first
  probe passes. Backends added through the admin API are probed before the call
  returns.
- **Passive:** connect errors and 502/503/504 from a proxied request count
  against the backend. After `passive_failure_threshold` in a row, it is
  ejected for `ejection_duration` without waiting for the next probe.
- **Retry:** if connecting to the chosen backend fails, the router retries on
  another backend (`routing.max_retries`) and rebinds the session there through
  the same CAS. It never retries once the backend has accepted the request,
  because generation may have started.
- **Load:** each backend has an in-flight count that is held until the
  response stream finishes, not just until the first byte.

## Running

```bash
cargo run --release -- --config config.example.yaml
```

The proxy listens on `listen` (default `:8080`) and the admin API on
`admin_listen` (default `:8081`). Set log verbosity with `RUST_LOG`.

### Admin API

| | |
|---|---|
| `GET /healthz` | liveness |
| `GET /backends` | health, ejection, draining, in-flight, indexed KV blocks |
| `PUT /backends/{id}` `{"url", "kv_events"?}` | add or replace a backend |
| `DELETE /backends/{id}` | remove it, its subscription, and its index entries |
| `POST /backends/{id}/drain` · `/undrain` | stop or resume new sessions |
| `GET` / `DELETE /sessions/{id}` | inspect or drop a binding |
| `POST /kv-events` | JSON envelope(s): `{"backend","seq"?,"events":[{"type":"block_stored",...}]}` |
| `POST /kv-events/{backend}` | raw engine msgpack `KVEventBatch` |
| `POST /route/explain` | scores and current binding for a request body (honors the session header) |

## Testing

```bash
cargo test                                   # unit + e2e (mock backends, real ZMQ)
REDIS_URL=redis://localhost:6379 cargo test  # also run the Redis store against a real server
cargo clippy --all-targets --all-features -- -Dwarnings
```

The e2e tests (`tests/e2e.rs`) run two router replicas sharing a store against
mock HTTP backends. They cover cross-replica stickiness, failover convergence,
connect-failure retry and rebind, KV-steered placement, draining, SSE
pass-through with in-flight accounting, admin backend changes, and a real ZMQ
publisher in vLLM's wire format.

## Next steps / open questions

- **Tokenization for text and chat requests**, so KV scoring applies to them.
  Options: a HF `tokenizers` + chat-template `TokenSource` that has to match
  the engine exactly, or clients sending token ids.
- **ZMQ replay.** vLLM exposes a replay socket. Requesting the missed range on
  a gap would beat both `ignore` and `clear`.
- **Data-parallel engines.** `data_parallel_rank` is ignored today, so all of an
  engine's ranks are treated as one cache. Either target ranks explicitly or
  run one backend entry per rank.
- **Shared load signal.** In-flight counts are per replica. Scraping engine
  `/metrics` (`vllm:num_requests_waiting`, `sglang:num_queue_reqs`, KV usage)
  would give every replica the same view.
- **Spill-over.** Optionally send one request from an overloaded sticky backend
  elsewhere without rebinding the session.
- **Discovery.** Watch Kubernetes EndpointSlices instead of static config and
  the admin API.
- Prometheus metrics: routing reasons, cache-hit ratio, store latency and
  fallbacks, event lag.
