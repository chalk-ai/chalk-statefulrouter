//! End-to-end tests: real HTTP mock backends behind real router listeners.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::get,
    Json,
};
use chalk_statefulrouter::{
    config::{BackendConfig, Config, HealthConfig, KvEventsEndpoint},
    session::{MemoryStore, SessionStore},
    Runtime, STICKY_SESSION_HEADER,
};
use futures::stream;
use serde_json::{json, Value};

struct MockBackend {
    id: String,
    addr: SocketAddr,
    healthy: Arc<AtomicBool>,
}

impl MockBackend {
    async fn start(id: &str) -> Self {
        let healthy = Arc::new(AtomicBool::new(true));
        let state = (id.to_string(), healthy.clone());
        let app = axum::Router::new()
            .route(
                "/health",
                get(
                    |State((_, healthy)): State<(String, Arc<AtomicBool>)>| async move {
                        if healthy.load(Ordering::SeqCst) {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    },
                ),
            )
            .route(
                "/v1/stream",
                get(|| async {
                    let chunks = stream::unfold(0, |i| async move {
                        if i == 3 {
                            return None;
                        }
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Some((Ok::<_, std::io::Error>(format!("data: {i}\n\n")), i + 1))
                    });
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(chunks),
                    )
                }),
            )
            .fallback(
                |State((id, _)): State<(String, Arc<AtomicBool>)>,
                 headers: HeaderMap,
                 uri: axum::http::Uri| async move {
                    Json(json!({
                        "backend": id,
                        "path": uri.path(),
                        "session": headers.get(STICKY_SESSION_HEADER).and_then(|v| v.to_str().ok()),
                    }))
                },
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            id: id.to_string(),
            addr,
            healthy,
        }
    }

    fn config(&self) -> BackendConfig {
        BackendConfig {
            id: self.id.clone(),
            url: format!("http://{}", self.addr),
            kv_events: None,
        }
    }
}

struct TestRouter {
    runtime: Runtime,
    proxy: String,
    admin: String,
}

fn config(backends: Vec<BackendConfig>, health_interval: Duration) -> Config {
    Config {
        backends,
        health: HealthConfig {
            interval: health_interval,
            timeout: Duration::from_millis(500),
            unhealthy_threshold: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

async fn start_router(config: Config, store: Option<Arc<dyn SessionStore>>) -> TestRouter {
    let runtime = Runtime::with_store(config, store).unwrap();
    let proxy = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let admin = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let urls = (
        format!("http://{}", proxy.local_addr().unwrap()),
        format!("http://{}", admin.local_addr().unwrap()),
    );
    let (proxy_app, admin_app) = (runtime.proxy_app(), runtime.admin_app());
    tokio::spawn(async move { axum::serve(proxy, proxy_app).await.unwrap() });
    tokio::spawn(async move { axum::serve(admin, admin_app).await.unwrap() });
    TestRouter {
        runtime,
        proxy: urls.0,
        admin: urls.1,
    }
}

impl TestRouter {
    async fn backends(&self) -> Vec<Value> {
        reqwest::get(format!("{}/backends", self.admin))
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    /// Wait until the router's view of each backend's health matches.
    async fn wait_health(&self, expected: &[(&str, bool)]) {
        for _ in 0..200 {
            let backends = self.backends().await;
            let ok = expected.iter().all(|(id, healthy)| {
                backends
                    .iter()
                    .any(|b| b["id"] == *id && b["healthy"] == *healthy)
            });
            if ok {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "backends never reached {expected:?}: {:?}",
            self.backends().await
        );
    }

    async fn post(&self, path: &str, session: Option<&str>, body: Value) -> reqwest::Response {
        let mut req = reqwest::Client::new()
            .post(format!("{}{}", self.proxy, path))
            .json(&body);
        if let Some(s) = session {
            req = req.header(STICKY_SESSION_HEADER, s);
        }
        req.send().await.unwrap()
    }
}

async fn served_by(resp: reqwest::Response) -> (String, Option<String>, Value) {
    assert_eq!(resp.status(), 200);
    let header_session = resp
        .headers()
        .get(STICKY_SESSION_HEADER)
        .map(|v| v.to_str().unwrap().to_string());
    let backend_header = resp.headers()["x-chalk-backend"]
        .to_str()
        .unwrap()
        .to_string();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["backend"], backend_header);
    (backend_header, header_session, body)
}

const FAST: Duration = Duration::from_millis(25);
const NEVER: Duration = Duration::from_secs(3600);

#[tokio::test]
async fn replicas_share_sticky_sessions() {
    let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
    assert_replicas_agree(store.clone(), store, "mem").await;
}

/// Same, with each replica holding its own connection to a real Redis/Valkey.
/// Runs when `REDIS_URL` is set.
#[tokio::test]
async fn replicas_share_sticky_sessions_via_redis() {
    use chalk_statefulrouter::session::RedisStore;
    let Ok(url) = std::env::var("REDIS_URL") else {
        eprintln!("REDIS_URL unset; skipping");
        return;
    };
    let prefix = "chalk-statefulrouter-e2e:";
    let s1 = Arc::new(RedisStore::connect(&url, prefix).await.unwrap());
    let s2 = Arc::new(RedisStore::connect(&url, prefix).await.unwrap());
    let run = chalk_statefulrouter::session::new_session_id();
    assert_replicas_agree(s1, s2, &run).await;
}

async fn assert_replicas_agree(
    store1: Arc<dyn SessionStore>,
    store2: Arc<dyn SessionStore>,
    run: &str,
) {
    let mocks = [
        MockBackend::start("a").await,
        MockBackend::start("b").await,
        MockBackend::start("c").await,
    ];
    let cfg = config(mocks.iter().map(MockBackend::config).collect(), FAST);
    let r1 = start_router(cfg.clone(), Some(store1)).await;
    let r2 = start_router(cfg, Some(store2)).await;
    let all_up = [("a", true), ("b", true), ("c", true)];
    r1.wait_health(&all_up).await;
    r2.wait_health(&all_up).await;

    let mut used = std::collections::HashSet::new();
    for i in 0..30 {
        let session = format!("{run}-conv-{i}");
        let mut seen = None;
        for round in 0..4 {
            let router = if round % 2 == 0 { &r1 } else { &r2 };
            let resp = router
                .post(
                    "/v1/chat/completions",
                    Some(&session),
                    json!({"messages": []}),
                )
                .await;
            let (backend, echoed, body) = served_by(resp).await;
            assert_eq!(echoed.as_deref(), Some(session.as_str()));
            assert_eq!(body["session"], session, "backend sees the session header");
            assert_eq!(*seen.get_or_insert(backend.clone()), backend);
        }
        used.insert(seen.unwrap());
    }
    assert!(used.len() > 1, "sessions should spread across backends");
}

#[tokio::test]
async fn generates_session_id_when_missing() {
    let mocks = [MockBackend::start("a").await, MockBackend::start("b").await];
    let r = start_router(
        config(mocks.iter().map(MockBackend::config).collect(), FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true), ("b", true)]).await;

    let (backend, session, body) =
        served_by(r.post("/v1/completions", None, json!({})).await).await;
    let session = session.expect("router mints a session id");
    assert_eq!(body["session"], session);
    for _ in 0..5 {
        let (again, _, _) =
            served_by(r.post("/v1/completions", Some(&session), json!({})).await).await;
        assert_eq!(again, backend);
    }

    let resp = r.post("/v1/completions", Some("bad id"), json!({})).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn unhealthy_backend_sessions_move_together() {
    let mocks = [MockBackend::start("a").await, MockBackend::start("b").await];
    let cfg = config(mocks.iter().map(MockBackend::config).collect(), FAST);
    let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
    let r1 = start_router(cfg.clone(), Some(store.clone())).await;
    let r2 = start_router(cfg, Some(store)).await;
    r1.wait_health(&[("a", true), ("b", true)]).await;
    r2.wait_health(&[("a", true), ("b", true)]).await;

    let (first, _, _) = served_by(r1.post("/v1/completions", Some("s"), json!({})).await).await;
    let dying = mocks.iter().find(|m| m.id == first).unwrap();
    let other = if first == "a" { "b" } else { "a" };
    dying.healthy.store(false, Ordering::SeqCst);
    r1.wait_health(&[(first.as_str(), false)]).await;
    r2.wait_health(&[(first.as_str(), false)]).await;

    let (moved, _, _) = served_by(r2.post("/v1/completions", Some("s"), json!({})).await).await;
    assert_eq!(moved, other);
    let (moved, _, _) = served_by(r1.post("/v1/completions", Some("s"), json!({})).await).await;
    assert_eq!(moved, other);

    // Recovery doesn't move the session back: its cache is on the new backend.
    dying.healthy.store(true, Ordering::SeqCst);
    r1.wait_health(&[(first.as_str(), true)]).await;
    let (stays, _, _) = served_by(r1.post("/v1/completions", Some("s"), json!({})).await).await;
    assert_eq!(stays, other);
}

#[tokio::test]
async fn connect_failure_retries_and_rebinds() {
    let live = MockBackend::start("live").await;
    let dead_addr = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let dead = BackendConfig {
        id: "dead".into(),
        url: format!("http://{dead_addr}"),
        kv_events: None,
    };
    let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
    // Probe once at startup, then not again during the test.
    let r = start_router(
        config(vec![live.config(), dead], NEVER),
        Some(store.clone()),
    )
    .await;
    r.wait_health(&[("live", true), ("dead", false)]).await;
    // Pretend the active checker hasn't noticed the backend died yet.
    let registry = &r.runtime.state.router.registry;
    registry
        .get("dead")
        .unwrap()
        .record_probe(Ok(()), &HealthConfig::default());
    store
        .bind_if_absent("s", "dead", Duration::from_secs(60))
        .await
        .unwrap();

    let (backend, _, _) = served_by(r.post("/v1/completions", Some("s"), json!({})).await).await;
    assert_eq!(backend, "live");
    let rebound = store.get("s", Duration::from_secs(60)).await.unwrap();
    assert_eq!(rebound.as_deref(), Some("live"));
}

#[tokio::test]
async fn kv_events_steer_new_sessions() {
    let mocks = [
        MockBackend::start("a").await,
        MockBackend::start("b").await,
        MockBackend::start("c").await,
    ];
    let r = start_router(
        config(mocks.iter().map(MockBackend::config).collect(), FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true), ("b", true), ("c", true)])
        .await;

    let prompt: Vec<u32> = (0..64).collect();
    let resp = reqwest::Client::new()
        .post(format!("{}/kv-events", r.admin))
        .json(&json!({
            "backend": "c",
            "seq": 1,
            "events": [{
                "type": "block_stored",
                "block_hashes": [101, 102, 103],
                "parent_block_hash": null,
                "token_ids": &prompt[..48],
                "block_size": 16,
            }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Events are applied asynchronously.
    for _ in 0..100 {
        let backends = r.backends().await;
        if backends
            .iter()
            .any(|b| b["id"] == "c" && b["kv_blocks"] == 3)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let explain: Value = reqwest::Client::new()
        .post(format!("{}/route/explain", r.admin))
        .json(&json!({"prompt": prompt}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(explain["scores"][0]["backend"], "c");
    assert_eq!(explain["scores"][0]["overlap_ratio"], 0.75);

    for i in 0..10 {
        let resp = r
            .post(
                "/v1/completions",
                Some(&format!("new-{i}")),
                json!({"prompt": prompt}),
            )
            .await;
        assert_eq!(served_by(resp).await.0, "c");
    }
}

#[tokio::test]
async fn msgpack_ingest_and_drain() {
    let mocks = [MockBackend::start("a").await, MockBackend::start("b").await];
    let r = start_router(
        config(mocks.iter().map(MockBackend::config).collect(), FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true), ("b", true)]).await;

    // vLLM wire format: [ts, [["BlockStored", hashes, parent, tokens, block_size, lora, medium]]]
    let mut payload = Vec::new();
    let batch = rmpv::Value::Array(vec![
        rmpv::Value::F64(0.0),
        rmpv::Value::Array(vec![rmpv::Value::Array(vec![
            "BlockStored".into(),
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![1; 32])]),
            rmpv::Value::Nil,
            rmpv::Value::Array((0..4).map(rmpv::Value::from).collect()),
            4.into(),
        ])]),
    ]);
    rmpv::encode::write_value(&mut payload, &batch).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/kv-events/a", r.admin))
        .header("x-chalk-kv-seq", "1")
        .body(payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let (bound, _, _) = served_by(r.post("/v1/completions", Some("keep"), json!({})).await).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/backends/{bound}/drain", r.admin))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Existing session stays; new ones avoid the draining backend.
    assert_eq!(
        served_by(r.post("/v1/completions", Some("keep"), json!({})).await)
            .await
            .0,
        bound
    );
    for i in 0..10 {
        let resp = r
            .post("/v1/completions", Some(&format!("fresh-{i}")), json!({}))
            .await;
        assert_ne!(served_by(resp).await.0, bound);
    }

    let backends = r.backends().await;
    assert!(backends
        .iter()
        .any(|b| b["id"] == "a" && b["kv_blocks"] == 1));
}

#[tokio::test]
async fn streams_pass_through_and_release_load() {
    let mock = MockBackend::start("a").await;
    let r = start_router(
        config(vec![mock.config()], FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true)]).await;

    let resp = reqwest::get(format!("{}/v1/stream", r.proxy))
        .await
        .unwrap();
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let backend = r.runtime.state.router.registry.get("a").unwrap();
    assert_eq!(backend.inflight(), 1, "in-flight while the stream is open");
    assert_eq!(
        resp.text().await.unwrap(),
        "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    );
    for _ in 0..50 {
        if backend.inflight() == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("in-flight count not released");
}

#[tokio::test]
async fn admin_adds_and_removes_backends() {
    let a = MockBackend::start("a").await;
    let r = start_router(
        config(vec![a.config()], FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true)]).await;

    let b = MockBackend::start("b").await;
    let added: Value = reqwest::Client::new()
        .put(format!("{}/backends/b", r.admin))
        .json(&json!({"url": format!("http://{}", b.addr)}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        added["healthy"], true,
        "new backends are probed before the call returns"
    );

    let resp = reqwest::Client::new()
        .delete(format!("{}/backends/a", r.admin))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
    assert_eq!(
        served_by(r.post("/v1/completions", Some("s"), json!({})).await)
            .await
            .0,
        "b"
    );
}

#[tokio::test]
async fn no_healthy_backends_is_503() {
    let a = MockBackend::start("a").await;
    a.healthy.store(false, Ordering::SeqCst);
    let r = start_router(
        config(vec![a.config()], FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", false)]).await;
    assert_eq!(
        r.post("/v1/completions", None, json!({})).await.status(),
        503
    );
}

#[cfg(feature = "zmq")]
#[tokio::test]
async fn subscribes_to_engine_zmq_publisher() {
    use zeromq::{Socket, SocketSend};

    let mut publisher = zeromq::PubSocket::new();
    let endpoint = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
    let mock = MockBackend::start("a").await;
    let mut backend = mock.config();
    backend.kv_events = Some(KvEventsEndpoint {
        zmq_endpoint: endpoint.to_string(),
        topic: "kv-events".into(),
    });
    let r = start_router(
        config(vec![backend], FAST),
        Some(Arc::new(MemoryStore::new())),
    )
    .await;
    r.wait_health(&[("a", true)]).await;

    let mut payload = Vec::new();
    let batch = rmpv::Value::Array(vec![
        rmpv::Value::F64(0.0),
        rmpv::Value::Array(vec![rmpv::Value::Array(vec![
            "BlockStored".into(),
            rmpv::Value::Array(vec![7.into(), 8.into()]),
            rmpv::Value::Nil,
            rmpv::Value::Array((0..4).map(rmpv::Value::from).collect()),
            2.into(),
            rmpv::Value::Nil,
            "GPU".into(),
        ])]),
    ]);
    rmpv::encode::write_value(&mut payload, &batch).unwrap();

    // PUB drops messages until the subscriber has joined, so keep publishing
    // (as an engine would) until the router has indexed the blocks.
    for seq in 0u64..200 {
        let mut msg = zeromq::ZmqMessage::from("kv-events");
        msg.push_back(seq.to_be_bytes().to_vec().into());
        msg.push_back(payload.clone().into());
        publisher.send(msg).await.unwrap();
        if r.runtime.state.router.index.block_count("a") == 2 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("router never indexed blocks published over zmq");
}
