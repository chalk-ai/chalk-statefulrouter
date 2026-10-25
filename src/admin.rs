//! Admin / control API, served on its own port.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    backend::BackendStatus,
    config::{BackendConfig, KvEventsEndpoint},
    kv::{event::decode_msgpack_batch, KvEventEnvelope},
    session::valid_session_id,
    AppState,
};

/// Optional publisher sequence number for raw msgpack batches.
pub const KV_SEQ_HEADER: &str = "x-chalk-kv-seq";

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/backends", get(list_backends))
        .route("/backends/{id}", put(upsert_backend).delete(remove_backend))
        .route("/backends/{id}/drain", post(drain))
        .route("/backends/{id}/undrain", post(undrain))
        .route("/sessions/{id}", get(get_session).delete(remove_session))
        .route("/kv-events", post(ingest_json))
        .route("/kv-events/{backend}", post(ingest_msgpack))
        .route("/route/explain", post(explain))
        .with_state(state)
}

fn status_with_kv(state: &AppState, mut status: BackendStatus) -> BackendStatus {
    status.kv_blocks = state.router.index.block_count(&status.id);
    status
}

async fn list_backends(State(state): State<Arc<AppState>>) -> Json<Vec<BackendStatus>> {
    Json(
        state
            .router
            .registry
            .all()
            .iter()
            .map(|b| status_with_kv(&state, b.status()))
            .collect(),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertBackend {
    url: String,
    #[serde(default)]
    kv_events: Option<KvEventsEndpoint>,
}

async fn upsert_backend(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<UpsertBackend>,
) -> Json<BackendStatus> {
    let config = BackendConfig {
        id,
        url: body.url,
        kv_events: body.kv_events,
    };
    let previous = state.router.registry.get(&config.id);
    let backend = state.router.registry.upsert(config);
    let replaced = previous.as_ref().is_some_and(|p| !Arc::ptr_eq(p, &backend));
    if previous.is_none() || replaced {
        if replaced {
            // A new URL is a different engine; its cache isn't the old one's.
            state.router.index.remove_backend(&backend.id);
        }
        state.subscriptions.sync(&backend);
        crate::health::probe(&backend, &state.client, &state.router.config.health).await;
    }
    Json(status_with_kv(&state, backend.status()))
}

async fn remove_backend(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> StatusCode {
    match state.router.registry.remove(&id) {
        Some(_) => {
            state.subscriptions.stop(&id);
            state.router.index.remove_backend(&id);
            StatusCode::NO_CONTENT
        }
        None => StatusCode::NOT_FOUND,
    }
}

async fn set_draining(state: &AppState, id: &str, draining: bool) -> Response {
    match state.router.registry.get(id) {
        Some(b) => {
            b.set_draining(draining);
            Json(status_with_kv(state, b.status())).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn drain(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    set_draining(&state, &id, true).await
}

async fn undrain(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    set_draining(&state, &id, false).await
}

async fn get_session(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(store) = &state.router.sessions else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "rendezvous mode stores no bindings",
        )
            .into_response();
    };
    match store.get(&id, state.router.config.sessions.ttl).await {
        Ok(Some(backend)) => Json(json!({"session": id, "backend": backend})).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(err) => (StatusCode::BAD_GATEWAY, err.to_string()).into_response(),
    }
}

async fn remove_session(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(store) = &state.router.sessions else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "rendezvous mode stores no bindings",
        )
            .into_response();
    };
    match store.remove(&id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(err) => (StatusCode::BAD_GATEWAY, err.to_string()).into_response(),
    }
}

/// Accepts one envelope or an array of them.
#[derive(Deserialize)]
#[serde(untagged)]
enum Envelopes {
    One(KvEventEnvelope),
    Many(Vec<KvEventEnvelope>),
}

async fn ingest_json(State(state): State<Arc<AppState>>, Json(body): Json<Envelopes>) -> Response {
    let envelopes = match body {
        Envelopes::One(e) => vec![e],
        Envelopes::Many(es) => es,
    };
    let sink = state.subscriptions.sink();
    let count = envelopes.len();
    for envelope in envelopes {
        if sink.send(envelope).await.is_err() {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    }
    Json(json!({"accepted": count})).into_response()
}

/// Raw engine msgpack `KVEventBatch`, as published on the engine's ZMQ socket.
async fn ingest_msgpack(
    State(state): State<Arc<AppState>>,
    Path(backend): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let events = match decode_msgpack_batch(&body) {
        Ok(events) => events,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    let seq = headers
        .get(KV_SEQ_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    let count = events.len();
    let envelope = KvEventEnvelope {
        backend,
        seq,
        events,
    };
    if state.subscriptions.sink().send(envelope).await.is_err() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    Json(json!({"accepted_events": count})).into_response()
}

/// Show how a request would be routed, without forwarding it or touching
/// session bindings.
async fn explain(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let session = headers
        .get(&state.session_header)
        .and_then(|v| v.to_str().ok())
        .filter(|id| valid_session_id(id));
    let path = body
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("/v1/completions");
    let request = body.get("body").unwrap_or(&body);
    let tokens = state.tokens.tokens(path, request);
    match state.router.explain(session, tokens.as_deref()).await {
        Ok(explanation) => Json(explanation).into_response(),
        Err(err) => (StatusCode::BAD_GATEWAY, err.to_string()).into_response(),
    }
}
