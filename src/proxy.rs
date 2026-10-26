//! The proxied inference API: every path is forwarded to the chosen backend,
//! with responses (including SSE streams) passed through as they arrive.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use serde_json::{json, Value};

use crate::{
    inspect::{inspect, InspectedRequest},
    routing::{Reason, RouteError},
    session::derive,
    AppState,
};

pub const BACKEND_HEADER: &str = "x-chalk-backend";
/// Where the session key came from: explicit, declared, inferred, generated.
pub const SESSION_SOURCE_HEADER: &str = "x-chalk-session-source";

/// Connection-scoped headers that must not be forwarded, plus ones the client
/// library recomputes.
const SKIP_HEADERS: &[HeaderName] = &[
    header::CONNECTION,
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
    header::HOST,
    header::CONTENT_LENGTH,
];

fn copy_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for (name, value) in from {
        if !SKIP_HEADERS.contains(name) && name.as_str() != "keep-alive" {
            to.append(name.clone(), value.clone());
        }
    }
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": "router_error"}});
    (status, Json(body)).into_response()
}

pub async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let config = state.router.config.clone();
    let (parts, body) = request.into_parts();
    let body = match axum::body::to_bytes(body, config.routing.max_body_bytes).await {
        Ok(body) => body,
        Err(_) => return error(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
    };

    let json = serde_json::from_slice::<Value>(&body).ok();
    let inspected = match &json {
        Some(json) => inspect(parts.uri.path(), json),
        None => InspectedRequest::default(),
    };
    let session_header = state.session_header.clone();
    let session = match derive::resolve(
        &config.sessions,
        &session_header,
        &parts.headers,
        json.as_ref(),
        &inspected,
    ) {
        Ok(session) => session,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    let tokens = state.tokens.tokens(&inspected);
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");

    let mut forward_headers = HeaderMap::new();
    copy_headers(&parts.headers, &mut forward_headers);
    if let Some(key) = &session {
        // Explicit ids were validated; derived and generated ones are hex.
        forward_headers.insert(
            session_header.clone(),
            HeaderValue::from_str(&key.id).unwrap(),
        );
    }

    let mut exclude = Vec::new();
    for attempt in 0..=config.routing.max_retries {
        let decision = match state
            .router
            .route(session.as_ref(), tokens.as_deref(), &exclude)
            .await
        {
            Ok(decision) => decision,
            Err(RouteError::NoBackends) => {
                return error(StatusCode::SERVICE_UNAVAILABLE, "no healthy backends")
            }
        };
        let backend = decision.backend;
        if decision.reason != Reason::Sticky {
            tracing::debug!(session = ?session, backend = %backend.id, reason = ?decision.reason, "routed");
        }

        let guard = backend.inflight_guard();
        let sent = state
            .client
            .request(
                parts.method.clone(),
                format!("{}{}", backend.url, path_and_query),
            )
            .headers(forward_headers.clone())
            .body(body.clone())
            .send()
            .await;

        let upstream = match sent {
            Ok(upstream) => upstream,
            Err(err) => {
                backend.record_request(Err(err.to_string()), &config.health);
                // Connection failures are safe to retry elsewhere: the backend
                // never saw the request. Anything later may have started work.
                if err.is_connect() && attempt < config.routing.max_retries {
                    tracing::warn!(backend = %backend.id, error = %err, "connect failed; retrying");
                    exclude.push(backend.id.clone());
                    continue;
                }
                return error(
                    StatusCode::BAD_GATEWAY,
                    format!("backend {}: {err}", backend.id),
                );
            }
        };

        let status = upstream.status();
        let outcome = match status.as_u16() {
            502..=504 => Err(format!("backend returned {status}")),
            _ => Ok(()),
        };
        backend.record_request(outcome, &config.health);

        let mut response_headers = HeaderMap::new();
        copy_headers(upstream.headers(), &mut response_headers);
        if let Some(key) = &session {
            response_headers.insert(
                session_header.clone(),
                HeaderValue::from_str(&key.id).unwrap(),
            );
        }
        if config.routing.expose_backend_header {
            if let Ok(value) = HeaderValue::from_str(&backend.id) {
                response_headers.insert(BACKEND_HEADER, value);
            }
            if let Some(key) = &session {
                response_headers.insert(
                    SESSION_SOURCE_HEADER,
                    HeaderValue::from_static(key.source.as_str()),
                );
            }
        }

        // The guard rides along with the body so in-flight load covers the
        // whole stream, not just time to first byte.
        let stream = upstream.bytes_stream().map(move |chunk| {
            let _ = &guard;
            chunk
        });
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        *response.headers_mut() = response_headers;
        return response;
    }
    unreachable!("the final attempt always returns")
}
