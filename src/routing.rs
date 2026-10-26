//! Backend selection.
//!
//! 1. A request with a session id goes to the session's bound backend while
//!    that backend is healthy.
//! 2. Otherwise (new session, dead binding, or no session) backends that accept
//!    new work are scored: the fraction of the prompt already in their KV cache
//!    counts for them, their in-flight load against them. Ties break by
//!    rendezvous hash so replicas with the same view make the same choice.
//! 3. For a session, the choice is written to the shared store atomically;
//!    if another replica bound the session first, its choice wins.
//! 4. Bounded load: a request whose sticky backend is far busier than average
//!    may spill to the next-best backend, leaving the binding in place (only
//!    for the session sources `routing.bounded_load.applies_to` lists).

use std::sync::Arc;

use serde::Serialize;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

use crate::{
    backend::{Backend, BackendId, BackendRegistry},
    config::Config,
    kv::{index::Overlap, KvIndex},
    session::{SessionKey, SessionStore, StoreError},
};

pub struct Router {
    pub registry: Arc<BackendRegistry>,
    /// `None` in rendezvous mode.
    pub sessions: Option<Arc<dyn SessionStore>>,
    pub index: Arc<KvIndex>,
    pub config: Arc<Config>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reason {
    /// Session was already bound to this backend.
    Sticky,
    /// Session was unbound; this router bound it.
    NewSession,
    /// Session was bound to a backend that can't serve it; moved here.
    Rebound { from: String },
    /// Session mapped by rendezvous hash (rendezvous mode, or store failure).
    Rendezvous,
    /// No session; best-scoring backend.
    Stateless,
    /// The session's backend was over its bounded-load cap; this request went
    /// elsewhere and the binding stayed put.
    Spilled { from: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct Score {
    pub backend: String,
    pub overlap_ratio: f64,
    pub inflight: u64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Explanation {
    pub session: Option<SessionKey>,
    /// Backend the session is bound to (or would map to, in rendezvous mode).
    pub binding: Option<String>,
    pub binding_usable: bool,
    /// The bound backend is over its bounded-load cap and this session may
    /// spill, so the request would go to the best-scoring other backend.
    pub would_spill: bool,
    pub request_tokens: usize,
    /// How a new binding would be chosen, best first.
    pub scores: Vec<Score>,
}

pub struct Decision {
    pub backend: Arc<Backend>,
    pub reason: Reason,
    /// Candidate scores, best first. Empty when no scoring happened.
    pub scores: Vec<Score>,
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("no healthy backends")]
    NoBackends,
}

impl Router {
    /// Pick a backend. `exclude` lists backends that already failed for this
    /// request; a session bound to one of them is rebound.
    pub async fn route(
        &self,
        session: Option<&SessionKey>,
        tokens: Option<&[u32]>,
        exclude: &[BackendId],
    ) -> Result<Decision, RouteError> {
        let Some(key) = session else {
            let tiebreak = crate::session::new_session_id();
            return self.best(tokens, exclude, &tiebreak, Reason::Stateless);
        };
        let decision = self.route_bound(&key.id, tokens, exclude).await?;
        if !self.should_spill(key, &decision) {
            return Ok(decision);
        }
        // Spilled requests of one session break ties the same way, so they
        // pile onto one second backend rather than seeding the prefix
        // everywhere.
        let mut skip = exclude.to_vec();
        skip.push(decision.backend.id.clone());
        let from = decision.backend.id.to_string();
        Ok(self
            .best(tokens, &skip, &key.id, Reason::Spilled { from })
            .unwrap_or(decision))
    }

    fn should_spill(&self, key: &SessionKey, decision: &Decision) -> bool {
        let bounded = &self.config.routing.bounded_load;
        matches!(decision.reason, Reason::Sticky | Reason::Rendezvous)
            && bounded.applies_to.contains(&key.source)
            && self.over_load_cap(&decision.backend)
    }

    /// Consistent hashing with bounded loads: a backend may hold at most
    /// `ceil(factor × average)` in-flight requests, counting this one.
    fn over_load_cap(&self, backend: &Backend) -> bool {
        let Some(factor) = self.config.routing.bounded_load.factor else {
            return false;
        };
        let live: Vec<_> = self
            .registry
            .all()
            .into_iter()
            .filter(|b| b.accepts_existing())
            .collect();
        if live.len() < 2 {
            return false;
        }
        let total: u64 = live.iter().map(|b| b.inflight()).sum();
        let cap = (factor * (total + 1) as f64 / live.len() as f64).ceil();
        (backend.inflight() + 1) as f64 > cap
    }

    /// The session's backend, binding the session if needed.
    async fn route_bound(
        &self,
        session: &str,
        tokens: Option<&[u32]>,
        exclude: &[BackendId],
    ) -> Result<Decision, RouteError> {
        let Some(store) = &self.sessions else {
            return self.rendezvous(session, exclude);
        };
        match self
            .route_session(store.as_ref(), session, tokens, exclude)
            .await
        {
            Ok(decision) => Ok(decision),
            // Degrade to rendezvous: replicas still agree with each other as
            // long as their health views do, which beats failing the request.
            Err(Failure::Store(err)) => {
                tracing::warn!(session, error = %err, "session store failed; using rendezvous");
                self.rendezvous(session, exclude)
            }
            Err(Failure::Route(err)) => Err(err),
        }
    }

    /// What [`Router::route`] would consider, without binding anything.
    pub async fn explain(
        &self,
        key: Option<&SessionKey>,
        tokens: Option<&[u32]>,
    ) -> Result<Explanation, StoreError> {
        let session = key.map(|k| k.id.as_str());
        let binding = match (session, &self.sessions) {
            (Some(s), Some(store)) => store.get(s, self.config.sessions.ttl).await?,
            (Some(s), None) => self
                .rendezvous(s, &[])
                .ok()
                .map(|d| d.backend.id.to_string()),
            (None, _) => None,
        };
        let usable = binding.as_deref().and_then(|id| self.usable(id, &[]));
        let binding_usable = usable.is_some();
        let would_spill = match (key, usable) {
            (Some(key), Some(backend)) => self.should_spill(
                key,
                &Decision {
                    backend,
                    reason: Reason::Sticky,
                    scores: Vec::new(),
                },
            ),
            _ => false,
        };
        let scores = self
            .best(tokens, &[], session.unwrap_or_default(), Reason::Stateless)
            .map(|d| d.scores)
            .unwrap_or_default();
        Ok(Explanation {
            session: key.cloned(),
            binding,
            binding_usable,
            would_spill,
            request_tokens: tokens.map_or(0, <[u32]>::len),
            scores,
        })
    }

    async fn route_session(
        &self,
        store: &dyn SessionStore,
        session: &str,
        tokens: Option<&[u32]>,
        exclude: &[BackendId],
    ) -> Result<Decision, Failure> {
        let ttl = self.config.sessions.ttl;
        let current = store.get(session, ttl).await?;
        if let Some(backend) = current.as_deref().and_then(|id| self.usable(id, exclude)) {
            return Ok(Decision {
                backend,
                reason: Reason::Sticky,
                scores: Vec::new(),
            });
        }

        let reason = match &current {
            None => Reason::NewSession,
            Some(from) => Reason::Rebound { from: from.clone() },
        };
        let best = self.best(tokens, exclude, session, reason)?;
        let winner = match &current {
            None => store.bind_if_absent(session, &best.backend.id, ttl).await?,
            Some(old) => {
                store
                    .compare_and_swap(session, old, &best.backend.id, ttl)
                    .await?
            }
        };
        if winner == *best.backend.id {
            return Ok(best);
        }

        // Another replica bound or rebound the session first. Adopt its
        // choice unless this replica can't use it (e.g. the health views
        // disagree), in which case take the binding over.
        if let Some(backend) = self.usable(&winner, exclude) {
            return Ok(Decision {
                backend,
                reason: Reason::Sticky,
                scores: best.scores,
            });
        }
        let winner = store
            .compare_and_swap(session, &winner, &best.backend.id, ttl)
            .await?;
        if winner != *best.backend.id {
            tracing::warn!(session, %winner, chosen = %best.backend.id, "session binding contended; serving from local choice");
        }
        Ok(best)
    }

    fn usable(&self, id: &str, exclude: &[BackendId]) -> Option<Arc<Backend>> {
        self.registry
            .get(id)
            .filter(|b| b.accepts_existing() && !exclude.contains(&b.id))
    }

    fn candidates(&self, exclude: &[BackendId]) -> Vec<Arc<Backend>> {
        self.registry
            .all()
            .into_iter()
            .filter(|b| b.accepts_new() && !exclude.contains(&b.id))
            .collect()
    }

    fn rendezvous(&self, session: &str, exclude: &[BackendId]) -> Result<Decision, RouteError> {
        let backend = self
            .candidates(exclude)
            .into_iter()
            .max_by_key(|b| rendezvous_weight(session, &b.id))
            .ok_or(RouteError::NoBackends)?;
        Ok(Decision {
            backend,
            reason: Reason::Rendezvous,
            scores: Vec::new(),
        })
    }

    fn best(
        &self,
        tokens: Option<&[u32]>,
        exclude: &[BackendId],
        tiebreak: &str,
        reason: Reason,
    ) -> Result<Decision, RouteError> {
        let candidates = self.candidates(exclude);
        let overlap = match tokens {
            Some(tokens) if self.config.kv.enabled => self.index.overlap(tokens),
            _ => Overlap::default(),
        };
        let max_inflight = candidates.iter().map(|b| b.inflight()).max().unwrap_or(0);
        let routing = &self.config.routing;

        let mut ranked: Vec<(f64, u64, Arc<Backend>, Score)> = candidates
            .into_iter()
            .map(|b| {
                let overlap_ratio = overlap.ratio(&b.id);
                let inflight = b.inflight();
                let load = if max_inflight == 0 {
                    0.0
                } else {
                    inflight as f64 / max_inflight as f64
                };
                let score = routing.overlap_weight * overlap_ratio - routing.load_weight * load;
                let score_row = Score {
                    backend: b.id.to_string(),
                    overlap_ratio,
                    inflight,
                    score,
                };
                (score, rendezvous_weight(tiebreak, &b.id), b, score_row)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));

        let backend = ranked
            .first()
            .map(|r| r.2.clone())
            .ok_or(RouteError::NoBackends)?;
        Ok(Decision {
            backend,
            reason,
            scores: ranked.into_iter().map(|r| r.3).collect(),
        })
    }
}

enum Failure {
    Store(StoreError),
    Route(RouteError),
}

impl From<StoreError> for Failure {
    fn from(e: StoreError) -> Self {
        Failure::Store(e)
    }
}

impl From<RouteError> for Failure {
    fn from(e: RouteError) -> Self {
        Failure::Route(e)
    }
}

/// Highest-random-weight hash: every replica ranks backends for a key the same
/// way, and removing a backend only moves the keys that were on it.
pub fn rendezvous_weight(key: &str, backend: &str) -> u64 {
    xxh3_64_with_seed(key.as_bytes(), xxh3_64(backend.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{BackendConfig, GapPolicy, HealthConfig},
        kv::{KvEvent, KvEventEnvelope},
        session::{MemoryStore, SessionSource},
    };

    fn k(id: &str) -> SessionKey {
        SessionKey::explicit(id)
    }

    fn inferred(id: &str) -> SessionKey {
        SessionKey {
            id: id.to_string(),
            source: SessionSource::Inferred,
        }
    }

    fn router(ids: &[&str], store: Option<Arc<dyn SessionStore>>) -> Router {
        let registry = Arc::new(BackendRegistry::new(ids.iter().map(|id| BackendConfig {
            id: id.to_string(),
            url: format!("http://{id}"),
            kv_events: None,
        })));
        for b in registry.all() {
            b.record_probe(Ok(()), &HealthConfig::default());
        }
        Router {
            registry,
            sessions: store,
            index: Arc::new(KvIndex::new(GapPolicy::Ignore)),
            config: Arc::new(Config::default()),
        }
    }

    fn kill(r: &Router, id: &str) {
        let cfg = HealthConfig {
            unhealthy_threshold: 1,
            ..Default::default()
        };
        r.registry
            .get(id)
            .unwrap()
            .record_probe(Err("down".into()), &cfg);
    }

    #[tokio::test]
    async fn replicas_sharing_a_store_agree() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r1 = router(&["a", "b", "c"], Some(store.clone()));
        let r2 = router(&["a", "b", "c"], Some(store.clone()));

        for i in 0..50 {
            let s = format!("session-{i}");
            let d1 = r1.route(Some(&k(&s)), None, &[]).await.unwrap();
            assert_eq!(d1.reason, Reason::NewSession);
            let d2 = r2.route(Some(&k(&s)), None, &[]).await.unwrap();
            assert_eq!(d2.reason, Reason::Sticky);
            assert_eq!(d1.backend.id, d2.backend.id);
        }
    }

    #[tokio::test]
    async fn failover_rebinds_once() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r1 = router(&["a", "b", "c"], Some(store.clone()));
        let r2 = router(&["a", "b", "c"], Some(store.clone()));

        let first = r1
            .route(Some(&k("s")), None, &[])
            .await
            .unwrap()
            .backend
            .id
            .clone();
        kill(&r1, &first);
        kill(&r2, &first);
        let d1 = r1.route(Some(&k("s")), None, &[]).await.unwrap();
        assert_eq!(
            d1.reason,
            Reason::Rebound {
                from: first.to_string()
            }
        );
        assert_ne!(d1.backend.id, first);
        // The other replica sees the new binding instead of rebinding again.
        let d2 = r2.route(Some(&k("s")), None, &[]).await.unwrap();
        assert_eq!(d2.reason, Reason::Sticky);
        assert_eq!(d2.backend.id, d1.backend.id);
    }

    #[tokio::test]
    async fn lost_bind_race_adopts_winner() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r = router(&["a", "b", "c"], Some(store.clone()));
        // Simulate another replica's prior binding to a backend this replica
        // wouldn't have picked by score.
        store
            .bind_if_absent("s", "c", r.config.sessions.ttl)
            .await
            .unwrap();
        let d = r.route(Some(&k("s")), None, &[]).await.unwrap();
        assert_eq!(&*d.backend.id, "c");
    }

    #[tokio::test]
    async fn excluded_backend_is_rebound() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r = router(&["a", "b"], Some(store.clone()));
        let first = r
            .route(Some(&k("s")), None, &[])
            .await
            .unwrap()
            .backend
            .id
            .clone();
        let retry = r
            .route(Some(&k("s")), None, std::slice::from_ref(&first))
            .await
            .unwrap();
        assert_ne!(retry.backend.id, first);
        let after = r.route(Some(&k("s")), None, &[]).await.unwrap();
        assert_eq!(after.backend.id, retry.backend.id);
    }

    #[tokio::test]
    async fn draining_keeps_sessions_but_takes_no_new_ones() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r = router(&["a", "b"], Some(store));
        let first = r.route(Some(&k("s")), None, &[]).await.unwrap().backend;
        first.set_draining(true);
        assert_eq!(
            r.route(Some(&k("s")), None, &[]).await.unwrap().backend.id,
            first.id
        );
        for i in 0..20 {
            let d = r
                .route(Some(&k(&format!("new-{i}"))), None, &[])
                .await
                .unwrap();
            assert_ne!(d.backend.id, first.id);
        }
    }

    #[tokio::test]
    async fn rendezvous_mode_is_stable_and_minimal() {
        let r1 = router(&["a", "b", "c"], None);
        let r2 = router(&["a", "b", "c"], None);
        let mut before = Vec::new();
        for i in 0..100 {
            let s = format!("s{i}");
            let d1 = r1.route(Some(&k(&s)), None, &[]).await.unwrap();
            let d2 = r2.route(Some(&k(&s)), None, &[]).await.unwrap();
            assert_eq!(d1.backend.id, d2.backend.id);
            before.push(d1.backend.id.clone());
        }
        kill(&r1, "a");
        for (i, prev) in before.iter().enumerate() {
            let d = r1
                .route(Some(&k(&format!("s{i}"))), None, &[])
                .await
                .unwrap();
            if &**prev != "a" {
                assert_eq!(
                    &d.backend.id, prev,
                    "sessions not on the dead backend must not move"
                );
            }
        }
    }

    #[tokio::test]
    async fn kv_overlap_wins_for_new_work() {
        let r = router(&["a", "b", "c"], Some(Arc::new(MemoryStore::new())));
        r.index.apply(&KvEventEnvelope {
            backend: "b".into(),
            seq: None,
            events: vec![KvEvent::BlockStored {
                block_hashes: vec![1, 2],
                parent_block_hash: None,
                token_ids: vec![1, 2, 3, 4],
                block_size: 2,
                lora_id: None,
            }],
        });
        let tokens = [1, 2, 3, 4, 5];
        for i in 0..10 {
            let d = r
                .route(Some(&k(&format!("s{i}"))), Some(&tokens), &[])
                .await
                .unwrap();
            assert_eq!(&*d.backend.id, "b");
            assert_eq!(d.scores[0].backend, "b");
        }
        let d = r.route(None, Some(&tokens), &[]).await.unwrap();
        assert_eq!(&*d.backend.id, "b");
    }

    #[tokio::test]
    async fn load_breaks_ties() {
        let r = router(&["a", "b"], None);
        let _busy = r.registry.get("a").unwrap().inflight_guard();
        for _ in 0..10 {
            assert_eq!(&*r.route(None, None, &[]).await.unwrap().backend.id, "b");
        }
    }

    #[tokio::test]
    async fn no_backends() {
        let r = router(&["a"], None);
        kill(&r, "a");
        assert!(matches!(
            r.route(None, None, &[]).await,
            Err(RouteError::NoBackends)
        ));
    }

    #[tokio::test]
    async fn hot_inferred_sessions_spill_without_rebinding() {
        let store: Arc<dyn SessionStore> = Arc::new(MemoryStore::new());
        let r = router(&["a", "b", "c"], Some(store.clone()));
        let home = r
            .route(Some(&inferred("s")), None, &[])
            .await
            .unwrap()
            .backend;

        // Load the home backend past ceil(1.5 × average).
        let guards: Vec<_> = (0..4).map(|_| home.inflight_guard()).collect();
        let d = r.route(Some(&inferred("s")), None, &[]).await.unwrap();
        assert_eq!(
            d.reason,
            Reason::Spilled {
                from: home.id.to_string()
            }
        );
        assert_ne!(d.backend.id, home.id);
        // Spilled requests of one session agree on where they go.
        let again = r.route(Some(&inferred("s")), None, &[]).await.unwrap();
        assert_eq!(again.backend.id, d.backend.id);
        // The binding didn't move.
        let ttl = r.config.sessions.ttl;
        assert_eq!(
            store.get("s", ttl).await.unwrap().as_deref(),
            Some(&*home.id)
        );

        // Explicit sessions don't spill by default.
        store.bind_if_absent("x", &home.id, ttl).await.unwrap();
        let d = r.route(Some(&k("x")), None, &[]).await.unwrap();
        assert_eq!(
            (d.reason, d.backend.id.clone()),
            (Reason::Sticky, home.id.clone())
        );

        drop(guards);
        let d = r.route(Some(&inferred("s")), None, &[]).await.unwrap();
        assert_eq!(
            (d.reason, d.backend.id.clone()),
            (Reason::Sticky, home.id.clone())
        );
    }

    #[tokio::test]
    async fn bounded_load_cap() {
        let mut r = router(&["a", "b"], None);
        let a = r.registry.get("a").unwrap();
        // factor 1.5, two backends, `k` in flight on a and none on b: the cap
        // counting this request is ceil(1.5 × (k + 1) / 2).
        let mut guards = Vec::new();
        for k in 0..3 {
            assert!(!r.over_load_cap(&a), "k={k} fits");
            guards.push(a.inflight_guard());
        }
        // k=3: cap ceil(3.0) = 3, but a would hold 4.
        assert!(r.over_load_cap(&a));
        // Load elsewhere raises the average: a=3, b=1 gives cap ceil(3.75) = 4.
        let _b = r.registry.get("b").unwrap().inflight_guard();
        assert!(!r.over_load_cap(&a));

        let config = Arc::get_mut(&mut r.config).unwrap();
        config.routing.bounded_load.factor = None;
        assert!(!r.over_load_cap(&a));
    }
}
