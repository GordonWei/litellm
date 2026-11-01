//! [`Host`]: the router core's only boundary. Everything the core needs from the embedding
//! process goes through it, and every attempt runs through [`Host::invoke`]: the core never runs
//! an operation itself. `PythonHost` serves it in a Python process (attempts call `litellm.<op>`);
//! `NativeHost` serves it in the Rust server (attempts call the `inference-*` crates).

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::{
    cache::RouterCache,
    config::{Op, RetryPolicy, RoutingStrategy},
    error::{AttemptError, Error, FailureKind, Payload},
    fallback::Fallbacks,
    health::Health,
    mock::MockFailure,
    registry::{Deployment, Registry},
    settings::Settings,
};

/// One call into the router.
#[derive(Clone, Debug, Default)]
pub struct RouterRequest {
    /// The requested name: a model group, alias, routing group, deployment id or wildcard match.
    pub model: String,
    /// The request params, for native executors and plugins that read them.
    pub body: Value,
    /// `user_api_key_team_id`.
    pub team_id: Option<String>,
    pub specific_deployment: bool,
    /// A per-request `num_retries`, which wins over the router's.
    pub num_retries: Option<u32>,
    pub disable_fallbacks: bool,
    /// `_router_weights`: model group to deployment id to weight.
    pub router_weights: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    /// Fallback lists the request passed (`Some(None)` when it passed `None`), which replace
    /// the router's.
    pub fallbacks: Option<Option<Fallbacks>>,
    pub context_window_fallbacks: Option<Option<Fallbacks>>,
    pub content_policy_fallbacks: Option<Option<Fallbacks>>,
    /// A per-request `max_fallbacks`, which replaces the router's.
    pub max_fallbacks: Option<u32>,
    /// A per-request `cooldown_time`: the call's own params carry it, so it cools a failed
    /// deployment down for that long unless its `model_info` sets one.
    pub cooldown_time: Option<f64>,
    /// A `mock_testing_*` flag the request set.
    pub mock_failure: Option<MockFailure>,
    /// A per-request `model_group_retry_policy`, which replaces the router's.
    pub model_group_retry_policy: Option<BTreeMap<String, RetryPolicy>>,
    /// What the host needs to run the call itself, such as the Python kwargs.
    pub payload: Option<Payload>,
}

/// One attempt against one deployment.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub op: Op,
    pub request: Arc<RouterRequest>,
    pub deployment: Arc<Deployment>,
    /// The name this attempt resolved: the requested name, or the fallback target.
    pub name: String,
    /// The model group being served, after fallbacks.
    pub model_group: String,
    /// The `order` level an order fallback targets in the same group.
    pub target_order: Option<i64>,
    /// Every deployment `name` resolved to for this caller, the picked one included.
    pub candidate_ids: Vec<String>,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
    pub fallback_depth: u32,
    /// How many fallbacks deep the walk is: 0 for the requested name.
    pub fallback_level: u32,
    pub timeout: Option<Duration>,
    /// Request params the fallback walk's dict targets merged in (`kwargs.update(target)`).
    pub params: Option<Arc<Map<String, Value>>>,
    /// The walk level whose request metadata this attempt carries (see [`RouterEvent::Fallback`]).
    pub scope: u32,
    /// The retry budget the level's loop stamps on the request before this attempt.
    pub max_retries: u32,
    /// How many deployments `get_model_list` lists for the level's group (`model_group_size`).
    pub model_group_size: usize,
}

/// What a successful attempt returns.
#[derive(Clone, Debug)]
pub struct Output {
    pub value: Payload,
    /// Prompt plus completion tokens, counted against the deployment's per-minute usage.
    pub total_tokens: u64,
}

/// What a plugin or strategy sees: never the router itself.
pub struct RoutingContext<'a> {
    pub registry: &'a Registry,
    pub settings: &'a Settings,
    pub health: &'a Health,
    pub request: &'a RouterRequest,
    pub op: &'a Op,
    pub model_group: &'a str,
}

/// Which selection step the host is asked to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectStage {
    /// Pre-call checks and sub-routers registered as plugins: return the deployments that remain.
    Plugins,
    /// A strategy not yet ported: return the candidates with the pick first.
    Strategy(RoutingStrategy),
}

/// Routing events, delivered through [`Host::observe`].
#[derive(Clone, Debug, PartialEq)]
pub enum RouterEvent {
    Selected {
        model_group: String,
        deployment_id: String,
    },
    Succeeded {
        deployment_id: String,
    },
    AttemptFailed {
        deployment_id: String,
        kind: FailureKind,
        status: Option<u16>,
    },
    CooldownSet {
        deployment_id: String,
        cooldown: Duration,
    },
    Retry {
        model_group: String,
        retry: u32,
        delay: Duration,
    },
    Fallback {
        from: String,
        to: String,
        /// The order level of `to` a same-group hop walks to.
        order: Option<i64>,
        depth: u32,
        /// The fallback level `to` is tried at: 1 for the request's own fallbacks.
        level: u32,
        /// The new level's request metadata scope, copied from `parent`'s at the hop.
        scope: u32,
        parent: u32,
        /// The `max_fallbacks` the walk reads, stamped on the hop's request.
        max_fallbacks: u32,
    },
    /// `log_retry`: the router recorded a failure in `scope`'s request metadata
    /// (`previous_models`, `request_retry_count`) before retrying or falling back.
    RetryLogged {
        scope: u32,
        level: u32,
        /// The group the failed level served (`kwargs["model"]`).
        model: String,
        /// The retry number stamped before the failed try; `None` keeps the scope's own.
        attempted_retries: Option<u32>,
        cause: RetryCause,
    },
    /// A level of the fallback walk ended on an error no attempt raised: the group had no
    /// deployment for the caller (`unknown`) or none available. Python raises that error for
    /// the level, whatever its earlier attempts raised.
    LevelFailed {
        level: u32,
        model: String,
        unknown: bool,
    },
    /// The router raised a `mock_testing_*` failure for `model` instead of calling it.
    MockFailed {
        mock: MockFailure,
        level: u32,
        model: String,
        num_retries: Option<u32>,
    },
    /// A group's retry loop ran every retry and the last one failed: Python stamps
    /// `num_retries` and `max_retries` on that error.
    RetriesExhausted {
        model_group: String,
        attempted: u32,
        max: u32,
    },
}

/// Which error a [`RouterEvent::RetryLogged`] records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetryCause {
    /// The last attempt's failure.
    Attempt,
    /// The error the level ended on (`run_async_fallback`'s `original_exception`).
    Level,
    /// A retry found no deployment: the group was unknown or had none available.
    NoDeployment { model: String, unknown: bool },
}

#[async_trait]
pub trait Host: Send + Sync {
    /// Run one attempt against the picked deployment. The core records its outcome (cooldowns,
    /// allowed-fails counters, usage) itself; the host only runs it.
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError>;

    /// Run a selection stage the core does not implement.
    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error>;

    /// Access and budget checks before a cross-group fallback target.
    async fn allow_fallback(&self, request: &RouterRequest, target: &str) -> bool;

    async fn sleep(&self, duration: Duration);

    fn observe(&self, event: &RouterEvent);

    /// The shared store routing state lives in.
    fn cache(&self) -> Arc<dyn RouterCache>;

    /// Whether [`Self::invoke`] can run `op` on `provider`. Checked for every served op and
    /// deployment when the router is built or changed, so an unsupported pair fails the config,
    /// never a request. `PythonHost` runs every op; `NativeHost` only those with a Rust route.
    fn supports(&self, _op: &Op, _provider: &str) -> bool {
        true
    }

    /// Whether [`Self::select`] can run strategies and plugins.
    fn can_select(&self) -> bool {
        true
    }
}
