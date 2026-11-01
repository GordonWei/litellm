//! [`Host`]: the router core's only boundary. Everything the core needs from the embedding
//! process goes through it, and every attempt runs through [`Host::invoke`]: the core never runs
//! an operation itself. `PythonHost` serves it in a Python process (attempts call `litellm.<op>`);
//! `NativeHost` serves it in the Rust server (attempts call the `inference-*` crates).

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    cache::RouterCache,
    config::{Op, RoutingStrategy},
    error::{AttemptError, Error, FailureKind, Payload},
    health::Health,
    precall::PreCallFacts,
    prerouting::{PreRoute, PreRouted},
    registry::{Deployment, Registry},
    settings::Settings,
    strategy::StrategyScope,
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
    /// The deployment a call on an existing resource (a video id) names: picked whenever the
    /// filters keep it (`_narrow_to_video_deployment`).
    pub pinned_deployment_id: Option<String>,
    /// A per-request `routing_strategy` override, as Python normalizes it.
    pub routing_strategy: Option<String>,
    /// Whether tag routing may apply to this request (router, request or model setting), so the
    /// host's tag stage runs.
    pub filter_tags: bool,
    /// A pass-through pick: only `use_in_pass_through` deployments may serve it.
    pub pass_through: Option<crate::select::PassThrough>,
    /// `_target_order`: the order level to pick from instead of the lowest.
    pub target_order: Option<i64>,
    /// `_excluded_deployment_ids`: never picked, even when that leaves nothing.
    pub excluded_ids: Vec<String>,
    /// `_retry_skipped_deployment_ids`: skipped unless that leaves nothing.
    pub retry_skipped_ids: Vec<String>,
    /// The request carries an image (`_request_carries_image`), so deployments that declare no
    /// vision are skipped while any other remains.
    pub has_image: bool,
    /// An async pick: the host's pre-routing step runs before each pick resolves its name.
    pub pre_routes: bool,
    /// The requested name is no model group (a search tool name): nothing is selected, every
    /// attempt is handed to the host as is, and no outcome is recorded for a deployment
    /// (`_asearch_with_fallbacks`, whose helper picks among the router's search tools).
    pub unrouted: bool,
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
    /// The name the request resolved from (Python's `model`), which strategies key state by.
    pub name: &'a str,
    /// Whose selector a strategy stage reads.
    pub scope: StrategyScope,
    /// The deployments the name resolved to for this caller, before health filters
    /// (`request_deployments`), for the tag stage.
    pub pool: &'a [Arc<Deployment>],
}

/// Which selection step the host is asked to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectStage {
    /// Pre-call checks and sub-routers registered as plugins: return the deployments that remain.
    Plugins,
    /// A strategy not yet ported: return the candidates with the pick first.
    Strategy(RoutingStrategy),
    /// Tag routing (`get_deployments_for_tag`), not yet ported: return the deployments that remain.
    Tags,
    /// `_deployment_declares_no_vision`: return the candidates that declare no vision support.
    TextOnly,
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
    },
    /// A group's retry loop ran every retry and the last one failed: Python stamps
    /// `num_retries` and `max_retries` on that error.
    RetriesExhausted {
        model_group: String,
        attempted: u32,
        max: u32,
    },
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

    /// Pre-routing before a pick resolves `ask.name`: the model group to resolve instead, a
    /// custom strategy's deployment to serve, or `None` to resolve the name itself. Asked only
    /// for picks that pre-route (`pre_routes`).
    async fn pre_route(&self, ask: &PreRoute<'_>) -> Result<Option<PreRouted>, Error> {
        let _ = ask;
        Ok(None)
    }

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

    /// The facts the `enable_pre_call_checks` filter decides by (token count, context windows,
    /// param support), for `candidates` and the deployments in `ctx.pool`. A host without them
    /// reports no input, so nothing is checked.
    /// Run the routing plugins for `ctx.name` when `run` (the async pick's pre-routing hook), and
    /// report the candidate models they left on the request, if any.
    async fn routing_plugin_models(
        &self,
        _ctx: &RoutingContext<'_>,
        _run: bool,
    ) -> Result<Option<Vec<String>>, Error> {
        Ok(None)
    }

    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        _candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, Error> {
        Ok(PreCallFacts::default())
    }
}
