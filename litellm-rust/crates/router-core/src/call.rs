//! The call loop: select, run the attempt, then retry in the same group or fall back to another
//! until it succeeds or the policy is exhausted. Mirrors `Router.async_function_with_fallbacks`,
//! `async_function_with_fallbacks_common_utils`, `async_function_with_retries`,
//! `should_retry_this_error` and `_time_to_sleep_before_retry` in `litellm/router.py`, and
//! `run_async_fallback` in `router_utils/fallback_event_handlers.py`.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use serde_json::{Map, Value};

use tracing::warn;

use crate::{
    Core, Snapshot,
    config::Op,
    error::{AttemptError, ConfigError, Error, FailureKind, Unsupported},
    fallback::{FallbackKeys, FallbackTarget, Fallbacks},
    health::{GroupShape, should_retry_status},
    host::{Attempt, Output, RouterEvent, RouterRequest},
    registry::ResolveKind,
    retry::{RetryContext, resolve_retry_policy, should_retry},
    select::{Candidates, SelectInputs},
};

/// `INITIAL_RETRY_DELAY`, `MAX_RETRY_DELAY` and `JITTER` in `litellm/constants.py`.
const INITIAL_RETRY_DELAY: f64 = 0.5;
const MAX_RETRY_DELAY: f64 = 8.0;
const JITTER: f64 = 0.75;
/// `calculate_retry_after` obeys a provider's retry-after up to a minute.
const MAX_HONOURED_RETRY_AFTER: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub struct RouterResponse {
    pub output: Output,
    pub deployment_id: String,
    pub model_group: String,
    pub attempted_retries: u32,
    /// The retry budget of the loop a retry succeeded in (`x-litellm-max-retries`); `None`
    /// when the first attempt succeeded.
    pub max_retries: Option<u32>,
    pub attempted_fallbacks: u32,
}

/// A failed attempt with the deployment it ran on, when one was picked.
struct Failed {
    error: Error,
    deployment_id: Option<String>,
}

impl From<Error> for Failed {
    fn from(error: Error) -> Self {
        Self {
            error,
            deployment_id: None,
        }
    }
}

/// Where a level of the fallback walk sits: `depth` is `run_async_fallback`'s counter, one more
/// for every target tried, and `level` how many fallbacks deep the walk is.
#[derive(Clone, Copy, Debug, Default)]
struct Hop {
    depth: u32,
    level: u32,
}

/// What one level of the walk serves: a group, or the same group at a later `order`, with the
/// request params the walk's dict targets merged so far (`kwargs.update(target)`).
#[derive(Clone, Debug)]
struct Target {
    group: String,
    order: Option<i64>,
    params: Option<Arc<Map<String, Value>>>,
}

/// A step of a level's fallback chain: a configured target, or an order level of the group.
#[derive(Clone, Debug)]
enum Next {
    Fallback(FallbackTarget),
    Order(String, i64),
}

/// An attempt the walk already made (`fallback_attempt_key`).
#[derive(Clone, Debug, PartialEq)]
enum Tried {
    Target(FallbackTarget),
    Order(String, i64),
}

/// The fallback lists a call reads: the request's own where it passed them, else the router's.
struct Lists<'a> {
    fallbacks: Option<&'a Fallbacks>,
    context_window: Option<&'a Fallbacks>,
    content_policy: Option<&'a Fallbacks>,
    keys: &'a FallbackKeys,
}

impl<'a> Lists<'a> {
    fn of(snapshot: &'a Snapshot, request: &'a RouterRequest) -> Self {
        let dynamic = &snapshot.settings.dynamic;
        let pick = |own: &'a Option<Option<Fallbacks>>, router: Option<&'a Fallbacks>| match own {
            Some(own) => own.as_ref(),
            None => router,
        };
        Self {
            fallbacks: pick(&request.fallbacks, Some(&dynamic.fallbacks)),
            context_window: pick(
                &request.context_window_fallbacks,
                dynamic.context_window_fallbacks.as_ref(),
            ),
            content_policy: pick(
                &request.content_policy_fallbacks,
                dynamic.content_policy_fallbacks.as_ref(),
            ),
            keys: &dynamic.fallback_keys,
        }
    }

    fn has_regular(&self) -> bool {
        self.fallbacks.is_some_and(|list| !list.is_empty())
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl Core {
    pub(crate) async fn call(
        &self,
        op: Op,
        request: Arc<RouterRequest>,
    ) -> Result<RouterResponse, Error> {
        let snapshot = self.snapshot.load_full();
        let mut attempted = Vec::new();
        let target = Target {
            group: request.model.clone(),
            order: None,
            params: None,
        };
        self.with_fallbacks(
            &snapshot,
            &op,
            &request,
            target,
            Hop::default(),
            &mut attempted,
        )
        .await
    }

    /// One level of `async_function_with_fallbacks`: the target's retry loop, then its
    /// fallbacks. A level that fails raises its own error, whatever its fallbacks raised.
    fn with_fallbacks<'a>(
        &'a self,
        snapshot: &'a Snapshot,
        op: &'a Op,
        request: &'a Arc<RouterRequest>,
        target: Target,
        at: Hop,
        attempted: &'a mut Vec<Tried>,
    ) -> BoxFuture<'a, Result<RouterResponse, Error>> {
        Box::pin(async move {
            let error = match self.with_retries(snapshot, op, request, &target, at).await {
                Ok(response) => return Ok(response),
                Err(error) => error,
            };
            if let Error::NoHealthyDeployments { model } | Error::UnknownModel { model } = &error {
                self.host.observe(&RouterEvent::LevelFailed {
                    level: at.level,
                    model: model.clone(),
                    unknown: matches!(error, Error::UnknownModel { .. }),
                });
            }
            let Some(targets) = self.fallback_targets(snapshot, request, &target, &error) else {
                return Err(error);
            };
            // `run_async_fallback`: the depth is checked once on entry, then every target
            // tried at this level counts one more.
            let max_fallbacks = request
                .max_fallbacks
                .unwrap_or(snapshot.settings.max_fallbacks);
            if at.depth >= max_fallbacks {
                return Err(error);
            }
            let failed = Tried::Target(FallbackTarget::Name(target.group.clone()));
            if !attempted.contains(&failed) {
                attempted.push(failed);
            }
            let mut params = target.params.clone();
            let mut hop = at.depth;
            for next in targets {
                let (tried, group, order) = match &next {
                    Next::Order(group, order) => (
                        Tried::Order(group.clone(), *order),
                        group.clone(),
                        Some(*order),
                    ),
                    Next::Fallback(FallbackTarget::Name(name)) if *name == target.group => continue,
                    Next::Fallback(fallback) => {
                        if let Some(model) = fallback.model()
                            && model != target.group
                            && !self.host.allow_fallback(request, model).await
                        {
                            continue;
                        }
                        let group = fallback.model().unwrap_or(&target.group).to_owned();
                        (Tried::Target(fallback.identity()), group, None)
                    }
                };
                if attempted.contains(&tried) {
                    continue;
                }
                attempted.push(tried);
                if let Next::Fallback(fallback) = &next
                    && let Some(merged) = fallback.params()
                {
                    let mut combined = params.as_deref().cloned().unwrap_or_default();
                    combined.extend(merged.iter().map(|(k, v)| (k.clone(), v.clone())));
                    params = Some(Arc::new(combined));
                }
                hop += 1;
                self.host.observe(&RouterEvent::Fallback {
                    from: target.group.clone(),
                    to: group.clone(),
                    order,
                    depth: hop,
                    level: at.level + 1,
                });
                let next = Target {
                    group,
                    order,
                    params: params.clone(),
                };
                let at = Hop {
                    depth: hop,
                    level: at.level + 1,
                };
                if let Ok(mut response) = self
                    .with_fallbacks(snapshot, op, request, next, at, attempted)
                    .await
                {
                    response.attempted_fallbacks = hop;
                    return Ok(response);
                }
            }
            Err(error)
        })
    }

    /// The fallback targets for a failure, or `None` when it must be raised as is
    /// (`async_function_with_fallbacks_common_utils`).
    fn fallback_targets(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        target: &Target,
        error: &Error,
    ) -> Option<Vec<Next>> {
        if request.disable_fallbacks {
            return None;
        }
        let failure = match error {
            Error::Attempt(failure) => Some(failure),
            Error::NoHealthyDeployments { .. } => None,
            Error::Config(_) | Error::UnknownModel { .. } | Error::Cache(_) | Error::Retired => {
                return None;
            }
        };
        if failure.is_some_and(|f| f.kind == FailureKind::GuardrailIntervention) {
            return None;
        }
        let is_a = |name: &str| failure.is_some_and(|f| f.is_a(name));
        let lists = Lists::of(snapshot, request);
        // `fallback_lookup_groups`: the group being served, then the group the request was for.
        let mut lookup = vec![target.group.as_str()];
        if request.model != target.group {
            lookup.push(request.model.as_str());
        }
        let chain = |list: &Fallbacks| {
            if list.is_client_side(lists.keys) {
                Some(list.as_chain())
            } else {
                list.lookup_groups(&lookup, lists.keys)
            }
        };
        let fallbacks_of = |targets: Vec<FallbackTarget>| {
            targets.into_iter().map(Next::Fallback).collect::<Vec<_>>()
        };
        let skip_order = is_a("ContextWindowExceededError") || is_a("ContentPolicyViolationError");
        if !skip_order {
            let mut orders: Vec<i64> = snapshot
                .registry
                .group(&target.group)
                .unwrap_or_default()
                .iter()
                .filter(|d| d.usable_by_team(request.team_id.as_deref()))
                .filter_map(|d| d.order())
                .collect();
            orders.sort_unstable();
            orders.dedup();
            if orders.len() > 1 {
                let current = target.order.unwrap_or(orders[0]);
                let mut targets: Vec<Next> = orders
                    .into_iter()
                    .filter(|order| *order > current)
                    .map(|order| Next::Order(target.group.clone(), order))
                    .collect();
                if let Some(list) = lists.fallbacks {
                    targets.extend(fallbacks_of(chain(list).unwrap_or_default()));
                }
                if !targets.is_empty() {
                    return Some(targets);
                }
            }
        }
        if let Some(list) = lists.fallbacks
            && list.is_client_side(lists.keys)
        {
            return Some(fallbacks_of(list.as_chain()));
        }
        let specific = if is_a("ContextWindowExceededError") {
            lists.context_window
        } else if is_a("ContentPolicyViolationError") {
            lists.content_policy
        } else {
            None
        };
        if let Some(list) = specific {
            return list.lookup_groups(&lookup, lists.keys).map(fallbacks_of);
        }
        lists
            .fallbacks
            .and_then(|list| list.lookup_groups(&lookup, lists.keys))
            .map(fallbacks_of)
    }

    async fn with_retries(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        target: &Target,
        at: Hop,
    ) -> Result<RouterResponse, Error> {
        let mut inputs = SelectInputs {
            target_order: target.order,
            ..SelectInputs::default()
        };
        let counting = SelectInputs {
            count_only: true,
            ..SelectInputs::default()
        };
        let first = self
            .attempt(snapshot, op, request, target, &inputs, 0, at)
            .await;
        let mut failed = match first {
            Ok(response) => return Ok(response),
            Err(failed) => failed,
        };
        let Error::Attempt(failure) = &failed.error else {
            return Err(failed.error);
        };
        if failure.kind == FailureKind::GuardrailIntervention {
            return Err(failed.error);
        }
        let settings = &snapshot.settings;
        let mut num_retries = request
            .num_retries
            .or(failure.num_retries)
            .unwrap_or(settings.num_retries);
        let group_policies = request
            .model_group_retry_policy
            .as_ref()
            .unwrap_or(&settings.model_group_retry_policy);
        let policy_retries = (request.num_retries != Some(0))
            .then(|| {
                resolve_retry_policy(
                    settings.retry_policy.as_ref(),
                    &target.group,
                    Some(group_policies),
                )?
                .retries_for(failure)
            })
            .flatten();
        let policy_applies = policy_retries.is_some();
        num_retries = policy_retries.unwrap_or(num_retries);
        // `_async_get_healthy_deployments` finds no group for a deployment id, so a call by id
        // counts no deployments when deciding whether to retry.
        let counted = |c: &Candidates, n: usize| if c.kind == ResolveKind::Id { 0 } else { n };
        let all = self
            .candidates(snapshot, op, request, &target.group, &counting)
            .await
            .map(|c| counted(&c, c.all.len()))
            .unwrap_or(0);

        for retry in 0..=num_retries {
            let healthy = self
                .candidates(snapshot, op, request, &target.group, &counting)
                .await
                .map(|c| counted(&c, c.healthy.len()))
                .unwrap_or(0);
            // A retry that found no deployment raised that error in Python's loop: a retry
            // policy keeps going, `should_retry_this_error` raises it (nothing is healthy).
            let failure = match &failed.error {
                Error::Attempt(failure) => Some(failure),
                Error::NoHealthyDeployments { .. } if policy_applies => None,
                _ => return Err(failed.error),
            };
            if let Some(failure) = failure
                && !policy_applies
                && !self.should_retry(snapshot, request, failure, healthy, all)
            {
                return Err(failed.error);
            }
            if num_retries == 0 {
                return Err(failed.error);
            }
            if let (Some(status), Some(id)) = (
                failure.and_then(|failure| failure.status),
                &failed.deployment_id,
            ) && !should_retry_status(status)
                && !inputs.skipped_ids.contains(id)
            {
                inputs.skipped_ids.push(id.clone());
            }
            let delay = self.time_to_sleep(
                snapshot,
                failure.and_then(|failure| failure.retry_after),
                num_retries - retry,
                num_retries,
                healthy,
                all,
            );
            self.host.sleep(delay).await;
            if retry == num_retries {
                self.host.observe(&RouterEvent::RetriesExhausted {
                    model_group: target.group.clone(),
                    attempted: num_retries,
                    max: num_retries,
                });
                break;
            }
            self.host.observe(&RouterEvent::Retry {
                model_group: target.group.clone(),
                retry: retry + 1,
                delay,
            });
            match self
                .attempt(snapshot, op, request, target, &inputs, retry + 1, at)
                .await
            {
                Ok(mut response) => {
                    response.max_retries = Some(num_retries);
                    return Ok(response);
                }
                Err(next) => failed = next,
            }
        }
        Err(failed.error)
    }

    fn should_retry(
        &self,
        snapshot: &Snapshot,
        request: &RouterRequest,
        failure: &AttemptError,
        healthy: usize,
        all: usize,
    ) -> bool {
        let lists = Lists::of(snapshot, request);
        should_retry(
            failure,
            RetryContext {
                healthy,
                all,
                has_context_window_fallbacks: lists.context_window.is_some(),
                has_content_policy_fallbacks: lists.content_policy.is_some(),
                has_regular_fallbacks: lists.has_regular(),
            },
        )
    }

    /// `_time_to_sleep_before_retry` and `calculate_retry_after`.
    fn time_to_sleep(
        &self,
        snapshot: &Snapshot,
        retry_after: Option<Duration>,
        remaining_retries: u32,
        num_retries: u32,
        healthy: usize,
        all: usize,
    ) -> Duration {
        if all != 1 && healthy > 0 {
            return Duration::ZERO;
        }
        let jitter = JITTER * self.rng.next_f64();
        if let Some(retry_after) = retry_after
            && !retry_after.is_zero()
            && retry_after <= MAX_HONOURED_RETRY_AFTER
        {
            return retry_after + Duration::from_secs_f64(jitter);
        }
        let exponent = num_retries.saturating_sub(remaining_retries);
        let backoff = (INITIAL_RETRY_DELAY * 2f64.powi(exponent as i32))
            .max(snapshot.settings.retry_after.as_secs_f64())
            .min(MAX_RETRY_DELAY);
        Duration::from_secs_f64(backoff + jitter)
    }

    /// One attempt: select a deployment, run it through the host, record the outcome.
    #[allow(clippy::too_many_arguments)]
    async fn attempt(
        &self,
        snapshot: &Snapshot,
        op: &Op,
        request: &Arc<RouterRequest>,
        target: &Target,
        inputs: &SelectInputs,
        retry: u32,
        at: Hop,
    ) -> Result<RouterResponse, Failed> {
        let candidates = self
            .candidates(snapshot, op, request, &target.group, inputs)
            .await?;
        if candidates.healthy.is_empty() {
            return Err(Error::NoHealthyDeployments {
                model: candidates.model_group,
            }
            .into());
        }
        let deployment = self.pick(snapshot, op, request, &candidates).await?;
        self.host.observe(&RouterEvent::Selected {
            model_group: candidates.model_group.clone(),
            deployment_id: deployment.id.clone(),
        });
        if !self.host.supports(op, deployment.provider()) {
            return Err(Error::Config(ConfigError::Unsupported(Unsupported::Op {
                op: op.clone(),
                provider: deployment.provider().to_owned(),
            }))
            .into());
        }
        let attempt = Attempt {
            op: op.clone(),
            request: request.clone(),
            deployment: deployment.clone(),
            name: target.group.clone(),
            model_group: candidates.model_group.clone(),
            target_order: target.order,
            candidate_ids: candidates.all.iter().map(|d| d.id.clone()).collect(),
            retry,
            fallback_depth: at.depth,
            fallback_level: at.level,
            timeout: snapshot.settings.timeout,
            params: target.params.clone(),
        };
        // Every attempt runs through the host; the core records every outcome itself.
        match self.host.invoke(&attempt).await {
            Ok(output) => {
                if let Err(error) = self
                    .health
                    .record_success(&deployment, output.total_tokens)
                    .await
                {
                    warn!(deployment_id = %deployment.id, error = %error, "usage was not recorded");
                }
                self.host.observe(&RouterEvent::Succeeded {
                    deployment_id: deployment.id.clone(),
                });
                Ok(RouterResponse {
                    output,
                    deployment_id: deployment.id.clone(),
                    model_group: candidates.model_group,
                    attempted_retries: retry,
                    max_retries: None,
                    attempted_fallbacks: at.depth,
                })
            }
            Err(failure) => {
                self.record_failure(snapshot, &candidates, &deployment, &failure)
                    .await;
                Err(Failed {
                    error: Error::Attempt(failure),
                    deployment_id: Some(deployment.id.clone()),
                })
            }
        }
    }

    async fn record_failure(
        &self,
        snapshot: &Snapshot,
        candidates: &Candidates,
        deployment: &crate::registry::Deployment,
        failure: &AttemptError,
    ) {
        self.host.observe(&RouterEvent::AttemptFailed {
            deployment_id: deployment.id.clone(),
            kind: failure.kind,
            status: failure.status,
        });
        let shape = GroupShape {
            // The deployment's own group decides, also when the call named the deployment by id.
            single_deployment: snapshot
                .registry
                .group(&deployment.model_name)
                .map_or(candidates.all.len() == 1, |group| group.len() == 1),
            has_fallback_group: snapshot
                .settings
                .dynamic
                .fallback_targets(&[candidates.model_group.as_str()])
                .is_some(),
        };
        match self
            .health
            .record_failure(&snapshot.settings, deployment, shape, failure)
            .await
        {
            Ok(Some(cooldown)) => self.host.observe(&RouterEvent::CooldownSet {
                deployment_id: deployment.id.clone(),
                cooldown,
            }),
            Ok(None) => {}
            Err(error) => {
                warn!(deployment_id = %deployment.id, error = %error, "cooldown was not recorded");
            }
        }
    }
}
