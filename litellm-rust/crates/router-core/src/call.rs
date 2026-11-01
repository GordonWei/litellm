//! The call loop: select, run the attempt, then retry in the same group or fall back to another
//! until it succeeds or the policy is exhausted. Mirrors `Router.async_function_with_fallbacks`,
//! `async_function_with_fallbacks_common_utils`, `async_function_with_retries`,
//! `should_retry_this_error` and `_time_to_sleep_before_retry` in `litellm/router.py`, and
//! `run_async_fallback` in `router_utils/fallback_event_handlers.py`.

use std::{collections::HashSet, future::Future, pin::Pin, sync::Arc, time::Duration};

use tracing::warn;

use crate::{
    Core, Snapshot,
    config::Op,
    error::{AttemptError, ConfigError, Error, FailureKind, Unsupported},
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

/// A fallback target: a group, or the same group at a later `order`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Target {
    group: String,
    order: Option<i64>,
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl Core {
    pub(crate) async fn call(
        &self,
        op: Op,
        request: Arc<RouterRequest>,
    ) -> Result<RouterResponse, Error> {
        let snapshot = self.snapshot.load_full();
        let mut attempted = HashSet::new();
        let target = Target {
            group: request.model.clone(),
            order: None,
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
        attempted: &'a mut HashSet<Target>,
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
            if at.depth >= snapshot.settings.max_fallbacks {
                return Err(error);
            }
            attempted.insert(Target {
                group: target.group.clone(),
                order: None,
            });
            let mut hop = at.depth;
            for next in targets {
                if next.order.is_none() && next.group == target.group {
                    continue;
                }
                if next.group != target.group
                    && !self.host.allow_fallback(request, &next.group).await
                {
                    continue;
                }
                if !attempted.insert(next.clone()) {
                    continue;
                }
                hop += 1;
                self.host.observe(&RouterEvent::Fallback {
                    from: target.group.clone(),
                    to: next.group.clone(),
                    order: next.order,
                    depth: hop,
                    level: at.level + 1,
                });
                if let Ok(mut response) = self
                    .with_fallbacks(
                        snapshot,
                        op,
                        request,
                        next,
                        Hop {
                            depth: hop,
                            level: at.level + 1,
                        },
                        attempted,
                    )
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
    ) -> Option<Vec<Target>> {
        if request.disable_fallbacks {
            return None;
        }
        let kind = match error {
            Error::Attempt(failure) => failure.kind,
            Error::NoHealthyDeployments { .. } => FailureKind::RateLimit,
            Error::Config(_) | Error::UnknownModel { .. } | Error::Cache(_) | Error::Retired => {
                return None;
            }
        };
        if kind == FailureKind::GuardrailIntervention {
            return None;
        }
        let dynamic = &snapshot.settings.dynamic;
        // `fallback_lookup_groups`: the group being served, then the group the request was for.
        let mut lookup = vec![target.group.as_str()];
        if request.model != target.group {
            lookup.push(request.model.as_str());
        }
        let groups = |names: Vec<String>| {
            names
                .into_iter()
                .map(|group| Target { group, order: None })
                .collect::<Vec<_>>()
        };
        let skip_order = matches!(
            kind,
            FailureKind::ContextWindowExceeded | FailureKind::ContentPolicyViolation
        );
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
                let mut targets: Vec<Target> = orders
                    .into_iter()
                    .filter(|order| *order > current)
                    .map(|order| Target {
                        group: target.group.clone(),
                        order: Some(order),
                    })
                    .collect();
                targets.extend(groups(
                    dynamic
                        .fallbacks
                        .lookup_groups(&lookup, &dynamic.fallback_keys)
                        .unwrap_or_default(),
                ));
                if !targets.is_empty() {
                    return Some(targets);
                }
            }
        }
        let specific = match kind {
            FailureKind::ContextWindowExceeded => dynamic.context_window_fallbacks.as_ref(),
            FailureKind::ContentPolicyViolation => dynamic.content_policy_fallbacks.as_ref(),
            _ => None,
        };
        if let Some(fallbacks) = specific {
            return fallbacks
                .lookup_groups(&lookup, &dynamic.fallback_keys)
                .map(groups);
        }
        dynamic.fallback_targets(&lookup).map(groups)
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
                && !self.should_retry(snapshot, failure, healthy, all)
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
        failure: &AttemptError,
        healthy: usize,
        all: usize,
    ) -> bool {
        let dynamic = &snapshot.settings.dynamic;
        should_retry(
            failure,
            RetryContext {
                healthy,
                all,
                has_context_window_fallbacks: dynamic.context_window_fallbacks.is_some(),
                has_content_policy_fallbacks: dynamic.content_policy_fallbacks.is_some(),
                has_regular_fallbacks: dynamic.has_regular_fallbacks(),
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
