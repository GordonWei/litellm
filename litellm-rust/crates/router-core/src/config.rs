//! What the router is told: [`RouterConfig`], fixed for the router's life, and
//! [`RouterDynamicConfig`], a versioned snapshot read once per call. The router never reads
//! process globals.

use std::{collections::BTreeMap, time::Duration};

use serde::Deserialize;

use crate::{
    error::AttemptError,
    registry::{DeploymentSpec, RoutingGroupSpec},
};

/// `DEFAULT_COOLDOWN_TIME_SECONDS` in `litellm/constants.py`.
pub const DEFAULT_COOLDOWN_TIME: Duration = Duration::from_secs(5);
/// `ROUTER_MAX_FALLBACKS` in `litellm/constants.py`.
pub const DEFAULT_MAX_FALLBACKS: u32 = 5;

/// The router operation a call runs, spelled as the Python router's endpoint method.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
)]
pub enum Op {
    #[strum(serialize = "acompletion")]
    Completion,
    #[strum(serialize = "atext_completion")]
    TextCompletion,
    #[strum(serialize = "aembedding")]
    Embedding,
    #[strum(serialize = "aresponses")]
    Responses,
    #[strum(serialize = "aanthropic_messages")]
    Messages,
    #[strum(serialize = "aimage_generation")]
    ImageGeneration,
    #[strum(serialize = "atranscription")]
    Transcription,
    #[strum(serialize = "aspeech")]
    Speech,
    #[strum(serialize = "aocr")]
    Ocr,
    #[strum(default, transparent)]
    Other(String),
}

/// `routing_strategy`. Only simple-shuffle is native; the rest go through `Host::select`.
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
)]
pub enum RoutingStrategy {
    #[default]
    #[strum(serialize = "simple-shuffle")]
    SimpleShuffle,
    #[strum(serialize = "least-busy")]
    LeastBusy,
    #[strum(serialize = "usage-based-routing")]
    UsageBased,
    #[strum(serialize = "usage-based-routing-v2")]
    UsageBasedV2,
    #[strum(serialize = "latency-based-routing")]
    LatencyBased,
    #[strum(serialize = "cost-based-routing")]
    CostBased,
    #[strum(default, transparent)]
    Other(String),
}

impl RoutingStrategy {
    pub fn is_native(&self) -> bool {
        matches!(self, Self::SimpleShuffle)
    }
}

/// `PluginKind` in `litellm/router_api.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, strum::Display)]
pub enum PluginKind {
    #[serde(rename = "auto")]
    #[strum(serialize = "auto")]
    Auto,
    #[serde(rename = "complexity")]
    #[strum(serialize = "complexity")]
    Complexity,
    #[serde(rename = "quality")]
    #[strum(serialize = "quality")]
    Quality,
    #[serde(rename = "adaptive")]
    #[strum(serialize = "adaptive")]
    Adaptive,
    #[serde(rename = "routing")]
    #[strum(serialize = "routing")]
    Routing,
    #[serde(rename = "budget_limiter")]
    #[strum(serialize = "budget_limiter")]
    BudgetLimiter,
    #[serde(rename = "prompt_caching_pins")]
    #[strum(serialize = "prompt_caching_pins")]
    PromptCachingPins,
}

/// One registered plugin. Its behaviour lives in the host until it is ported.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PluginSpec {
    pub kind: PluginKind,
    pub name: String,
    #[serde(default)]
    pub config: serde_json::Value,
}

/// `RetryPolicy` in `litellm/types/router.py`: retries per exception type.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[allow(non_snake_case)]
pub struct RetryPolicy {
    pub BadRequestErrorRetries: Option<u32>,
    pub AuthenticationErrorRetries: Option<u32>,
    pub TimeoutErrorRetries: Option<u32>,
    pub RateLimitErrorRetries: Option<u32>,
    pub ContentPolicyViolationErrorRetries: Option<u32>,
    pub InternalServerErrorRetries: Option<u32>,
    pub ServiceUnavailableErrorRetries: Option<u32>,
    pub NotFoundErrorRetries: Option<u32>,
    pub DefaultRetries: Option<u32>,
}

/// `AllowedFailsPolicy` in `litellm/types/router.py`: allowed fails per minute per exception type.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[allow(non_snake_case)]
pub struct AllowedFailsPolicy {
    pub BadRequestErrorAllowedFails: Option<u32>,
    pub AuthenticationErrorAllowedFails: Option<u32>,
    pub TimeoutErrorAllowedFails: Option<u32>,
    pub RateLimitErrorAllowedFails: Option<u32>,
    pub ContentPolicyViolationErrorAllowedFails: Option<u32>,
    pub InternalServerErrorAllowedFails: Option<u32>,
    pub ServiceUnavailableErrorAllowedFails: Option<u32>,
    pub BadGatewayErrorAllowedFails: Option<u32>,
    pub NotFoundErrorAllowedFails: Option<u32>,
}

impl AllowedFailsPolicy {
    /// `_core_get_allowed_fails_from_policy`: the first set field whose exception class the
    /// failure is an instance of, in Python's order.
    pub fn allowed_fails_for(&self, failure: &AttemptError) -> Option<u32> {
        [
            ("AuthenticationError", self.AuthenticationErrorAllowedFails),
            ("Timeout", self.TimeoutErrorAllowedFails),
            ("RateLimitError", self.RateLimitErrorAllowedFails),
            (
                "ContentPolicyViolationError",
                self.ContentPolicyViolationErrorAllowedFails,
            ),
            ("BadRequestError", self.BadRequestErrorAllowedFails),
            ("InternalServerError", self.InternalServerErrorAllowedFails),
            (
                "ServiceUnavailableError",
                self.ServiceUnavailableErrorAllowedFails,
            ),
            ("BadGatewayError", self.BadGatewayErrorAllowedFails),
            ("NotFoundError", self.NotFoundErrorAllowedFails),
        ]
        .into_iter()
        .find_map(|(class, allowed)| allowed.filter(|_| failure.is_a(class)))
    }
}

pub use crate::fallback::{FallbackKeys, Fallbacks};

/// Static config, fixed for the router's life.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    pub model_list: Vec<DeploymentSpec>,
    pub routing_strategy: RoutingStrategy,
    pub routing_groups: Vec<RoutingGroupSpec>,
    pub plugins: Vec<PluginSpec>,
    /// Router default `cooldown_time`; `None` is `DEFAULT_COOLDOWN_TIME_SECONDS`.
    pub cooldown_time: Option<f64>,
    pub disable_cooldowns: bool,
    pub allowed_fails_policy: Option<AllowedFailsPolicy>,
    pub retry_policy: Option<RetryPolicy>,
    /// `model_group_retry_policy`: a group's own policy, which wins over `retry_policy`.
    pub model_group_retry_policy: BTreeMap<String, RetryPolicy>,
    /// `retry_after`: the minimum backoff between retries, in seconds.
    pub retry_after: f64,
    pub timeout: Option<f64>,
    pub request_timeout: Option<f64>,
    pub drop_params: Option<bool>,
    pub enable_pre_call_checks: bool,
    /// `Router(plugins=[...])` is set: the host runs the routing-plugin pipeline.
    pub routing_plugins: bool,
    /// The host's `CustomLogger` deployment filters run on every async candidate list
    /// (`async_callback_filter_deployments`).
    pub callback_filters: bool,
    pub enable_health_check_routing: bool,
    /// Deployment ids that are a provider's default and never cool down
    /// (`provider_default_deployment_ids`).
    pub provider_default_deployment_ids: Vec<String>,
    /// The operations this router must serve: `Host::supports` must accept each one for every
    /// configured provider, or the router fails to build.
    pub served_ops: Vec<Op>,
}

/// Dynamic config: a versioned snapshot read once per call.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RouterDynamicConfig {
    pub version: u64,
    /// `model_group_alias` as Python spells it: each alias maps to a group name, or to
    /// `{"model": <group>, "hidden": <bool>}`.
    #[serde(alias = "model_group_alias", deserialize_with = "alias_targets")]
    pub model_alias_map: BTreeMap<String, String>,
    pub fallbacks: Fallbacks,
    pub default_fallbacks: Vec<String>,
    pub context_window_fallbacks: Option<Fallbacks>,
    pub content_policy_fallbacks: Option<Fallbacks>,
    pub num_retries: Option<u32>,
    pub max_fallbacks: u32,
    /// `allowed_fails`. `Some` means it was set explicitly, which switches cooldowns from the
    /// failure-ratio rule to the allowed-fails counter (`_is_allowed_fails_set_on_router`).
    pub allowed_fails: Option<u32>,
    /// Other spellings a fallback key matches.
    pub fallback_keys: FallbackKeys,
}

/// One `model_group_alias` value: a group name, or `{"model": <group>, "hidden": <bool>}`.
#[derive(Deserialize)]
#[serde(untagged)]
enum AliasTarget {
    Group(String),
    Entry { model: String },
}

fn alias_targets<'de, D>(deserializer: D) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let aliases = BTreeMap::<String, AliasTarget>::deserialize(deserializer)?;
    Ok(aliases
        .into_iter()
        .map(|(alias, target)| match target {
            AliasTarget::Group(group) | AliasTarget::Entry { model: group } => (alias, group),
        })
        .collect())
}

impl Default for RouterDynamicConfig {
    fn default() -> Self {
        Self {
            version: 0,
            model_alias_map: BTreeMap::new(),
            fallbacks: Fallbacks::default(),
            default_fallbacks: Vec::new(),
            context_window_fallbacks: None,
            content_policy_fallbacks: None,
            num_retries: None,
            max_fallbacks: DEFAULT_MAX_FALLBACKS,
            allowed_fails: None,
            fallback_keys: FallbackKeys::default(),
        }
    }
}

impl RouterDynamicConfig {
    /// The regular fallback targets for the lookup groups, then `default_fallbacks`, which
    /// Python folds into `fallbacks` as `{"*": [...]}`: what `_has_fallback_group` reads.
    pub fn fallback_targets(&self, groups: &[&str]) -> Option<Vec<String>> {
        if groups
            .iter()
            .any(|group| self.fallbacks.has_keyed_chain(group, &self.fallback_keys))
        {
            return Some(Vec::new());
        }
        (!self.default_fallbacks.is_empty()).then(|| self.default_fallbacks.clone())
    }

    pub fn has_regular_fallbacks(&self) -> bool {
        !self.fallbacks.is_empty() || !self.default_fallbacks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::group_name(json!({"model_group_alias": {"gpt": "gpt-4"}}))]
    #[case::hidden_entry(json!({"model_group_alias": {"gpt": {"model": "gpt-4", "hidden": true}}}))]
    #[case::native_name(json!({"model_alias_map": {"gpt": "gpt-4"}}))]
    fn model_group_alias_reads_both_python_spellings(#[case] wire: serde_json::Value) {
        let dynamic: RouterDynamicConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(
            dynamic.model_alias_map,
            BTreeMap::from([("gpt".to_owned(), "gpt-4".to_owned())])
        );
    }

    #[rstest]
    #[case::native("simple-shuffle", RoutingStrategy::SimpleShuffle)]
    #[case::unknown("my-strategy", RoutingStrategy::Other("my-strategy".to_owned()))]
    fn routing_strategy_parses_its_wire_name(
        #[case] wire: &str,
        #[case] expected: RoutingStrategy,
    ) {
        assert_eq!(wire.parse::<RoutingStrategy>().unwrap(), expected);
        assert_eq!(expected.to_string(), wire);
    }
}
