"""
``RustCore``: the router core served by litellm-rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns. It reaches outside
itself only through its host, and the host never calls back into a Python router:

- every attempt runs through the host: ``router_attempt`` builds the request for the one
  deployment Rust picked and ``PythonHost.invoke(op)`` calls ``litellm.<op>`` with it
- the core records every outcome (cooldowns, allowed-fails counters, usage) itself, in the routing
  cache the host builds and owns (``PythonHost.routing_cache``)
- sleeps, fallback access and budget checks, and cache reads and writes are host steps

``RustCore`` does not build, wrap or read a ``PythonCore``. Settings are resolved from the
``RouterConfig`` and the host's process settings; the views (``registry()``, ``settings()``,
``health()``) are read from the Rust handle. A member the Rust core does not serve raises
``RustCoreUnsupported`` (an ``AttributeError``) instead of reaching a Python core, so deleting
``router_python_core.py`` cannot change what this core does.
"""

import contextlib
import dataclasses
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._internal_context import service_target
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    apply_remaining_usage_headers,
    apply_response_model_id,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import (
    format_content_policy_fallback_hint,
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    truncate_fallback_error_detail,
)
from litellm.litellm_core_utils.routing_groups import validate_routing_strategy
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.router_api import (
    SETTING_FIELDS,
    ApplyResult,
    ChangeSet,
    RouterHealth,
    RouterRegistry,
    RouterSettings,
    RoutingContext,
    RoutingObservers,
)
from litellm.router_attempt import for_requested_model, request_from_call
from litellm.router_config import RouterConfig, RouterDynamicConfig
from litellm.router_python_host import PythonHost
from litellm.router_rust_callbacks import OptionalChecks, RustOptionalChecks
from litellm.router_rust_endpoints import (
    AttemptCall,
    Plan,
    RustEndpoints,
    attempt_call,
    pinned_video_deployment,
    plan_for,
    stamp_error,
    unrouted_fields,
)
from litellm.router_rust_fallbacks import (
    MOCK_FLAGS,
    RustFallbacks,
    mock_error,
    mock_failure,
    refuse_malformed_fallbacks,
    rejected_fallback_target,
)
from litellm.router_rust_health import (
    RustHealthCallbacks,
    cooled_in_running_loop,
    failure_status,
)
from litellm.router_rust_members import RustRegistryMembers
from litellm.router_rust_plugins import SYNC_PATH_MESSAGE, candidate_models, run_plugins
from litellm.router_rust_precall import PreCallFacts, declares_no_vision, request_carries_image
from litellm.router_rust_prerouting import PreRouter, PreRoutingStrategies
from litellm.router_rust_registry import HostRegistry, RustRegistry, hosted_registry_view
from litellm.router_rust_retries import RustRetries, refuse_unserved_retry_policies, resolved_retry_policy
from litellm.router_rust_selection import (
    SELECTOR_ATTRIBUTES,
    StrategySelectors,
    group_documents,
    group_settings,
    normalize_strategy,
    parse_groups,
    pick_async,
    pick_sync,
    picked_id,
    routing_pre_call_checks,
)
from litellm.router_rust_settings import (
    MISSING,
    SettingsUpdate,
    apply_construction_effects,
    argument_attribute,
    default_litellm_params,
    initial_plain,
    read_settings_update,
    request_timeout,
    settings_view,
)
from litellm.router_rust_shaping import Shaping
from litellm.router_rust_slots import enter_deployment_slot
from litellm.router_rust_streaming import (
    StreamAttempt,
    continuation_flag,
    continuation_routing,
    open_attempt_stream,
    open_attempt_stream_sync,
    register_continuation_prefill_check,
    shape_attempt_kwargs,
)
from litellm.router_rust_walk import FallbackOutcomes, WalkMetadata, await_all
from litellm.router_utils.cooldown_cache import ROUTER_COOLDOWNS_TARGET
from litellm.router_utils.router_callbacks.track_deployment_metrics import find_deployment_metadata
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.router import Deployment, RouterErrors, RouterRateLimitError, RoutingGroup, RoutingStrategy
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.rust_bridge._native import (  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        RouterCall,
        RouterCoreHandle,
    )
    from litellm.types.router import CustomRoutingStrategyBase, RoutingPlugin

# Async endpoints the Rust core routes: select a deployment, then run the attempt on the host.
RUST_SERVED_OPS: Final[frozenset[str]] = frozenset(
    {
        "acompletion",
        "atext_completion",
        "aembedding",
        "aresponses",
        "aanthropic_messages",
        "aimage_generation",
        "atranscription",
        "aspeech",
        "arerank",
        "amoderation",
    }
)
_FAILURE_KINDS: Final = (
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GuardrailIntervention",
    "CallerCredentialError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_STEP: Final = TypeAdapter(Mapping[str, object])
_PROVIDERS: Final = tuple(provider.value for provider in LlmProviders)
_DEFAULT_COOLDOWN_SECONDS: Final = 5.0

# Constructor arguments this core reads. Any other argument set to something other than its
# unset value keeps the router on the Python core while both exist (make_core decides that from
# the config alone, before any core is built).
_SERVED_ARGUMENTS: Final = frozenset(
    {
        "model_list",
        "routing_strategy",
        "routing_strategy_args",
        "routing_groups",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
        "model_group_alias",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "cooldown_time",
        "disable_cooldowns",
        "allowed_fails_policy",
        "retry_after",
        "timeout",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
        "enable_mid_stream_fallback_continuation",
        "redis_url",
        "redis_host",
        "redis_port",
        "redis_password",
        "redis_db",
        "cache_kwargs",
        "set_verbose",
        "debug_level",
        # Read host-side by the endpoint plans (litellm/router_rust_endpoints.py).
        "stream_timeout",
        "cache_responses",
        "assistants_config",
        "search_tools",
        "guardrail_list",
        "default_priority",
        "polling_interval",
        "ignore_invalid_deployments",
        # Read host-side by the pre-routing strategies (litellm/router_rust_prerouting.py).
        "auto_router_capability_limit",
        # Decided in router-core (precall.rs); its facts come from litellm/router_rust_precall.py.
        "enable_pre_call_checks",
        # Run host-side (litellm/router_rust_plugins.py); router-core filters on what they leave.
        "plugins",
        "retry_policy",
        "model_group_retry_policy",
        # Built host-side as CustomLogger checks (litellm/router_rust_callbacks.py); router-core runs
        # their deployment filters through its callback_filters stage.
        "optional_pre_call_checks",
        "provider_budget_config",
        "deployment_affinity_ttl_seconds",
        "session_affinity_repin_on_failover",
        "model_group_affinity_config",
    }
)
# Arguments that may be passed at their default without asking for anything.
_UNSET_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "client_ttl": 3600,
        "cache_responses": False,
        "enable_pre_call_checks": False,
        "enable_tag_filtering": False,
        "retry_policy": None,
        "model_group_retry_policy": {},
    }
)
# Per-request routing overrides this core does not read yet: such a request is refused, not
# rerouted to another core.
_UNSERVED_REQUEST_KEYS: Final = frozenset(
    {
        "default_fallbacks",
        "retry_policy",
    }
)
# Settings a changeset may change. Static ones rebuild the Rust core (settings are fixed for a
# core's life); dynamic ones are handed to it as a new dynamic document.
_STATIC_SETTINGS: Final = frozenset(
    {
        "routing_strategy",
        "routing_groups",
        "cooldown_time",
        "disable_cooldowns",
        "allowed_fails_policy",
        "retry_after",
        "timeout",
        "enable_pre_call_checks",
    }
)
_DYNAMIC_SETTINGS: Final = frozenset(
    {
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "model_group_alias",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
        "routing_strategy_args",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
    }
)


# Plain settings `update_settings` sets that the Rust core's retry loop reads.
_RETRY_POLICY_SETTINGS: Final = ("retry_policy", "model_group_retry_policy")


# Settings converted by `Resolved._selection_setting`.
_SELECTION_SETTINGS: Final = frozenset(
    {
        "routing_strategy",
        "routing_strategy_args",
        "enable_tag_filtering",
        "tag_filtering_match_any",
        "tag_routing_prefix",
    }
)


class RustCoreUnsupported(AttributeError):
    """A member or request the Rust core does not serve. It is never forwarded to a Python core."""


def _cache_target(key: str) -> str:
    """The service target a routing cache read or write is attributed to, as the Python core's."""
    return ROUTER_COOLDOWNS_TARGET if key.endswith(":cooldown") else ROUTER_USAGE_TARGET


# --- failures -----------------------------------------------------------------------------------


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    if is_guardrail_intervention(error):  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
        return "GuardrailIntervention"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def _retry_after_seconds(error: BaseException) -> float | None:
    response: Final = getattr(error, "response", None)
    headers: Final[object] = getattr(error, "litellm_response_headers", None) or getattr(response, "headers", None)
    get: Final[object] = getattr(headers, "get", None)
    if headers is None or not callable(get):
        return None
    milliseconds: Final[object] = get("retry-after-ms")
    seconds: Final[object] = get("retry-after")
    try:
        value: Final = float(str(milliseconds)) / 1000 if milliseconds is not None else float(str(seconds))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "retry_after": _retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "detail": str(error)[:500],
        "classes": [cls.__name__ for cls in type(error).__mro__],
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


# --- settings -----------------------------------------------------------------------------------


def _json_default(value: object) -> object:
    dump: Final = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else str(value)


def _plain(value: object) -> object:
    """``value`` as plain JSON values, the shape the Rust core reads."""
    loaded: Final[object] = json.loads(json.dumps(value, default=_json_default))  # pyright: ignore[reportAny]  # plain JSON
    return loaded


def _document(value: object) -> Mapping[str, object]:
    """A mapping as the plain JSON document the Rust core reads."""
    return as_str_mapping(_plain(value)) or {}


def _as_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()  # pyright: ignore[reportUnknownVariableType]  # a list's items are read as objects


@dataclasses.dataclass(frozen=True, slots=True)
class Resolved:
    """The router's settings, resolved from its arguments and the process settings the way the
    Python router's constructor resolves them. Replaced, never changed, by ``apply``."""

    routing_strategy: str
    routing_strategy_args: Mapping[str, object]
    num_retries: int
    max_fallbacks: int
    allowed_fails: int
    explicit_allowed_fails: int | None
    cooldown_time: float
    disable_cooldowns: bool | None
    allowed_fails_policy: object
    retry_after: float
    timeout: float | None
    fallbacks: list[object] | None  # mutable-ok: the setting as Python spells it
    context_window_fallbacks: list[object] | None  # mutable-ok: same
    content_policy_fallbacks: list[object] | None  # mutable-ok: same
    model_group_alias: Mapping[str, object]
    default_litellm_params: Mapping[str, object]
    fallback_access_check: object
    fallback_budget_check: object
    enable_mid_stream_fallback_continuation: bool = False
    routing_groups: "tuple[RoutingGroup, ...]" = ()
    enable_tag_filtering: bool = False
    tag_filtering_match_any: bool = True
    tag_routing_prefix: str = ""
    enable_pre_call_checks: bool = False
    routing_plugins: "tuple[RoutingPlugin, ...]" = ()
    retry_policy: object = None
    model_group_retry_policy: object = None
    assigned: frozenset[str] = frozenset()

    @classmethod
    def of(cls, config: RouterConfig, dynamic: RouterDynamicConfig) -> "Resolved":
        get: Final = config.get
        # The process list is the router's own, as in the Python router: default_fallbacks are
        # appended to it in place.
        configured: Final[list[object] | None] = (  # mutable-ok: the router's live list, as Python keeps it
            list(_as_list(get("fallbacks"))) or (dynamic.fallbacks if dynamic.fallbacks else None)  # pyright: ignore[reportAssignmentType]  # the process list as given
        )
        refuse_malformed_fallbacks(configured)
        default_fallbacks: Final = get("default_fallbacks") or dynamic.default_fallbacks
        if isinstance(default_fallbacks, str):
            raise TypeError(
                f"default_fallbacks must be a list of model names, got '{default_fallbacks}'. Use ['{default_fallbacks}']"
            )
        if default_fallbacks is not None and configured is not None:
            configured.append({"*": default_fallbacks})
        fallbacks: Final[list[object] | None] = (  # mutable-ok: see above
            [{"*": default_fallbacks}] if default_fallbacks is not None and configured is None else configured
        )
        content_policy: Final = list(_as_list(get("content_policy_fallbacks"))) or (
            list(dynamic.content_policy_fallbacks) if dynamic.content_policy_fallbacks else None
        )
        refuse_malformed_fallbacks(content_policy)
        num_retries: Final = get("num_retries")
        max_fallbacks: Final = get("max_fallbacks")
        allowed_fails: Final = get("allowed_fails")
        timeout: Final = get("timeout")
        strategy: Final = get("routing_strategy")
        validate_routing_strategy(strategy)  # pyright: ignore[reportArgumentType]  # validated as given
        return cls(
            routing_strategy=normalize_strategy(strategy) or "simple-shuffle",
            routing_strategy_args=MappingProxyType(dict(as_str_mapping(get("routing_strategy_args")) or {})),
            num_retries=int(str(num_retries))
            if num_retries is not None
            else dynamic.num_retries
            if dynamic.num_retries is not None
            else 2,
            max_fallbacks=int(str(max_fallbacks))
            if max_fallbacks is not None
            else dynamic.max_fallbacks
            if dynamic.max_fallbacks is not None
            else dynamic.router_max_fallbacks,
            allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else dynamic.allowed_fails,
            explicit_allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else None,
            cooldown_time=float(str(get("cooldown_time") or _DEFAULT_COOLDOWN_SECONDS)),
            disable_cooldowns=_optional_bool(get("disable_cooldowns")),
            allowed_fails_policy=get("allowed_fails_policy"),
            retry_after=float(str(get("retry_after") or 0)),
            timeout=float(str(timeout)) if timeout else dynamic.request_timeout,
            fallbacks=fallbacks,
            context_window_fallbacks=list(_as_list(get("context_window_fallbacks")))
            or (list(dynamic.context_window_fallbacks) if dynamic.context_window_fallbacks else None),
            content_policy_fallbacks=content_policy,
            model_group_alias=MappingProxyType(dict(as_str_mapping(get("model_group_alias")) or {})),
            default_litellm_params=MappingProxyType(dict(as_str_mapping(get("default_litellm_params")) or {})),
            fallback_access_check=get("fallback_access_check"),
            fallback_budget_check=get("fallback_budget_check"),
            enable_mid_stream_fallback_continuation=continuation_flag(get("enable_mid_stream_fallback_continuation")),
            enable_tag_filtering=bool(get("enable_tag_filtering")),
            tag_filtering_match_any=_true_unless_given(get("tag_filtering_match_any")),
            tag_routing_prefix=str(get("tag_routing_prefix") or ""),
            enable_pre_call_checks=bool(get("enable_pre_call_checks")),
            routing_plugins=tuple(_as_list(get("plugins"))),  # pyright: ignore[reportArgumentType]  # the plugins as passed
            routing_groups=parse_groups(
                get("routing_groups"),
                frozenset(
                    str(name)
                    for entry in _as_list(get("model_list"))
                    if (name := (as_str_mapping(entry) or {}).get("model_name"))
                ),
                frozenset(as_str_mapping(get("model_group_alias")) or {}),
            ),
            retry_policy=resolved_retry_policy(get("retry_policy")),
            model_group_retry_policy=get("model_group_retry_policy"),
        )

    def changed(self, values: Mapping[str, object], updated: frozenset[str] = frozenset()) -> "Resolved":
        """``values`` applied; ``updated`` names those passed as ``ChangeSet.settings`` (which go
        through ``update_settings``: only they may name ``routing_strategy_args``, and only they
        validate a strategy)."""
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        unknown: Final = sorted(
            set(values)
            - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES}
            - (updated & {"routing_strategy_args", "routing_groups", "enable_tag_filtering", "tag_routing_prefix"})
        )
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        unserved: Final = sorted(set(values) - _STATIC_SETTINGS - _DYNAMIC_SETTINGS)
        if unserved:
            raise RustCoreUnsupported(f"the Rust router core does not serve the setting(s) {', '.join(unserved)} yet")
        updates: Final[dict[str, object]] = {}  # mutable-ok: built here, then applied once
        for name, value in values.items():
            if name in {"model_group_alias", "default_litellm_params"}:
                updates[name] = MappingProxyType(dict(as_str_mapping(value) or {}))
            elif name in ("num_retries", "max_fallbacks", "allowed_fails"):
                updates[name] = int(str(value))
                if name == "allowed_fails":
                    updates["explicit_allowed_fails"] = int(str(value))
            elif name in ("cooldown_time", "retry_after"):
                updates[name] = float(str(value or 0)) or (
                    _DEFAULT_COOLDOWN_SECONDS if name == "cooldown_time" else 0.0
                )
            elif name == "timeout":
                updates[name] = float(str(value)) if value is not None else None
            elif name in _SELECTION_SETTINGS:
                updates[name] = self._selection_setting(name, value, validate=name in updated)
            elif name == "disable_cooldowns":
                updates[name] = _optional_bool(value)
            elif name == "enable_pre_call_checks":
                updates[name] = bool(value)
            elif name in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
                updates[name] = list(_as_list(value)) or None
            else:
                updates[name] = value
        return dataclasses.replace(self, **updates, assigned=self.assigned | frozenset(values))  # pyright: ignore[reportArgumentType]  # each value converted to its field's type above

    def _selection_setting(self, name: str, value: object, *, validate: bool) -> object:
        """A selection setting converted to its field's type; a strategy changed through
        ``update_settings`` is validated."""
        if name == "routing_strategy":
            normalized: Final = normalize_strategy(value) or "simple-shuffle"
            if validate and normalized != self.routing_strategy:
                validate_routing_strategy(value)  # pyright: ignore[reportArgumentType]  # validated as given
            return normalized
        if name == "routing_strategy_args":
            return MappingProxyType(dict(as_str_mapping(value) or {}))
        if name == "tag_routing_prefix":
            return str(value or "")
        return bool(value)

    def static_document(self, model_list: Sequence[object]) -> Mapping[str, object]:
        values: Final = {
            "model_list": _plain(list(model_list)),
            "routing_strategy": self.routing_strategy,
            "cooldown_time": self.cooldown_time,
            "disable_cooldowns": self.disable_cooldowns,
            "allowed_fails_policy": _plain(self.allowed_fails_policy),
            "retry_after": self.retry_after,
            "timeout": self.timeout,
            "routing_groups": group_documents(self.routing_groups) if self.routing_groups else None,
            "enable_pre_call_checks": True if self.enable_pre_call_checks else None,
            "routing_plugins": True if self.routing_plugins else None,
            # `async_callback_filter_deployments` runs on every async candidate list.
            "callback_filters": True,
            "retry_policy": _plain(self.retry_policy),
            "model_group_retry_policy": _plain(self.model_group_retry_policy) or None,
        }
        return {name: value for name, value in values.items() if value is not None}

    def dynamic_document(self, version: int, model_names: Sequence[str]) -> Mapping[str, object]:
        values: Final = {
            "version": version,
            "model_group_alias": _plain(dict(self.model_group_alias)),
            "fallbacks": _plain(self.fallbacks),
            "context_window_fallbacks": _plain(self.context_window_fallbacks),
            "content_policy_fallbacks": _plain(self.content_policy_fallbacks),
            "num_retries": self.num_retries,
            "max_fallbacks": self.max_fallbacks,
            "allowed_fails": self._router_allowed_fails(),
            "fallback_keys": self._fallback_keys(model_names),
        }
        return {name: value for name, value in values.items() if value is not None}

    def _router_allowed_fails(self) -> int | None:
        """``allowed_fails`` when it counts as set on the router (``_is_allowed_fails_set_on_router``):
        passed explicitly, or different from the process default."""
        import litellm

        if self.explicit_allowed_fails is not None:
            return self.explicit_allowed_fails
        if self.allowed_fails != litellm.allowed_fails:
            return self.allowed_fails
        return None

    def _fallback_keys(self, model_names: Sequence[str]) -> Mapping[str, object]:
        """The spellings a fallback key matches besides the group name (`get_fallback_model_group`)."""
        from litellm.router_utils.fallback_event_handlers import (
            _provider_prefixed_model_group,  # pyright: ignore[reportPrivateUsage]  # the fallback walk's own spelling rule
        )

        keys: Final = tuple(  # comprehension-ok: the first group of each fallback entry, across the three lists
            str(next(iter(mapping)))
            for entries in (self.fallbacks, self.context_window_fallbacks, self.content_policy_fallbacks)
            for entry in _as_list(entries)
            if (mapping := as_str_mapping(entry))
        )
        names: Final = {*model_names, *self.model_group_alias}
        prefixed: Final = {
            name: spelled for name in names if (spelled := _provider_prefixed_model_group(name, keys)) is not None
        }
        from litellm.types.router import LiteLLMParamsTypedDict

        return {
            "providers": _PROVIDERS,
            "provider_prefixed": prefixed,
            "param_keys": list(LiteLLMParamsTypedDict.__annotations__),
        }


def _alias_target(target: object) -> str | None:
    mapping: Final = as_str_mapping(target)
    if mapping is not None:
        model: Final = mapping.get("model")
        return model if isinstance(model, str) else None
    return target if isinstance(target, str) else None


def _true_unless_given(value: object) -> bool:
    return True if value is None else bool(value)


def _group_selectors(groups: "Sequence[RoutingGroup]") -> Sequence[tuple[str, str, Mapping[str, object]]]:
    return [
        (
            group.group_name,
            group.routing_strategy,
            as_str_mapping(group.routing_strategy_args) or {},  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # RoutingGroup's untyped dict
        )
        for group in groups
    ]


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def unsupported_arguments(config: RouterConfig) -> frozenset[str]:
    """Arguments the Rust core does not serve."""
    unserved: Final = frozenset(
        name
        for name, value in config.arguments.items()
        if name not in _SERVED_ARGUMENTS and value not in (None, _UNSET_DEFAULTS.get(name))
    )
    # `simple_shuffle` raises on a non-numeric weight when it picks; only the Python core keeps that.
    odd_weight: Final = any(
        (params := as_str_mapping((as_str_mapping(deployment) or {}).get("litellm_params"))) is not None
        and params.get("weight") is not None
        and (isinstance(params.get("weight"), bool) or not isinstance(params.get("weight"), int | float))
        for deployment in _as_list(config.get("model_list"))
    )
    return unserved | ({"model_list"} if odd_weight else frozenset[str]())


def _team_id(request: Mapping[str, object]) -> str | None:
    metadata: Final = as_str_mapping(request.get("metadata") or request.get("litellm_metadata"))
    team_id: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
    return team_id if isinstance(team_id, str) else None


def _refuse_unserved_deployments(model_list: Sequence[object]) -> None:
    """Deployment fields the Rust core does not read yet: refused, never silently ignored."""
    for entry in model_list:
        model_info = as_str_mapping((as_str_mapping(_record(entry)) or {}).get("model_info")) or {}
        if model_info.get("allowed_fails_policy") is not None:
            raise RustCoreUnsupported(
                "a deployment's model_info.allowed_fails_policy is not served by the Rust router core yet"
            )


# Request fallback settings, which replace the router's for the call.
_REQUEST_FALLBACK_KEYS: Final = ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "max_fallbacks")


def _start_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    """The request fields the Rust core routes on."""
    fields: Final = {
        key: request[key]
        for key in ("specific_deployment", "num_retries", "cooldown_time")
        if request.get(key) is not None
    }
    # `record_disable_fallbacks(kwargs, disable_fallbacks is True)`: only True turns them off.
    disabled: Final = {"disable_fallbacks": True} if request.get("disable_fallbacks") is True else {}
    group_policies: Final = request.get("model_group_retry_policy")
    lists: Final = {key: _plain(request[key]) for key in _REQUEST_FALLBACK_KEYS if key in request}
    weights: Final = request.get("_router_weights")
    team_id: Final = _team_id(request)
    strategy: Final = request.get("routing_strategy")
    override: Final = normalize_strategy(strategy) if isinstance(strategy, str | RoutingStrategy) else None
    return {
        **fields,
        **({"routing_strategy": override} if override is not None else {}),
        **disabled,
        **({"router_weights": weights} if weights is not None else {}),
        **({"model_group_retry_policy": _plain(group_policies)} if group_policies is not None else {}),
        **lists,
        **({"team_id": team_id} if team_id else {}),
    }


# --- views --------------------------------------------------------------------------------------


class RustSettings(RouterSettings):
    """``settings()``: the resolved settings as a frozen snapshot."""

    def as_dict(self) -> Mapping[str, object]:
        router: Final = self._router
        if not isinstance(router, _SettingsSource):
            return super().as_dict()
        return router.settings_dict()


class _SettingsSource:
    def settings_dict(self) -> Mapping[str, object]:
        raise NotImplementedError


class RustHealth(RouterHealth):
    """``health()``: cooldowns, fail counts and usage, read through the host's routing cache and
    written through the Rust core."""

    __slots__ = ("_core",)

    def __init__(self, core: "RustCore") -> None:  # pyright: ignore[reportMissingSuperCall]  # reads the Rust core, not a Router
        self._core = core

    def __getattribute__(self, name: str) -> object:
        if name in _HEALTH_SERVED or name.startswith("__") or name == "_core":
            found: Final[object] = object.__getattribute__(self, name)  # pyright: ignore[reportAny]  # the view's own member
            return found
        raise RustCoreUnsupported(f"health().{name} is not served by the Rust router core", name=name, obj=self)

    @property
    def store(self) -> "DualCache":
        return self._core._cache  # pyright: ignore[reportPrivateUsage]  # the health view reads its own core

    @property
    def cooldowns(self) -> "CooldownCache":
        return self._core._cooldowns  # pyright: ignore[reportPrivateUsage]  # see above

    def failures_this_minute(self, deployment_id: str) -> int:
        count: Final[object] = self.store.get_cache(f"{deployment_id}:fails", local_only=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # DualCache's reads are untyped
        return int(count) if isinstance(count, int | float) else 0

    def record_failure(self, deployment_id: str, exception: Exception, exception_status: str | int) -> bool:
        failure: Final = {**attempt_failure(exception), "status": failure_status(exception_status)}
        handle: Final = self._core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        call: Final = handle.report_failure(deployment_id, failure, count_only=exception_status == "")
        done: Final = self._core._drive_sync_steps(call)  # pyright: ignore[reportPrivateUsage]  # see above
        return cooled_in_running_loop(bool(done.get("cooled")))

    def team_ptu_ceiling(self, team_id: str, model_group: str) -> "PTUTeamCeiling | None":
        return self._core._hosted._core_team_ptu_ceiling(team_id, model_group)  # pyright: ignore[reportPrivateUsage]  # the host registry's ported ceiling

    async def usage(  # pyright: ignore[reportIncompatibleMethodOverride]  # the same two kinds
        self, model_group: str, *, kind: Literal["current", "remaining"] = "current"
    ) -> tuple[int | None, int | None] | Mapping[str, int]:
        """The group's usage this minute, read from the routing cache over the host registry's
        live model list, as ``get_model_group_usage`` / ``get_remaining_model_group_usage`` read it."""
        hosted: Final = self._core._hosted  # pyright: ignore[reportPrivateUsage]  # see above
        if kind == "current":
            return await hosted._core_get_model_group_usage(model_group)  # pyright: ignore[reportPrivateUsage]  # the ported usage read
        if kind == "remaining":
            return await hosted._core_get_remaining_model_group_usage(model_group)  # pyright: ignore[reportPrivateUsage]  # see above
        raise ValueError(f"unknown usage kind {kind!r}")


_HEALTH_SERVED: Final = frozenset(
    {"store", "cooldowns", "failures_this_minute", "record_failure", "usage", "team_ptu_ceiling"}
)


# --- the core -----------------------------------------------------------------------------------


_NO_INPUT: Final = object()
_EndpointImpl: TypeAlias = Callable[..., Awaitable[object]]
# A strategy select step's strategy, selector, keyed name and candidate deployments.
_StrategyInputs: TypeAlias = tuple[str, object, str, list[dict[str, object]]]


class RustCore(
    _SettingsSource, RustRegistryMembers, RustRetries, RustFallbacks, RustHealthCallbacks, RustOptionalChecks
):
    """A ``RouterCore`` served by the Rust router core over a ``PythonHost``. It holds no Python core."""

    def __init__(self, config: RouterConfig, host: PythonHost, resolved: Resolved) -> None:
        from litellm.router_utils.cooldown_cache import CooldownCache

        self._config: Final = config
        self._host: Final = host
        apply_construction_effects(config, host)
        self._resolved: Resolved = resolved
        # Settings the Python core keeps as plain attributes, and settings deleted with `del`.
        self._plain: Final = initial_plain(config)
        self._deleted: frozenset[str] = frozenset()
        self._request_timeout: Final = request_timeout(config)
        legacy: Final[dict[str, object]] = {"deployment_stats": {}}  # mutable-ok: the core's own legacy attributes
        self._legacy: Final = legacy
        self._health_view: RustHealth | None = None
        self._cache: Final = host.routing_cache(config)
        self._cooldowns: CooldownCache = CooldownCache(cache=self._cache, default_cooldown_time=resolved.cooldown_time)
        self._version: int = 1
        self._retired: bool = False
        self._settings: RustSettings | None = None
        self._tag_filtering_models: tuple[int, bool] | None = None
        self._selectors: Final = StrategySelectors(host, self._cache)
        self._selectors.init_default(resolved.routing_strategy, resolved.routing_strategy_args)
        self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        self._endpoints: Final = RustEndpoints(self, config)
        self._registry_view: RouterRegistry | None = None
        self._hosted: Final = HostRegistry(
            host,
            model_group_alias=dict(resolved.model_group_alias),  # pyright: ignore[reportArgumentType]  # the configured alias map
            ignore_invalid_deployments=bool(config.get("ignore_invalid_deployments")),
            on_change=self._sync_registry,
        )
        self._hosted._core_cache = self._cache  # pyright: ignore[reportPrivateUsage]  # usage reads come from the cache the Rust core writes
        ctx: Final = RoutingContext(self)  # pyright: ignore[reportArgumentType]  # the strategies read registry(), settings(), health() and select()
        strategies: Final = PreRoutingStrategies(
            self._hosted, host, ctx, config.get("auto_router_capability_limit"), self._cache
        )
        self._hosted.strategies = strategies
        self._prerouter: Final = PreRouter(strategies, self._hosted, host, ctx, self._cache)
        # `register_plugin`: a custom strategy that replaces every pick (the host answers the
        # Rust core's pre_route step with its deployment).
        self._custom_strategy: CustomRoutingStrategyBase | None = None
        self._hosted.build(_as_list(config.get("model_list")) if config.get("model_list") is not None else None)
        self._hosted.set_routing_groups(resolved.routing_groups)
        self._handle: RouterCoreHandle = self._new_handle(list(self._hosted._core_model_list))  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry
        self._join_callback_holders()
        self._checks: Final = OptionalChecks(self, host, self._cache, self._hosted, ctx, config.arguments)
        self._hosted.checks = self._checks
        strategies.ensure_affinity = self._checks.ensure_deployment_affinity_callback
        self._checks.build(config.get("optional_pre_call_checks"), config.get("model_list"))
        if resolved.enable_mid_stream_fallback_continuation:
            register_continuation_prefill_check(host)
        if resolved.routing_strategy == "lar1":
            from litellm.router_strategy.lar1_routing import apply_lar1_routing_strategy

            apply_lar1_routing_strategy(ctx, resolved.routing_strategy_args)  # pyright: ignore[reportArgumentType]  # the core's RoutingContext

    @property
    def _registry_state(self) -> HostRegistry:
        return self._hosted

    @property
    def _python_host(self) -> PythonHost:
        return self._host

    @classmethod
    def build(cls, config: RouterConfig, host: PythonHost) -> "RustCore":
        from litellm.rust_bridge._native import RustBridgeDeclined

        extra: Final = unsupported_arguments(config)
        if extra:
            raise RustBridgeDeclined(f"router arguments not served by the Rust core: {sorted(extra)}")
        RustEndpoints.prepare_host(config, host)
        return cls(config, host, Resolved.of(config, host.dynamic()))

    @property
    def enable_mid_stream_fallback_continuation(self) -> bool:
        return self._resolved.enable_mid_stream_fallback_continuation

    def _set_continuation(self, enabled: bool) -> None:
        """``_set_mid_stream_fallback_continuation``: an enabled flag registers the process-wide prefill check."""
        self._resolved = dataclasses.replace(self._resolved, enable_mid_stream_fallback_continuation=enabled)
        self._settings = None
        if enabled:
            register_continuation_prefill_check(self._host)

    def __getattr__(self, name: str) -> object:
        if name in SELECTOR_ATTRIBUTES.values():
            return self._selectors.attribute(name)
        # Only reached for members this class does not define: the endpoints served through the
        # host, else refused. Never forwarded anywhere.
        endpoints: Final[object] = self.__dict__.get("_endpoints")
        endpoint: Final = endpoints.member(name) if isinstance(endpoints, RustEndpoints) else None
        if endpoint is not None:
            return endpoint
        attribute: Final = self._settings_attribute(name) if "_resolved" in self.__dict__ else MISSING
        if attribute is not MISSING:
            return attribute
        raise RustCoreUnsupported(f"'{name}' is not served by the Rust router core", name=name, obj=self)

    def _settings_attribute(self, name: str) -> object:
        """A router attribute that reads a setting, the constructor's arguments or a constant."""
        r: Final = self._resolved
        resolved: Final[Mapping[str, object]] = {
            "num_retries": r.num_retries,
            "max_fallbacks": r.max_fallbacks,
            "timeout": r.timeout,
            "retry_after": r.retry_after,
            "explicit_allowed_fails": r.explicit_allowed_fails,
            "routing_strategy_args": r.routing_strategy_args,
            "request_timeout": self._request_timeout,
            "default_priority": self._endpoints._default_priority,  # pyright: ignore[reportPrivateUsage]  # parsed by the endpoint plans
        }
        if name in resolved:
            return resolved[name]
        if name in self._plain:
            return self._plain[name]
        if name in self._legacy:
            return self._legacy[name]
        if name == "healthy_deployments":
            return self._hosted._core_model_list  # pyright: ignore[reportPrivateUsage]  # the live list, as the core keeps it
        if name == "get_model_group_io_token_usage":
            return self._hosted.get_model_group_io_token_usage
        if name == "scheduler":
            return self._endpoints._get_scheduler()  # pyright: ignore[reportPrivateUsage]  # the endpoint plans' scheduler
        if name == "chat":
            from litellm.main import Chat

            self._legacy["chat"] = Chat(params=default_litellm_params(self._config), router_obj=self)  # pyright: ignore[reportArgumentType]  # Chat calls the router's endpoints, which this core serves
            return self._legacy["chat"]
        return argument_attribute(self._config, name)

    @property
    def _shaping(self) -> Shaping:
        return self._endpoints.shaping(self._resolved)

    # -- building --------------------------------------------------------------------------------

    def _new_handle(self, model_list: Sequence[object]) -> "RouterCoreHandle":
        from litellm.rust_bridge._native import RouterCoreHandle

        _refuse_unserved_deployments(model_list)
        names: Final = [
            str(name)
            for entry in model_list
            if isinstance(name := (as_str_mapping(entry) or {}).get("model_name"), str)
        ]
        return RouterCoreHandle(
            self._resolved.static_document(model_list), self._resolved.dynamic_document(self._version, names)
        )

    def _rebuild(self) -> None:
        """A new Rust core for changed static settings, over the same deployments."""
        old: Final = self._handle
        self._handle = self._new_handle(list(self._hosted._core_model_list))  # pyright: ignore[reportPrivateUsage]  # see __init__
        old.retire()

    def _sync_registry(self) -> None:
        """Hand the host registry's model_list to the Rust core after a write."""
        from litellm.rust_bridge._native import RouterCoreHandle

        self._tag_filtering_models = None
        handle: Final[object] = self.__dict__.get("_handle")
        if isinstance(handle, RouterCoreHandle):
            _ = handle.apply({"replace": _plain(list(self._hosted._core_model_list))})  # pyright: ignore[reportPrivateUsage]  # see __init__

    # -- entry points ----------------------------------------------------------------------------

    def registry(self) -> RouterRegistry:
        """The registry view, built once (``registry()`` returns the same object each time)."""
        if self._registry_view is None:
            self._registry_view = hosted_registry_view(self._hosted)
        return self._registry_view

    def settings(self) -> RustSettings:
        """A cached snapshot, replaced when a changeset changes a setting."""
        if self._settings is None:
            self._settings = self._settings_snapshot()
        return self._settings

    def _settings_snapshot(self) -> RustSettings:
        from litellm.types.router import RouterGeneralSettings

        r: Final = self._resolved
        snapshot: Final = RustSettings(
            routing_strategy=r.routing_strategy,
            fallbacks=r.fallbacks,
            context_window_fallbacks=r.context_window_fallbacks,
            content_policy_fallbacks=r.content_policy_fallbacks,
            default_litellm_params=dict(r.default_litellm_params)
            if "default_litellm_params" in r.assigned
            else default_litellm_params(self._config),
            cooldown_time=r.cooldown_time,
            allowed_fails=r.allowed_fails,
            allowed_fails_policy=r.allowed_fails_policy,  # pyright: ignore[reportArgumentType]  # the policy as passed
            disable_cooldowns=r.disable_cooldowns,
            enable_pre_call_checks=r.enable_pre_call_checks,
            enable_health_check_routing=False,
            enable_tag_filtering=r.enable_tag_filtering,
            tag_filtering_match_any=r.tag_filtering_match_any,
            tag_routing_prefix=r.tag_routing_prefix,
            fallback_access_check=r.fallback_access_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            fallback_budget_check=r.fallback_budget_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            cache_responses=self._endpoints.cache_responses,
            search_tools=self._endpoints.search_tools,  # pyright: ignore[reportArgumentType]  # the configured entries, as passed
            guardrail_list=self._endpoints.guardrail_list,  # pyright: ignore[reportArgumentType]  # same
            provider_default_deployment_ids=self._hosted._core_provider_default_deployment_ids,  # pyright: ignore[reportPrivateUsage]  # the host registry's live list
            health_check_ignore_transient_errors=False,
            router_general_settings=self._config.get("router_general_settings") or RouterGeneralSettings(),  # pyright: ignore[reportArgumentType]  # the argument as passed
            _router=self,  # pyright: ignore[reportArgumentType]  # the snapshot reads as_dict from this core
            _unset=frozenset({"kubernetes_pod_discovery"}),
        )
        if not self._deleted:
            return snapshot
        # A deleted setting reads as unset.
        return dataclasses.replace(
            snapshot,
            **dict.fromkeys(self._deleted, None),  # pyright: ignore[reportArgumentType]  # settings fields only
            _unset=snapshot._unset | self._deleted,  # pyright: ignore[reportPrivateUsage]  # the snapshot's own unset set
        )

    def settings_dict(self) -> Mapping[str, object]:
        r: Final = self._resolved
        latency: Final = self._selectors.latency_args() if r.routing_strategy == "latency-based-routing" else None
        # `get_settings` reports the latency selector's full routing args.
        args: Final = latency if latency is not None else dict(r.routing_strategy_args)
        values: Final = {
            "routing_strategy_args": args,
            "routing_strategy": r.routing_strategy,
            "allowed_fails": r.allowed_fails,
            "cooldown_time": r.cooldown_time,
            "num_retries": r.num_retries,
            "timeout": r.timeout,
            "retry_after": r.retry_after,
            "fallbacks": r.fallbacks,
            "context_window_fallbacks": r.context_window_fallbacks,
            "model_group_alias": dict(r.model_group_alias),
            "enable_mid_stream_fallback_continuation": r.enable_mid_stream_fallback_continuation,
            "enable_tag_filtering": r.enable_tag_filtering,
            "tag_routing_prefix": r.tag_routing_prefix,
            "routing_groups": group_settings(r.routing_groups),
        }
        return settings_view(values, self._plain, self._deleted)

    def health(self) -> RustHealth:
        """The health view, built once."""
        if self._health_view is None:
            self._health_view = RustHealth(self)
        return self._health_view

    def observe(self, logging_obj: object = None, *, request_scoped: bool = False) -> RoutingObservers:
        """The routing observers. This core registers no outcome callbacks and builds no alerting
        (``alerting_config`` is declined), so there is no Slack alerting logger."""
        if request_scoped:
            self._serve_through_request_logging()
        if logging_obj is not None:
            raise RustCoreUnsupported("observe(logging_obj) is not served by the Rust router core yet")
        return RoutingObservers(slack_alerting_logger=None)

    def register_plugin(self, plugin: "CustomRoutingStrategyBase") -> None:
        """``_core_set_custom_routing_strategy``: ``plugin`` picks every deployment from now on, until
        the routing strategy changes."""
        self._custom_strategy = plugin

    def set_custom_routing_strategy(self, CustomRoutingStrategy: "CustomRoutingStrategyBase") -> None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the deprecated alias notes its use like Python's
        )

        _note("set_custom_routing_strategy", "register_plugin(strategy)")
        self.register_plugin(CustomRoutingStrategy)

    def retire(self) -> None:
        self._retired = True
        self._retire_callbacks()
        if self._hosted.strategies is not None:
            self._hosted.strategies.retire()
        self._handle.retire()

    def apply(self, changes: ChangeSet) -> ApplyResult:
        """``apply_changes``: assigned settings, then ``update_settings``, upserts, deletes and the
        Redis cache, in that order. A refused changeset changes nothing."""
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if changes.refresh_model_info:
            raise TypeError("ChangeSet(refresh_model_info=True) needs aapply()")
        unknown: Final = sorted(set(changes.assign) - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES})
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        update: Final = read_settings_update(changes.settings)
        if update.optional_pre_call_checks is not MISSING:
            self._checks.set(update.optional_pre_call_checks)
        if update.continuation is not MISSING:
            self._set_continuation(continuation_flag(update.continuation))
        self._apply_settings(changes.assign, update)
        # Upserts first, then deletes (`apply_changes`), through the host registry, which hands
        # each change to the Rust core.
        upserted: Final = [self._hosted._core_upsert_deployment(deployment=d) for d in changes.upsert]  # pyright: ignore[reportPrivateUsage]  # see __init__
        deleted: Final = [self._hosted._core_delete_deployment(id=i) for i in changes.delete]  # pyright: ignore[reportPrivateUsage]  # see __init__
        if changes.redis_cache is not None:
            # `_core_update_redis_cache`: the routing state's cache gains the Redis cache.
            self._cache.attach_redis_cache(changes.redis_cache)
            self._checks.prompt_caching_pin_cache.attach_redis_cache(changes.redis_cache)
        return ApplyResult(
            version=self._hosted._registry_version,  # pyright: ignore[reportPrivateUsage]  # see __init__
            upserted=upserted,
            deleted=deleted,  # pyright: ignore[reportArgumentType]  # the removed model_list entries, as the core returns them
        )

    async def aapply(self, changes: ChangeSet) -> ApplyResult:
        """``apply``, then the model info refresh, which needs I/O."""
        result: Final = self.apply(dataclasses.replace(changes, refresh_model_info=False, refresh_client=None))
        if changes.refresh_model_info:
            await self._hosted.arefresh_model_info(changes.refresh_client)
        return result

    def _apply_settings(self, assign: Mapping[str, object], update: SettingsUpdate) -> None:
        policies: Final = {name: update.plain[name] for name in _RETRY_POLICY_SETTINGS if name in update.plain}
        refuse_unserved_retry_policies(policies.get("model_group_retry_policy"))
        values: Final = {**assign, **update.values}
        given: Final = dict(update.values)  # the settings passed through `update_settings`
        if update.routing_groups is not MISSING:
            given["routing_groups"] = update.routing_groups
            values["routing_groups"] = parse_groups(
                update.routing_groups,
                frozenset(str(d.get("model_name")) for d in self._handle.deployments()),
                frozenset(self._resolved.model_group_alias),
            )
        self._plain.update(update.plain)
        if policies:
            self._set_retry_policies(policies)
        if self._deleted & set(values):
            self._deleted = self._deleted - set(values)
            self._settings = None
        if values:
            previous: Final = self._resolved
            resolved: Final = previous.changed(values, frozenset(given))
            static_changed: Final = bool(_STATIC_SETTINGS & set(values))
            self._resolved = resolved
            if set(values) & set(SETTING_FIELDS):
                self._settings = None  # the snapshot holds settings fields, not the plain attributes
            if "model_group_alias" in values:
                self._hosted._core_model_group_alias = dict(resolved.model_group_alias)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # see __init__
            self._apply_strategy(previous, resolved, given)
            if "routing_groups" in values:
                self._settings = None
                self._hosted.set_routing_groups(resolved.routing_groups)
            if "cooldown_time" in values:
                from litellm.router_utils.cooldown_cache import CooldownCache

                self._cooldowns = CooldownCache(cache=self._cache, default_cooldown_time=resolved.cooldown_time)
            if static_changed:
                self._rebuild()
            else:
                names: Final = [str(d.get("model_name")) for d in self._handle.deployments()]
                self._handle.apply({"settings": resolved.dynamic_document(self._version, names)})

    def _set_retry_policies(self, policies: Mapping[str, object]) -> None:
        """``update_settings``' ``retry_policy`` and ``model_group_retry_policy``, which the retry
        loop reads from then on: router-core holds them with its static settings."""
        previous: Final = self._resolved
        resolved: Final = dataclasses.replace(
            previous,
            retry_policy=resolved_retry_policy(policies.get("retry_policy", previous.retry_policy)),
            model_group_retry_policy=policies.get("model_group_retry_policy", previous.model_group_retry_policy),
        )
        if resolved != previous:
            self._resolved = resolved
            self._rebuild()

    def _apply_strategy(self, previous: Resolved, resolved: Resolved, values: Mapping[str, object]) -> None:
        """``_core_update_settings``: a changed strategy re-inits the selectors with the args passed
        alongside it (or none); changed args alone rebuild the default selector."""
        if "routing_strategy" in values and resolved.routing_strategy != previous.routing_strategy:
            # `routing_strategy_init` drops a custom strategy.
            self._custom_strategy = None
            self._selectors.init_default(
                resolved.routing_strategy, as_str_mapping(values.get("routing_strategy_args")) or {}
            )
            self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif "routing_groups" in values:
            self._selectors.init_groups(_group_selectors(resolved.routing_groups))
        elif "routing_strategy_args" in values and resolved.routing_strategy_args != previous.routing_strategy_args:
            if resolved.routing_strategy == "lar1":
                from litellm.router_strategy.lar1_routing import apply_lar1_routing_strategy

                apply_lar1_routing_strategy(RoutingContext(self), resolved.routing_strategy_args)  # pyright: ignore[reportArgumentType]  # the core's RoutingContext
                return
            self._selectors.rebuild_default(resolved.routing_strategy_args)

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("upsert_deployment", "apply(ChangeSet(upsert=[deployment]))")
        return self.apply(ChangeSet(upsert=[deployment])).upserted[0]

    def delete_deployment(self, id: str) -> Deployment | None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see above

        _note("delete_deployment", "apply(ChangeSet(delete=[id]))")
        return self.apply(ChangeSet(delete=[id])).deleted[0]

    def update_settings(self, **kwargs: object) -> None:  # kwargs-ok: the deprecated signature, unchanged
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("update_settings", "apply(ChangeSet(settings={...}))")
        _ = self.apply(ChangeSet(settings=kwargs))

    def update_redis_cache(self, cache: "RedisCache") -> None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("update_redis_cache", "apply(ChangeSet(redis_cache=cache))")
        _ = self.apply(ChangeSet(redis_cache=cache))

    async def arefresh_model_info(self, *, client: "AsyncHTTPHandler | None" = None) -> None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see upsert_deployment

        _note("arefresh_model_info", "aapply(ChangeSet(refresh_model_info=True))")
        _ = await self.aapply(ChangeSet(refresh_model_info=True, refresh_client=client))

    def flush_cache(self) -> None:
        """Drop the host's response cache and the routing state."""
        self._host.set_global_cache(None)
        self._cache.flush_cache()
        self._checks.prompt_caching_pin_cache.in_memory_cache.flush_cache()

    def reset(self) -> None:
        """Empty the host's success and failure callback lists (every entry, not only this core's),
        drop the retry policy and flush the caches."""
        for name in ("success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback"):
            self._host.set_callback_list(name, [])
        self._plain["retry_policy"] = None
        self._set_retry_policies({"retry_policy": None})
        self.flush_cache()

    def __delattr__(self, name: str) -> None:
        # `del router.<setting>` unsets it: the settings view reports it unset from then on.
        from litellm.router_api import SETTING_FIELDS

        if name not in SETTING_FIELDS:
            object.__delattr__(self, name)
            return
        if name in self._deleted:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        self._deleted = self._deleted | {name}
        self._settings = None

    def __setattr__(self, name: str, value: object) -> None:
        # A setting write is a changeset (`router.num_retries = 2` is `apply(ChangeSet(assign=...))`),
        # a `model_list` write swaps the live list, and any other attribute is kept as written.
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        if name in PLAIN_SETTING_ATTRIBUTES or name in SETTING_FIELDS or name in ASSIGNABLE_ALIASES:
            _ = self.apply(ChangeSet(assign={name: value}))
        elif name == "model_list":
            self._assign_model_list(_as_list(value))
        else:
            object.__setattr__(self, name, value)

    async def select(  # pyright: ignore[reportIncompatibleMethodOverride]  # the RouterCore selection
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the RouterCore signature
        *,
        kind: str = "pick",
        specific_deployment: bool | None = None,
        messages: object = None,
        input: object = None,
        **_: object,  # kwargs-ok: the RouterCore select options this core ignores
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Pick a deployment for ``model`` without calling it (``pick``), list the deployments a
        pick would choose among (``healthy``), or pick one for a pass-through route
        (``pass_through``, which picks with the sync strategies, as Python does)."""
        if kind not in ("pick", "healthy", "pass_through"):
            raise ValueError(f"unknown select kind {kind!r}")
        request: Final = request_kwargs if request_kwargs is not None else {}
        selection: Final = _Selection(self, model, request, kind, specific_deployment, messages, input)
        if kind == "pass_through":
            return selection.run_sync()
        return await selection.run()

    async def async_get_available_deployment_for_pass_through(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the legacy signature
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
    ) -> Mapping[str, object]:
        """The full healthy chain, then only ``use_in_pass_through`` deployments, picked with the
        async strategies."""
        selection: Final = _Selection(
            self, model, request_kwargs, "async_pass_through", specific_deployment, messages, input
        )
        picked: Final = await selection.run()
        return picked if isinstance(picked, Mapping) else {}

    def get_available_deployment(
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the legacy signature
    ) -> Mapping[str, object]:
        """The sync pick: no pre-routing, the sync strategies."""
        selection: Final = _Selection(
            self,
            model,
            request_kwargs if request_kwargs is not None else {},
            "sync_pick",
            specific_deployment,
            messages,
            input,
        )
        picked: Final = selection.run_sync()
        return picked if isinstance(picked, Mapping) else {}

    def get_available_deployment_for_pass_through(
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the legacy signature
    ) -> Mapping[str, object]:
        """Deprecated: ``select(model, kind="pass_through")``."""
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("get_available_deployment_for_pass_through", 'select(model, kind="pass_through")')
        selection: Final = _Selection(
            self,
            model,
            request_kwargs if request_kwargs is not None else {},
            "pass_through",
            specific_deployment,
            messages,
            input,
        )
        picked: Final = selection.run_sync()
        return picked if isinstance(picked, Mapping) else {}

    # -- endpoints -------------------------------------------------------------------------------

    async def call(self, op: str, request: Mapping[str, object]) -> object:
        """Run one endpoint with the request's keyword payload, as ``getattr(core, op)(**request)``."""
        member: Final[object] = getattr(self, op)  # pyright: ignore[reportAny]  # litellm's untyped helpers and request values
        if not callable(member):
            raise TypeError(f"{op!r} is not an endpoint")
        result: Final[object] = member(**request)
        return await result if inspect.isawaitable(result) else result

    async def acompletion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.acompletion(request_from_call("acompletion", args, kwargs))

    async def atext_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.atext_completion(request_from_call("atext_completion", args, kwargs))

    async def aembedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("aembedding"), request_from_call("aembedding", args, kwargs))

    async def aresponses(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.generic(plan_for("aresponses"), request_from_call("aresponses", args, kwargs))

    async def aanthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.generic(plan_for("aanthropic_messages"), kwargs)

    async def anthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        """The proxy's ``/v1/messages`` route name for ``aanthropic_messages``; async, as in PythonCore."""
        return await self._endpoints.generic(plan_for("anthropic_messages"), kwargs)

    async def aimage_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("aimage_generation"), request_from_call("aimage_generation", args, kwargs))

    async def atranscription(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._drive(plan_for("atranscription"), request_from_call("atranscription", args, kwargs))

    async def aspeech(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        request: Final = dict(request_from_call("aspeech", args, kwargs))
        request.setdefault("voice", None)
        return await self._drive(plan_for("aspeech"), request)

    async def arerank(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        # `arerank` sets kwargs["input"] = input: the Python builtin, which reaches the deployment call.
        request: Final = {**request_from_call("arerank", args, kwargs), "input": input}
        return await self._drive(plan_for("arerank"), request)

    async def amoderation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self._endpoints.amoderation(request_from_call("amoderation", args, kwargs))

    def completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("completion"), request_from_call("completion", args, kwargs))

    def text_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._endpoints.text_completion(request_from_call("text_completion", args, kwargs))

    def embedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("embedding"), request_from_call("embedding", args, kwargs))

    def image_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync(plan_for("image_generation"), request_from_call("image_generation", args, kwargs))

    @property
    def total_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.total_calls

    @property
    def success_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.success_calls

    @property
    def fail_calls(self) -> Mapping[str, int]:
        return self._endpoints.counters.fail_calls

    # -- driving a call --------------------------------------------------------------------------

    def _start(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        prepared_already: bool = False,
    ) -> "tuple[RouterCall, Mapping[str, object]]":
        if self._retired:
            raise RuntimeError("this router core was retired")
        unserved: Final = sorted(
            key
            for key in request
            if key in _UNSERVED_REQUEST_KEYS or (key.startswith("mock_testing_") and key not in MOCK_FLAGS)
        )
        if unserved:
            raise RustCoreUnsupported(f"request keys not served by the Rust router core yet: {unserved}")
        refuse_unserved_retry_policies(request.get("model_group_retry_policy"))
        model: Final = request.get("model")
        if not isinstance(model, str):
            raise TypeError(f"{plan.endpoint}() needs a model name")
        field: Final = plan.prepare_field or "metadata"
        if request.get(field) is not None and not isinstance(request.get(field), Mapping):
            raise TypeError(f"{field} must be a mapping")
        prepared: Final = dict(request) if prepared_already else self._endpoints.prepare(plan, request, model)
        mock: Final = mock_failure(request)
        fields: Final[Mapping[str, object]] = {
            **self._routing_fields(prepared),
            **pinned_video_deployment(plan, request),
            **unrouted_fields(plan),
            # The sync path never runs pre-routing; a custom strategy picks on both.
            **({} if plan.sync and self._custom_strategy is None else {"pre_routes": True}),
            **({"mock_failure": mock} if mock is not None else {}),
            # Every target of the walk is refused.
            **({"fallbacks": []} if rejected_fallback_target(request) is not None else {}),
            **(overrides or {}),
        }
        return self._handle.start(plan.endpoint, _document({"model": model, **fields})), prepared

    def _routing_fields(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """The request fields the Rust core routes on, and whether its tag stage may apply: the
        router's ``enable_tag_filtering``, the request's own, or a deployment's
        ``model_info.enable_tag_filtering`` (``get_deployments_for_tag``'s effective setting)."""
        filter_tags: Final = (
            self._resolved.enable_tag_filtering
            or request.get("enable_tag_filtering") is True
            or self._model_tag_filtering()
        )
        return {
            **_start_fields(request),
            **({"filter_tags": True} if filter_tags else {}),
            **({"has_image": True} if request_carries_image(None, request) else {}),
        }

    def _model_tag_filtering(self) -> bool:
        """Whether any deployment sets ``model_info.enable_tag_filtering``, cached per registry version."""
        cached: Final = self._tag_filtering_models
        if cached is not None and cached[0] == self._version:
            return cached[1]
        found: Final = any(
            (as_str_mapping((as_str_mapping(record) or {}).get("model_info")) or {}).get("enable_tag_filtering")
            is not None
            for record in self._handle.deployments()
        )
        self._tag_filtering_models = (self._version, found)
        return found

    async def _drive(
        self,
        plan: Plan,
        request: Mapping[str, object],
        overrides: Mapping[str, object] | None = None,
        *,
        passthrough: "Callable[[Mapping[str, object]], Awaitable[object]] | None" = None,
        prepared_already: bool = False,
    ) -> object:
        call, prepared = self._start(plan, request, overrides, prepared_already=prepared_already)
        driver, step = await self._run(plan, call, prepared)
        if step["kind"] == "failed":
            if passthrough is not None and driver.attempted == 0:
                return await passthrough(prepared)
            raise driver.failed(step)
        return await driver.finish_async(step)

    async def _run(
        self, plan: Plan, call: "RouterCall", prepared: Mapping[str, object]
    ) -> "tuple[_Driver, Mapping[str, object]]":
        """Drive ``call`` to its ``done`` or ``failed`` step, answering every step on the host."""
        driver: Final = _Driver(self, plan, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome against the step it answers
            driver.take_events(step.get("events"))
            await await_all(driver.outcomes.take_pending())
            match step["kind"]:
                case "attempt":
                    outcome = await driver.attempt(step)
                case "sleep":
                    await self._host.sleep(float(str(step["seconds"])))
                    outcome = None
                case "allow_fallback":
                    outcome = await driver.allow_fallback(str(step["target"]))
                case "cache":
                    outcome = await self._answer_cache(step)
                case "select":
                    outcome = await driver.select(step)
                case "pre_route":
                    outcome = await self._pre_route(step, prepared)
                case "done" | "failed":
                    return driver, step
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _call_sync(
        self, plan: Plan, request: Mapping[str, object], overrides: Mapping[str, object] | None = None
    ) -> object:
        call, prepared = self._start(plan, request, overrides)
        driver, step = self._run_sync(plan, call, prepared)
        if step["kind"] == "failed":
            raise driver.failed(step)
        return driver.finish(step)

    def _run_sync(
        self, plan: Plan, call: "RouterCall", prepared: Mapping[str, object]
    ) -> "tuple[_Driver, Mapping[str, object]]":
        driver: Final = _Driver(self, plan, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _run
            driver.take_events(step.get("events"))
            if pending := driver.outcomes.take_pending():
                from litellm.litellm_core_utils import asyncify

                asyncify.run_async_function(await_all, pending)  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge
            match step["kind"]:
                case "attempt":
                    outcome = driver.attempt_sync(step)
                case "sleep":
                    self._host.sleep_sync(float(str(step["seconds"])))
                    outcome = None
                case "allow_fallback":
                    from litellm.litellm_core_utils import asyncify

                    outcome = asyncify.run_async_function(driver.allow_fallback, str(step["target"]))  # pyright: ignore[reportUnknownMemberType]  # the untyped sync bridge
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "select":
                    outcome = driver.select_sync(step)
                case "pre_route":
                    outcome = self._custom_pick_sync(step, prepared)
                case "done" | "failed":
                    return driver, step
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    # -- stream recovery (the host's stream wrappers, router_rust_streaming) ---------------------

    def _recovery_call(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> "RouterCall":
        start: Final = self._handle.retry_stream if recovery == "retry" else self._handle.fallback_stream
        routing: Final = _document(
            {"model": request.get("model"), **self._routing_fields(request), **continuation_routing(self, request)}
        )
        return start(op, routing, _document(origin), attempt_failure(failure))

    def stream_policy(
        self, request: Mapping[str, object], origin: Mapping[str, object], trigger: BaseException | None
    ) -> Mapping[str, object]:
        routing: Final = _document({"model": request.get("model"), **_start_fields(request)})
        kind: Final = None if trigger is None else str(attempt_failure(trigger)["kind"])
        return self._handle.stream_policy(routing, _document(origin), kind)

    async def recover_stream(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> object:
        """The fallback walk for a stream that failed before content: the fallback's response with the
        router's headers, or the error the walk ended on."""
        call: Final = self._recovery_call(recovery, op, request, origin, failure)
        driver, step = await self._run(plan_for(op), call, request)
        if step["kind"] == "failed":
            raise driver.failed(step)
        return await driver.finish_async(step)

    def recover_stream_sync(
        self,
        recovery: str,
        op: str,
        request: Mapping[str, object],
        origin: Mapping[str, object],
        failure: BaseException,
    ) -> object:
        call: Final = self._recovery_call(recovery, op, request, origin, failure)
        driver, step = self._run_sync(plan_for(op), call, request)
        if step["kind"] == "failed":
            raise driver.failed(step)
        return driver.finish(step)

    async def retry_stream(
        self, op: str, request: Mapping[str, object], origin: Mapping[str, object], failure: BaseException
    ) -> "tuple[object, int, int] | BaseException":
        """A same-group retry of a stream that failed before content: the retried response (without the
        router's headers) with the retry count and budget, or the newest failure (``failure`` itself
        when no retry ran)."""
        call: Final = self._recovery_call("retry", op, request, origin, failure)
        driver, step = await self._run(plan_for(op), call, request)
        if step["kind"] == "failed":
            return driver.last_error if driver.last_error is not None else failure
        attempted: Final = step.get("attempted_retries")
        budget: Final = step.get("max_retries")
        return (
            driver.response,
            attempted if isinstance(attempted, int) else 0,
            budget if isinstance(budget, int) else 0,
        )

    def params_sent_to_deployment(self, request: Mapping[str, object], deployment_id: object) -> Mapping[str, object]:
        """The request as a deployment received it: its params, the router defaults, its tool choice, then
        the request (``_params_sent_to_deployment``)."""
        record: Final = self._handle.deployment(deployment_id) if isinstance(deployment_id, str) else None
        params: Final = as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}
        defaults: Final = {k: v for k, v in self._resolved.default_litellm_params.items() if v is not None}
        tool_choice: Final = {"tool_choice": params["tool_choice"]} if params.get("tool_choice") is not None else {}
        return MappingProxyType({**params, **defaults, **tool_choice, **request})

    def _drive_sync_steps(self, call: "RouterCall") -> Mapping[str, object]:
        """Drive a call whose only steps are cache ops (``report_failure``) to its end."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            match step["kind"]:
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    async def _drive_cache_steps(self, call: "RouterCall") -> Mapping[str, object]:
        """``_drive_sync_steps`` answered through the async cache (``report_success``)."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            match step["kind"]:
                case "cache":
                    outcome = await self._answer_cache(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    # -- pre-route steps -------------------------------------------------------------------------

    async def _pre_route(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object = _NO_INPUT,
        input: object = _NO_INPUT,
        specific_deployment: object = _NO_INPUT,
    ) -> object:
        """The host's pre-routing of one pick (router_rust_prerouting): the model group to resolve
        instead of the step's ``model``, a custom strategy's deployment, or ``None``."""
        custom: Final = self._custom_strategy
        if custom is not None:
            picked: Final[object] = await custom.async_get_available_deployment(  # pyright: ignore[reportUnknownMemberType]  # the strategy's untyped pick
                model=str(step.get("model")),
                messages=request.get("messages") if messages is _NO_INPUT else messages,  # pyright: ignore[reportArgumentType]  # the request's messages, as given
                input=request.get("input") if input is _NO_INPUT else input,  # pyright: ignore[reportArgumentType]  # the request's input, as given
                specific_deployment=request.get("specific_deployment")
                if specific_deployment is _NO_INPUT
                else specific_deployment,  # pyright: ignore[reportArgumentType]  # same
                request_kwargs=request,  # pyright: ignore[reportArgumentType]  # the request kwargs, as given
            )
            return _custom_pick(picked)
        if not isinstance(request, dict):
            return None
        depth: Final = step.get("fallback_depth")
        r: Final = self._resolved
        owner: Final[object] = self.__dict__.get("_core_owner")
        public: Final[object] = owner() if callable(owner) else None
        return await self._prerouter.pre_route(
            str(step.get("model")),
            request,  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the hook stamps
            messages=request.get("messages") if messages is _NO_INPUT else messages,
            input=request.get("input") if input is _NO_INPUT else input,
            fallback_depth=depth if isinstance(depth, int) else 0,
            router_strategy=normalize_strategy(r.routing_strategy),
            enable_tag_filtering=r.enable_tag_filtering,
            match_any=r.tag_filtering_match_any,
            public_router=public if public is not None else self,
            run_plugins=partial(
                self._run_plugins, request, request.get("messages") if messages is _NO_INPUT else messages
            )  # pyright: ignore[reportUnknownArgumentType]  # the request kwargs the plugins stamp
            if r.routing_plugins
            else None,
        )

    async def _run_plugins(
        self,
        request: dict[str, object],  # mutable-ok: the plugins stamp their outcome on the caller's request
        messages: object,
        model: str,
    ) -> None:
        """``_run_routing_plugins`` over the pre-routing hook's session-resolved ``model``."""
        await run_plugins(
            self._resolved.routing_plugins,
            self._hosted,  # pyright: ignore[reportArgumentType]  # the host registry serves the listing
            self._resolved.model_group_alias,
            model,
            request,
            messages,
        )

    def _custom_pick_sync(
        self,
        step: Mapping[str, object],
        request: Mapping[str, object],
        messages: object = _NO_INPUT,
        input: object = _NO_INPUT,
        specific_deployment: object = _NO_INPUT,
    ) -> object:
        """A custom strategy's sync pick (``get_available_deployment``); without one, no pre-routing."""
        custom: Final = self._custom_strategy
        if custom is None:
            return None
        picked: Final[object] = custom.get_available_deployment(  # pyright: ignore[reportUnknownMemberType]  # the strategy's untyped pick
            model=str(step.get("model")),
            messages=request.get("messages") if messages is _NO_INPUT else messages,  # pyright: ignore[reportArgumentType]  # see _pre_route
            input=request.get("input") if input is _NO_INPUT else input,  # pyright: ignore[reportArgumentType]  # same
            specific_deployment=request.get("specific_deployment")
            if specific_deployment is _NO_INPUT
            else specific_deployment,  # pyright: ignore[reportArgumentType]  # same
            request_kwargs=request,  # pyright: ignore[reportArgumentType]  # same
        )
        return _custom_pick(picked)

    # -- select steps ----------------------------------------------------------------------------

    def _strategy_inputs(self, step: Mapping[str, object], request: Mapping[str, object]) -> "_StrategyInputs | None":
        """The strategy, its selector, the name it keys state by and the candidate deployments of a
        ``strategy`` select step; ``None`` for a stage the host passes through."""
        if step.get("stage") != "strategy":
            return None
        strategy: Final = str(step.get("strategy"))
        group: Final = step.get("group")
        selector: Final = self._selectors.selector_for(
            strategy, str(step.get("scope")), str(group) if group is not None else None, request
        )
        candidates: Final = [
            dict(as_str_mapping(self._handle.deployment(str(id))) or {}) for id in _as_list(step.get("candidates"))
        ]
        return strategy, selector, str(step.get("model")), candidates

    def _records_of(self, ids: object) -> list[dict[str, object]]:  # mutable-ok: the deployment dicts filters take
        return [dict(as_str_mapping(self._handle.deployment(str(id))) or {}) for id in _as_list(ids)]

    async def _answer_tags(self, step: Mapping[str, object], request: Mapping[str, object]) -> object:
        """``get_deployments_for_tag`` over the step's candidates (bridged: not ported to Rust yet)."""
        from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
        from litellm.router_strategy.tag_based_routing import get_deployments_for_tag

        kept: Final = await get_deployments_for_tag(
            ctx=RoutingContext(self),  # pyright: ignore[reportArgumentType]  # the filter reads registry() and settings()
            model=str(step.get("model")),
            healthy_deployments=self._records_of(step.get("candidates")),  # pyright: ignore[reportArgumentType]  # deployment dicts
            request_kwargs=dict(request),  # pyright: ignore[reportArgumentType]  # the request kwargs
            metadata_variable_name=get_metadata_variable_name_from_kwargs(dict(request)),
            request_deployments=self._records_of(step.get("pool")),  # pyright: ignore[reportArgumentType]  # deployment dicts
        )
        return [found for record in kept if (found := picked_id(record)) is not None]

    def _pre_call_facts(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
    ) -> PreCallFacts:
        hosted: Final = self._hosted

        def record(id: str) -> Mapping[str, object]:
            return hosted._core_get_model_info(id) or as_str_mapping(self._handle.deployment(id)) or {}  # pyright: ignore[reportPrivateUsage]  # see __init__

        return PreCallFacts(self._host, hosted, step, record, request, messages, input)  # pyright: ignore[reportArgumentType]  # the host registry serves the reads

    def _text_only(self, step: Mapping[str, object]) -> list[str]:  # mutable-ok: the step's answer
        """The candidates that declare no vision support (bridged: needs the cost map)."""
        group: Final = str(step.get("model_group"))
        return [
            str(id)
            for id in _as_list(step.get("candidates"))
            if declares_no_vision(as_str_mapping(self._handle.deployment(str(id))) or {}, group)
        ]

    async def _answer_select(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
    ) -> object:
        if step.get("stage") == "pre_call":
            return await self._pre_call_facts(step, request, messages, input).answer_async()
        if step.get("stage") == "routing_plugins":
            # The plugins already ran in this pick's pre_route step (see `_pre_route`), on the
            # session-resolved name and before a strategy was selected, as Python's hook runs them.
            return candidate_models(request)
        if step.get("stage") == "tags":
            return await self._answer_tags(step, request)
        if step.get("stage") == "text_only":
            return self._text_only(step)
        inputs: Final = self._strategy_inputs(step, request)
        if inputs is None:
            return step["candidates"]
        strategy, selector, model, candidates = inputs
        if selector is None:
            return []
        picked: Final = await pick_async(
            strategy,
            selector,  # pyright: ignore[reportArgumentType]  # a selector from selector_for
            model,
            candidates,
            messages,
            input,
            dict(request),
        )
        if picked is not None and self._selectors.is_override(strategy, selector):
            await selector.async_pre_call_check(dict(as_str_mapping(picked) or {}), None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every selector is a CustomLogger
        found: Final = picked_id(picked)
        return [found] if found is not None else []

    def _answer_select_sync(
        self, step: Mapping[str, object], request: Mapping[str, object], messages: object, input: object
    ) -> object:
        if step.get("stage") == "pre_call":
            return self._pre_call_facts(step, request, messages, input).answer()
        if step.get("stage") == "routing_plugins":
            return candidate_models(request)
        if step.get("stage") == "callback_filters":
            # Only the async candidate lists run the callback filters.
            return step["candidates"]
        if step.get("stage") == "text_only":
            return self._text_only(step)
        inputs: Final = self._strategy_inputs(step, request)
        if inputs is None:
            return step["candidates"]
        strategy, selector, model, candidates = inputs
        if selector is None:
            return []
        picked: Final = pick_sync(
            strategy,
            selector,  # pyright: ignore[reportArgumentType]  # a selector from selector_for
            model,
            candidates,
            messages,
            input,
            dict(request),
        )
        if picked is not None and self._selectors.is_override(strategy, selector):
            selector.pre_call_check(dict(as_str_mapping(picked) or {}))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # see above
        found: Final = picked_id(picked)
        return [found] if found is not None else []

    # -- cache steps -----------------------------------------------------------------------------

    def _store_for(self, key: str) -> "DualCache":
        """Cooldowns live in the cooldown store; every other routing key in the routing cache."""
        return self._cooldowns.cooldown_store if key.endswith(":cooldown") else self._cache

    async def _answer_cache(self, step: Mapping[str, object]) -> object:
        with service_target(_cache_target(str(step.get("key")))):
            return await self._answer_cache_in_target(step)

    async def _answer_cache_in_target(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        local: Final = step.get("local") is True
        match step.get("op"):
            case "get":
                return _plain(await self._store_for(key).async_get_cache(key, local_only=local))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain(
                    [await self._store_for(str(k)).async_get_cache(str(k)) for k in _as_list(step.get("keys"))]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
                )
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    await store.async_set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    await store.async_set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                total: Final = (
                    await counter.async_increment_cache(key, by, local_only=local)  # pyright: ignore[reportUnknownMemberType]  # see above
                    if ttl is None
                    else await counter.async_increment_cache(key, by, local_only=local, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                )
                return by if total is None else total
            case "delete":
                await self._store_for(key).async_delete_cache(key)
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    def _answer_cache_sync(self, step: Mapping[str, object]) -> object:
        with service_target(_cache_target(str(step.get("key")))):
            return self._answer_cache_sync_in_target(step)

    def _answer_cache_sync_in_target(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        local: Final = step.get("local") is True
        match step.get("op"):
            case "get":
                return _plain(self._store_for(key).get_cache(key, local_only=local))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain([self._store_for(str(k)).get_cache(str(k)) for k in _as_list(step.get("keys"))])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    store.set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    store.set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                # DualCache.increment_cache is annotated int but adds floats as given.
                return (
                    counter.increment_cache(key, by, local_only=local)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                    if ttl is None
                    else counter.increment_cache(key, by, local_only=local, ttl=ttl)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                )
            case "delete":
                self._store_for(key).delete_cache(key)  # pyright: ignore[reportUnknownMemberType]  # DualCache's delete is untyped
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    # -- errors the core raises when no attempt failed ------------------------------------------

    def _failure_error(
        self, step: Mapping[str, object], model: str, last: BaseException | None, *, pass_through: str = ""
    ) -> BaseException:
        """The error a call or pick that failed before any attempt raises; ``pass_through`` names
        the selection kind (``pass_through``/``async_pass_through``) whose errors it words."""
        import litellm.exceptions as litellm_exceptions

        kind: Final = step.get("error_kind")
        aliases: Final = self._resolved.model_group_alias
        if kind == "paused":
            # The message names the deployment's litellm model, not the id that was asked for.
            record: Final = as_str_mapping(self._handle.deployment(model)) or {}
            litellm_model: Final = str((as_str_mapping(record.get("litellm_params")) or {}).get("model"))
            return litellm_exceptions.ServiceUnavailableError(
                message=f"Model '{litellm_model}' is currently paused and cannot accept requests.",
                model=litellm_model,
                llm_provider="",
            )
        if kind == "only_markers":
            named_model: Final = str(step.get("error_model") or model)
            return litellm_exceptions.BadRequestError(
                message=f"You passed in model={named_model}. {RouterErrors.only_strategy_marker_deployments.value}",
                model=named_model,
                llm_provider="",
            )
        if kind == "ptu_reserved":
            reserved: Final = str(step.get("error_model") or model)
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {reserved} is reserved for the teams holding a PTU share of it",
                model=reserved,
                llm_provider="",
            )
        if kind == "not_pass_through":
            return litellm_exceptions.BadRequestError(
                message=f"Deployment {model} does not support pass-through endpoint (use_in_pass_through=False)",
                model=model,
                llm_provider="",
            )
        if kind == "no_pass_through_deployments":
            message: Final = (
                f"Model {model} has no deployments configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true to the deployment configuration"
                if pass_through == "async_pass_through"
                else f"Model {model} has no deployment configured with use_in_pass_through=True. "
                "Please add use_in_pass_through: true in the deployment configuration"
            )
            return litellm_exceptions.BadRequestError(message=message, model=model, llm_provider="")
        if kind == "context_window_exceeded":
            return litellm_exceptions.ContextWindowExceededError(
                message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context "
                f"window large enough for this call.\n{step.get('error_detail')}",
                model=str(step.get("error_model") or model),
                llm_provider="",
            )
        if kind == "unsupported_params":
            params: Final = sorted(str(param) for param in _as_list(step.get("error_params")))
            unsupported_model: Final = str(step.get("error_model") or model)
            return litellm_exceptions.UnsupportedParamsError(
                message=(
                    f"no deployment of model={unsupported_model} supports {params}. "
                    "To drop these, set `litellm.drop_params=True` or for proxy `litellm_settings: drop_params: true`, "
                    f"or send allowed_openai_params={params} to pass them through"
                ),
                model=unsupported_model,
            )
        if kind == "routing_plugins_left_nothing":
            left: Final = [str(name) for name in _as_list(step.get("error_params"))]
            return ValueError(f"No deployments left after routing-plugin filtering. candidate_models={left}")
        if kind == "routing_plugins_need_async":
            return ValueError(SYNC_PATH_MESSAGE)
        if kind == "unknown_model":
            named: Final = _alias_target(aliases.get(model)) or model if model in aliases else model
            return litellm_exceptions.BadRequestError(
                message=f"You passed in model={named}. {RouterErrors.no_healthy_deployments.value}",
                model=named,
                llm_provider="",
            )
        if kind == "no_healthy_deployments":
            group: Final = self.registry().resolve(model)
            group_ids: Final = self.registry().names(str(group), kind="ids")
            # A listed pass-through pick names only the pass-through deployments.
            ids: Final = (
                [id for id in group_ids if _uses_pass_through(self._handle.deployment(id))]
                if pass_through == "pass_through"
                else list(group_ids)
            )
            active: Final = self._cooldowns.get_active_cooldowns(list(ids), None)  # pyright: ignore[reportArgumentType]  # a list of ids
            cooling: Final = [id for id, _ in active]
            min_cooldown: Final = min(
                (_seconds(entry.get("cooldown_time"), self._resolved.cooldown_time) for _, entry in active),
                default=_seconds(None, self._resolved.cooldown_time),
            )
            return RouterRateLimitError(
                model=model,
                cooldown_time=min_cooldown,
                enable_pre_call_checks=self._resolved.enable_pre_call_checks,
                cooldown_list=cooling,
                model_ids=list(ids),  # pyright: ignore[reportArgumentType]  # a list of ids
            )
        if last is not None:
            return last
        return RuntimeError(f"router-core ({kind}): {step.get('error')}")


def _uses_pass_through(record: object) -> bool:
    params: Final = as_str_mapping((as_str_mapping(record) or {}).get("litellm_params")) or {}
    return bool(params.get("use_in_pass_through"))


def _seconds(value: object, default: float) -> float | int:
    """A cooldown time as stored: an int stays an int, so error texts read ``60`` like Python's."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return int(default) if float(default).is_integer() else default


def _ttl(step: Mapping[str, object]) -> float | None:
    ttl: Final = step.get("ttl")
    return ttl if isinstance(ttl, int | float) else None


def _amount(value: object) -> float | int:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 1


def _record(deployment: object) -> object:
    dump: Final = getattr(deployment, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else deployment


class _NotSynchronous(TypeError):
    """A sync attempt's host endpoint returned an awaitable: a host bug, never an attempt failure."""


class _Selection:
    """One ``select``: the Rust selection call, its select and cache steps answered on the host."""

    def __init__(
        self,
        core: RustCore,
        model: str,
        request: dict[str, object],  # mutable-ok: the caller's request, whose narrowing keys are popped
        kind: str,
        specific_deployment: bool | None,
        messages: object,
        input: object,
    ) -> None:
        self._core: Final = core
        self._model: Final = model
        self._request: Final = request
        self._kind: Final = kind
        self._messages: Final = messages
        self._input: Final = input
        fields: Final = {
            **core._routing_fields(request),  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            **_narrowing_fields(request, kind),
            **({"has_image": True} if request_carries_image(messages, request) else {}),
        }  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
        if specific_deployment:
            fields["specific_deployment"] = True
        if kind == "pick" or (kind == "sync_pick" and core._custom_strategy is not None):  # pyright: ignore[reportPrivateUsage]  # the selection serves its own core
            fields["pre_routes"] = True
        self._specific_deployment: Final = specific_deployment
        # The error a callback filter raised, which a failed selection raises.
        self._filter_error: BaseException | None = None
        if kind == "pass_through":
            fields["pass_through"] = "listed"
        elif kind == "async_pass_through":
            fields["pass_through"] = "healthy"
        handle: Final = core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        self._call: Final = (handle.healthy if kind == "healthy" else handle.select)(model, _document(fields))

    def _done(self, step: Mapping[str, object]) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        if self._kind == "healthy":
            # The deployments as served: a priority group's members carry their priority as order.
            return [
                for_requested_model(as_str_mapping(record) or {}, self._model)
                for record in _as_list(step.get("deployments"))
            ]
        return for_requested_model(as_str_mapping(step.get("deployment")) or {}, self._model)

    def _failed(self, step: Mapping[str, object]) -> BaseException:
        error: Final = self._core._failure_error(step, self._model, self._filter_error, pass_through=self._kind)  # pyright: ignore[reportPrivateUsage]  # see above
        logging_obj: Final = self._request.get("litellm_logging_obj")
        dispatch: Final[object] = getattr(logging_obj, "dispatch_failure_handlers", None)
        if callable(dispatch) and self._kind not in ("pass_through", "sync_pick"):
            import asyncio
            import traceback

            # A refused pick is logged to the request's logging object (`_async_get_available_deployment`).
            trace: Final = "Traceback (most recent call last):\n" + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            pending: Final[object] = dispatch(exception=error, traceback_exception=trace, prefer_async_handlers=True)
            if inspect.isawaitable(pending):
                _ = asyncio.ensure_future(pending)
        return error

    async def run(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        core: Final = self._core
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(self._call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome
            match step["kind"]:
                case "cache":
                    outcome = await core._answer_cache(step)  # pyright: ignore[reportPrivateUsage]  # see above
                case "select" if step.get("stage") == "callback_filters":
                    outcome, self._filter_error = await core._answer_callback_filters(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages
                    )
                case "select":
                    outcome = await core._answer_select(step, self._request, self._messages, self._input)  # pyright: ignore[reportPrivateUsage]  # see above
                case "pre_route":
                    outcome = await core._pre_route(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages, self._input, self._specific_deployment
                    )
                case "done":
                    return self._done(step)
                case "failed":
                    raise self._failed(step)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def run_sync(self) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        core: Final = self._core
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(self._call.next(outcome))  # pyright: ignore[reportArgumentType]  # see run
            match step["kind"]:
                case "cache":
                    outcome = core._answer_cache_sync(step)  # pyright: ignore[reportPrivateUsage]  # see above
                case "select":
                    outcome = core._answer_select_sync(step, self._request, self._messages, self._input)  # pyright: ignore[reportPrivateUsage]  # see above
                case "pre_route":
                    outcome = core._custom_pick_sync(  # pyright: ignore[reportPrivateUsage]  # see above
                        step, self._request, self._messages, self._input, self._specific_deployment
                    )
                case "done":
                    return self._done(step)
                case "failed":
                    raise self._failed(step)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")


def _custom_pick(picked: object) -> Mapping[str, object]:
    """A custom strategy's answer as the deployment document the Rust core serves."""
    document: Final = as_str_mapping(picked)
    if document is None:
        raise TypeError(f"the custom routing strategy returned {type(picked).__name__}, not a deployment")
    return _document(document)


def _narrowing_fields(request: dict[str, object], kind: str) -> Mapping[str, object]:  # mutable-ok: popped
    """``_target_order``, ``_excluded_deployment_ids`` and ``_retry_skipped_deployment_ids``, popped
    from the request as the healthy chain pops them (a listed pass-through pick only reads the order)."""
    from litellm.litellm_core_utils.routing_common_utils import (
        as_excluded_deployment_ids,
        as_retry_skipped_deployment_ids,
    )

    if kind == "pass_through":
        order: Final = request.get("_target_order")
        return {"target_order": order} if isinstance(order, int) and not isinstance(order, bool) else {}
    target_order: Final = request.pop("_target_order", None)
    excluded: Final = as_excluded_deployment_ids(request.pop("_excluded_deployment_ids", None))
    skipped: Final = as_retry_skipped_deployment_ids(request.pop("_retry_skipped_deployment_ids", None))
    return {
        **(
            {"target_order": target_order}
            if isinstance(target_order, int) and not isinstance(target_order, bool)
            else {}
        ),
        **({"excluded_ids": list(excluded)} if excluded else {}),
        **({"retry_skipped_ids": list(skipped)} if skipped else {}),
    }


_NO_ROUTING_PRE_CALL_CHECKS: Final = frozenset({"search", "rerank", "moderation"})


class _Driver:
    """Answers the attempt and fallback steps of one call, and shapes its result.

    Each attempt is one host call against the deployment Rust picked. The driver keeps the error
    each fallback level ended on: a level that fails raises its own error, whatever its fallbacks
    raised, as the Python router does.
    """

    def __init__(self, core: RustCore, plan: Plan, request: Mapping[str, object]) -> None:
        self._core: Final = core
        self._plan: Final = plan
        self._op: Final = plan.endpoint
        self._request: Final = request
        self._last_response: object = None
        self._last_error: BaseException | None = None
        self._level_errors: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        self._served: tuple[str | None, Mapping[str, object]] = (None, MappingProxyType({}))
        self._attempts: int = 0
        self._walk: Final = WalkMetadata(request, plan.metadata_field, self._params_base_model)
        self.outcomes: Final = FallbackOutcomes(request)
        # The fallback level the walk is at, from its fallback events.
        self._level: int = 0

    @property
    def attempted(self) -> int:
        """How many attempts reached the host."""
        return self._attempts

    def _blocked(self, response: object, endpoint_name: str) -> str | None:
        """How the provider blocked a served response, the first check of
        ``_should_raise_content_policy_error`` (a chat response with no choices, or whose first
        choice was content-filtered) and of ``_should_raise_anthropic_refusal_error`` (a
        /v1/messages safeguard refusal carrying ``stop_details``)."""
        from litellm.llms.anthropic.messages.native.utils import get_safeguard_refusal_stop_details
        from litellm.types.utils import ModelResponse

        if endpoint_name == "anthropic_messages" and get_safeguard_refusal_stop_details(response) is not None:
            return "refusal"
        if self._op not in ("acompletion", "completion") or not isinstance(response, ModelResponse):
            return None
        return (
            "content_filter" if not response.choices or response.choices[0].finish_reason == "content_filter" else None
        )

    def _blocked_error(self, event: Mapping[str, object]) -> BaseException:
        """The error the router raises for a blocked response a fallback would serve."""
        import litellm.exceptions as litellm_exceptions
        from litellm.llms.anthropic.messages.native.utils import (
            get_safeguard_refusal_stop_details,
            safeguard_refusal_error,
        )

        model: Final = str(event.get("model"))
        stop_details: Final = get_safeguard_refusal_stop_details(self._last_response)
        if event.get("blocked") == "refusal" and stop_details is not None:
            return safeguard_refusal_error(model=model, stop_details=stop_details)
        return litellm_exceptions.ContentPolicyViolationError(
            message="Response output was blocked.", model=model, llm_provider=""
        )

    def _params_base_model(self, deployment_id: str) -> str | None:
        """``_deployment_params_base_model``: the ``base_model`` under the deployment's ``litellm_params``."""
        record: Final = as_str_mapping(self._core._handle.deployment(deployment_id)) or {}  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        base_model: Final = (as_str_mapping(record.get("litellm_params")) or {}).get("base_model")
        return base_model if isinstance(base_model, str) else None

    def _logged_error(self, event: Mapping[str, object]) -> BaseException | None:
        """The error a ``retry_logged`` event records."""
        match event.get("cause"):
            case "level":
                level: Final = event.get("level")
                return self._level_errors.get(level if isinstance(level, int) else 0, self._last_error)
            case "no_deployment":
                kind: Final = "unknown_model" if event.get("unknown") else "no_healthy_deployments"
                return self._core._failure_error({"error_kind": kind}, str(event.get("failed_model")), None)  # pyright: ignore[reportPrivateUsage]  # see above
            case _:
                return self._last_error

    def take_events(self, events: object) -> None:
        for raw_event in _as_list(events):
            event = as_str_mapping(raw_event)
            if event is None:
                continue
            if event.get("kind") == "mock_failed":
                resolved = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                mocked = mock_error(
                    str(event.get("mock")),
                    str(event.get("model")),
                    event.get("num_retries"),
                    self._request,
                    {
                        "fallbacks": resolved.fallbacks,
                        "context_window_fallbacks": resolved.context_window_fallbacks,
                        "content_policy_fallbacks": resolved.content_policy_fallbacks,
                    },
                )
                level = event.get("level")
                self._last_error = mocked
                self._level_errors[level if isinstance(level, int) else 0] = mocked
                continue
            if event.get("kind") == "level_failed":
                kind = "unknown_model" if event.get("unknown") else "no_healthy_deployments"
                level = event.get("level")
                self._level_errors[level if isinstance(level, int) else 0] = self._core._failure_error(  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                    {"error_kind": kind}, str(event.get("model")), None
                )
                continue
            if event.get("kind") == "fallback":
                self._walk.hop(event)
                hop_level = event.get("level")
                self._level = hop_level if isinstance(hop_level, int) else self._level
                continue
            if event.get("kind") == "blocked":
                level = event.get("level")
                blocked = self._blocked_error(event)
                self._last_error = blocked
                self._level_errors[level if isinstance(level, int) else 0] = blocked
                continue
            if event.get("kind") == "fallback_done":
                level = event.get("level")
                at = level if isinstance(level, int) else 0
                self._level = at
                if event.get("succeeded") is not True:
                    self._hint_nested_failure(self._level_errors.get(at + 1), str(event.get("to")))
                self.outcomes.done(
                    event,
                    self._level_errors.get(at, self._last_error),
                    self._level_errors.get(at + 1),
                    self._walk.hop_kwargs(event),
                )
                continue
            if event.get("kind") == "retry_logged":
                self._walk.log_retry(event, self._logged_error(event))
                continue
            if event.get("kind") != "retries_exhausted":
                continue
            error: BaseException | None = self._last_error
            # `async_function_with_retries` stamps the error its exhausted retry loop raises.
            if error is not None and type(error) in self._core._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                error.max_retries = event.get("max")  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
                error.num_retries = event.get("attempted")  # pyright: ignore[reportAttributeAccessIssue]  # same

    def _call(self, step: Mapping[str, object]) -> "tuple[Mapping[str, object], AttemptCall]":
        name: Final = step.get("model")
        requested: Final = str(name) if name is not None else str(self._request.get("model"))
        deployment: Final = for_requested_model(as_str_mapping(step.get("deployment")) or {}, requested)
        self._attempts += 1
        level: Final = step.get("fallback_level")
        # The request params the walk's dict targets merged in (`kwargs.update(target)`).
        params: Final = as_str_mapping(step.get("params"))
        call: Final = attempt_call(
            self._core,
            self._plan,
            self._request if params is None else {**self._request, **params},
            deployment,
            group=requested,
            requested=requested,
            fallback_level=level if isinstance(level, int) else 0,
        )
        shaped: Final = shape_attempt_kwargs(call.kwargs, deployment)
        # The walk's metadata for the attempt's level (retries, fallback hops) and its hop kwargs.
        call.kwargs = self._walk.stamp(step, shaped)
        field: Final = self._plan.metadata_field
        if isinstance(call.served, dict) and field in call.kwargs:
            call.served[field] = call.kwargs[field]
        self._served = (requested, call.served)
        self._core._endpoints.counters.started(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        return deployment, call

    @property
    def response(self) -> object:
        return self._last_response

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def _stream_attempt(
        self, step: Mapping[str, object], kwargs: Mapping[str, object], slot: "contextlib.ExitStack | None" = None
    ) -> StreamAttempt:
        def number(key: str) -> int:
            value: Final = step.get(key)
            return value if isinstance(value, int) else 0

        order: Final = step.get("target_order")
        deployment_id: Final = step.get("deployment_id")
        return StreamAttempt(
            core=self._core,
            op=self._op,
            request=self._request,
            kwargs=kwargs,
            group=str(step.get("model") or step.get("model_group")),
            order=order if isinstance(order, int) else None,
            depth=number("fallback_depth"),
            level=number("fallback_level"),
            deployment_id=deployment_id if isinstance(deployment_id, str) else None,
            attempted_retries=number("retry"),
            slot=slot,
        )

    def _failed(
        self,
        step: Mapping[str, object],
        deployment: Mapping[str, object],
        call: "AttemptCall | None",
        error: BaseException,
    ) -> Mapping[str, object]:
        served: Final = call.served if call is not None else {}  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        stamp_error(self._plan, error, deployment, served)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
        self._core._endpoints.attempt_failed(self._plan, error, deployment, call, str(step.get("model")))  # pyright: ignore[reportPrivateUsage]  # see above
        self._last_error = error
        level: Final = step.get("fallback_level")
        self._level_errors[level if isinstance(level, int) else 0] = error
        return attempt_failure(error)

    def _succeeded(self, call: "AttemptCall", response: object, host_op: str) -> Mapping[str, object]:
        self._core._endpoints.counters.succeeded(self._plan, call)  # pyright: ignore[reportPrivateUsage]  # see above
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response), "blocked": self._blocked(response, host_op)}

    async def _refused(
        self,
        step: Mapping[str, object],
        deployment: Mapping[str, object],
        call: "AttemptCall",
        error: BaseException,
    ) -> Mapping[str, object]:
        """A routing pre-call check refused the attempt (``async_routing_strategy_pre_call_checks``):
        a rate limit cools the deployment down (``set_cooldown_deployments``); the attempt's failure
        is recorded by nothing else."""
        import litellm.exceptions as litellm_exceptions

        core: Final = self._core
        if isinstance(error, litellm_exceptions.RateLimitError):
            deployment_id: Final = (as_str_mapping(deployment.get("model_info")) or {}).get("id")
            if isinstance(deployment_id, str):
                report: Final = core._handle.report_failure(deployment_id, dict(attempt_failure(error)), uncounted=True)  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                await core._drive_cache_steps(report)  # pyright: ignore[reportPrivateUsage]  # see above
        return {**self._failed(step, deployment, call, error), "recorded": True}

    async def attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: the picked one
        call: AttemptCall | None = None  # rebind-ok: built inside the try
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            host: Final = self._core._host  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            # The deployment's max_parallel_requests slot; an open chat stream takes it along.
            with enter_deployment_slot(self._core, self._plan, deployment, call.kwargs) as slot:
                # A search tool is no deployment: nothing to check; Python's `_arerank` and
                # `amoderation` run no routing pre-call checks either.
                if self._plan.kind not in _NO_ROUTING_PRE_CALL_CHECKS:
                    try:
                        await routing_pre_call_checks(
                            self._core._checks.routing_callbacks(),  # pyright: ignore[reportPrivateUsage]  # see above
                            as_str_mapping(step.get("deployment")) or {},
                        )
                    except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop as the attempt's failure
                        return await self._refused(step, deployment, call, refused)
                pending: Final[object] = (
                    self._core._endpoints.create_file_everywhere(  # pyright: ignore[reportPrivateUsage]  # see above
                        call.kwargs, call.model_name, self._request.get("specific_deployment")
                    )
                    if self._plan.kind == "create_file"
                    else host.invoke(call.host_op)(**call.kwargs)
                )
                # The generic helper awaits whatever the handler returns, as the Python router does.
                awaited: Final = self._plan.kind == "generic" or inspect.isawaitable(pending)
                answered: Final[object] = await pending if awaited else pending  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]  # see above
                self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]  # see above
                response: Final = await open_attempt_stream(self._stream_attempt(step, call.kwargs, slot), answered)  # pyright: ignore[reportUnknownArgumentType]  # see above
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            return self._failed(step, deployment, call, error)
        return self._succeeded(call, response, call.host_op)

    def attempt_sync(self, step: Mapping[str, object]) -> Mapping[str, object]:
        deployment: Mapping[str, object] = as_str_mapping(step.get("deployment")) or {}  # rebind-ok: see attempt
        call: AttemptCall | None = None  # rebind-ok: see attempt
        try:
            deployment, call = self._call(step)  # rebind-ok: rebound as the Python router rebinds it
            try:
                self._core._endpoints.sync_pre_call_checks(self._plan, deployment, self._request)  # pyright: ignore[reportPrivateUsage]  # see above
            except Exception as refused:  # noqa: BLE001  # handed to the Rust call loop like any attempt failure
                # The sync path records no outcome for a refused pre-call check: no cooldown.
                return {**self._failed(step, deployment, call, refused), "recorded": True}
            answered: Final = self._core._host.invoke(call.host_op)(**call.kwargs)  # pyright: ignore[reportPrivateUsage]  # a sync endpoint
            if inspect.isawaitable(answered):
                raise _NotSynchronous(f"the host's {call.host_op} endpoint is not synchronous")
            self._core._endpoints.check_response(self._plan, answered, call)  # pyright: ignore[reportPrivateUsage]  # see above
            response: Final = open_attempt_stream_sync(self._stream_attempt(step, call.kwargs), answered)
        except _NotSynchronous:
            raise
        except Exception as error:  # noqa: BLE001  # see attempt
            return self._failed(step, deployment, call, error)
        return self._succeeded(call, response, "")

    async def select(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        if step.get("stage") == "callback_filters":
            answer, error = await self._core._answer_callback_filters(step, request, request.get("messages"))  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            if error is not None:
                # The try failed with the filter's error: the level raises it if nothing else serves.
                self._last_error = error
                self._level_errors[self._level] = error
            return answer
        return await self._core._answer_select(step, request, request.get("messages"), request.get("input"))  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core

    def select_sync(self, step: Mapping[str, object]) -> object:
        request: Final = self._request
        return self._core._answer_select_sync(step, request, request.get("messages"), request.get("input"))  # pyright: ignore[reportPrivateUsage]  # see above

    async def allow_fallback(self, target: str) -> bool:
        """`_is_fallback_target_authorized`, then `_is_fallback_target_within_budget`."""
        if target == self._request.get("model"):
            return True
        from litellm.types.router import FallbackSkipReason

        ctx: Final = RoutingContext(self._core)  # pyright: ignore[reportArgumentType]  # the checks read registry() and settings()
        request: Final = dict(self._request)
        if not await ctx.fallback_access_allowed(model=target, request_kwargs=request):
            self._walk.targets().record_skip(target, FallbackSkipReason.NOT_AUTHORIZED)
            return False
        if not await ctx.fallback_budget_allowed(model=target, request_kwargs=request):
            self._walk.targets().record_skip(target, FallbackSkipReason.OVER_BUDGET)
            return False
        return True

    def finish(self, step: Mapping[str, object]) -> object:
        """The response with the router's headers: the deployment id, the model group it was
        served as, and the retry and fallback counts (``set_response_headers``)."""
        if not self._plan.sets_headers:
            return self._last_response
        response: object = prepare_response_for_header_attachment(self._last_response)  # rebind-ok: gains headers
        if response is None:
            return self._last_response
        name, kwargs = self._served
        additional: Final = ensure_response_additional_headers(response)
        apply_response_model_id(response, find_deployment_metadata(kwargs))
        additional["x-litellm-model-group"] = name
        retries: Final = step.get("attempted_retries")
        if isinstance(retries, int):
            max_retries: Final = step.get("max_retries")
            response = add_retry_headers_to_response(  # rebind-ok: see above
                response=response,
                attempted_retries=retries,
                max_retries=max_retries if isinstance(max_retries, int) and retries > 0 else None,
            )
        fallbacks: Final = step.get("attempted_fallbacks")
        attempted: Final = fallbacks if isinstance(fallbacks, int) else 0
        return add_fallback_headers_to_response(
            response=self.outcomes.add_headers(response, attempted), attempted_fallbacks=attempted
        )

    async def finish_async(self, step: Mapping[str, object]) -> object:
        response: Final = self.finish(step)
        name: Final = self._served[0]
        if name is not None and response is not None and self._plan.sets_headers:
            remaining: Final = await self._core.health().usage(name, kind="remaining")
            if isinstance(remaining, Mapping):
                apply_remaining_usage_headers(ensure_response_additional_headers(response), dict(remaining))
        return response

    def failed(self, step: Mapping[str, object]) -> BaseException:
        model: Final = str(self._request.get("model"))
        primary: Final = self._level_errors.get(0)
        if primary is None:
            return self._core._failure_error(step, model, self._last_error)  # pyright: ignore[reportPrivateUsage]  # see above
        if self._core._host.dynamic().expose_router_debug_in_errors and hasattr(primary, "message"):  # pyright: ignore[reportPrivateUsage]  # see above
            primary.message += self._explanation(primary, model)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
        return primary

    def _hint_nested_failure(self, error: BaseException | None, model: str) -> None:
        """A fallback level that fails on a context-window or content-policy error with no list
        of its own appends the hint to set one, as every level of the Python walk does."""
        from litellm.router_utils.fallback_event_handlers import check_non_standard_fallback_format

        if (
            error is None
            or not hasattr(error, "message")
            or not self._core._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        ):
            return
        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # see above
        fallbacks: Final = self._request.get("fallbacks", resolved.fallbacks)
        if check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            return
        hint: Final = self._kind_hint(error, model, fallbacks)
        if hint:
            error.message += hint  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the router appends to the error it raises

    def _kind_hint(self, error: BaseException, model: str, fallbacks: object) -> str:
        """The hint a context-window or content-policy error gets when its own list is unset."""
        import litellm.exceptions as litellm_exceptions

        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        if isinstance(error, litellm_exceptions.ContextWindowExceededError):
            if self._request.get("context_window_fallbacks", resolved.context_window_fallbacks) is not None:
                return ""
            return (
                f"\nmodel={model}. context_window_fallbacks=None. fallbacks={fallbacks}.\n\n"
                "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            )
        if isinstance(error, litellm_exceptions.ContentPolicyViolationError):
            if self._request.get("content_policy_fallbacks", resolved.content_policy_fallbacks) is not None:
                return ""
            return "\n" + format_content_policy_fallback_hint(model, None, fallbacks)  # pyright: ignore[reportArgumentType]  # the list as configured
        return ""

    def _explanation(self, primary: BaseException, model: str) -> str:
        """What ``_async_function_with_fallbacks_common_utils`` appends to the error it re-raises
        under ``expose_router_debug_in_errors``, from the resolved settings and what the call did."""
        import litellm.exceptions as litellm_exceptions
        from litellm.router_utils.fallback_event_handlers import (
            check_non_standard_fallback_format,
            get_fallback_model_group_for_lookup_groups,
        )

        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        request: Final = self._request
        # The request's own lists replace the router's (`kwargs.get("fallbacks", router)`).
        fallbacks: Final = request.get("fallbacks", resolved.fallbacks)
        window_list: Final = request.get("context_window_fallbacks", resolved.context_window_fallbacks)
        policy_list: Final = request.get("content_policy_fallbacks", resolved.content_policy_fallbacks)
        lookup: Final = (model,)
        if check_non_standard_fallback_format(fallbacks=_as_list(fallbacks)):
            chain_list: Final = _as_list(fallbacks)
            rejected: Final = rejected_fallback_target(request)
            detail: Final = (
                truncate_fallback_error_detail(redact_string(str(rejected)))
                if rejected is not None
                else self._fallback_detail(chain_list)
            )
            return format_fallback_outcome_message(model, chain_list, detail, skipped=self._walk.targets().skipped)
        context_window: Final = isinstance(primary, litellm_exceptions.ContextWindowExceededError)
        content_policy: Final = isinstance(primary, litellm_exceptions.ContentPolicyViolationError)
        # The error kind's own list when it is set, else the generic one with a hint to set it.
        own: Final = window_list if context_window else policy_list if content_policy else None
        hint: Final = (
            f"\nmodel={model}. context_window_fallbacks=None. fallbacks={fallbacks}.\n\n"
            "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            if context_window and own is None
            else "\n" + format_content_policy_fallback_hint(model, None, fallbacks)  # pyright: ignore[reportArgumentType]  # the list as configured
            if content_policy and own is None
            else ""
        )
        source: Final = own if own is not None else fallbacks
        entries: Final = [dict(m) for entry in _as_list(source) if (m := as_str_mapping(entry))]
        found, generic = (
            get_fallback_model_group_for_lookup_groups(entries, lookup)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
            if entries
            else (None, None)
        )
        chain: Final[Sequence[object] | None] = (
            found if found is not None else _as_list(entries[generic].get("*")) if generic is not None else None
        )
        if chain is None and entries and own is None:
            return hint + format_no_fallback_group_message(lookup, entries)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
        if chain is None:
            return hint + format_fallback_outcome_message(model, None, "")
        return hint + format_fallback_outcome_message(
            model, chain, self._fallback_detail(chain), skipped=self._walk.targets().skipped
        )

    def _fallback_detail(self, chain: Sequence[object]) -> str:
        """The error the fallback walk ended on: the first fallback level's last error, or the
        unknown-model error of a target with no deployment, which fails before any attempt."""
        last: Final = str(chain[-1]) if chain else None
        attempted: Final = self._level_errors.get(1)
        unknown: Final = last is not None and self._core._handle.resolve(last) is None  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        error: Final = (
            attempted
            if attempted is not None or not unknown or last is None
            else self._core._failure_error({"error_kind": "unknown_model"}, last, None)  # pyright: ignore[reportPrivateUsage]  # see above
        )
        return truncate_fallback_error_detail(redact_string(str(error))) if error is not None else ""


__all__ = [
    "RUST_SERVED_OPS",
    "Resolved",
    "RustCore",
    "RustCoreUnsupported",
    "RustHealth",
    "RustRegistry",
    "RustSettings",
    "attempt_failure",
    "failure_kind",
    "total_tokens",
    "unsupported_arguments",
]
