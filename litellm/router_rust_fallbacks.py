"""
Fallback configuration checks ``RustCore`` runs on the host: ``validate_fallbacks``, with the
Python router's error texts. router-core walks the fallbacks; this module only refuses a list it
could not walk the way the Python router would, before the core is built.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from litellm.router_utils.fallback_event_handlers import check_non_standard_fallback_format


def validate_fallbacks(fallback_param: Sequence[object] | None) -> None:
    """``Router.validate_fallbacks``: every entry is a one-key dict whose targets are a list (or a
    client-side dict target)."""
    if fallback_param is None:
        return
    for fallback_dict in fallback_param:
        if not isinstance(fallback_dict, dict):
            raise ValueError(f"Item '{fallback_dict}' is not a dictionary.")  # noqa: TRY004  # the Python router's error class
        if len(fallback_dict) != 1:  # pyright: ignore[reportUnknownArgumentType]  # only the size is read
            raise ValueError(
                f"Dictionary '{fallback_dict}' must have exactly one key, but has {len(fallback_dict)} keys."  # pyright: ignore[reportUnknownArgumentType]  # see above
            )
        ((model_group, targets),) = fallback_dict.items()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # an entry as configured
        entry: list[object] = [fallback_dict]  # mutable-ok: the one-entry list the check reads
        if isinstance(targets, list) or check_non_standard_fallback_format(fallbacks=entry):
            continue
        raise TypeError(
            f"Fallbacks for '{model_group}' must be a list of model names, got '{targets}'. "
            f"Use {{'{model_group}': ['{targets}']}}"
        )


def rejected_fallback_target(request: Mapping[str, object]) -> Exception | None:
    """The error ``run_async_fallback`` raises before its walk when a client-side request list
    carries a dict target the request may not set (a server-owned federation field, an ``oidc/``
    api key): every target of that walk is refused."""
    from litellm.litellm_core_utils.credential_accessor import reject_api_key_secret_reference
    from litellm.types.router import reject_server_owned_wif_params

    fallbacks: Final = request.get("fallbacks")
    if not isinstance(fallbacks, list) or not check_non_standard_fallback_format(fallbacks=fallbacks):  # pyright: ignore[reportUnknownArgumentType]  # a request list as given
        return None
    for target in fallbacks:  # pyright: ignore[reportUnknownVariableType]  # see above
        if isinstance(target, dict):
            try:
                reject_server_owned_wif_params(target)  # pyright: ignore[reportUnknownArgumentType]  # see above
                reject_api_key_secret_reference(target)  # pyright: ignore[reportUnknownArgumentType]  # see above
            except Exception as error:  # noqa: BLE001  # the walk's own refusal, reported as its failure
                return error
    return None


class RustFallbacks:
    """Fallback members served on the host."""

    def validate_fallbacks(self, fallback_param: Sequence[object] | None) -> None:
        validate_fallbacks(fallback_param)


__all__ = ["RustFallbacks", "rejected_fallback_target", "validate_fallbacks"]
