"""
The retry members ``RustCore`` serves: ``should_retry_this_error`` and
``get_num_retries_from_retry_policy``. router-core decides both; this module only describes the
Python exception the way an attempt failure is described and reads the answer back.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.types.router import RetryPolicy

if TYPE_CHECKING:
    from litellm.rust_bridge._native import (  # pyright: ignore[reportPrivateUsage]  # the stub's handle
        RouterCoreHandle,
    )


def resolved_retry_policy(value: object) -> RetryPolicy | None:
    """The router's ``retry_policy`` as the Python router's constructor keeps it: a dict is read
    into a ``RetryPolicy``, a ``RetryPolicy`` is kept, anything else is no policy."""
    if isinstance(value, RetryPolicy):
        return value
    if isinstance(value, Mapping):
        fields: Final[Mapping[str, int | None]] = value  # pyright: ignore[reportUnknownVariableType]  # the policy's fields
        return RetryPolicy(**fields)  # pyright: ignore[reportArgumentType]  # validated by the model
    return None


def refuse_unserved_retry_policies(policies: object) -> None:
    """A request ``model_group_retry_policy`` whose entries are not policies is not served."""
    from litellm.router_rust_core import RustCoreUnsupported

    if policies is None:
        return
    entries: Final = policies.values() if isinstance(policies, Mapping) else None  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # read as objects
    if entries is None or not all(isinstance(entry, RetryPolicy | Mapping) for entry in entries):  # pyright: ignore[reportUnknownVariableType]  # see above
        raise RustCoreUnsupported("a request model_group_retry_policy that is not a policy per group is not served yet")


def _count(deployments: object) -> int:
    return len(deployments) if isinstance(deployments, list) else 0  # pyright: ignore[reportUnknownArgumentType]  # only the length is read


def _filled(fallbacks: object) -> bool:
    return isinstance(fallbacks, Sequence) and len(fallbacks) > 0  # pyright: ignore[reportUnknownArgumentType]  # only the length is read


class RustRetries:
    """Retry members read from the Rust core's handle."""

    _handle: "RouterCoreHandle"

    def should_retry_this_error(
        self,
        error: Exception,
        healthy_deployments: list[object] | None = None,  # mutable-ok: the Python router's signature
        all_deployments: list[object] | None = None,  # mutable-ok: same
        context_window_fallbacks: list[object] | None = None,  # mutable-ok: same
        content_policy_fallbacks: list[object] | None = None,  # mutable-ok: same
        regular_fallbacks: list[object] | None = None,  # mutable-ok: same
    ) -> bool:
        """``True`` when the error is retried in the same group; otherwise the error is raised."""
        from litellm.router_rust_core import attempt_failure

        retried: Final = self._handle.should_retry(
            dict(attempt_failure(error)),
            (_count(healthy_deployments), _count(all_deployments)),
            (context_window_fallbacks is not None, content_policy_fallbacks is not None, _filled(regular_fallbacks)),
        )
        if not retried:
            raise error
        return True

    def get_num_retries_from_retry_policy(self, exception: Exception, model_group: str | None = None) -> int | None:
        from litellm.router_rust_core import attempt_failure

        return self._handle.retry_policy_retries(dict(attempt_failure(exception)), model_group)


__all__ = ["RustRetries", "refuse_unserved_retry_policies", "resolved_retry_policy"]
