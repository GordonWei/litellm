//! Pre-routing: before an async pick resolves its name, the host may route the request to another
//! model group (Python's `_core_async_pre_routing_hook`). The complexity, auto, adaptive and
//! quality routers registered from `auto_router/...` marker deployments, and the Claude Code
//! session binding, live host-side; the core asks once per pick, before resolution, and resolves
//! the name the host answers. The attempt still serves the requested name (its `model_group`):
//! only the deployments come from the routed group.

use crate::{Core, config::Op, error::Error, host::RouterRequest};

/// What the host's pre-routing step sees.
#[derive(Clone, Copy, Debug)]
pub struct PreRoute<'a> {
    pub op: &'a Op,
    pub request: &'a RouterRequest,
    /// The name this pick resolves: the requested name or the fallback target.
    pub name: &'a str,
    /// How many fallbacks deep the walk is (Python's `fallback_depth`): only a first attempt
    /// binds a Claude Code session.
    pub fallback_depth: u32,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
}

impl Core {
    /// The name a pick of `ask.name` resolves: the host's pre-routed group, else the name itself.
    /// Only the async picks pre-route (`request.pre_routes`); the sync path never does.
    pub(crate) async fn pre_routed(&self, ask: PreRoute<'_>) -> Result<String, Error> {
        if !ask.request.pre_routes {
            return Ok(ask.name.to_owned());
        }
        Ok(self
            .host
            .pre_route(&ask)
            .await?
            .unwrap_or_else(|| ask.name.to_owned()))
    }
}
