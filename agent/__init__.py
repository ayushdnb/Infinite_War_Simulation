# codex_bellum/agent/__init__.py
from __future__ import annotations

# Always available
from .brain import ActorCriticBrain, scripted_brain

# Back-compat: try to import TinyActorCritic; if missing, alias it.
try:
    from .brain import TinyActorCritic  # type: ignore
except Exception:
    class TinyActorCritic(ActorCriticBrain):  # type: ignore
        pass

__all__ = ["ActorCriticBrain", "TinyActorCritic", "scripted_brain"]
