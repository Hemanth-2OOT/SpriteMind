"""Animation name resolution.

The renderer supports a fixed animation taxonomy. Historically any unrecognised
name was silently rewritten to ``walk``, which made ``attack`` and ``hurt``
render an identical walk cycle with no indication that the request had been
discarded. This module keeps the accepted vocabulary explicit:

* canonical names are returned unchanged;
* a small set of documented aliases is mapped onto a canonical name and the
  caller is told that a substitution happened;
* anything else raises, so the API can return 400 instead of a wrong animation.
"""
from typing import Optional, Tuple

from spritemind.dataset.temporal_dataset import ANIMATION_CATEGORIES

SUPPORTED_ANIMATIONS: Tuple[str, ...] = tuple(ANIMATION_CATEGORIES)

# Deliberate, documented substitutions. Natural-language verbs map onto the
# canonical motion profile that already implements the intended movement.
ANIMATION_ALIASES = {
    # combat
    "attack": "melee_attack",
    "melee": "melee_attack",
    "slash": "melee_attack",
    "swing": "melee_attack",
    "strike": "melee_attack",
    "shoot": "ranged_attack",
    "fire": "ranged_attack",
    "bolt": "ranged_attack",
    "throw": "ranged_attack",
    "hurt": "hit_reaction",
    "damage": "hit_reaction",
    "damaged": "hit_reaction",
    "hit": "hit_reaction",
    "block": "blocking",
    "guard": "blocking",
    "shield": "blocking",
    "magic": "cast",
    "spell": "cast",
    "magic_attack": "cast",
    "channel": "cast",
    "special": "special_ability",
    "ability": "special_ability",
    "ultimate": "special_ability",
    # movement
    "stand": "idle",
    "rest": "idle",
    "sprint": "run",
    "leap": "jump",
    "teleport": "dash",
    "die": "death",
    "dead": "death",
}


class UnknownAnimationError(ValueError):
    """Raised when an animation name is neither supported nor a known alias."""


def resolve_animation_name(name: Optional[str]) -> Tuple[str, bool]:
    """Returns ``(canonical_name, aliased)``.

    ``None``/empty defaults to ``walk`` (the historical default for a missing
    field). An unsupported *value* is never silently coerced.
    """
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not key:
        return "walk", False
    if key in SUPPORTED_ANIMATIONS:
        return key, False
    if key in ANIMATION_ALIASES:
        return ANIMATION_ALIASES[key], True
    raise UnknownAnimationError(
        f"Unsupported animation '{name}'. Supported: {', '.join(SUPPORTED_ANIMATIONS)}"
    )
