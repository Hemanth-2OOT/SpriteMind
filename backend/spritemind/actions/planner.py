"""Prompt -> action set planner.

Turns a free-text character prompt into a coherent list of animations
(archetype, element, weapon and action ordering) that a game asset needs.
"""
import re
from typing import Dict, List, Optional, Sequence

from spritemind.actions.catalog import (
    ACTION_KEYWORDS,
    ARCHETYPE_ACTIONS,
    ARCHETYPE_KEYWORDS,
    CATEGORY_ARCHETYPE,
    ELEMENT_DEFAULT_BY_CATEGORY,
    ELEMENT_KEYWORDS,
    WEAPON_BY_ARCHETYPE,
    WEAPON_KEYWORDS,
    ActionPlan,
    ActionSpec,
)
from spritemind.dataset.generator import CATEGORIES, STYLES
from spritemind.utils.palettes import PALETTES

PALETTE_NAMES = set(PALETTES.keys())
CATEGORY_ALIASES = {
    "knight": ("knight", "paladin", "warrior", "soldier", "guard"),
    "wizard": ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "druid"),
    "rogue": ("rogue", "assassin", "thief", "ninja", "ranger", "archer"),
    "skeleton": ("skeleton", "bone", "undead"),
    "zombie": ("zombie", "ghoul", "walker"),
    "goblin": ("goblin", "orc", "imp"),
    "robot": ("robot", "mech", "android", "cyborg", "droid"),
    "slime": ("slime", "blob", "ooze"),
    "animal": ("wolf", "bear", "cat", "dog", "animal", "beast"),
    "fantasy_creature": ("dragon", "demon", "golem", "elemental", "creature", "titan"),
}

SIGNATURE_ORDER = ["cast", "cast_burst", "shoot", "attack", "attack_heavy", "block", "hurt", "jump", "fall"]


def _find(words: Sequence[str], text: str) -> Optional[str]:
    for word in words:
        if re.search(rf"\b{re.escape(word)}", text):
            return word
    return None


class ActionSetPlanner:
    """Deterministic prompt interpretation for multi-action sprite sets."""

    @classmethod
    def plan(
        cls,
        prompt: str = "",
        category: Optional[str] = None,
        style: Optional[str] = None,
        palette: Optional[str] = None,
        width: int = 32,
        height: int = 32,
        requested_actions: Optional[Sequence[str]] = None,
        max_actions: int = 12,
    ) -> ActionPlan:
        text = (prompt or "").lower().strip()
        rationale: List[str] = []

        final_category = category if category in CATEGORIES else cls._infer_category(text)
        if final_category:
            rationale.append(f"category '{final_category}'")
        else:
            final_category = "knight"
            rationale.append("category defaulted to 'knight'")

        final_style = style if style in STYLES else cls._infer_style(text)
        final_palette = palette if palette in PALETTE_NAMES else cls._infer_palette(text)

        archetype = cls._infer_archetype(text) or CATEGORY_ARCHETYPE.get(final_category, "warrior")
        rationale.append(f"archetype '{archetype}'")

        weapon = cls._infer_weapon(text) or WEAPON_BY_ARCHETYPE.get(archetype, "none")
        element = cls._infer_element(text) or ELEMENT_DEFAULT_BY_CATEGORY.get(final_category, "none")
        if element == "none" and final_style == "cyberpunk":
            element = "lightning"
        if element != "none":
            rationale.append(f"element '{element}' on attack effects")

        actions = cls._select_actions(text, archetype, requested_actions, max_actions)
        rationale.append(f"{len(actions)} actions: " + ", ".join(a.key for a in actions))

        return ActionPlan(
            prompt=prompt,
            category=final_category,
            style=final_style,
            palette=final_palette,
            width=width,
            height=height,
            archetype=archetype,
            element=element,
            weapon=weapon,
            actions=actions,
            rationale=rationale,
        )

    @classmethod
    def _infer_category(cls, text: str) -> Optional[str]:
        for cat, aliases in CATEGORY_ALIASES.items():
            if _find(aliases, text):
                return cat
        return None

    @classmethod
    def _infer_style(cls, text: str) -> str:
        hints = {
            "dark_fantasy": ("dark fantasy", "gothic", "grim", "blasphemous", "bloodborne"),
            "cyberpunk": ("cyberpunk", "neon", "sci-fi", "scifi"),
            "retro_8bit": ("8-bit", "8 bit", "retro", "nes"),
            "flat": ("flat", "minimal", "clean"),
        }
        for style, words in hints.items():
            if _find(words, text):
                return style
        return "classic_16bit"

    @classmethod
    def _infer_palette(cls, text: str) -> str:
        hints = {
            "gameboy": ("gameboy", "game boy", "green mono"),
            "nes": ("nes", "nintendo"),
            "sweetie16": ("sweetie", "pastel"),
            "dawnbringer": ("dawnbringer",),
        }
        for name, words in hints.items():
            if name in PALETTE_NAMES and _find(words, text):
                return name
        return "pico8"

    @classmethod
    def _infer_archetype(cls, text: str) -> Optional[str]:
        for archetype, words in ARCHETYPE_KEYWORDS.items():
            if _find(words, text):
                return archetype
        return None

    @classmethod
    def _infer_weapon(cls, text: str) -> Optional[str]:
        for weapon, words in WEAPON_KEYWORDS:
            if _find(words, text):
                return weapon
        return None

    @classmethod
    def _infer_element(cls, text: str) -> Optional[str]:
        for element, words in ELEMENT_KEYWORDS.items():
            if _find(words, text):
                return element
        return None

    @classmethod
    def _select_actions(
        cls,
        text: str,
        archetype: str,
        requested: Optional[Sequence[str]],
        max_actions: int,
    ) -> List[ActionSpec]:
        pool: Dict[str, ActionSpec] = {a.key: a for a in ARCHETYPE_ACTIONS[archetype]}

        hinted = {key for key, words in ACTION_KEYWORDS.items() if _find(words, text) and key in pool}

        ordered_keys: List[str] = ["idle", "walk", "run"]
        ordered_keys += [k for k in SIGNATURE_ORDER if k in pool]
        ordered_keys += ["dash", "fall", "death"]

        keep: List[str] = []
        for key in ordered_keys:
            if key in pool and key not in keep:
                keep.append(key)

        if hinted:
            for key in hinted:
                if key not in keep:
                    keep.append(key)

        if requested:
            for key in requested:
                if key in pool and key not in keep:
                    keep.append(key)
            keep = [k for k in keep if k in (set(requested) | {"idle"})]

        return [pool[k] for k in keep[:max_actions]]
