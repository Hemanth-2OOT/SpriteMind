"""Builds a full game-ready animation set from a single prompt.

Every action shares one base character program, so the sprite keeps its identity
(proportions, palette, gear) across idle, movement, attacks and death.
"""
import base64
from typing import Any, Dict, List, Optional

import numpy as np

from spritemind.actions.effects import build_effects
from spritemind.actions.planner import ActionSetPlanner
from spritemind.actions.catalog import ActionPlan, ActionSpec
from spritemind.dataset.generator import CharacterGenerator
from spritemind.ml.temporal import TemporalAnimator
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.transform import mirror_script
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.export import export_gif


class ActionSetBuilder:
    """Generates a coherent multi-action sprite set for one character."""

    @classmethod
    def build(
        cls,
        prompt: str = "",
        category: Optional[str] = None,
        style: Optional[str] = None,
        palette: Optional[str] = None,
        width: int = 32,
        height: int = 32,
        engine: str = "procedural_baseline",
        seed: int = 42,
        actions: Optional[List[str]] = None,
        max_actions: int = 12,
        include_gif: bool = True,
        facing: str = "right",
    ) -> Dict[str, Any]:
        plan = ActionSetPlanner.plan(
            prompt=prompt,
            category=category,
            style=style,
            palette=palette,
            width=width,
            height=height,
            requested_actions=actions,
            max_actions=max_actions,
            seed=seed,
        )
        return cls.build_from_plan(
            plan, engine=engine, seed=seed, include_gif=include_gif, facing=facing
        )

    @classmethod
    def build_from_plan(
        cls,
        plan: ActionPlan,
        engine: str = "procedural_baseline",
        seed: int = 42,
        include_gif: bool = True,
        facing: str = "right",
    ) -> Dict[str, Any]:
        base_program = CharacterGenerator.generate_character(
            category=plan.category,
            style=plan.style,
            pose="idle",
            palette_name=plan.palette,
            width=plan.width,
            height=plan.height,
            seed=seed,
            proportion=plan.proportion,
            accessory=plan.accessory,
            weapon=None if plan.weapon == "none" else plan.weapon,
        )
        base_script = PixelScriptSerializer.serialize_program(base_program)
        # Mirror only after animation: the animator is direction-agnostic, so
        # flipping its input and its output would cancel out.
        shown_base_script = (
            mirror_script(base_script, plan.width) if facing == "left" else base_script
        )

        results: List[Dict[str, Any]] = []
        for spec in plan.actions:
            results.append(
                cls._build_action(
                    spec,
                    plan,
                    base_script,
                    engine=engine,
                    seed=seed,
                    include_gif=include_gif,
                    facing=facing,
                )
            )

        return {
            "status": "success",
            "prompt": plan.prompt,
            "plan": plan.to_dict(),
            "base_script": shown_base_script,
            "facing": "left" if facing == "left" else "right",
            "num_actions": len(results),
            "actions": results,
        }

    @classmethod
    def _build_action(
        cls,
        spec: ActionSpec,
        plan: ActionPlan,
        base_script: str,
        engine: str,
        seed: int,
        include_gif: bool,
        facing: str = "right",
    ) -> Dict[str, Any]:
        anim = TemporalAnimator.generate_animation(
            base_script_or_prompt=base_script,
            category=plan.category,
            style=plan.style,
            anim_category=spec.anim,
            palette_name=plan.palette,
            width=plan.width,
            height=plan.height,
            num_frames=spec.frames,
            fps=spec.fps,
            engine_mode=engine,
            seed=seed,
        )

        frames: List[Dict[str, Any]] = []
        rgba_frames: List[np.ndarray] = []
        for frame in anim["frames"]:
            script = frame["script_text"]
            effect_phase = frame["phase"]
            for effect in spec.effects:
                fx = build_effects(
                    effect,
                    effect_phase,
                    palette_name=plan.palette,
                    width=plan.width,
                    height=plan.height,
                    element=plan.element,
                )
                if fx:
                    script = f"{script}\n{fx}"
            if facing == "left":
                script = mirror_script(script, plan.width)

            result = PixelScriptEngine.execute(script, width=plan.width, height=plan.height, palette_name=plan.palette)
            data = result.to_dict(include_passes=True)
            data["script_text"] = script
            data["frame_index"] = frame["frame_index"]
            data["phase"] = frame["phase"]
            frames.append(data)
            rgba_frames.append(result.composite_array)

        entry: Dict[str, Any] = {
            **spec.to_dict(),
            "engine_used": anim["engine_used"],
            "is_looping": spec.loop,
            "num_frames": len(frames),
            "frames": frames,
        }

        if include_gif and rgba_frames:
            entry["gif_base64"] = base64.b64encode(
                export_gif(rgba_frames, fps=spec.fps, scale=4, loop=0 if spec.loop else 0)
            ).decode("ascii")
        return entry
