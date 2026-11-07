"""Procedural Baseline Generator.

Provides deterministic algorithmic pixel-art synthesis to serve as a rigorous
baseline comparison for neural models.
"""
from typing import Dict, Any, Optional
from spritemind.dataset.generator import CharacterGenerator
from spritemind.pixelscript.ast import PixelScriptProgram


class ProceduralBaseline:
    """Deterministic rule-based baseline."""

    @classmethod
    def generate(
        cls,
        category: str = "knight",
        style: str = "classic_16bit",
        pose: str = "idle",
        palette_name: str = "pico8",
        width: int = 32,
        height: int = 32,
        seed: int = 42,
        colour: Optional[str] = None,
        element: Optional[str] = None,
    ) -> PixelScriptProgram:
        return CharacterGenerator.generate_character(
            category=category,
            style=style,
            pose=pose,
            palette_name=palette_name,
            width=width,
            height=height,
            frame_idx=0,
            total_frames=1,
            seed=seed,
            colour=colour,
            element=element,
        )
