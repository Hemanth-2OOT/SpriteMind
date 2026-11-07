"""PixelScript validation logic."""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_actions: int = 0
    invalid_action_count: int = 0
    invalid_action_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_actions": self.total_actions,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_rate": self.invalid_action_rate,
        }


class PixelScriptValidator:
    """Validates PixelScript programs against canvas dimensions and logical rules."""

    @classmethod
    def validate(
        cls,
        program: PixelScriptProgram,
        canvas_width: int = 32,
        canvas_height: int = 32,
        max_palette_size: int = 32,
    ) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        defined_parts: Set[str] = set()
        defined_layers: Set[int] = {0}  # Layer 0 is default
        defined_palette_indices: Set[int] = set()

        total = len(program.actions)
        invalid_count = 0

        for idx, action in enumerate(program.actions):
            line_ref = f"Action #{idx + 1}" + (f" (line {action.line_number})" if action.line_number else "")
            p = action.params
            has_error = False

            if action.action_type == ActionType.PALETTE:
                c_idx = p.get("index")
                if c_idx is None or not (0 <= c_idx < max_palette_size):
                    errors.append(f"{line_ref}: PALETTE index {c_idx} out of range [0..{max_palette_size-1}]")
                    has_error = True
                else:
                    defined_palette_indices.add(c_idx)

            elif action.action_type == ActionType.LAYER:
                lid = p.get("id")
                if lid is None or lid < 0:
                    errors.append(f"{line_ref}: LAYER id must be non-negative integer")
                    has_error = True
                else:
                    defined_layers.add(lid)

            elif action.action_type == ActionType.PART:
                pname = p.get("name")
                if not pname:
                    errors.append(f"{line_ref}: PART requires a non-empty name")
                    has_error = True
                else:
                    defined_parts.add(pname)
                parent = p.get("parent")
                if parent and parent not in defined_parts:
                    warnings.append(f"{line_ref}: PART '{pname}' references unknown parent '{parent}'")

            elif action.action_type == ActionType.PIXEL:
                x, y = p.get("x"), p.get("y")
                if x is None or y is None:
                    errors.append(f"{line_ref}: PIXEL missing x or y coordinates")
                    has_error = True
                elif not (0 <= x < canvas_width and 0 <= y < canvas_height):
                    warnings.append(f"{line_ref}: PIXEL ({x}, {y}) out of canvas bounds ({canvas_width}x{canvas_height})")

            elif action.action_type == ActionType.RECT:
                x, y, w, h = p.get("x"), p.get("y"), p.get("w"), p.get("h")
                if any(v is None for v in (x, y, w, h)):
                    errors.append(f"{line_ref}: RECT missing required dimension params")
                    has_error = True
                elif w <= 0 or h <= 0:
                    errors.append(f"{line_ref}: RECT width ({w}) and height ({h}) must be > 0")
                    has_error = True
                elif x >= canvas_width or y >= canvas_height or (x + w) <= 0 or (y + h) <= 0:
                    warnings.append(f"{line_ref}: RECT completely out of canvas bounds")

            elif action.action_type == ActionType.LINE:
                coords = (p.get("x0"), p.get("y0"), p.get("x1"), p.get("y1"))
                if any(c is None for c in coords):
                    errors.append(f"{line_ref}: LINE missing coordinate params")
                    has_error = True

            elif action.action_type == ActionType.FILL:
                x, y = p.get("x"), p.get("y")
                if x is None or y is None:
                    errors.append(f"{line_ref}: FILL missing coordinates")
                    has_error = True
                elif not (0 <= x < canvas_width and 0 <= y < canvas_height):
                    warnings.append(f"{line_ref}: FILL origin ({x}, {y}) out of canvas bounds")

            elif action.action_type == ActionType.ERASE:
                x, y = p.get("x"), p.get("y")
                if x is None or y is None:
                    errors.append(f"{line_ref}: ERASE missing coordinates")
                    has_error = True

            elif action.action_type == ActionType.REUSE:
                target_part = p.get("part_name")
                if not target_part:
                    errors.append(f"{line_ref}: REUSE missing part_name")
                    has_error = True
                elif target_part not in defined_parts:
                    warnings.append(f"{line_ref}: REUSE target part '{target_part}' not yet defined")

            elif action.action_type == ActionType.TRANSFORM:
                target_part = p.get("part_name")
                if not target_part:
                    errors.append(f"{line_ref}: TRANSFORM missing part_name")
                    has_error = True
                elif target_part not in defined_parts:
                    warnings.append(f"{line_ref}: TRANSFORM target part '{target_part}' not yet defined")

            elif action.action_type in (ActionType.OUTLINE, ActionType.SHADE, ActionType.AURA):
                x, y, w, h = p.get("x"), p.get("y"), p.get("w"), p.get("h")
                if any(v is None for v in (x, y, w, h)):
                    errors.append(f"{line_ref}: {action.action_type.value} missing region params")
                    has_error = True
                elif w <= 0 or h <= 0:
                    errors.append(f"{line_ref}: {action.action_type.value} width ({w}) and height ({h}) must be > 0")
                    has_error = True
                elif x >= canvas_width or y >= canvas_height or (x + w) <= 0 or (y + h) <= 0:
                    warnings.append(f"{line_ref}: {action.action_type.value} completely out of canvas bounds")
                if p.get("color") is None:
                    errors.append(f"{line_ref}: {action.action_type.value} missing color")
                    has_error = True
                if action.action_type == ActionType.SHADE and p.get("direction") not in (0, 1, 2, 3):
                    errors.append(f"{line_ref}: SHADE direction must be 0..3")
                    has_error = True
                if action.action_type == ActionType.AURA and p.get("density", 2) not in (1, 2, 3, 4):
                    errors.append(f"{line_ref}: AURA density must be 1..4")
                    has_error = True

            if has_error:
                invalid_count += 1

        rate = (invalid_count / total) if total > 0 else 0.0
        return ValidationReport(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            total_actions=total,
            invalid_action_count=invalid_count,
            invalid_action_rate=round(rate, 4),
        )
