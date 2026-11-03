"""Synthetic Defect Generator for Visual Critic training and evaluation."""
import random
import copy
from typing import List, Dict, Any, Tuple
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram


DEFECT_TYPES = [
    "stray_pixel",
    "disconnected_part",
    "palette_clash",
    "silhouette_hole",
    "missing_feature",
]


class DefectGenerator:
    """Injects controlled defects into PixelScript programs to train the Visual Critic."""

    @classmethod
    def inject_defect(
        cls,
        program: PixelScriptProgram,
        defect_type: str = None,
        canvas_width: int = 32,
        canvas_height: int = 32,
        seed: int = None,
    ) -> Tuple[PixelScriptProgram, Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        if defect_type is None:
            defect_type = random.choice(DEFECT_TYPES)

        corrupted_actions = copy.deepcopy(program.actions)
        defect_info: Dict[str, Any] = {
            "defect_type": defect_type,
            "description": "",
            "bbox": None,
            "suggested_fix": None,
        }

        if defect_type == "stray_pixel":
            # Add 1-2 random pixels floating far from character center (e.g. at canvas margins)
            sx = random.choice([random.randint(1, 4), random.randint(canvas_width - 5, canvas_width - 2)])
            sy = random.choice([random.randint(1, 4), random.randint(canvas_height - 5, canvas_height - 2)])
            corrupted_actions.append(PixelAction(ActionType.PIXEL, {"x": sx, "y": sy, "color": 9}))
            defect_info["description"] = f"Stray floating pixel detected at ({sx}, {sy})"
            defect_info["bbox"] = {"min_x": sx, "min_y": sy, "max_x": sx, "max_y": sy}
            defect_info["suggested_fix"] = f"ERASE {sx} {sy} 1 1"

        elif defect_type == "disconnected_part":
            # Move arm_r or leg_r away by altering coordinates in rect or adding transform
            found = False
            for act in corrupted_actions:
                if act.action_type == ActionType.PART and ("arm" in str(act.params.get("name")) or "leg" in str(act.params.get("name"))):
                    pname = act.params["name"]
                    corrupted_actions.append(PixelAction(ActionType.TRANSFORM, {"part_name": pname, "dx": 5, "dy": -3, "rot_deg": 0}))
                    defect_info["description"] = f"Disconnected limb '{pname}' separated from torso"
                    defect_info["bbox"] = {"min_x": 20, "min_y": 10, "max_x": 28, "max_y": 20}
                    defect_info["suggested_fix"] = f"TRANSFORM {pname} -5 3 0"
                    found = True
                    break
            if not found:
                # Fallback to stray pixel
                return cls.inject_defect(program, "stray_pixel", canvas_width, canvas_height)

        elif defect_type == "palette_clash":
            # Change color of an interior RECT to an eye-jarring unharmonized color
            for act in corrupted_actions:
                if act.action_type == ActionType.RECT:
                    orig_color = act.params.get("color")
                    # Replace with bright neon color string or index 15
                    act.params["color"] = 15 if orig_color != 15 else 11
                    defect_info["description"] = f"Palette clash: dissonant color {act.params['color']} replaces harmonious tone"
                    defect_info["bbox"] = {"min_x": act.params.get("x", 0), "min_y": act.params.get("y", 0), "max_x": act.params.get("x", 0) + act.params.get("w", 1), "max_y": act.params.get("y", 0) + act.params.get("h", 1)}
                    defect_info["suggested_fix"] = f"RECT {act.params['x']} {act.params['y']} {act.params['w']} {act.params['h']} {orig_color} 1"
                    break

        elif defect_type == "silhouette_hole":
            # Punch an erase hole into center of torso/canvas
            hx, hy = canvas_width // 2, canvas_height // 2
            corrupted_actions.append(PixelAction(ActionType.ERASE, {"x": hx - 1, "y": hy - 1, "w": 3, "h": 3}))
            defect_info["description"] = f"Unintended silhouette void / hole in body at ({hx-1}, {hy-1})"
            defect_info["bbox"] = {"min_x": hx - 1, "min_y": hy - 1, "max_x": hx + 1, "max_y": hy + 1}
            defect_info["suggested_fix"] = f"RECT {hx - 1} {hy - 1} 3 3 2 1"

        elif defect_type == "missing_feature":
            # Filter out eyes PART or eye pixels
            filtered = []
            removed = False
            for act in corrupted_actions:
                if act.action_type == ActionType.PART and act.params.get("name") == "eyes":
                    removed = True
                    continue
                if removed and act.action_type == ActionType.END and act.params.get("target") == "PART":
                    removed = False
                    continue
                if not removed:
                    filtered.append(act)
            corrupted_actions = filtered
            defect_info["description"] = "Facial feature absence: missing eye contrast pixels"
            defect_info["bbox"] = {"min_x": 14, "min_y": 10, "max_x": 18, "max_y": 12}
            defect_info["suggested_fix"] = "PART eyes head 14 10\nPIXEL 14 11 9\nPIXEL 17 11 9\nEND PART"

        new_prog = PixelScriptProgram(actions=corrupted_actions, metadata=dict(program.metadata))
        new_prog.metadata["has_defect"] = True
        new_prog.metadata["defect_type"] = defect_type
        return new_prog, defect_info
