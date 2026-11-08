"""Asset Composition Engine for SpriteMind Stage 5.

Combines characters, weapons, armor, accessories, and visual effects into
game-ready PixelScript assets and synchronized multi-frame animations.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np

from spritemind.assets.library import AssetLibrary, AssetDefinition, CHARACTER_SOCKETS, WEAPON_SOCKETS
from spritemind.ml.composition_model import SocketPredictorManager
from spritemind.renderer.engine import PixelScriptEngine, RenderResult
from spritemind.renderer.export import array_to_base64
from spritemind.ml.temporal import TemporalAnimator
from spritemind.pixelscript.scaling import DESIGN_SIZE, scale_script



class AssetComposer:
    """Orchestrates asset composition, socket alignment, and animated layer synchronization."""

    @classmethod
    def compose(
        cls,
        character_asset_or_script: str = "character_player_soldier",
        weapon_id: Optional[str] = "weapon_pump_shotgun",
        helmet_id: Optional[str] = None,
        backpack_id: Optional[str] = None,
        fx_id: Optional[str] = None,
        engine_mode: str = "trained_model",  # "trained_model" | "untrained_model" | "procedural_baseline" | "hybrid"
        palette_name: str = "pico8",
        width: int = 64,
        height: int = 64,
    ) -> Dict[str, Any]:
        """Composes a static character sprite with equipped items and optional effects."""
        # 1. Resolve base character
        base_asset = AssetLibrary.get(character_asset_or_script)
        if base_asset:
            char_script = base_asset.script_text
            char_name = base_asset.name
            char_sockets = dict(base_asset.sockets)
            char_hitboxes = dict(base_asset.hitboxes)
        else:
            char_script = character_asset_or_script
            char_name = "Custom Character"
            char_sockets = {}
            char_hitboxes = {}

        scale = max(1, min(width, height) // DESIGN_SIZE)

        # 2. Render base character to extract visual features for socket prediction
        base_render = PixelScriptEngine.execute(
            char_script, width=DESIGN_SIZE, height=DESIGN_SIZE, palette_name=palette_name
        )
        base_rgba = base_render.composite_array

        # 3. Determine attachment sockets
        if engine_mode in ("trained_model", "hybrid", "untrained_model"):
            predicted_sockets = SocketPredictorManager.predict_sockets(
                base_rgba,
                engine_mode="untrained_model" if engine_mode == "untrained_model" else "trained_model",
            )
            # Merge or override with predicted sockets
            for k, v in predicted_sockets.items():
                char_sockets[k] = v
        else:
            # Procedural baseline socket defaults
            if not char_sockets:
                char_sockets = SocketPredictorManager.predict_sockets(base_rgba, engine_mode="procedural_baseline")

        # 4. Assemble composite PixelScript lines with layer ordering
        composite_lines: List[str] = []
        layers_used: List[str] = []
        composed_sockets: Dict[str, Tuple[int, int]] = dict(char_sockets)
        composed_hitboxes: Dict[str, List[int]] = dict(char_hitboxes)

        # Back Equipment (Layer -1)
        if backpack_id:
            bp_asset = AssetLibrary.get(backpack_id)
            if bp_asset:
                target_socket = char_sockets.get("back", (16, 16))
                origin_socket = bp_asset.sockets.get("back", bp_asset.pivot)
                dx = target_socket[0] - origin_socket[0]
                dy = target_socket[1] - origin_socket[1]
                bp_lines = cls._translate_script_primitives(bp_asset.script_text, dx, dy, layer_id=1, layer_name="back_equip")
                composite_lines.extend(bp_lines)
                layers_used.append(f"backpack:{bp_asset.name}")

        # Character Body (Layer 0)
        body_lines = [ln.strip() for ln in char_script.splitlines() if ln.strip() and not ln.strip().startswith("END PROGRAM")]
        composite_lines.extend(body_lines)
        layers_used.append(f"body:{char_name}")

        # Helmet (Layer 2)
        if helmet_id:
            helm_asset = AssetLibrary.get(helmet_id)
            if helm_asset:
                target_socket = char_sockets.get("head", (16, 8))
                origin_socket = helm_asset.sockets.get("head", helm_asset.pivot)
                dx = target_socket[0] - origin_socket[0]
                dy = target_socket[1] - origin_socket[1]
                helm_lines = cls._translate_script_primitives(helm_asset.script_text, dx, dy, layer_id=2, layer_name="helmet")
                composite_lines.extend(helm_lines)
                layers_used.append(f"helmet:{helm_asset.name}")

        # Held Weapon (Layer 3)
        muzzle_world_pos = None
        if weapon_id:
            wep_asset = AssetLibrary.get(weapon_id)
            if wep_asset:
                target_socket = char_sockets.get("hand_r", (21, 20))
                origin_socket = wep_asset.sockets.get("grip", wep_asset.pivot)
                dx = target_socket[0] - origin_socket[0]
                dy = target_socket[1] - origin_socket[1]
                wep_lines = cls._translate_script_primitives(wep_asset.script_text, dx, dy, layer_id=3, layer_name="weapon")
                composite_lines.extend(wep_lines)
                layers_used.append(f"weapon:{wep_asset.name}")

                # Update muzzle socket in world coordinates
                if "muzzle" in wep_asset.sockets:
                    muzzle_orig = wep_asset.sockets["muzzle"]
                    muzzle_world_pos = (muzzle_orig[0] + dx, muzzle_orig[1] + dy)
                    composed_sockets["muzzle"] = muzzle_world_pos

                # Merge weapon hitboxes
                for hb_name, bounds in wep_asset.hitboxes.items():
                    composed_hitboxes[f"weapon_{hb_name}"] = [bounds[0] + dx, bounds[1] + dy, bounds[2], bounds[3]]

        # Active FX / Muzzle Flash (Layer 4)
        if fx_id:
            fx_asset = AssetLibrary.get(fx_id)
            if fx_asset:
                target_socket = muzzle_world_pos or char_sockets.get("hand_r", (21, 20))
                origin_socket = fx_asset.sockets.get("muzzle", fx_asset.pivot)
                dx = target_socket[0] - origin_socket[0]
                dy = target_socket[1] - origin_socket[1]
                fx_lines = cls._translate_script_primitives(fx_asset.script_text, dx, dy, layer_id=4, layer_name="fx")
                composite_lines.extend(fx_lines)
                layers_used.append(f"fx:{fx_asset.name}")

        composite_lines.append("END PROGRAM")
        design_script = "\n".join(composite_lines)
        final_script = scale_script(design_script, scale)

        # 5. Render at design size, then enlarge the pixels. Rendering the scaled
        # script instead would thin every one-pixel stroke and open seams.
        rendered = PixelScriptEngine.execute(
            design_script, width=DESIGN_SIZE, height=DESIGN_SIZE, palette_name=palette_name
        )
        r_dict = rendered.to_dict(scale=scale, shade=True, clean=True)

        return {
            "status": "success",
            "name": f"Composed {char_name}",
            "engine_used": engine_mode,
            "script_text": final_script,
            "image_base64": r_dict["image_base64"],
            "pixel_count": r_dict["pixel_count"],
            "sockets": {k: (v[0] * scale, v[1] * scale) for k, v in composed_sockets.items()},
            "hitboxes": {
                k: [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for k, b in composed_hitboxes.items()
            },
            "layers_used": layers_used,
            "width": width,
            "height": height,
        }

    @classmethod
    def compose_animation(
        cls,
        character_asset_or_script: str = "character_zombie_walker",
        animation_category: str = "walk",
        num_frames: int = 6,
        fps: int = 8,
        weapon_id: Optional[str] = "weapon_pump_shotgun",
        helmet_id: Optional[str] = None,
        backpack_id: Optional[str] = None,
        fx_id: Optional[str] = None,
        engine_mode: str = "trained_model",
        palette_name: str = "pico8",
        width: int = 64,
        height: int = 64,
    ) -> Dict[str, Any]:
        """Synthesizes a multi-frame animation with dynamically synchronized equipment sockets."""
        base_asset = AssetLibrary.get(character_asset_or_script)
        char_script = base_asset.script_text if base_asset else character_asset_or_script

        scale = max(1, min(width, height) // DESIGN_SIZE)

        # 1. Generate base character animation frames
        char_anim = TemporalAnimator.generate_animation(
            base_script_or_prompt=char_script,
            anim_category=animation_category,
            num_frames=num_frames,
            fps=fps,
            engine_mode=engine_mode,
            palette_name=palette_name,
            width=DESIGN_SIZE,
            height=DESIGN_SIZE,
        )
        base_frames = char_anim.get("frames", [])

        composed_frames = []
        for f_idx, frame_data in enumerate(base_frames):
            frame_script = frame_data.get("script_text", "")

            # Extract dynamic limb transformations from frame script
            arm_r_offset = cls._extract_part_transform(frame_script, "arm_r") or cls._extract_part_transform(frame_script, "arm_right") or (0, 0)
            head_offset = cls._extract_part_transform(frame_script, "head") or (0, 0)
            torso_offset = cls._extract_part_transform(frame_script, "torso") or (0, 0)

            # Compose frame with synchronized dynamic offsets
            comp_res = cls._compose_frame_with_offsets(
                char_frame_script=frame_script,
                arm_r_offset=arm_r_offset,
                head_offset=head_offset,
                torso_offset=torso_offset,
                weapon_id=weapon_id,
                helmet_id=helmet_id,
                backpack_id=backpack_id,
                fx_id=fx_id if f_idx in (1, 2) else None,  # Muzzle flash during active strike/fire frames
                palette_name=palette_name,
                width=DESIGN_SIZE,
                height=DESIGN_SIZE,
            )
            if scale > 1:
                design_script = comp_res["script_text"]
                frame_render = PixelScriptEngine.execute(
                    design_script, width=DESIGN_SIZE, height=DESIGN_SIZE, palette_name=palette_name
                ).to_dict(scale=scale, shade=True, clean=True)
                sockets = {k: (v[0] * scale, v[1] * scale) for k, v in (comp_res.get("sockets") or {}).items()}
                hitboxes = {
                    k: [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
                    for k, b in (comp_res.get("hitboxes") or {}).items()
                }
                frame_render.update(
                    {
                        "script_text": scale_script(design_script, scale),
                        "sockets": sockets,
                        "hitboxes": hitboxes,
                        "name": comp_res.get("name"),
                        "engine_used": comp_res.get("engine_used", engine_mode),
                    }
                )
                frame_render["width"], frame_render["height"] = width, height
                comp_res = frame_render
            comp_res["frame_index"] = f_idx
            composed_frames.append(comp_res)

        return {
            "status": "success",
            "name": f"Animated {character_asset_or_script} ({animation_category})",
            "animation_category": animation_category,
            "engine_used": char_anim.get("engine_used", engine_mode),
            "num_frames": len(composed_frames),
            "fps": fps,
            "frames": composed_frames,
            "identity_confidence": char_anim.get("identity_confidence"),
        }

    @classmethod
    def _compose_frame_with_offsets(
        cls,
        char_frame_script: str,
        arm_r_offset: Tuple[int, int],
        head_offset: Tuple[int, int],
        torso_offset: Tuple[int, int],
        weapon_id: Optional[str],
        helmet_id: Optional[str],
        backpack_id: Optional[str],
        fx_id: Optional[str],
        palette_name: str,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """Composes a single animation frame taking dynamic part transformations into account."""
        base_render = PixelScriptEngine.execute(char_frame_script, width=width, height=height, palette_name=palette_name)
        base_sockets = SocketPredictorManager.predict_sockets(base_render.composite_array)

        # Dynamic kinematic socket displacement
        dyn_hand_r = (base_sockets["hand_r"][0] + arm_r_offset[0], base_sockets["hand_r"][1] + arm_r_offset[1])
        dyn_head = (base_sockets["head"][0] + head_offset[0], base_sockets["head"][1] + head_offset[1])
        dyn_back = (base_sockets["back"][0] + torso_offset[0], base_sockets["back"][1] + torso_offset[1])

        composite_lines: List[str] = []

        # Back Equipment
        if backpack_id:
            bp = AssetLibrary.get(backpack_id)
            if bp:
                orig = bp.sockets.get("back", bp.pivot)
                dx = dyn_back[0] - orig[0]
                dy = dyn_back[1] - orig[1]
                composite_lines.extend(cls._translate_script_primitives(bp.script_text, dx, dy, layer_id=1, layer_name="back_equip"))

        # Base Frame
        body_lines = [ln.strip() for ln in char_frame_script.splitlines() if ln.strip() and not ln.strip().startswith("END PROGRAM")]
        composite_lines.extend(body_lines)

        # Helmet
        if helmet_id:
            helm = AssetLibrary.get(helmet_id)
            if helm:
                orig = helm.sockets.get("head", helm.pivot)
                dx = dyn_head[0] - orig[0]
                dy = dyn_head[1] - orig[1]
                composite_lines.extend(cls._translate_script_primitives(helm.script_text, dx, dy, layer_id=2, layer_name="helmet"))

        # Weapon
        muzzle_pos = None
        if weapon_id:
            wep = AssetLibrary.get(weapon_id)
            if wep:
                orig = wep.sockets.get("grip", wep.pivot)
                dx = dyn_hand_r[0] - orig[0]
                dy = dyn_hand_r[1] - orig[1]
                composite_lines.extend(cls._translate_script_primitives(wep.script_text, dx, dy, layer_id=3, layer_name="weapon"))
                if "muzzle" in wep.sockets:
                    muzzle_pos = (wep.sockets["muzzle"][0] + dx, wep.sockets["muzzle"][1] + dy)

        # FX
        if fx_id and muzzle_pos:
            fx = AssetLibrary.get(fx_id)
            if fx:
                orig = fx.sockets.get("muzzle", fx.pivot)
                dx = muzzle_pos[0] - orig[0]
                dy = muzzle_pos[1] - orig[1]
                composite_lines.extend(cls._translate_script_primitives(fx.script_text, dx, dy, layer_id=4, layer_name="fx"))

        composite_lines.append("END PROGRAM")
        final_script = "\n".join(composite_lines)
        rendered = PixelScriptEngine.execute(final_script, width=width, height=height, palette_name=palette_name)
        r_dict = rendered.to_dict()
        r_dict["script_text"] = final_script
        return r_dict

    @classmethod
    def _translate_script_primitives(
        cls,
        script_text: str,
        dx: int,
        dy: int,
        layer_id: int = 1,
        layer_name: str = "equipment",
    ) -> List[str]:
        """Parses script lines, translates 2D coordinate primitives by (dx, dy), and injects layer."""
        out = [f"LAYER {layer_id} {layer_name} 1.0"]
        for line in script_text.splitlines():
            line = line.strip()
            if not line or line.startswith("LAYER") or line.startswith("END PROGRAM"):
                continue

            parts = line.split()
            cmd = parts[0].upper()

            try:
                if cmd == "PIXEL" and len(parts) >= 4:
                    x, y, c = int(parts[1]) + dx, int(parts[2]) + dy, parts[3]
                    out.append(f"PIXEL {x} {y} {c}")
                elif cmd == "RECT" and len(parts) >= 7:
                    x, y, w, h, c, filled = int(parts[1]) + dx, int(parts[2]) + dy, parts[3], parts[4], parts[5], parts[6]
                    out.append(f"RECT {x} {y} {w} {h} {c} {filled}")
                elif cmd == "LINE" and len(parts) >= 6:
                    x0, y0, x1, y1, c = int(parts[1]) + dx, int(parts[2]) + dy, int(parts[3]) + dx, int(parts[4]) + dy, parts[5]
                    out.append(f"LINE {x0} {y0} {x1} {y1} {c}")
                elif cmd == "CIRCLE" and len(parts) >= 6:
                    cx, cy, r, c, filled = int(parts[1]) + dx, int(parts[2]) + dy, parts[3], parts[4], parts[5]
                    out.append(f"CIRCLE {cx} {cy} {r} {c} {filled}")
                elif cmd == "PART" and len(parts) >= 5:
                    pname, parent, ax, ay = parts[1], parts[2], int(parts[3]) + dx, int(parts[4]) + dy
                    out.append(f"PART {pname} {parent} {ax} {ay}")
                elif cmd in ("END", "PALETTE"):
                    out.append(line)
                else:
                    out.append(line)
            except Exception:
                out.append(line)

        return out

    @staticmethod
    def _extract_part_transform(script_text: str, part_name: str) -> Optional[Tuple[int, int]]:
        """Extracts (dx, dy) from TRANSFORM part_name dx dy rot if present."""
        pattern = re.compile(rf"TRANSFORM\s+{re.escape(part_name)}\s+(-?\d+)\s+(-?\d+)")
        for line in script_text.splitlines():
            m = pattern.search(line.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
        return None
