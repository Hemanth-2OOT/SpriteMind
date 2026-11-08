"""Procedural & Synthetic Pixel Art Dataset Generator.

Generates structured character anatomy programs in PixelScript across 10 diverse categories:
Zombies, Knights, Rogues, Wizards, Robots, Slimes, Skeletons, Goblins, Animals, and Fantasy Creatures.
Supports multiple proportions (standard, stocky, slender, chibi, quadruped), clothing, weapons,
accessories, facial features, and animation poses.
"""
import random
from typing import Dict, Any, List, Optional, Tuple
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram
from spritemind.utils.palettes import PALETTES

CATEGORIES = [
    "knight", "zombie", "rogue", "wizard", "robot",
    "slime", "skeleton", "goblin", "animal", "fantasy_creature"
]
STYLES = ["retro_8bit", "classic_16bit", "flat", "dark_fantasy", "cyberpunk"]
POSES = ["idle", "walk", "run", "attack", "death"]
PERSPECTIVES = ["front", "side", "three_quarter", "top_down"]
PROPORTIONS = ["standard", "stocky", "slender", "chibi"]
WEAPONS = ["sword", "staff", "axe", "dagger", "shield", "bow", "none"]
ACCESSORIES = ["helmet", "cape", "horns", "wings", "backpack", "pauldrons", "none"]


def _clip_ops(ops, y_lo: int, y_hi: int):
    """Restricts drawing ops to the rows ``[y_lo, y_hi)``.

    Used to split a limb drawn as one rectangle into an upper and a lower
    segment without changing a single output pixel.
    """
    clipped = []
    for kind, params in ops:
        if kind == ActionType.RECT:
            top = max(params["y"], y_lo)
            bottom = min(params["y"] + params["h"], y_hi)
            if bottom <= top:
                continue
            clipped.append((kind, {**params, "y": top, "h": bottom - top}))
        elif kind == ActionType.PIXEL:
            if y_lo <= params["y"] < y_hi:
                clipped.append((kind, dict(params)))
        elif kind == ActionType.LINE:
            y0, y1 = params["y0"], params["y1"]
            if y0 == y1:
                if y_lo <= y0 < y_hi:
                    clipped.append((kind, dict(params)))
                continue
            top = max(min(y0, y1), y_lo)
            bottom = min(max(y0, y1) + 1, y_hi)
            if bottom <= top:
                continue
            clipped.append((kind, {**params, "y0": top, "y1": bottom - 1}))
    return clipped


class CharacterGenerator:
    """Generates complete structured PixelScript programs for characters, animals, and fantasy creatures."""

    @classmethod
    def _emit_two_segment_limb(
        cls,
        actions: List[PixelAction],
        upper_name: str,
        lower_name: str,
        parent: str,
        x: int,
        y: int,
        split: int,
        height: int,
        ops: List[Tuple[ActionType, Dict[str, Any]]],
    ):
        """Emits one limb as an upper and a lower part sharing the same pixel band.

        Splitting the rectangle at ``split`` is pixel-neutral, but it gives the
        elbow/knee a real pivot: the lower part is anchored on the split row and
        parented to the upper part, so rig.py can rotate it.
        """
        for name, band_top, band_bottom, band_parent in (
            (upper_name, y, y + split, parent),
            (lower_name, y + split, y + height, upper_name),
        ):
            actions.append(PixelAction(ActionType.PART, {
                "name": name, "parent": band_parent, "anchor_x": x, "anchor_y": band_top,
            }))
            for kind, params in _clip_ops(ops, band_top, band_bottom):
                actions.append(PixelAction(kind, params))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

    @classmethod
    def _finishing_pass(
        cls,
        actions: List[PixelAction],
        c_outline: int,
        width: int,
        height: int,
        element: Optional[str] = None,
        palette_name: str = "pico8",
        seed: Optional[int] = None,
    ):
        """Returns to the base layer, then adds a rim shadow, a 1px silhouette outline and,
        for elemental characters, a dithered aura in the element's hue."""
        actions.append(PixelAction(ActionType.LAYER, {"id": 0, "name": "base", "opacity": 1.0}))
        actions.append(PixelAction(ActionType.SHADE, {
            "x": 0, "y": 0, "w": width, "h": height, "color": c_outline, "direction": 1,
        }))
        actions.append(PixelAction(ActionType.OUTLINE, {
            "x": 0, "y": 0, "w": width, "h": height, "color": c_outline,
        }))
        if element:
            from spritemind.dataset.conditioning import ELEMENT_RGB, ramp

            tint = ELEMENT_RGB.get(element)
            if tint is not None:
                _base, highlight, _shadow = ramp(palette_name, tint)
                actions.append(PixelAction(ActionType.AURA, {
                    "x": 0, "y": 0, "w": width, "h": height,
                    "color": highlight,
                    "density": 2 if (seed or 0) % 2 == 0 else 3,
                }))

    @classmethod
    def generate_character(
        cls,
        category: str = "knight",
        style: str = "classic_16bit",
        pose: str = "idle",
        palette_name: str = "pico8",
        width: int = 32,
        height: int = 32,
        frame_idx: int = 0,
        total_frames: int = 1,
        seed: Optional[int] = None,
        proportion: Optional[str] = None,
        weapon: Optional[str] = None,
        accessory: Optional[str] = None,
        colour: Optional[str] = None,
        element: Optional[str] = None,
    ) -> PixelScriptProgram:
        if seed is not None:
            rng = random.Random(seed + frame_idx * 1000)
        else:
            rng = random.Random()

        # Category normalization
        if category not in CATEGORIES:
            category = "knight"

        prop = proportion or rng.choice(PROPORTIONS)
        wep = weapon or rng.choice(WEAPONS)
        acc = accessory or rng.choice(ACCESSORIES)

        actions: List[PixelAction] = []
        meta = {
            "category": category,
            "style": style,
            "pose": pose,
            "palette": palette_name,
            "width": width,
            "height": height,
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "proportion": prop,
            "weapon": wep,
            "accessory": acc,
            "colour": colour,
            "element": element,
            "seed": seed or 0,
        }

        # Color indices selection based on category & seed variation
        c_outline = 1  # black
        if category == "zombie":
            c_skin = rng.choice([4, 12, 11])      # green / olive / sick yellow
            c_skin_hi = 12 if c_skin == 4 else 11
            c_cloth = rng.choice([5, 6, 2])       # brown / dark grey / navy torn shirt
            c_pants = rng.choice([6, 1, 5])
            c_eyes = rng.choice([9, 10, 8])       # red / orange / white
        elif category == "knight":
            c_skin = rng.choice([16, 15, 6])      # peach / light / tanned
            c_skin_hi = 8
            c_cloth = rng.choice([7, 6, 8, 10])   # steel / dark metal / gold trim
            c_pants = rng.choice([2, 1, 6])       # navy / black / grey
            c_eyes = rng.choice([1, 2, 9])
        elif category == "rogue":
            c_skin = 16
            c_skin_hi = 15
            c_cloth = rng.choice([3, 1, 2])       # dark purple / black / dark blue cloak
            c_pants = 1
            c_eyes = rng.choice([11, 10, 12])     # yellow / amber glowing
        elif category == "wizard":
            c_skin = 16
            c_skin_hi = 8
            c_cloth = rng.choice([13, 3, 9])      # blue / violet / red robes
            c_pants = 2
            c_eyes = rng.choice([14, 12, 11])     # arcane glow
        elif category == "robot":
            c_skin = rng.choice([7, 6, 13])       # metallic grey / dark steel / cobalt
            c_skin_hi = 8
            c_cloth = rng.choice([6, 2, 1])       # dark chassis
            c_pants = 2
            c_eyes = rng.choice([12, 13, 9])      # neon cyan / green / red sensor
        elif category == "slime":
            c_skin = rng.choice([12, 13, 9, 14])  # lime / blue / red / indigo
            c_skin_hi = 8 if c_skin != 8 else 11
            c_cloth = c_skin
            c_pants = c_skin
            c_eyes = 1
        elif category == "skeleton":
            c_skin = 8                            # bone white
            c_skin_hi = 7
            c_cloth = rng.choice([6, 1, 5, 3])    # tattered rags
            c_pants = 1
            c_eyes = rng.choice([9, 12, 14])      # hollow red / cyan fire
        elif category == "goblin":
            c_skin = rng.choice([12, 4])
            c_skin_hi = 11
            c_cloth = rng.choice([5, 10])         # leather
            c_pants = 5
            c_eyes = 10
        elif category == "animal": # wolf, beast, bear, cat
            c_skin = rng.choice([6, 7, 5, 2, 8])  # grey wolf / brown bear / shadow panther / arctic
            c_skin_hi = 8 if c_skin != 8 else 7
            c_cloth = c_skin
            c_pants = c_skin
            c_eyes = rng.choice([11, 9, 12])      # amber / feral red / blue
        else: # fantasy_creature: dragon / golem / demon
            c_skin = rng.choice([9, 14, 2, 4, 10])# crimson dragon / void / forest drake
            c_skin_hi = 10 if c_skin == 9 else 8
            c_cloth = c_skin
            c_pants = c_skin
            c_eyes = rng.choice([11, 9, 8])

        cx = width // 2
        cy = height // 2

        # Conditioned colour override: prompt words win over the category default ramp.
        if colour or element:
            from spritemind.dataset.conditioning import COLOUR_RGB, ELEMENT_RGB, ramp

            tint = COLOUR_RGB.get(colour or "") or ELEMENT_RGB.get(element or "")
            if tint is not None:
                base, hi, shadow = ramp(palette_name, tint)
                c_skin, c_skin_hi, c_cloth, c_pants = base, hi, base, shadow
            if element and element in ELEMENT_RGB:
                e_base, e_hi, _ = ramp(palette_name, ELEMENT_RGB[element])
                c_eyes = e_hi
                if not colour:
                    c_skin, c_skin_hi = e_base, e_hi
                    c_cloth = c_pants = e_base

        # Pose & animation offsets
        bob_y = 0
        leg_l_dx, leg_r_dx = 0, 0
        arm_l_dy, arm_r_dy = 0, 0

        if pose == "idle":
            bob_y = 1 if (frame_idx % 2 == 1) else 0
        elif pose == "walk":
            step = frame_idx % 4
            if step == 0:
                leg_l_dx, leg_r_dx = 1, -1
                arm_l_dy, arm_r_dy = -1, 1
            elif step == 1:
                bob_y = -1
            elif step == 2:
                leg_l_dx, leg_r_dx = -1, 1
                arm_l_dy, arm_r_dy = 1, -1
        elif pose == "run":
            step = frame_idx % 4
            bob_y = -1 if step % 2 == 0 else 1
            leg_l_dx = 2 if step < 2 else -2
            leg_r_dx = -2 if step < 2 else 2
            arm_l_dy = -2 if step < 2 else 2
            arm_r_dy = 2 if step < 2 else -2
        elif pose == "attack":
            if frame_idx == 0:
                arm_r_dy = -3
            elif frame_idx == 1:
                arm_r_dy = 3
                bob_y = 1
        elif pose == "death":
            bob_y = min(8, frame_idx * 3)

        # Base layer definition
        actions.append(PixelAction(ActionType.LAYER, {"id": 0, "name": "base", "opacity": 1.0}))

        # -------------------------------------------------------------
        # 1. SPECIAL CASE: QUADRUPED ANIMAL (wolf, bear, beast)
        # -------------------------------------------------------------
        if category == "animal":
            # Torso: Horizontal body 12x6
            body_w, body_h = 12, 6
            bx = cx - (body_w // 2)
            by = cy + bob_y

            actions.append(PixelAction(ActionType.PART, {"name": "torso", "parent": None, "anchor_x": cx, "anchor_y": by}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx, "y": by, "w": body_w, "h": body_h, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.LINE, {"x0": bx + 1, "y0": by, "x1": bx + body_w - 2, "y1": by, "color": c_skin_hi}))
            # Detail pass: spine ridge and fur shadow
            actions.append(PixelAction(ActionType.LINE, {"x0": bx + 2, "y0": by + 2, "x1": bx + body_w - 3, "y1": by + 2, "color": c_skin_hi}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx + 1, "y": by + body_h - 1, "w": body_w - 2, "h": 1, "color": c_cloth, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # Head with snout & ears (front-facing right)
            hx, hy = bx + body_w - 2, by - 4
            actions.append(PixelAction(ActionType.PART, {"name": "head", "parent": "torso", "anchor_x": hx, "anchor_y": hy}))
            actions.append(PixelAction(ActionType.RECT, {"x": hx, "y": hy, "w": 6, "h": 6, "color": c_skin, "filled": True}))
            # Ears
            actions.append(PixelAction(ActionType.PIXEL, {"x": hx + 1, "y": hy - 1, "color": c_skin_hi}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": hx + 4, "y": hy - 1, "color": c_skin_hi}))
            # Snout
            actions.append(PixelAction(ActionType.RECT, {"x": hx + 4, "y": hy + 3, "w": 3, "h": 2, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # Eyes
            actions.append(PixelAction(ActionType.PART, {"name": "eyes", "parent": "head", "anchor_x": hx + 2, "anchor_y": hy + 2}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": hx + 2, "y": hy + 2, "color": c_eyes}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # Tail
            actions.append(PixelAction(ActionType.PART, {"name": "tail", "parent": "torso", "anchor_x": bx, "anchor_y": by}))
            # Stepped columns keep the tail in one piece; a 45 degree LINE breaks
            # into dots that only touch at the corners.
            for step in range(3):
                actions.append(PixelAction(ActionType.RECT, {"x": bx - 1 - step, "y": by - step, "w": 1, "h": 2, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": bx - 3, "y": by - 3, "color": c_skin_hi}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # 4 Legs: Front-left, Front-right, Hind-left, Hind-right
            leg_h = 5
            actions.append(PixelAction(ActionType.PART, {"name": "leg_left", "parent": "torso", "anchor_x": bx + 1, "anchor_y": by + body_h}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx + 1 + leg_l_dx, "y": by + body_h, "w": 2, "h": leg_h, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            actions.append(PixelAction(ActionType.PART, {"name": "leg_right", "parent": "torso", "anchor_x": bx + body_w - 3, "anchor_y": by + body_h}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx + body_w - 3 + leg_r_dx, "y": by + body_h, "w": 2, "h": leg_h, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            cls._finishing_pass(actions, c_outline, width, height, element, palette_name, seed)
            actions.append(PixelAction(ActionType.END, {"target": "PROGRAM"}))
            return PixelScriptProgram(actions=actions, metadata=meta)

        # -------------------------------------------------------------
        # 2. SPECIAL CASE: SLIME (blob)
        # -------------------------------------------------------------
        if category == "slime":
            r = 7 if prop == "stocky" else (4 if prop == "chibi" else 6)
            actions.append(PixelAction(ActionType.PART, {"name": "torso", "parent": None, "anchor_x": cx, "anchor_y": cy}))
            actions.append(PixelAction(ActionType.RECT, {"x": cx - r, "y": cy - r + 3 + bob_y, "w": r * 2, "h": r * 2 - 2, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.RECT, {"x": cx - r + 1, "y": cy - r + 1 + bob_y, "w": r * 2 - 2, "h": 2, "color": c_skin_hi, "filled": True}))
            # Detail pass: glossy highlight and drip
            actions.append(PixelAction(ActionType.PIXEL, {"x": cx - r + 2, "y": cy - r + 2 + bob_y, "color": 8}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": cx + r - 3, "y": cy + r - 3 + bob_y, "color": c_skin_hi}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # Eyes
            actions.append(PixelAction(ActionType.PART, {"name": "eyes", "parent": "torso", "anchor_x": cx, "anchor_y": cy}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": cx - 3, "y": cy + bob_y, "color": c_eyes}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": cx + 2, "y": cy + bob_y, "color": c_eyes}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            if wep in ("sword", "dagger"):
                actions.append(PixelAction(ActionType.PART, {"name": "weapon", "parent": "torso", "anchor_x": cx + r + 1, "anchor_y": cy}))
                actions.append(PixelAction(ActionType.LINE, {"x0": cx + r, "y0": cy + 2, "x1": cx + r, "y1": cy - 4, "color": 7}))
                actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            cls._finishing_pass(actions, c_outline, width, height, element, palette_name, seed)
            actions.append(PixelAction(ActionType.END, {"target": "PROGRAM"}))
            return PixelScriptProgram(actions=actions, metadata=meta)

        # -------------------------------------------------------------
        # 3. BIPED HUMAN/CREATURE ANATOMY (Knights, Zombies, Rogues, Wizards, Robots, Skeletons, Goblins, Fantasy Creatures)
        # -------------------------------------------------------------
        # Proportion tuning
        if prop == "stocky":
            torso_w, torso_h = 10, 6
            head_w, head_h = 7, 6
            leg_w, leg_h = 4, 5
            arm_w, arm_h = 3, 5
        elif prop == "slender":
            torso_w, torso_h = 6, 8
            head_w, head_h = 6, 6
            leg_w, leg_h = 2, 7
            arm_w, arm_h = 2, 7
        elif prop == "chibi":
            torso_w, torso_h = 6, 5
            head_w, head_h = 8, 7
            leg_w, leg_h = 2, 4
            arm_w, arm_h = 2, 4
        else: # standard
            torso_w, torso_h = 8, 7
            head_w, head_h = 6, 6
            leg_w, leg_h = 3, 6
            arm_w, arm_h = 2, 6

        tx = cx - (torso_w // 2)
        ty = cy - 2 + bob_y

        # Torso
        actions.append(PixelAction(ActionType.PART, {"name": "torso", "parent": None, "anchor_x": cx, "anchor_y": ty}))
        actions.append(PixelAction(ActionType.RECT, {"x": tx, "y": ty, "w": torso_w, "h": torso_h, "color": c_cloth, "filled": True}))
        # Belt line
        actions.append(PixelAction(ActionType.LINE, {"x0": tx, "y0": ty + torso_h - 1, "x1": tx + torso_w - 1, "y1": ty + torso_h - 1, "color": c_pants}))
        # Detail pass: shoulder trim, chest highlight, strap and belt buckle
        actions.append(PixelAction(ActionType.RECT, {"x": tx, "y": ty, "w": torso_w, "h": 1, "color": c_skin_hi, "filled": True}))
        actions.append(PixelAction(ActionType.LINE, {"x0": tx + 1, "y0": ty + 2, "x1": tx + 1, "y1": ty + torso_h - 2, "color": 8}))
        actions.append(PixelAction(ActionType.LINE, {"x0": tx + 2, "y0": ty + 1, "x1": tx + torso_w - 2, "y1": ty + torso_h - 2, "color": c_pants}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx, "y": ty + torso_h - 1, "color": 10}))
        # Neck and shoulder shelf live inside the torso part: if they were drawn
        # outside any part they would not move with the body and would tear a
        # hole between head and shoulders on every animated frame.
        neck_w = max(2, head_w // 3)
        actions.append(PixelAction(ActionType.RECT, {"x": cx - neck_w // 2, "y": ty - 1, "w": neck_w, "h": 1, "color": c_skin, "filled": True}))
        actions.append(PixelAction(ActionType.RECT, {"x": tx - 1, "y": ty, "w": torso_w + 2, "h": 1, "color": c_cloth, "filled": True}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": tx - 1, "y": ty, "color": c_skin_hi}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": tx + torso_w, "y": ty, "color": c_skin_hi}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Head
        hx = cx - (head_w // 2)
        hy = ty - head_h - 1
        actions.append(PixelAction(ActionType.PART, {"name": "head", "parent": "torso", "anchor_x": cx, "anchor_y": hy}))
        actions.append(PixelAction(ActionType.RECT, {"x": hx, "y": hy, "w": head_w, "h": head_h, "color": c_skin, "filled": True}))

        # Helmet / hair / hood
        if acc == "helmet" or category in ("knight", "robot"):
            actions.append(PixelAction(ActionType.RECT, {"x": hx, "y": hy, "w": head_w, "h": 2, "color": 7 if category != "robot" else 8, "filled": True}))
            actions.append(PixelAction(ActionType.LINE, {"x0": hx, "y0": hy + 1, "x1": hx, "y1": hy + head_h - 1, "color": 7}))
        elif category == "rogue": # dark hood
            actions.append(PixelAction(ActionType.RECT, {"x": hx - 1, "y": hy - 1, "w": head_w + 2, "h": 3, "color": c_cloth, "filled": True}))
        elif category == "wizard": # pointed wizard hat
            actions.append(PixelAction(ActionType.LINE, {"x0": hx - 1, "y0": hy, "x1": hx + head_w, "y1": hy, "color": c_cloth}))
            actions.append(PixelAction(ActionType.RECT, {"x": cx - 2, "y": hy - 3, "w": 4, "h": 3, "color": c_cloth, "filled": True}))
            actions.append(PixelAction(ActionType.PIXEL, {"x": cx, "y": hy - 4, "color": 11}))
        else: # general hair
            actions.append(PixelAction(ActionType.RECT, {"x": hx, "y": hy, "w": head_w, "h": 2, "color": c_skin_hi, "filled": True}))
        # Detail pass: jaw shadow and mouth
        actions.append(PixelAction(ActionType.LINE, {"x0": hx + 1, "y0": hy + head_h - 1, "x1": hx + head_w - 2, "y1": hy + head_h - 1, "color": c_skin_hi}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx, "y": hy + head_h - 2, "color": c_outline}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Horns (for fantasy_creature or demon)
        if acc == "horns" or category == "fantasy_creature":
            actions.append(PixelAction(ActionType.PART, {"name": "horns", "parent": "head", "anchor_x": cx, "anchor_y": hy - 2}))
            # Vertical stacks, not diagonals: a 1px diagonal line leaves the horn
            # as three dots that only touch at the corners.
            for horn_x, curve in ((hx, -1), (hx + head_w - 1, 1)):
                actions.append(PixelAction(ActionType.RECT, {"x": horn_x, "y": hy - 3, "w": 1, "h": 3, "color": 11, "filled": True}))
                actions.append(PixelAction(ActionType.PIXEL, {"x": horn_x + curve, "y": hy - 2, "color": 11}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Eyes
        actions.append(PixelAction(ActionType.PART, {"name": "eyes", "parent": "head", "anchor_x": cx, "anchor_y": hy + 3}))
        eye_y = hy + (head_h // 2)
        eye_spacing = 3 if head_w >= 8 else 2
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx - eye_spacing, "y": eye_y, "color": c_eyes}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx + eye_spacing - 1, "y": eye_y, "color": c_eyes}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx - eye_spacing, "y": eye_y - 1, "color": 8}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Wings (if fantasy_creature or accessory)
        if acc == "wings" or category == "fantasy_creature":
            actions.append(PixelAction(ActionType.PART, {"name": "wings", "parent": "torso", "anchor_x": cx, "anchor_y": ty}))
            # Overlapping runs so the wing is a solid feather mass instead of a
            # diagonal of isolated pixels.
            for step, (y, x, w) in enumerate(((ty - 2, tx - 3, 3), (ty - 1, tx - 2, 3), (ty, tx - 1, 3))):
                actions.append(PixelAction(ActionType.RECT, {"x": x, "y": y, "w": w, "h": 1, "color": 14, "filled": True}))
                mirror_x = tx + torso_w - x + tx - 1
                actions.append(PixelAction(ActionType.RECT, {"x": mirror_x, "y": y, "w": w, "h": 1, "color": 14, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Cape
        if acc == "cape":
            actions.append(PixelAction(ActionType.LINE, {"x0": tx - 1, "y0": ty + 1, "x1": tx - 2, "y1": ty + torso_h + 2, "color": 9}))
            actions.append(PixelAction(ActionType.LINE, {"x0": tx + torso_w, "y0": ty + 1, "x1": tx + torso_w + 1, "y1": ty + torso_h + 2, "color": 9}))

        # Arms. Each arm is a single pixel band split into an upper arm and a
        # forearm: identical composite pixels, but the elbow now has a pivot.
        arm_split = max(1, min(arm_h - 1, arm_h // 2))
        al_x = tx - arm_w
        al_y = ty + 1 + arm_l_dy
        left_arm_ops = [
            (ActionType.RECT, {"x": al_x, "y": al_y, "w": arm_w, "h": arm_h, "color": c_cloth, "filled": True}),
            (ActionType.PIXEL, {"x": al_x, "y": al_y + arm_h - 1, "color": c_skin}),
            (ActionType.PIXEL, {"x": al_x, "y": al_y, "color": c_skin_hi}),
            (ActionType.LINE, {"x0": al_x, "y0": al_y + arm_h - 2, "x1": al_x + arm_w - 1, "y1": al_y + arm_h - 2, "color": 8}),
            (ActionType.LINE, {"x0": al_x + arm_w - 1, "y0": al_y + 1, "x1": al_x + arm_w - 1, "y1": al_y + arm_h - 1, "color": c_pants}),
        ]
        cls._emit_two_segment_limb(
            actions, "arm_left", "forearm_left", "torso", al_x, al_y, arm_split, arm_h, left_arm_ops
        )

        ar_x = tx + torso_w
        ar_y = ty + 1 + arm_r_dy
        right_arm_ops = [
            (ActionType.RECT, {"x": ar_x, "y": ar_y, "w": arm_w, "h": arm_h, "color": c_cloth, "filled": True}),
            (ActionType.PIXEL, {"x": ar_x + arm_w - 1, "y": ar_y + arm_h - 1, "color": c_skin}),
            (ActionType.PIXEL, {"x": ar_x + arm_w - 1, "y": ar_y, "color": c_skin_hi}),
            (ActionType.LINE, {"x0": ar_x, "y0": ar_y + 1, "x1": ar_x, "y1": ar_y + arm_h - 1, "color": c_pants}),
            (ActionType.LINE, {"x0": ar_x, "y0": ar_y + arm_h - 2, "x1": ar_x + arm_w - 1, "y1": ar_y + arm_h - 2, "color": 8}),
        ]
        cls._emit_two_segment_limb(
            actions, "arm_right", "forearm_right", "torso", ar_x, ar_y, arm_split, arm_h, right_arm_ops
        )

        # Legs, split the same way so the knee can bend.
        leg_split = max(1, min(leg_h - 1, leg_h // 2))
        ll_x = tx + leg_l_dx
        ll_y = ty + torso_h
        left_leg_ops = [
            (ActionType.RECT, {"x": ll_x, "y": ll_y, "w": leg_w, "h": leg_h, "color": c_pants, "filled": True}),
            (ActionType.RECT, {"x": ll_x - 1, "y": ll_y + leg_h - 1, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}),
            (ActionType.PIXEL, {"x": ll_x, "y": ll_y + 1, "color": 8}),
            (ActionType.RECT, {"x": ll_x - 1, "y": ll_y + leg_h - 2, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}),
        ]
        cls._emit_two_segment_limb(
            actions, "leg_left", "shin_left", "torso", ll_x, ll_y, leg_split, leg_h, left_leg_ops
        )

        lr_x = tx + torso_w - leg_w + leg_r_dx
        lr_y = ty + torso_h
        right_leg_ops = [
            (ActionType.RECT, {"x": lr_x, "y": lr_y, "w": leg_w, "h": leg_h, "color": c_pants, "filled": True}),
            (ActionType.RECT, {"x": lr_x, "y": lr_y + leg_h - 1, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}),
            (ActionType.PIXEL, {"x": lr_x + leg_w - 1, "y": lr_y + 1, "color": 8}),
            (ActionType.RECT, {"x": lr_x, "y": lr_y + leg_h - 2, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}),
        ]
        cls._emit_two_segment_limb(
            actions, "leg_right", "shin_right", "torso", lr_x, lr_y, leg_split, leg_h, right_leg_ops
        )

        # Weapon
        if wep != "none":
            actions.append(PixelAction(ActionType.LAYER, {"id": 1, "name": "equipment", "opacity": 1.0}))
            # A shield is carried on the left arm: anchoring it to the right arm
            # left it floating in mid-air and made it swing with the sword hand.
            offhand = wep == "shield"
            actions.append(PixelAction(ActionType.PART, {
                "name": "weapon",
                "parent": "forearm_left" if offhand else "forearm_right",
                "anchor_x": al_x - 3 if offhand else ar_x + arm_w,
                "anchor_y": al_y if offhand else ar_y + 2,
            }))

            if wep == "sword":
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y + 2, "x1": ar_x + arm_w, "y1": ar_y - 6, "color": 7}))
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w - 1, "y0": ar_y, "x1": ar_x + arm_w + 1, "y1": ar_y, "color": 10}))
            elif wep == "staff":
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y + 6, "x1": ar_x + arm_w, "y1": ar_y - 7, "color": 5}))
                actions.append(PixelAction(ActionType.PIXEL, {"x": ar_x + arm_w, "y": ar_y - 8, "color": 14}))
            elif wep == "axe":
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y + 4, "x1": ar_x + arm_w, "y1": ar_y - 5, "color": 5}))
                actions.append(PixelAction(ActionType.RECT, {"x": ar_x + arm_w - 1, "y": ar_y - 5, "w": 3, "h": 2, "color": 7, "filled": True}))
            elif wep == "dagger":
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y + 2, "x1": ar_x + arm_w, "y1": ar_y - 2, "color": 7}))
            elif wep == "shield":
                actions.append(PixelAction(ActionType.RECT, {"x": al_x - 3, "y": al_y + 2, "w": 3, "h": 5, "color": 2, "filled": True}))
                actions.append(PixelAction(ActionType.PIXEL, {"x": al_x - 2, "y": al_y + 4, "color": 10}))
            elif wep == "bow":
                # Stave as three touching runs so the curve stays connected, then the string.
                for x_off, y_off, run in ((2, -4, 3), (1, -2, 5), (2, 2, 3)):
                    actions.append(PixelAction(ActionType.RECT, {
                        "x": ar_x + arm_w + x_off, "y": ar_y + y_off, "w": 1, "h": run, "color": 5, "filled": True,
                    }))
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y - 4, "x1": ar_x + arm_w, "y1": ar_y + 4, "color": 7}))

            # Detail pass: weapon glint and grip wrap. The glint has to sit on the
            # weapon itself; a fixed offset left a lone pixel floating in the air
            # above short blades and above the left-hand shield.
            if wep != "shield":
                glint_y = ar_y - 1 if wep == "dagger" else ar_y - 5
                actions.append(PixelAction(ActionType.PIXEL, {"x": ar_x + arm_w, "y": glint_y, "color": 8}))
            actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y + 1, "x1": ar_x + arm_w, "y1": ar_y + 2, "color": c_outline}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        cls._finishing_pass(actions, c_outline, width, height, element, palette_name, seed)
        actions.append(PixelAction(ActionType.END, {"target": "PROGRAM"}))
        return PixelScriptProgram(actions=actions, metadata=meta)
