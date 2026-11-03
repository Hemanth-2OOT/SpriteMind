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


class CharacterGenerator:
    """Generates complete structured PixelScript programs for characters, animals, and fantasy creatures."""

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
            actions.append(PixelAction(ActionType.LINE, {"x0": bx, "y0": by + 1, "x1": bx - 3, "y1": by - 2, "color": c_skin}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            # 4 Legs: Front-left, Front-right, Hind-left, Hind-right
            leg_h = 5
            actions.append(PixelAction(ActionType.PART, {"name": "leg_left", "parent": "torso", "anchor_x": bx + 1, "anchor_y": by + body_h}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx + 1 + leg_l_dx, "y": by + body_h, "w": 2, "h": leg_h, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

            actions.append(PixelAction(ActionType.PART, {"name": "leg_right", "parent": "torso", "anchor_x": bx + body_w - 3, "anchor_y": by + body_h}))
            actions.append(PixelAction(ActionType.RECT, {"x": bx + body_w - 3 + leg_r_dx, "y": by + body_h, "w": 2, "h": leg_h, "color": c_skin, "filled": True}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

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

            actions.append(PixelAction(ActionType.END, {"target": "PROGRAM"}))
            return PixelScriptProgram(actions=actions, metadata=meta)

        # -------------------------------------------------------------
        # 3. BIPED HUMAN/CREATURE ANATOMY (Knights, Zombies, Rogues, Wizards, Robots, Skeletons, Goblins, Fantasy Creatures)
        # -------------------------------------------------------------
        # Proportion tuning
        if prop == "stocky":
            torso_w, torso_h = 10, 6
            head_w, head_h = 8, 6
            leg_w, leg_h = 4, 5
            arm_w, arm_h = 3, 5
        elif prop == "slender":
            torso_w, torso_h = 6, 8
            head_w, head_h = 6, 6
            leg_w, leg_h = 2, 7
            arm_w, arm_h = 2, 7
        elif prop == "chibi":
            torso_w, torso_h = 6, 5
            head_w, head_h = 10, 8
            leg_w, leg_h = 2, 4
            arm_w, arm_h = 2, 4
        else: # standard
            torso_w, torso_h = 8, 7
            head_w, head_h = 8, 7
            leg_w, leg_h = 3, 6
            arm_w, arm_h = 2, 6

        tx = cx - (torso_w // 2)
        ty = cy - 2 + bob_y

        # Torso
        actions.append(PixelAction(ActionType.PART, {"name": "torso", "parent": None, "anchor_x": cx, "anchor_y": ty}))
        actions.append(PixelAction(ActionType.RECT, {"x": tx, "y": ty, "w": torso_w, "h": torso_h, "color": c_cloth, "filled": True}))
        # Belt line
        actions.append(PixelAction(ActionType.LINE, {"x0": tx, "y0": ty + torso_h - 1, "x1": tx + torso_w - 1, "y1": ty + torso_h - 1, "color": c_pants}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Head
        hx = cx - (head_w // 2)
        hy = ty - head_h
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
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Horns (for fantasy_creature or demon)
        if acc == "horns" or category == "fantasy_creature":
            actions.append(PixelAction(ActionType.PART, {"name": "horns", "parent": "head", "anchor_x": cx, "anchor_y": hy - 2}))
            actions.append(PixelAction(ActionType.LINE, {"x0": hx, "y0": hy, "x1": hx - 1, "y1": hy - 2, "color": 11}))
            actions.append(PixelAction(ActionType.LINE, {"x0": hx + head_w - 1, "y0": hy, "x1": hx + head_w, "y1": hy - 2, "color": 11}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Eyes
        actions.append(PixelAction(ActionType.PART, {"name": "eyes", "parent": "head", "anchor_x": cx, "anchor_y": hy + 3}))
        eye_y = hy + (head_h // 2)
        eye_spacing = 3 if head_w >= 8 else 2
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx - eye_spacing, "y": eye_y, "color": c_eyes}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": cx + eye_spacing - 1, "y": eye_y, "color": c_eyes}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Wings (if fantasy_creature or accessory)
        if acc == "wings" or category == "fantasy_creature":
            actions.append(PixelAction(ActionType.PART, {"name": "wings", "parent": "torso", "anchor_x": cx, "anchor_y": ty}))
            actions.append(PixelAction(ActionType.LINE, {"x0": tx, "y0": ty + 1, "x1": tx - 4, "y1": ty - 3, "color": 14}))
            actions.append(PixelAction(ActionType.LINE, {"x0": tx + torso_w, "y0": ty + 1, "x1": tx + torso_w + 4, "y1": ty - 3, "color": 14}))
            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Cape
        if acc == "cape":
            actions.append(PixelAction(ActionType.LINE, {"x0": tx - 1, "y0": ty + 1, "x1": tx - 2, "y1": ty + torso_h + 2, "color": 9}))
            actions.append(PixelAction(ActionType.LINE, {"x0": tx + torso_w, "y0": ty + 1, "x1": tx + torso_w + 1, "y1": ty + torso_h + 2, "color": 9}))

        # Arms
        al_x = tx - arm_w
        al_y = ty + 1 + arm_l_dy
        actions.append(PixelAction(ActionType.PART, {"name": "arm_left", "parent": "torso", "anchor_x": al_x, "anchor_y": al_y}))
        actions.append(PixelAction(ActionType.RECT, {"x": al_x, "y": al_y, "w": arm_w, "h": arm_h, "color": c_cloth, "filled": True}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": al_x, "y": al_y + arm_h - 1, "color": c_skin}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        ar_x = tx + torso_w
        ar_y = ty + 1 + arm_r_dy
        actions.append(PixelAction(ActionType.PART, {"name": "arm_right", "parent": "torso", "anchor_x": ar_x, "anchor_y": ar_y}))
        actions.append(PixelAction(ActionType.RECT, {"x": ar_x, "y": ar_y, "w": arm_w, "h": arm_h, "color": c_cloth, "filled": True}))
        actions.append(PixelAction(ActionType.PIXEL, {"x": ar_x + arm_w - 1, "y": ar_y + arm_h - 1, "color": c_skin}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Legs
        ll_x = tx + 1 + leg_l_dx
        ll_y = ty + torso_h
        actions.append(PixelAction(ActionType.PART, {"name": "leg_left", "parent": "torso", "anchor_x": ll_x, "anchor_y": ll_y}))
        actions.append(PixelAction(ActionType.RECT, {"x": ll_x, "y": ll_y, "w": leg_w, "h": leg_h, "color": c_pants, "filled": True}))
        actions.append(PixelAction(ActionType.RECT, {"x": ll_x - 1, "y": ll_y + leg_h - 1, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        lr_x = tx + torso_w - 1 - leg_w + leg_r_dx
        lr_y = ty + torso_h
        actions.append(PixelAction(ActionType.PART, {"name": "leg_right", "parent": "torso", "anchor_x": lr_x, "anchor_y": lr_y}))
        actions.append(PixelAction(ActionType.RECT, {"x": lr_x, "y": lr_y, "w": leg_w, "h": leg_h, "color": c_pants, "filled": True}))
        actions.append(PixelAction(ActionType.RECT, {"x": lr_x, "y": lr_y + leg_h - 1, "w": leg_w + 1, "h": 1, "color": c_outline, "filled": True}))
        actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        # Weapon
        if wep != "none":
            actions.append(PixelAction(ActionType.LAYER, {"id": 1, "name": "equipment", "opacity": 1.0}))
            actions.append(PixelAction(ActionType.PART, {"name": "weapon", "parent": "arm_right", "anchor_x": ar_x + arm_w, "anchor_y": ar_y + 2}))

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
                actions.append(PixelAction(ActionType.RECT, {"x": al_x - 3, "y": al_y, "w": 3, "h": 5, "color": 2, "filled": True}))
                actions.append(PixelAction(ActionType.PIXEL, {"x": al_x - 2, "y": al_y + 2, "color": 10}))
            elif wep == "bow":
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y - 4, "x1": ar_x + arm_w + 2, "y1": ar_y, "color": 5}))
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w + 2, "y0": ar_y, "x1": ar_x + arm_w, "y1": ar_y + 4, "color": 5}))
                actions.append(PixelAction(ActionType.LINE, {"x0": ar_x + arm_w, "y0": ar_y - 4, "x1": ar_x + arm_w, "y1": ar_y + 4, "color": 7}))

            actions.append(PixelAction(ActionType.END, {"target": "PART"}))

        actions.append(PixelAction(ActionType.END, {"target": "PROGRAM"}))
        return PixelScriptProgram(actions=actions, metadata=meta)
