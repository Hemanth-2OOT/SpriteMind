"""Temporal Animation Dataset Generator and Storage.

Generates reproducible multi-frame animation sequences for pixel-art characters.
Guarantees strict separation by character identity to prevent train/test leakage.
"""
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from spritemind.dataset.generator import (
    CharacterGenerator,
    CATEGORIES,
    STYLES,
    PROPORTIONS,
    WEAPONS,
    ACCESSORIES,
)
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine


ANIMATION_CATEGORIES = [
    # Basic Animations
    "idle",
    "walk",
    "run",
    "jump",
    "fall",
    "death",
    # Combat Animations
    "melee_attack",
    "ranged_attack",
    "hit_reaction",
    "taking_damage",
    "blocking",
    "special_ability",
    "cast",
    "dash",
    # Creature Animations
    "slime_bouncing",
    "zombie_movement",
    "skeleton_walking",
    "robot_movement",
    "animal_running",
    "flying_creatures",
]

CYCLIC_ANIMATIONS = {
    "idle",
    "walk",
    "run",
    "slime_bouncing",
    "zombie_movement",
    "skeleton_walking",
    "robot_movement",
    "animal_running",
    "flying_creatures",
}


class TemporalDatasetGenerator:
    """Generates structured character animation sequences with complete provenance."""

    PARTS_LIST = [
        "torso",
        "head",
        "arm_left",
        "arm_right",
        "leg_left",
        "leg_right",
        "weapon",
        "accessory",
    ]

    @staticmethod
    def _clamp_down(dy: int, limit: int = 1) -> int:
        """Arms that travel further down than this sink behind the torso and the
        sprite visibly loses pixels from one frame to the next."""
        return min(dy, limit)

    @classmethod
    def compute_motion_offsets(
        cls,
        anim_category: str,
        phase: float,  # normalized in [0.0, 1.0]
        category: str = "knight",
    ) -> Dict[str, Tuple[int, int, int]]:
        """Computes anatomical part offsets (dx, dy, rot_deg) for a given motion phase."""
        offsets = {p: (0, 0, 0) for p in cls.PARTS_LIST}
        rad = phase * 2.0 * math.pi

        if anim_category == "idle":
            # Breathing is read through the shoulders and head only: lifting the
            # torso would leave an empty row above the legs and make the sprite
            # pulse in size from frame to frame.
            breathe = 1 if math.sin(rad) > 0.3 else 0
            offsets["head"] = (0, -breathe, 0)
            offsets["arm_left"] = (0, -breathe, 0)
            offsets["arm_right"] = (0, -breathe, 0)

        elif anim_category == "walk":
            # Alternating leg stride, foot lift, and counter-arm swing with rotation
            stride = int(round(math.sin(rad) * 2.0))
            arm_swing = int(round(math.cos(rad) * 2.0))
            bob = -1 if abs(stride) > 1 else 0
            offsets["torso"] = (0, bob, 0)
            offsets["head"] = (0, bob, 0)
            offsets["leg_left"] = (stride, -1 if stride > 0 else 0, stride * 8)
            offsets["leg_right"] = (-stride, -1 if stride < 0 else 0, -stride * 8)
            offsets["arm_left"] = (0, cls._clamp_down(arm_swing), arm_swing * 12)
            offsets["arm_right"] = (0, cls._clamp_down(-arm_swing), -arm_swing * 12)
            offsets["weapon"] = (0, cls._clamp_down(-arm_swing), -arm_swing * 15)

        elif anim_category == "run":
            # Dynamic leaning stride with exaggerated arm drive and rotation
            stride = int(round(math.sin(rad) * 3.5))
            arm_swing = int(round(math.cos(rad) * 3.0))
            bob = -1 if math.sin(rad * 2.0) > 0 else 1
            offsets["torso"] = (1, bob, 6)
            offsets["head"] = (1, bob, 3)
            offsets["leg_left"] = (stride, -1 if stride > 0 else 1, stride * 12)
            offsets["leg_right"] = (-stride, -1 if stride < 0 else 1, -stride * 12)
            offsets["arm_left"] = (-1, cls._clamp_down(arm_swing), arm_swing * 20)
            offsets["arm_right"] = (1, cls._clamp_down(-arm_swing), -arm_swing * 20)
            offsets["weapon"] = (1, cls._clamp_down(-arm_swing), 10 - arm_swing * 15)

        elif anim_category == "jump":
            # Anticipation -> Apex flight -> Descent & Landing
            if phase < 0.2:
                # Squash anticipation
                offsets["torso"] = (0, 2, 0)
                offsets["head"] = (0, 2, 0)
                offsets["leg_left"] = (-1, 1, 5)
                offsets["leg_right"] = (1, 1, -5)
                offsets["arm_left"] = (0, 1, 0)
                offsets["arm_right"] = (0, 1, 0)
            elif phase < 0.7:
                # Airborne apex with limb extension
                apex_y = -int(round(math.sin((phase - 0.2) / 0.5 * math.pi) * 6.0))
                offsets["torso"] = (0, apex_y, 0)
                offsets["head"] = (0, apex_y - 1, 0)
                offsets["leg_left"] = (-1, apex_y + 2, -10)
                offsets["leg_right"] = (1, apex_y + 2, 10)
                offsets["arm_left"] = (-1, apex_y - 2, 20)
                offsets["arm_right"] = (1, apex_y - 2, -20)
                offsets["weapon"] = (2, apex_y - 3, -15)
            else:
                # Landing compression
                offsets["torso"] = (0, 1, 0)
                offsets["head"] = (0, 1, 0)
                offsets["leg_left"] = (0, 0, 0)
                offsets["leg_right"] = (0, 0, 0)

        elif anim_category == "fall":
            drop_y = int(round(phase * 6.0))
            offsets["torso"] = (0, drop_y, 0)
            offsets["head"] = (0, drop_y - 1, 0)
            offsets["arm_left"] = (-1, drop_y - 2, 15)
            offsets["arm_right"] = (1, drop_y - 2, -15)
            offsets["leg_left"] = (-1, drop_y, -5)
            offsets["leg_right"] = (1, drop_y, 5)

        elif anim_category == "death":
            # Collapse to ground with rotation
            fall_prog = min(1.0, phase * 1.3)
            dy = int(round(fall_prog * 8.0))
            rot = int(round(fall_prog * 75.0))
            offsets["torso"] = (-int(fall_prog * 3), dy, -rot)
            offsets["head"] = (-int(fall_prog * 5), dy + 2, -rot)
            offsets["arm_left"] = (-2, dy + 2, -rot // 2)
            offsets["arm_right"] = (1, dy + 3, rot // 2)
            offsets["leg_left"] = (2, dy, 0)
            offsets["leg_right"] = (4, dy, 0)
            offsets["weapon"] = (-int(fall_prog * 6), dy + 4, -rot - 15)

        elif anim_category == "dash":
            # Crouch -> explosive forward burst -> settle, with lean and trailing limbs
            if phase < 0.2:
                prog = phase / 0.2
                offsets["torso"] = (-1, 1 + int(round(prog)), -6)
                offsets["head"] = (-1, 1 + int(round(prog)), -4)
                offsets["leg_left"] = (-1, 1, 8)
                offsets["leg_right"] = (1, 1, -8)
                offsets["arm_left"] = (-2, 0, -10)
                offsets["arm_right"] = (-2, 0, 10)
            elif phase < 0.65:
                prog = (phase - 0.2) / 0.45
                burst = math.sin(prog * math.pi)
                lean = int(round(burst * 4.0))
                offsets["torso"] = (lean, -1, 14 + int(round(burst * 8)))
                offsets["head"] = (lean, -1, 8)
                offsets["leg_left"] = (-lean - 2, 0, -12)
                offsets["leg_right"] = (-lean - 1, -1, 6)
                offsets["arm_left"] = (-3, 1, -25)
                offsets["arm_right"] = (-2, 0, 20)
                offsets["weapon"] = (-1, 1, 10)
                offsets["accessory"] = (-2, 0, 18)
            else:
                prog = (phase - 0.65) / 0.35
                settle = int(round((1 - prog) * 2))
                offsets["torso"] = (settle, 0, 4)
                offsets["head"] = (settle, 0, 2)
                offsets["leg_left"] = (-1, 0, -4)
                offsets["leg_right"] = (1, 0, 2)

        elif anim_category == "cast":
            # Gather -> raise arm with orb -> release -> recover
            if phase < 0.35:
                prog = phase / 0.35
                offsets["arm_right"] = (-1, -int(round(prog * 2)), -12 - int(round(prog * 10)))
                offsets["weapon"] = (-1, -int(round(prog * 3)), 10 - int(round(prog * 18)))
                offsets["torso"] = (-1, 0, -3)
                offsets["head"] = (0, 0, -2)
            elif phase < 0.6:
                offsets["arm_right"] = (0, -3, -26)
                offsets["weapon"] = (1, -5, -30)
                offsets["torso"] = (0, -1, 4)
                offsets["head"] = (0, -1, 3)
                offsets["arm_left"] = (-2, -1, -12)
            elif phase < 0.8:
                prog = (phase - 0.6) / 0.2
                offsets["arm_right"] = (2 + int(round(prog)), 1, 18 + int(round(prog * 10)))
                offsets["weapon"] = (3 + int(round(prog * 2)), 1, 24)
                offsets["torso"] = (1, 0, 6)
                offsets["head"] = (1, 0, 4)
                offsets["leg_left"] = (-1, 0, 0)
                offsets["leg_right"] = (1, 0, 0)
            else:
                offsets["arm_right"] = (1, 0, 6)
                offsets["weapon"] = (1, 0, 8)
                offsets["torso"] = (0, 0, 1)

        elif anim_category == "melee_attack":
            # 4 distinct progressive keyframe phases: Windup -> Apex -> Slash Strike -> Follow-through
            if phase < 0.25:
                # 1. Windup pull-back
                prog = phase / 0.25
                offsets["arm_right"] = (-2 - int(round(prog)), -1 - int(round(2 * prog)), -15 - int(round(20 * prog)))
                offsets["weapon"] = (-3 - int(round(2 * prog)), -2 - int(round(2 * prog)), -25 - int(round(25 * prog)))
                offsets["torso"] = (-1, 0, -5)
                offsets["head"] = (-1, 0, -3)
            elif phase < 0.50:
                # 2. Coiled apex windup ready to strike
                prog = (phase - 0.25) / 0.25
                offsets["arm_right"] = (-3, -3, -35 - int(round(10 * prog)))
                offsets["weapon"] = (-5, -4, -50 - int(round(15 * prog)))
                offsets["torso"] = (-1, 0, -8)
                offsets["head"] = (0, 0, 0)
            elif phase < 0.75:
                # 3. Explosive forward slash strike & impact
                prog = (phase - 0.50) / 0.25
                offsets["arm_right"] = (4 + int(round(2 * prog)), 1, 40 + int(round(15 * prog)))
                offsets["weapon"] = (6 + int(round(3 * prog)), 2, 55 + int(round(15 * prog)))
                offsets["torso"] = (2, 0, 10)
                offsets["head"] = (1, 0, 5)
                offsets["leg_left"] = (-1, 0, 0)
                offsets["leg_right"] = (1, 0, 0)
            else:
                # 4. Recovery & follow-through
                prog = (phase - 0.75) / 0.25
                offsets["arm_right"] = (2 - int(round(prog)), 1, 15 - int(round(10 * prog)))
                offsets["weapon"] = (3 - int(round(prog)), 1, 20 - int(round(15 * prog)))
                offsets["torso"] = (1, 0, 2)
                offsets["head"] = (0, 0, 0)

        elif anim_category == "ranged_attack":
            # Draw bow/wand -> Aim -> Release -> Recoil
            if phase < 0.35:
                prog = phase / 0.35
                offsets["arm_left"] = (2 + int(round(prog)), 0, 0)
                offsets["arm_right"] = (-1 - int(round(2 * prog)), 0, -10)
                offsets["weapon"] = (3 + int(round(prog)), 0, 5)
                offsets["torso"] = (-1, 0, -3)
            elif phase < 0.65:
                offsets["arm_left"] = (4, 0, 0)
                offsets["arm_right"] = (-4, 0, -20)
                offsets["weapon"] = (5, 0, 10)
                offsets["torso"] = (-1, 0, -5)
            elif phase < 0.85:
                offsets["arm_left"] = (4, 0, 0)
                offsets["arm_right"] = (-2, -1, 10)
                offsets["weapon"] = (6, 0, -10)
                offsets["torso"] = (1, 0, 5)
            else:
                offsets["arm_left"] = (1, 0, 0)
                offsets["arm_right"] = (0, 0, 0)
                offsets["weapon"] = (1, 0, 0)
                offsets["torso"] = (0, 0, 0)

        elif anim_category in ("hit_reaction", "taking_damage"):
            # Recoil jerk back with flash
            recoil = int(round(math.sin(phase * math.pi) * 3.5))
            offsets["torso"] = (-recoil, 0, -10)
            offsets["head"] = (-recoil - 1, -1, -15)
            offsets["arm_left"] = (-recoil + 1, -1, 0)
            offsets["arm_right"] = (-recoil, -1, 0)

        elif anim_category == "blocking":
            # Shield raise and brace stance
            brace = 1 if phase > 0.2 and phase < 0.8 else 0
            offsets["arm_left"] = (2, -1, 0)
            offsets["torso"] = (-brace, brace, -5)
            offsets["head"] = (-brace, brace, 0)

        elif anim_category == "special_ability":
            # Arcane levitation & power surge
            lev = -int(round(math.sin(rad) * 3.0))
            offsets["torso"] = (0, lev, 0)
            offsets["head"] = (0, lev, 0)
            offsets["arm_left"] = (-2, lev - 2, -20)
            offsets["arm_right"] = (2, lev - 2, 20)
            offsets["weapon"] = (3, lev - 4, 30)

        elif anim_category == "slime_bouncing":
            # Elastic squash and stretch bounce
            if phase < 0.4:
                # Squashed on ground
                offsets["torso"] = (0, 2, 0)
            elif phase < 0.8:
                # Stretched high in air
                jump_y = -int(round(math.sin((phase - 0.4) / 0.4 * math.pi) * 5.0))
                offsets["torso"] = (0, jump_y, 0)
            else:
                offsets["torso"] = (0, 1, 0)

        elif anim_category == "zombie_movement":
            # Shuffling limp with sagging torso
            limp_x = int(round(math.sin(rad) * 1.5))
            limp_y = 1 if math.sin(rad) > 0 else 0
            offsets["torso"] = (limp_x, limp_y, 10)
            offsets["head"] = (limp_x + 1, limp_y + 1, 15)
            offsets["arm_left"] = (3, 1, 0)
            offsets["arm_right"] = (3, 0, 0)
            offsets["leg_left"] = (limp_x, 0, 0)

        elif anim_category == "skeleton_walking":
            # Jerky clacking gait
            step = int(round(math.sin(rad) * 2.0))
            offsets["leg_left"] = (step, 0, 0)
            offsets["leg_right"] = (-step, 0, 0)
            offsets["arm_left"] = (-step, 0, 0)
            offsets["arm_right"] = (step, 0, 0)

        elif anim_category == "robot_movement":
            # Mechanical stepped translation (quantized)
            step_q = 2 if math.sin(rad) > 0.5 else (-2 if math.sin(rad) < -0.5 else 0)
            offsets["leg_left"] = (step_q, 0, 0)
            offsets["leg_right"] = (-step_q, 0, 0)
            offsets["head"] = (0, 1 if step_q != 0 else 0, 0)

        elif anim_category == "animal_running":
            # Quadruped gallop
            stride = int(round(math.sin(rad) * 3.0))
            bob = -1 if math.sin(rad) > 0 else 1
            offsets["torso"] = (0, bob, 0)
            offsets["leg_left"] = (stride, 0, 0)
            offsets["leg_right"] = (-stride, 0, 0)
            offsets["accessory"] = (-stride, -bob, 0)  # tail

        elif anim_category == "flying_creatures":
            # Wing flap sinusoidal lift
            flap = int(round(math.sin(rad) * 3.0))
            lift = -int(round(math.cos(rad) * 2.0))
            offsets["torso"] = (0, lift, 0)
            offsets["head"] = (0, lift, 0)
            offsets["accessory"] = (0, flap, flap * 5)  # wings

        return offsets

    @classmethod
    def generate_sequence(
        cls,
        character_id: str,
        category: str = "knight",
        anim_category: str = "walk",
        num_frames: int = 4,
        style: str = "classic_16bit",
        palette_name: str = "pico8",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates an end-to-end animation sequence for a specific character."""
        # 1. Generate identity-defining Frame 0 (Base Pose)
        base_prog = CharacterGenerator.generate_character(
            category=category,
            style=style,
            pose="idle",
            palette_name=palette_name,
            seed=seed,
        )
        base_script = PixelScriptSerializer.serialize_program(base_prog)

        is_looping = anim_category in CYCLIC_ANIMATIONS
        frame_records = []

        for f_idx in range(num_frames):
            phase = f_idx / float(num_frames) if is_looping else f_idx / float(max(1, num_frames - 1))
            offsets = cls.compute_motion_offsets(anim_category, phase, category=category)

            # Apply kinematic part transforms to base program (stripping any prior TRANSFORM actions)
            frame_lines = []
            for ln in base_script.splitlines():
                stripped = ln.strip()
                if stripped and not stripped.startswith("TRANSFORM"):
                    frame_lines.append(stripped)

            # Append dynamic kinematic TRANSFORM actions before rendering
            for part_name, (dx, dy, rot) in offsets.items():
                if dx != 0 or dy != 0 or rot != 0:
                    frame_lines.append(f"TRANSFORM {part_name} {dx} {dy} {rot}")

            frame_script = "\n".join(frame_lines)
            res = PixelScriptEngine.execute(frame_script, width=32, height=32, palette_name=palette_name)

            frame_records.append({
                "frame_index": f_idx,
                "phase": round(phase, 4),
                "script_text": frame_script,
                "offsets": offsets,
                "image_base64": res.to_dict()["image_base64"],
                "pixel_count": int(np.count_nonzero(res.composite_array[:, :, 3] > 0)),
            })

        return {
            "character_id": character_id,
            "category": category,
            "animation_category": anim_category,
            "num_frames": num_frames,
            "is_looping": is_looping,
            "palette": palette_name,
            "style": style,
            "seed": seed,
            "frames": frame_records,
        }

    @classmethod
    def build_dataset(
        cls,
        output_dir: str = "data/temporal_dataset_1k",
        total_sequences: int = 1000,
        val_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """Builds a complete, reproducible multi-sequence animation dataset split strictly by character."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        rng = random.Random(1337)
        # Partition character identities: 80% train, 20% val
        num_characters = max(100, total_sequences // 4)
        num_train_chars = int(num_characters * (1.0 - val_ratio))
        char_ids = [f"char_{i:04d}" for i in range(num_characters)]
        train_char_set = set(char_ids[:num_train_chars])

        train_records = []
        val_records = []

        seq_counter = 0
        while seq_counter < total_sequences:
            cid = rng.choice(char_ids)
            cat = rng.choice(CATEGORIES)
            anim = rng.choice(ANIMATION_CATEGORIES)
            n_frames = rng.choice([4, 6, 8])
            pal = rng.choice(["pico8", "dawnbringer16", "gameboy", "cyberpunk"])
            seed = int(cid.split("_")[1]) * 1000 + rng.randint(0, 99)

            seq = cls.generate_sequence(
                character_id=cid,
                category=cat,
                anim_category=anim,
                num_frames=n_frames,
                palette_name=pal,
                seed=seed,
            )

            if cid in train_char_set:
                train_records.append(seq)
            else:
                val_records.append(seq)

            seq_counter += 1

        # Save to JSONL
        train_file = out_path / "train_sequences.jsonl"
        val_file = out_path / "val_sequences.jsonl"
        meta_file = out_path / "dataset_metadata.json"

        with open(train_file, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r) + "\n")

        meta = {
            "dataset_name": "temporal_dataset_1k",
            "total_sequences": total_sequences,
            "train_sequences": len(train_records),
            "val_sequences": len(val_records),
            "total_characters": num_characters,
            "train_characters": num_train_chars,
            "val_characters": num_characters - num_train_chars,
            "categories": CATEGORIES,
            "animation_categories": ANIMATION_CATEGORIES,
            "leakage_free_split": True,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta
