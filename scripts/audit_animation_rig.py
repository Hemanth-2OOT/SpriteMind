"""Animation audit: weapon-to-hand distance, frame identity and silhouette.

Reproduces the Milestone A measurements (see docs/ANIMATION_RIG.md):

    cd backend && PYTHONPATH=. python ../scripts/audit_animation_rig.py --json /tmp/audit.json

Rendering only: no checkpoint, dataset or generated asset is touched.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from spritemind.ml.metrics import alpha_mask, connected_component_count  # noqa: E402
from spritemind.ml.temporal import TemporalAnimator  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402

ANIMATIONS = [
    "idle",
    "walk",
    "run",
    "attack",
    "hurt",
    "cast",
    "dash",
    "melee_attack",
    "hit_reaction",
]


def _min_distance(a, b):
    if not a or not b:
        return None
    return round(min(math.hypot(ax - bx, ay - by) for ax, ay in a for bx, by in b), 3)


def audit(frames=6, mode="socketed", seed=3):
    report = {
        "frames": frames,
        "rig_mode": mode,
        "category": "wizard",
        "weapon": "staff",
        "accessory": "cape",
        "seed": seed,
        "animations": {},
        "identical_to_walk_frames": {},
    }
    signatures = {}

    for anim in ANIMATIONS:
        out = TemporalAnimator.generate_animation(
            base_script_or_prompt="",
            category=report["category"],
            style="classic_16bit",
            anim_category=anim,
            palette_name="pico8",
            width=32,
            height=32,
            num_frames=frames,
            fps=8,
            engine_mode="procedural_baseline",
            seed=seed,
            weapon=report["weapon"],
            accessory=report["accessory"],
            rig_mode=mode,
        )
        rows, sigs = [], []
        for frame in out["frames"]:
            res = PixelScriptEngine.execute(
                frame["script_text"], width=32, height=32, palette_name="pico8"
            )
            weapon = res.canvas.parts.get("weapon")
            hand = res.canvas.parts.get("arm_right")
            comp = res.composite_array
            rows.append(
                {
                    "frame": frame["frame_index"],
                    "weapon_hand_distance": _min_distance(
                        weapon.pixels if weapon else set(), hand.pixels if hand else set()
                    ),
                    "islands": connected_component_count(alpha_mask(comp)),
                    "opaque_pixels": int(np.count_nonzero(comp[:, :, 3] > 0)),
                }
            )
            sigs.append(np.asarray(comp).tobytes())
        signatures[anim] = sigs

        dists = [r["weapon_hand_distance"] for r in rows if r["weapon_hand_distance"] is not None]
        report["animations"][anim] = {
            "requested_animation": out.get("requested_animation"),
            "reported_animation_category": out["animation_category"],
            "animation_aliased": out.get("animation_aliased"),
            "max_weapon_hand_distance": max(dists) if dists else None,
            "mean_weapon_hand_distance": round(sum(dists) / len(dists), 3) if dists else None,
            "frames": rows,
        }

    walk = signatures["walk"]
    for anim, sigs in signatures.items():
        if anim == "walk":
            continue
        same = sum(1 for a, b in zip(sigs, walk) if a == b)
        report["identical_to_walk_frames"][anim] = f"{same}/{len(walk)}"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--mode", choices=["socketed", "flat"], default="socketed")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    report = audit(frames=args.frames, mode=args.mode, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text)
        print(f"wrote {args.json}")
    print(f"rig_mode={report['rig_mode']} identical_to_walk={report['identical_to_walk_frames']}")
    for anim, row in report["animations"].items():
        print(
            f"{anim:14s} resolved={row['reported_animation_category']:14s} "
            f"max_d={row['max_weapon_hand_distance']} mean_d={row['mean_weapon_hand_distance']}"
        )


if __name__ == "__main__":
    main()
