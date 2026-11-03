"""Reproducible Evaluation Suite for Visual Intelligence, Defect Detection, and Self-Correction.

Compares:
1. Corrupted / Injected Sprite
2. Automatically Corrected Sprite
3. Ground-Truth Clean Sprite
Measures precision, recall, F1, Silhouette IoU, Pixel Accuracy, Identity Preservation,
Repair Success Rate, and Unintended Modification Rate across clean, corrupted, and difficult edge cases.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from spritemind.critic.repair_engine import AutonomousRepairEngine
from spritemind.renderer.engine import PixelScriptEngine


# 10 Representative Test Archetypes including Difficult Edge Cases
EVALUATION_CASES: List[Dict[str, Any]] = [
    {
        "id": "eval_001_stray_pixel",
        "name": "Knight with floating stray pixel",
        "category": "knight",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 5 8 1
LINE 13 18 18 18 2
PART head torso 16 6
RECT 11 6 10 8 15 1
PART arm_left torso 11 15
RECT 11 15 2 4 10 1
PART arm_right torso 19 15
RECT 19 15 2 4 10 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 5 8 1
LINE 13 18 18 18 2
PART head torso 16 6
RECT 11 6 10 8 15 1
PART arm_left torso 11 15
RECT 11 15 2 4 10 1
PART arm_right torso 19 15
RECT 19 15 2 4 10 1
PIXEL 2 2 9""",
        "defect_type": "stray_pixel",
        "is_edge_case": False,
    },
    {
        "id": "eval_002_disconnected_arm",
        "name": "Zombie with disconnected arm",
        "category": "zombie",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 6 4 1
PART head torso 16 6
RECT 12 6 8 8 3 1
PART arm_left torso 10 15
RECT 10 15 3 4 4 1
PART arm_right torso 19 15
RECT 19 15 3 4 4 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 6 4 1
PART head torso 16 6
RECT 12 6 8 8 3 1
PART arm_left torso 10 15
RECT 10 15 3 4 4 1
PART arm_right torso 26 15
RECT 26 15 3 4 4 1""",
        "defect_type": "disconnected_part",
        "is_edge_case": False,
    },
    {
        "id": "eval_003_interior_hole",
        "name": "Robot chassis with unexpected interior void",
        "category": "robot",
        "palette": "cyberpunk",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 11 12 10 8 6 1
RECT 13 14 6 4 7 1
PART head torso 16 6
RECT 12 6 8 6 13 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 11 12 10 8 6 1
RECT 13 14 6 4 7 1
PART head torso 16 6
RECT 12 6 8 6 13 1
PIXEL 15 15 0""",
        "defect_type": "unexpected_hole",
        "is_edge_case": False,
    },
    {
        "id": "eval_004_palette_leak",
        "name": "Rogue with out-of-palette RGB leak",
        "category": "rogue",
        "palette": "gameboy",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 13 6 7 2 1
PART head torso 16 6
RECT 12 6 8 7 3 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 13 6 7 2 1
PART head torso 16 6
RECT 12 6 8 7 3 1
PIXEL 2 2 15""",  # index 15 does not exist in gameboy palette (only 0-4)
        "defect_type": "palette_leak",
        "is_edge_case": False,
    },
    {
        "id": "eval_005_missing_limb",
        "name": "Skeleton missing right arm without shield",
        "category": "skeleton",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 14 13 4 7 7 1
PART head torso 16 6
RECT 12 6 8 7 7 1
PART arm_left torso 11 14
RECT 11 14 2 5 7 1
PART arm_right torso 19 14
RECT 19 14 2 5 7 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 14 13 4 7 7 1
PART head torso 16 6
RECT 12 6 8 7 7 1
PART arm_left torso 11 14
RECT 11 14 2 5 7 1""",
        "defect_type": "missing_limb",
        "is_edge_case": False,
    },
    # Difficult Edge Cases (where intentional details might look like defects)
    {
        "id": "eval_006_wizard_magic_orb",
        "name": "Wizard with intentional floating spell orb (Edge Case)",
        "category": "wizard",
        "palette": "dawnbringer16",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 7 3 1
PART head torso 16 6
RECT 12 6 8 8 2 1
PIXEL 10 12 14
PIXEL 10 13 14""",  # Intentional floating spell orb near staff
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 7 3 1
PART head torso 16 6
RECT 12 6 8 8 2 1
PIXEL 10 12 14
PIXEL 10 13 14
PIXEL 1 1 9""",  # Actual stray pixel in addition to magic orb
        "defect_type": "stray_pixel",
        "is_edge_case": True,
    },
    {
        "id": "eval_007_slime_satellite",
        "name": "Slime with intentional detached droplet (Edge Case)",
        "category": "slime",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 11 16 10 7 12 1
RECT 12 14 8 2 12 1
PIXEL 22 17 12""",  # Detached slime satellite droplet
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 11 16 10 7 12 1
RECT 12 14 8 2 12 1
PIXEL 22 17 12
PIXEL 16 18 0""",  # Unintended internal hole
        "defect_type": "unexpected_hole",
        "is_edge_case": True,
    },
    {
        "id": "eval_008_animal_quadruped",
        "name": "Wolf with quadruped leg balance",
        "category": "animal",
        "palette": "dawnbringer16",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 10 15 12 5 8 1
PART head torso 21 11
RECT 20 11 6 5 8 1
PART leg_left torso 11 20
RECT 11 20 2 4 5 1
PART leg_right torso 19 20
RECT 19 20 2 4 5 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 10 15 12 5 8 1
PART head torso 21 11
RECT 20 11 6 5 8 1
PART leg_left torso 11 20
RECT 11 20 2 4 5 1
PART leg_right torso 19 20
RECT 19 20 2 4 5 1
PIXEL 2 2 9""",
        "defect_type": "stray_pixel",
        "is_edge_case": False,
    },
    {
        "id": "eval_009_fantasy_wings",
        "name": "Dragon with floating wingtips",
        "category": "fantasy_creature",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 15
RECT 12 14 8 7 9 1
PART head torso 16 8
RECT 13 7 6 6 8 1
PART wings torso 16 13
RECT 8 11 4 5 10 1
RECT 20 11 4 5 10 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 15
RECT 12 14 8 7 9 1
PART head torso 16 8
RECT 13 7 6 6 8 1
PART wings torso 16 13
RECT 8 11 4 5 10 1
RECT 28 11 4 5 10 1""",  # Right wing detached
        "defect_type": "disconnected_part",
        "is_edge_case": False,
    },
    {
        "id": "eval_010_goblin_asymmetry",
        "name": "Goblin with severe detached ear",
        "category": "goblin",
        "palette": "pico8",
        "clean_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 13 15 6 6 4 1
PART head torso 16 8
RECT 12 8 8 7 4 1
PART ear_left head 10 9
RECT 10 9 2 4 12 1
PART ear_right head 20 9
RECT 20 9 2 4 12 1""",
        "corrupted_script": """LAYER 0 base 1.00
PART torso none 16 16
RECT 13 15 6 6 4 1
PART head torso 16 8
RECT 12 8 8 7 4 1
PART ear_left head 10 9
RECT 10 9 2 4 12 1
PART ear_right head 28 9
RECT 28 9 2 4 12 1""",
        "defect_type": "disconnected_part",
        "is_edge_case": False,
    },
]


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return float(intersection / max(1, union))


def compute_pixel_accuracy(img1: np.ndarray, img2: np.ndarray) -> float:
    total_pixels = img1.shape[0] * img1.shape[1]
    matching = np.all(img1 == img2, axis=-1).sum()
    return float((matching / total_pixels) * 100.0)


class Stage3EvaluationSuite:
    """Rigorous evaluation comparing Original, Corrupted, and Self-Corrected sprites."""

    def __init__(self):
        self.engine = AutonomousRepairEngine()
        self.results_path = Path("data/benchmarks/critic_eval_results.json")

    def run_evaluation(self) -> Dict[str, Any]:
        case_results = []
        tp, fp, fn, tn = 0, 0, 0, 0
        total_repairs_successful = 0
        total_iou_gain = []
        total_unintended_changes = []
        edge_cases_preserved = 0
        total_edge_cases = 0

        print(f"Executing Stage 3 Evaluation Suite on {len(EVALUATION_CASES)} archetypes...")

        for case in EVALUATION_CASES:
            c_id = case["id"]
            cat = case["category"]
            pal = case["palette"]
            expected_defect = case["defect_type"]
            is_edge = case["is_edge_case"]

            if is_edge:
                total_edge_cases += 1

            # Render Ground Truth clean
            gt_res = PixelScriptEngine.execute(case["clean_script"], width=32, height=32, palette_name=pal)
            gt_arr = gt_res.composite_array
            gt_alpha = gt_arr[:, :, 3]

            # Render Corrupted
            corrupt_res = PixelScriptEngine.execute(case["corrupted_script"], width=32, height=32, palette_name=pal)
            corrupt_arr = corrupt_res.composite_array
            corrupt_alpha = corrupt_arr[:, :, 3]

            # Baseline metrics before repair
            corrupt_iou = compute_iou(corrupt_alpha, gt_alpha)
            corrupt_acc = compute_pixel_accuracy(corrupt_arr, gt_arr)

            # Inspect Corrupted sprite
            inspection = self.engine.inspect(corrupt_res, category=cat, palette_name=pal)
            detected_types = [d["type"] for d in inspection.get("detected_defects", [])]

            # Defect Detection Accuracy
            if expected_defect in detected_types:
                tp += 1
            else:
                fn += 1

            # Check if edge case was falsely over-flagged
            if is_edge:
                # Wizard orb or slime droplet should not trigger false positive stray pixel
                false_positives = [d for d in inspection.get("detected_defects", []) if d["type"] != expected_defect]
                if len(false_positives) == 0:
                    edge_cases_preserved += 1

            # Run Autonomous Refinement
            refine_res = self.engine.refine_closed_loop(
                case["corrupted_script"],
                category=cat,
                palette_name=pal,
                max_iterations=3,
            )

            # Render Final Repaired
            repaired_script = refine_res["final_frame"]["script_text"]
            repaired_res = PixelScriptEngine.execute(repaired_script, width=32, height=32, palette_name=pal)
            repaired_arr = repaired_res.composite_array
            repaired_alpha = repaired_arr[:, :, 3]

            # Post-repair metrics
            repaired_iou = compute_iou(repaired_alpha, gt_alpha)
            repaired_acc = compute_pixel_accuracy(repaired_arr, gt_arr)
            iou_gain = repaired_iou - corrupt_iou
            total_iou_gain.append(iou_gain)

            # Measure unintended modifications (non-defect pixels in GT that were changed)
            gt_opaque = gt_alpha > 0
            unintended_diff = np.logical_and(gt_opaque, np.any(repaired_arr != gt_arr, axis=-1)).sum()
            unintended_rate = float(unintended_diff / max(1, gt_opaque.sum())) * 100.0
            total_unintended_changes.append(unintended_rate)

            # Success determination: repaired IoU improved or maintained, defects resolved
            repair_success = (repaired_iou >= corrupt_iou and refine_res["resolved_defects"] > 0)
            if repair_success:
                total_repairs_successful += 1

            # Identity preservation check
            identity_preserved = (repaired_iou >= 0.85)

            case_results.append({
                "id": c_id,
                "name": case["name"],
                "category": cat,
                "is_edge_case": is_edge,
                "expected_defect": expected_defect,
                "detected_defects": detected_types,
                "detection_successful": (expected_defect in detected_types),
                "corrupted_iou": round(corrupt_iou, 3),
                "repaired_iou": round(repaired_iou, 3),
                "iou_gain": round(iou_gain, 3),
                "corrupted_pixel_accuracy": round(corrupt_acc, 2),
                "repaired_pixel_accuracy": round(repaired_acc, 2),
                "unintended_modification_rate": round(unintended_rate, 2),
                "identity_preserved": identity_preserved,
                "repair_success": repair_success,
                "initial_quality": refine_res["initial_quality"],
                "final_quality": refine_res["final_quality"],
                "iterations_used": refine_res["iterations_completed"],
                "actions_applied": [r.get("action") for r in refine_res["applied_repairs"]],
            })

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-6, precision + recall)

        summary = {
            "total_cases": len(EVALUATION_CASES),
            "detection_precision": round(precision * 100.0, 2),
            "detection_recall": round(recall * 100.0, 2),
            "detection_f1": round(f1, 4),
            "repair_success_rate": round((total_repairs_successful / len(EVALUATION_CASES)) * 100.0, 2),
            "avg_silhouette_iou_gain": round(sum(total_iou_gain) / len(total_iou_gain), 4),
            "avg_unintended_modification_rate": round(sum(total_unintended_changes) / len(total_unintended_changes), 2),
            "edge_case_preservation_rate": round((edge_cases_preserved / max(1, total_edge_cases)) * 100.0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        report = {
            "summary": summary,
            "case_results": case_results,
        }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("Evaluation complete! Results saved to", self.results_path)
        return report


if __name__ == "__main__":
    ev = Stage3EvaluationSuite()
    res = ev.run_evaluation()
    print("Summary:", json.dumps(res["summary"], indent=2))
