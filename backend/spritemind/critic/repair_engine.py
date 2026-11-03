"""Autonomous Self-Correction & Closed-Loop Refinement Engine for SpriteMind.

Coordinates visual inspection, defect detection, repair planning, PixelScript transformation,
iterative re-rendering, and identity preservation guards.
"""
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from spritemind.critic.model import NeuralVisualCritic, DEFECT_CLASSES
from spritemind.critic.visual_understanding import VisualUnderstandingPipeline
from spritemind.renderer.engine import PixelScriptEngine, RenderResult
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.ast import PixelScriptProgram, PixelAction, ActionType


class RepairAction:
    """Represents a planned surgical modification to a PixelScript program."""

    def __init__(
        self,
        defect_id: str,
        defect_type: str,
        severity: str,
        description: str,
        pixelscript_patch: str,
        estimated_quality_gain: float = 10.0,
    ):
        self.defect_id = defect_id
        self.defect_type = defect_type
        self.severity = severity
        self.description = description
        self.pixelscript_patch = pixelscript_patch
        self.estimated_quality_gain = estimated_quality_gain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "description": self.description,
            "pixelscript_patch": self.pixelscript_patch,
            "estimated_quality_gain": self.estimated_quality_gain,
        }


class AutonomousRepairEngine:
    """Closed-loop visual critic and self-correction engine."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_critic: Optional[NeuralVisualCritic] = None
        self.visual_understanding = VisualUnderstandingPipeline()
        self._load_neural_critic()

    def _load_neural_critic(self):
        ckpt_path = self.checkpoints_dir / "visual_critic_best.pt"
        if ckpt_path.exists():
            try:
                model = NeuralVisualCritic().to(self.device)
                loaded = torch.load(ckpt_path, map_location=self.device)
                if isinstance(loaded, dict) and "model_state_dict" in loaded:
                    model.load_state_dict(loaded["model_state_dict"])
                else:
                    model.load_state_dict(loaded)
                model.eval()
                self.neural_critic = model
            except Exception as e:
                print("Could not load neural critic weights:", e)

    def inspect(
        self,
        render_res: RenderResult,
        category: str = "knight",
        palette_name: str = "pico8",
        prog: Optional[PixelScriptProgram] = None,
    ) -> Dict[str, Any]:
        """Runs hybrid inspection combining morphological visual understanding and neural critic."""
        # 1. Morphological / Component Analysis
        report = self.visual_understanding.analyze(render_res, category=category, palette_name=palette_name, prog=prog)

        # 2. Neural Critic Inference (if model loaded)
        neural_predictions: Dict[str, Any] = {}
        if self.neural_critic:
            try:
                comp = render_res.composite_array
                tensor_img = torch.from_numpy(
                    comp.transpose(2, 0, 1).astype(np.float32) / 255.0
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    n_out = self.neural_critic(tensor_img)
                    logits = n_out["defect_logits"][0]
                    probs = torch.softmax(logits, dim=-1)
                    pred_class_idx = int(torch.argmax(probs).item())
                    pred_class = DEFECT_CLASSES[pred_class_idx]
                    confidence = float(probs[pred_class_idx].item())

                    q_score = float(n_out["quality_score"][0, 0].item())
                    heatmap = n_out["heatmap"][0, 0].cpu().numpy().tolist()

                    neural_predictions = {
                        "predicted_class": pred_class,
                        "confidence": round(confidence, 3),
                        "quality_score": round(q_score, 1),
                        "has_defect": (pred_class != "clean"),
                        "heatmap_peak": [
                            int(np.unravel_index(np.argmax(n_out["heatmap"][0, 0].cpu().numpy()), (32, 32))[1]),
                            int(np.unravel_index(np.argmax(n_out["heatmap"][0, 0].cpu().numpy()), (32, 32))[0]),
                        ],
                    }

                    # Blend quality score with morphological score
                    report["quality_score"] = int(round(0.6 * report["quality_score"] + 0.4 * q_score))
            except Exception as e:
                neural_predictions = {"error": str(e)}

        report["neural_critic"] = neural_predictions

        # Formulate repair plans
        repair_actions = self.plan_repairs(report.get("detected_defects", []), prog)
        report["repair_plans"] = [a.to_dict() for a in repair_actions]

        return report

    def plan_repairs(
        self,
        defects: List[Dict[str, Any]],
        prog: Optional[PixelScriptProgram] = None,
    ) -> List[RepairAction]:
        """Prioritizes and synthesizes valid repair operations for detected defects."""
        repairs = []
        # Sort defects by type priority first, then severity
        type_priority = {
            "disconnected_part": 0,
            "stray_pixel": 1,
            "unexpected_hole": 2,
            "palette_leak": 3,
            "missing_limb": 4,
            "incorrect_proportions": 5,
        }
        severity_order = {"critical": 0, "moderate": 1, "minor": 2}
        sorted_defects = sorted(
            defects,
            key=lambda d: (
                type_priority.get(d.get("type"), 99),
                severity_order.get(d.get("severity", "minor"), 2),
            )
        )

        for d in sorted_defects:
            d_type = d.get("type")
            d_id = d.get("id", "def_0")
            fix = d.get("suggested_fix")
            sev = d.get("severity", "minor")

            if fix:
                repairs.append(RepairAction(
                    defect_id=d_id,
                    defect_type=d_type,
                    severity=sev,
                    description=d.get("description", f"Repair for {d_type}"),
                    pixelscript_patch=fix,
                    estimated_quality_gain=25.0 if sev == "critical" else (15.0 if sev == "moderate" else 8.0),
                ))

        return repairs

    def apply_patch_to_script(self, script_text: str, patch_cmd: str) -> str:
        """Applies a PixelScript repair command to an existing script."""
        patch_cmd = patch_cmd.strip()
        if "\n" in patch_cmd:
            cur = script_text
            for sub_cmd in patch_cmd.splitlines():
                if sub_cmd.strip():
                    cur = self.apply_patch_to_script(cur, sub_cmd.strip())
            return cur

        lines = [ln.strip() for ln in script_text.splitlines() if ln.strip()]

        if patch_cmd.startswith("ERASE"):
            # Appending ERASE overrides previously painted pixels at that coordinate
            lines.append(patch_cmd)
        elif patch_cmd.startswith("TRANSFORM"):
            # If an existing TRANSFORM for this part exists, replace it; else append before END
            parts = patch_cmd.split()
            if len(parts) >= 2:
                target_part = parts[1]
                found_existing = False
                for i, ln in enumerate(lines):
                    if ln.startswith(f"TRANSFORM {target_part} "):
                        lines[i] = patch_cmd
                        found_existing = True
                        break
                if not found_existing:
                    lines.append(patch_cmd)
        elif patch_cmd.startswith("PIXEL") or patch_cmd.startswith("RECT"):
            lines.append(patch_cmd)
        elif patch_cmd.startswith("PART"):
            # Insert part definition
            lines.append(patch_cmd)
        else:
            lines.append(patch_cmd)

        return "\n".join(lines)

    def refine_closed_loop(
        self,
        script_text: str,
        category: str = "knight",
        palette_name: str = "pico8",
        width: int = 32,
        height: int = 32,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Executes full closed-loop iterative self-correction with identity guards."""
        start_time = time.perf_counter()

        # Iteration 0: Baseline state
        curr_script = script_text
        curr_res = PixelScriptEngine.execute(curr_script, width=width, height=height, palette_name=palette_name)
        curr_prog = PixelScriptParser.parse_text(curr_script)
        initial_pixels = curr_res.composite_array[:, :, 3]
        initial_pixel_count = int(np.count_nonzero(initial_pixels > 0))

        initial_report = self.inspect(curr_res, category=category, palette_name=palette_name, prog=curr_prog)

        history = [{
            "iteration": 0,
            "script_text": curr_script,
            "image_base64": curr_res.to_dict()["image_base64"],
            "quality_score": initial_report["quality_score"],
            "defects_count": initial_report["defects_count"],
            "defects": initial_report["detected_defects"],
            "pixels_count": initial_pixel_count,
            "action_applied": "Initial State",
        }]

        applied_actions_log = []
        applied_patches = set()
        best_script = curr_script
        best_quality = initial_report["quality_score"]
        best_res = curr_res
        best_report = initial_report

        for it in range(1, max_iterations + 1):
            if best_report["defects_count"] == 0 or best_quality >= 96:
                break

            plans = best_report.get("repair_plans", [])
            if not plans:
                break

            # Find top prioritized repair plan that hasn't been tried yet
            selected_repair = None
            for p in plans:
                patch = p.get("pixelscript_patch", "").strip()
                if patch and patch not in applied_patches:
                    selected_repair = p
                    break

            if not selected_repair:
                break

            applied_patches.add(selected_repair["pixelscript_patch"].strip())
            candidate_script = self.apply_patch_to_script(best_script, selected_repair["pixelscript_patch"])

            # Re-render candidate
            cand_res = PixelScriptEngine.execute(candidate_script, width=width, height=height, palette_name=palette_name)
            cand_prog = PixelScriptParser.parse_text(candidate_script)
            cand_report = self.inspect(cand_res, category=category, palette_name=palette_name, prog=cand_prog)

            cand_pixels = cand_res.composite_array[:, :, 3]
            cand_pixel_count = int(np.count_nonzero(cand_pixels > 0))

            # Identity Preservation Guard:
            # Reject change if pixel count diverges by > 20%
            pixel_diff = abs(cand_pixel_count - initial_pixel_count)
            max_allowed_pixel_change = max(10, int(initial_pixel_count * 0.20))

            if pixel_diff > max_allowed_pixel_change:
                # Destructive repair rejected!
                applied_actions_log.append({
                    "iteration": it,
                    "action": selected_repair["pixelscript_patch"],
                    "status": "rejected_identity_guard",
                    "reason": f"Pixel count divergence ({pixel_diff}px > {max_allowed_pixel_change}px limit)",
                })
                continue

            # Monotonic Quality Guard:
            # Accept if defects count decreased or quality score increased
            if cand_report["defects_count"] <= best_report["defects_count"] or cand_report["quality_score"] >= best_quality:
                best_script = candidate_script
                best_quality = cand_report["quality_score"]
                best_res = cand_res
                best_report = cand_report

                applied_actions_log.append({
                    "iteration": it,
                    "action": selected_repair["pixelscript_patch"],
                    "description": selected_repair["description"],
                    "quality_before": history[-1]["quality_score"],
                    "quality_after": best_quality,
                    "defects_before": history[-1]["defects_count"],
                    "defects_after": best_report["defects_count"],
                    "status": "applied",
                })

                history.append({
                    "iteration": it,
                    "script_text": best_script,
                    "image_base64": best_res.to_dict()["image_base64"],
                    "quality_score": best_quality,
                    "defects_count": best_report["defects_count"],
                    "defects": best_report["detected_defects"],
                    "pixels_count": cand_pixel_count,
                    "action_applied": selected_repair["pixelscript_patch"],
                })
            else:
                applied_actions_log.append({
                    "iteration": it,
                    "action": selected_repair["pixelscript_patch"],
                    "status": "rejected_quality_guard",
                    "reason": f"Quality degraded from {best_quality} to {cand_report['quality_score']}",
                })
                continue

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        final_frame = best_res.to_dict(include_passes=True)
        final_frame["script_text"] = best_script

        return {
            "initial_quality": initial_report["quality_score"],
            "final_quality": best_quality,
            "quality_gain": best_quality - initial_report["quality_score"],
            "initial_defects": initial_report["defects_count"],
            "final_defects": best_report["defects_count"],
            "resolved_defects": max(0, initial_report["defects_count"] - best_report["defects_count"]),
            "iterations_completed": len(history) - 1,
            "applied_repairs": applied_actions_log,
            "history": history,
            "final_frame": final_frame,
            "final_critic": best_report,
            "elapsed_ms": round(total_time_ms, 2),
        }
