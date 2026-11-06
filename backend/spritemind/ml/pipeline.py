"""Generation Pipeline coordinating Neural SVAN and Procedural Engines."""
import os
import time
import re
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from spritemind.ml.svan import SVANModel
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.ml.procedural import ProceduralBaseline
from spritemind.ml.temporal import TemporalAnimator
from spritemind.dataset.generator import CATEGORIES, STYLES, POSES
from spritemind.pixelscript.ast import PixelScriptProgram, PixelAction, ActionType
from spritemind.renderer.engine import PixelScriptEngine, RenderResult
from spritemind.renderer.export import array_to_base64
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.parser import PixelScriptParser


class SpriteMindPipeline:
    """End-to-end generation, inference, criticism, and rendering pipeline."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        # Small tensors lose badly to thread-pool overhead on many-core boxes.
        torch.set_num_threads(int(os.environ.get("SPRITEMIND_THREADS", "2")))
        self.tokenizer = ActionTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.active_checkpoint_name: Optional[str] = None
        self.trained_model: Optional[SVANModel] = None

    def get_loaded_model_info(self) -> Dict[str, Any]:
        has_trained = self.trained_model is not None
        param_count = self.trained_model.count_parameters() if has_trained else 0
        return {
            "active_checkpoint": self.active_checkpoint_name,
            "has_trained_model": has_trained,
            "parameter_count": param_count,
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size,
        }

    def load_checkpoint(self, checkpoint_name: str) -> bool:
        ckpt_path = self.checkpoints_dir / checkpoint_name
        if not ckpt_path.exists():
            return False

        model = SVANModel(vocab_size=self.tokenizer.vocab_size, d_model=128)
        ckpt_data = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state_dict"])
            self.checkpoint_metadata = ckpt_data.get("metadata", {})
        else:
            model.load_state_dict(ckpt_data)
            self.checkpoint_metadata = {}

        model.to(self.device)
        model.eval()

        self.trained_model = model
        self.active_checkpoint_name = checkpoint_name
        return True

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parses keywords from natural language prompt into structured parameters."""
        prompt_lower = prompt.lower()

        # Category
        detected_cat = "knight"
        for cat in CATEGORIES:
            if cat in prompt_lower:
                detected_cat = cat
                break

        # Pose
        detected_pose = "idle"
        for pose in POSES:
            if pose in prompt_lower:
                detected_pose = pose
                break

        # Style
        detected_style = "classic_16bit"
        if "8-bit" in prompt_lower or "8bit" in prompt_lower or "retro" in prompt_lower:
            detected_style = "retro_8bit"
        elif "cyberpunk" in prompt_lower or "neon" in prompt_lower:
            detected_style = "cyberpunk"
        elif "dark" in prompt_lower or "fantasy" in prompt_lower:
            detected_style = "dark_fantasy"

        # Dimensions
        dim = 32
        match_dim = re.search(r"(\d+)\s*x\s*(\d+)", prompt_lower)
        if match_dim:
            dim = int(match_dim.group(1))
            if dim not in (16, 32, 64):
                dim = 32

        # Palette
        palette = "pico8"
        if "gameboy" in prompt_lower:
            palette = "gameboy"
        elif "cyberpunk" in prompt_lower:
            palette = "cyberpunk"
        elif "dawnbringer" in prompt_lower:
            palette = "dawnbringer16"

        return {
            "category": detected_cat,
            "style": detected_style,
            "pose": detected_pose,
            "dimension": dim,
            "palette": palette,
        }

    def generate(
        self,
        prompt: str = "",
        category: Optional[str] = None,
        style: Optional[str] = None,
        pose: Optional[str] = None,
        palette_name: Optional[str] = None,
        width: int = 32,
        height: int = 32,
        num_frames: int = 1,
        engine_mode: str = "procedural_baseline",  # "trained_model" | "untrained_model" | "procedural_baseline"
        seed: int = 42,
        max_actions: int = 64,
    ) -> Dict[str, Any]:
        """Executes generation pipeline with explicit backend attribution."""
        start_time = time.perf_counter()

        # Parse prompt if provided
        parsed = self.parse_prompt(prompt) if prompt else {}
        final_cat = category or parsed.get("category", "knight")
        final_style = style or parsed.get("style", "classic_16bit")
        final_pose = pose or parsed.get("pose", "idle")
        final_palette = palette_name or parsed.get("palette", "pico8")
        final_dim = parsed.get("dimension", width)

        frames_data: List[Dict[str, Any]] = []
        actual_engine = engine_mode

        # If user requested trained_model but none is loaded, fall back or handle honestly
        if engine_mode == "trained_model" and self.trained_model is None:
            best_ckpt = self.checkpoints_dir / "svan_checkpoint_best.pt"
            if best_ckpt.exists():
                self.load_checkpoint("svan_checkpoint_best.pt")
            else:
                checkpoints = list(self.checkpoints_dir.glob("*.pt"))
                if checkpoints:
                    self.load_checkpoint(checkpoints[0].name)
                else:
                    # Do NOT silently substitute procedural; switch to untrained model or raise
                    actual_engine = "untrained_model"

        # Generate frames
        if num_frames > 1:
            anim_result = TemporalAnimator.generate_animation(
                category=final_cat,
                style=final_style,
                anim_category=final_pose,
                palette_name=final_palette,
                width=final_dim,
                height=final_dim,
                num_frames=num_frames,
                fps=8,
                engine_mode=actual_engine,
                seed=seed,
            )
            frames_data = anim_result["frames"]
            actual_engine = anim_result["engine_used"]
            first_frame_script = frames_data[0]["script_text"]
            first_prog = PixelScriptParser.parse_text(first_frame_script)
            prog_frames = [first_prog]
        else:
            # Single frame generation
            if actual_engine == "procedural_baseline":
                prog = ProceduralBaseline.generate(
                    category=final_cat,
                    style=final_style,
                    pose=final_pose,
                    palette_name=final_palette,
                    width=final_dim,
                    height=final_dim,
                    seed=seed,
                )
            else:
                model = self.trained_model if actual_engine == "trained_model" else SVANModel(self.tokenizer.vocab_size).to(self.device)
                model.eval()
                prog = self._neural_generate(
                    model=model,
                    category=final_cat,
                    style=final_style,
                    pose=final_pose,
                    palette=final_palette,
                    max_actions=max_actions,
                    seed=seed,
                )
            prog_frames = [prog]
            res = PixelScriptEngine.execute(prog, width=final_dim, height=final_dim, palette_name=final_palette)
            f_dict = res.to_dict(include_passes=True)
            f_dict["frame_index"] = 0
            f_dict["script_text"] = PixelScriptSerializer.serialize_program(prog)
            frames_data = [f_dict]

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Visual Critic analysis on primary frame
        first_frame_res = PixelScriptEngine.execute(prog_frames[0], width=final_dim, height=final_dim, palette_name=final_palette)
        critic_analysis = self._run_critic(
            first_frame_res,
            prog_frames[0],
            category=final_cat,
            palette_name=final_palette,
        )

        return {
            "engine_used": actual_engine,
            "category": final_cat,
            "style": final_style,
            "pose": final_pose,
            "palette": final_palette,
            "dimensions": {"width": final_dim, "height": final_dim},
            "num_frames": len(frames_data),
            "frames": frames_data,
            "critic": critic_analysis,
            "inference_time_ms": round(total_time_ms, 2),
            "seed": seed,
        }

    def _neural_generate(
        self,
        model: SVANModel,
        category: str,
        style: str,
        pose: str,
        palette: str,
        max_actions: int,
        seed: int,
    ) -> PixelScriptProgram:
        """Runs autoregressive token generation with SVAN transformer."""
        torch.manual_seed(seed)
        cat_idx = torch.tensor([CATEGORIES.index(category) if category in CATEGORIES else 0], device=self.device)
        style_idx = torch.tensor([STYLES.index(style) if style in STYLES else 0], device=self.device)
        pose_idx = torch.tensor([POSES.index(pose) if pose in POSES else 0], device=self.device)
        pal_idx = torch.tensor([0], device=self.device)

        with torch.inference_mode():
            sem_cond = model.semantic_encoder(cat_idx, style_idx, pose_idx, pal_idx)
            struct_cond = torch.zeros_like(sem_cond)
            vis_cond = torch.zeros_like(sem_cond)
            memory = torch.stack([sem_cond, struct_cond, vis_cond], dim=1)

            curr_tokens = [self.tokenizer.sos_id]
            max_tokens = min(128, max_actions * 4)
            transformer = model.action_transformer
            use_cache = hasattr(transformer, "forward_step")
            cache = transformer.new_cache() if use_cache else None

            for step in range(max_tokens):
                if use_cache:
                    token_tensor = torch.tensor([[curr_tokens[-1]]], dtype=torch.long, device=self.device)
                    logits = transformer.forward_step(token_tensor, memory, cache, step)
                else:
                    tgt_tensor = torch.tensor([curr_tokens], dtype=torch.long, device=self.device)
                    logits = transformer(tgt_tensor, memory)
                next_token_logits = logits[0, -1, :]

                # Grammar mask: only tokens legal for the current action position.
                allowed = self.tokenizer.allowed_token_ids(curr_tokens)
                if allowed is not None:
                    masked = torch.full_like(next_token_logits, float("-inf"))
                    masked[allowed] = 0.0
                    next_token_logits = next_token_logits + masked

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_tokens.append(next_tok)

                if next_tok == self.tokenizer.eos_id:
                    break

        actions = self.tokenizer.decode_tokens(curr_tokens)
        meta = {"category": category, "style": style, "pose": pose, "palette": palette, "generator": "neural_svan"}
        return PixelScriptProgram(actions=actions, metadata=meta)

    def _run_critic(
        self,
        render_res: RenderResult,
        prog: PixelScriptProgram,
        category: str = "knight",
        palette_name: str = "pico8",
    ) -> Dict[str, Any]:
        """Evaluates rendered sprite through hybrid neural-geometric visual critic."""
        from spritemind.critic.repair_engine import AutonomousRepairEngine
        if not hasattr(self, "_critic_engine"):
            self._critic_engine = AutonomousRepairEngine(checkpoints_dir=str(self.checkpoints_dir))
        return self._critic_engine.inspect(render_res, category=category, palette_name=palette_name, prog=prog)
