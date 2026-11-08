"""Generation Pipeline coordinating Neural SVAN and Procedural Engines."""
import base64
import copy
import io
import os
import time
import re
from collections import OrderedDict
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from spritemind.ml.prompt_encoder import encode_prompt
from spritemind.ml.svan import SVANModel
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.ml.procedural import ProceduralBaseline
from spritemind.ml.temporal import TemporalAnimator
from spritemind.dataset.generator import CATEGORIES, STYLES, POSES
from spritemind.dataset.conditioning import parse_conditioning, colour_id, element_id
from spritemind.pixelscript.ast import PixelScriptProgram, PixelAction, ActionType
from spritemind.renderer.engine import PixelScriptEngine, RenderResult
from spritemind.renderer.export import array_to_base64
from PIL import Image
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.scaling import scale_script
from spritemind.renderer.lighting import apply_lighting
from spritemind.assets.faithfulness import COLOUR_TARGETS, colour_faithfulness
from spritemind.assets.faithfulness import _matches as _colour_matches
from spritemind.utils.palettes import PALETTES
from spritemind.paths import DEFAULT_CHECKPOINTS, resolve_checkpoints_dir


# Rerank terms. The reference values are what an "average" candidate scores, so a
# candidate that ignores the requested colour or that the critic rejects is pushed
# down, and one that satisfies both is pushed up.
COLOUR_RERANK_WEIGHT = 0.5
COLOUR_RERANK_REFERENCE = 15.0
CRITIC_RERANK_WEIGHT = 0.3
CRITIC_RERANK_REFERENCE = 60.0
# A pixel outside the palette is a hard defect, so a candidate loses this much per
# leaked share of its opaque area. Measured: without it the colour term let 0.18%
# of sprites through with off-palette pixels.
PALETTE_LEAK_PENALTY = 40.0
# R12: the share of pixels drawn in the requested colour, credited up to a target
# and capped there. Saturating means a candidate cannot win by flooding the
# sprite, and 0.0 keeps the shipped ranking byte-identical.
COVERAGE_RERANK_WEIGHT = 0.0
COVERAGE_RERANK_TARGET = 0.22


class SpriteMindPipeline:
    """End-to-end generation, inference, criticism, and rendering pipeline."""

    def __init__(self, checkpoints_dir: str = DEFAULT_CHECKPOINTS):
        # Small tensors lose badly to thread-pool overhead on many-core boxes.
        torch.set_num_threads(int(os.environ.get("SPRITEMIND_THREADS", "2")))
        self.tokenizer = ActionTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints_dir = resolve_checkpoints_dir(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.active_checkpoint_name: Optional[str] = None
        self.trained_model: Optional[SVANModel] = None
        self.checkpoint_error: Optional[str] = None
        self.coverage_rerank_weight = COVERAGE_RERANK_WEIGHT
        self.coverage_rerank_target = COVERAGE_RERANK_TARGET

        self._result_cache: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
        self._result_cache_size = int(os.environ.get("SPRITEMIND_RESULT_CACHE", "64"))
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_stats(self) -> Dict[str, int]:
        return {
            "size": len(self._result_cache),
            "capacity": self._result_cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
        }

    def clear_cache(self) -> None:
        self._result_cache.clear()

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

        ckpt_data = torch.load(ckpt_path, map_location=self.device)
        metadata = ckpt_data.get("metadata", {}) if isinstance(ckpt_data, dict) else {}
        model_config = dict(metadata.get("model_config") or {})
        model_config.setdefault("vocab_size", self.tokenizer.vocab_size)
        saved_vocab = int(model_config["vocab_size"])
        if saved_vocab != self.tokenizer.vocab_size:
            # A stale checkpoint would emit token ids the current vocabulary does
            # not cover, so refuse it loudly and let the caller fall back.
            self.checkpoint_error = (
                f"{checkpoint_name} was trained with vocab {saved_vocab}, "
                f"the tokenizer has {self.tokenizer.vocab_size}; retrain it"
            )
            return False
        model = SVANModel.from_config(model_config)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state_dict"])
            self.checkpoint_metadata = metadata
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

        conditioning = parse_conditioning(prompt)

        return {
            "category": detected_cat,
            "style": detected_style,
            "pose": detected_pose,
            "dimension": dim,
            "palette": palette,
            "colour": conditioning.get("colour"),
            "element": conditioning.get("element"),
            "weapon": conditioning.get("weapon"),
            "accessory": conditioning.get("accessory"),
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
        best_of: int = 4,
        temperature: float = 0.9,
        top_k: int = 0,
        use_critic: bool = True,
        colour_steer: float = 0.0,
        shade: bool = True,
    ) -> Dict[str, Any]:
        """Generates a sprite, replaying a previous identical request from cache."""
        key = (
            prompt.strip(),
            category,
            style,
            pose,
            palette_name,
            width,
            height,
            num_frames,
            engine_mode,
            seed,
            max_actions,
            best_of,
            round(float(temperature), 3),
            int(top_k),
            bool(use_critic),
            round(float(colour_steer), 3),
            bool(shade),
        )
        if self._result_cache_size <= 0:
            result = self._generate_uncached(
                prompt=prompt,
                category=category,
                style=style,
                pose=pose,
                palette_name=palette_name,
                width=width,
                height=height,
                num_frames=num_frames,
                engine_mode=engine_mode,
                seed=seed,
                max_actions=max_actions,
                best_of=best_of,
                use_critic=use_critic,
                colour_steer=colour_steer,
                shade=shade,
            )
            result["cache_hit"] = False
            return result

        start_time = time.perf_counter()
        cached = self._result_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._result_cache.move_to_end(key)
            result = copy.deepcopy(cached)
            result["cache_hit"] = True
            result["inference_time_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return result

        self.cache_misses += 1
        result = self._generate_uncached(
            prompt=prompt,
            category=category,
            style=style,
            pose=pose,
            palette_name=palette_name,
            width=width,
            height=height,
            num_frames=num_frames,
            engine_mode=engine_mode,
            seed=seed,
            max_actions=max_actions,
            best_of=best_of,
            temperature=temperature,
            top_k=top_k,
            use_critic=use_critic,
            shade=shade,
        )
        result["cache_hit"] = False
        self._result_cache[key] = copy.deepcopy(result)
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self._result_cache_size:
            self._result_cache.popitem(last=False)
        return result

    def _generate_uncached(
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
        best_of: int = 4,
        temperature: float = 0.9,
        top_k: int = 0,
        procedural_guard: bool = True,
        use_critic: bool = True,
        colour_steer: float = 0.0,
        shade: bool = True,
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
        final_colour = parsed.get("colour")
        final_element = parsed.get("element")
        final_weapon = parsed.get("weapon")
        final_accessory = parsed.get("accessory")

        frames_data: List[Dict[str, Any]] = []
        candidate_source = "procedural_baseline"
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
            if self.trained_model is None and actual_engine == "trained_model":
                # The checkpoint exists but cannot run with this tokenizer's vocabulary.
                actual_engine = "procedural_baseline"

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
                colour=final_colour,
                element=final_element,
                weapon=final_weapon,
                accessory=final_accessory,
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
                    colour=final_colour,
                    element=final_element,
                    weapon=final_weapon,
                    accessory=final_accessory,
                )
            else:
                model = self.trained_model if actual_engine == "trained_model" else SVANModel(self.tokenizer.vocab_size).to(self.device)
                model.eval()
                prog, candidate_source = self._neural_generate(
                    model=model,
                    category=final_cat,
                    style=final_style,
                    pose=final_pose,
                    palette=final_palette,
                    max_actions=max_actions,
                    seed=seed,
                    width=final_dim,
                    height=final_dim,
                    best_of=best_of,
                    temperature=temperature,
                    top_k=top_k,
                    procedural_guard=procedural_guard and actual_engine == "trained_model",
                    prompt_text=prompt,
                    colour=final_colour,
                    element=final_element,
                    use_critic=use_critic,
                    colour_steer=colour_steer,
                )
            prog_frames = [prog]
            res = PixelScriptEngine.execute(prog, width=final_dim, height=final_dim, palette_name=final_palette)
            f_dict = res.to_dict(include_passes=True)
            f_dict["frame_index"] = 0
            f_dict["script_text"] = PixelScriptSerializer.serialize_program(prog)
            frames_data = [f_dict]

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        frames_data = [self._upscale_frame(f, final_dim, width, height, shade=shade) for f in frames_data]
        out_scale = max(1, min(int(width), int(height)) // max(1, int(final_dim)))

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
            "colour": final_colour,
            "element": final_element,
            "dimensions": {"width": final_dim * out_scale, "height": final_dim * out_scale},
            "num_frames": len(frames_data),
            "frames": frames_data,
            "critic": critic_analysis,
            "candidate_source": candidate_source,
            "rerank": {
                "colour_weight": COLOUR_RERANK_WEIGHT,
                "critic_weight": CRITIC_RERANK_WEIGHT if use_critic else 0.0,
                "critic_used": bool(use_critic),
            },
            "checkpoint_error": self.checkpoint_error,
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
        width: int = 32,
        height: int = 32,
        best_of: int = 1,
        temperature: float = 0.9,
        top_k: int = 0,
        procedural_guard: bool = False,
        prompt_text: str = "",
        colour: Optional[str] = None,
        element: Optional[str] = None,
        use_critic: bool = True,
        use_colour: bool = True,
        colour_steer: float = 0.0,
    ) -> Tuple[PixelScriptProgram, str]:
        """Runs autoregressive token generation with SVAN transformer, optionally best-of-N reranked.

        With `procedural_guard` the procedural baseline enters the same rerank as
        one more candidate: it wins only when its score beats every sampled
        program, and the caller reports that through `candidate_source`.

        Candidates are ranked on the rendered pixels: the structural heuristic plus
        the share of pixels drawn in a colour the prompt asked for plus the visual
        critic's quality score. `use_critic=False` disables the last term so the
        critic's contribution can be measured on its own.
        """
        n = max(1, best_of)
        candidates = (
            self._sample_batch(
                model, category, style, pose, max_actions, seed, temperature, n, top_k,
                prompt_text, colour, element, palette=palette, colour_steer=colour_steer,
            )
            if n > 1
            else [
                self._sample_tokens(
                    model, category, style, pose, max_actions, seed, temperature, top_k,
                    prompt_text, colour, element, palette=palette, colour_steer=colour_steer,
                )
            ]
        )

        rank = self._rerank_score if (n > 1 or procedural_guard) else None
        best_tokens: Optional[List[int]] = None
        best_score = float("-inf")
        for tokens in candidates:
            actions_cand = self.tokenizer.decode_tokens(tokens)
            prog_cand = PixelScriptProgram(actions=actions_cand)
            score = (
                rank(
                    prog_cand, palette, width, height,
                    category=category, colour=colour, prompt_text=prompt_text,
                    use_critic=use_critic, use_colour=use_colour,
                )
                if rank is not None
                else 0.0
            )
            if score > best_score:
                best_score, best_tokens = score, tokens

        actions = self.tokenizer.decode_tokens(best_tokens or [self.tokenizer.sos_id])
        meta = {"category": category, "style": style, "pose": pose, "palette": palette, "generator": "neural_svan"}
        best_program = PixelScriptProgram(actions=actions, metadata=meta)

        if not procedural_guard:
            return best_program, "neural"

        guard_program = ProceduralBaseline.generate(
            category=category,
            style=style,
            pose=pose,
            palette_name=palette,
            width=width,
            height=height,
            seed=seed,
            colour=colour,
            element=element,
        )
        guard_score = rank(
            guard_program, palette, width, height,
            category=category, colour=colour, prompt_text=prompt_text,
            use_critic=use_critic, use_colour=use_colour,
        )
        if guard_score > best_score:
            return guard_program, "procedural_guard"
        return best_program, "neural"

    def _colour_steer_ids(self, palette: str, colour: Optional[str]) -> List[int]:
        """Token ids of palette entries that match a requested colour name."""
        if not colour:
            return []
        target = COLOUR_TARGETS.get(str(colour).lower())
        entries = PALETTES.get(palette)
        if target is None or not entries:
            return []
        ids = []
        for index, rgba in enumerate(entries):
            token = self.tokenizer.token_to_id.get(f"COL_{index}")
            if token is not None and index and _colour_matches(*rgba[:3], target):
                ids.append(token)
        return ids

    def _render_candidate(
        self, prog: PixelScriptProgram, palette: str, width: int = 32, height: int = 32
    ) -> Optional[RenderResult]:
        try:
            return PixelScriptEngine.execute(prog, width=width, height=height, palette_name=palette)
        except Exception:
            return None

    @staticmethod
    def _upscale_frame(
        frame: Dict[str, Any], design_dim: int, width: int, height: int, shade: bool = True
    ) -> Dict[str, Any]:
        """Lightens the rim, then enlarges a design-space frame by repeating pixels.

        Programs are authored at 32 px, so an enlarged frame is the same art at a
        larger canvas. Redrawing a scaled script instead would thin the one-pixel
        strokes and open seams between parts.
        """
        scale = max(1, min(int(width), int(height)) // max(1, int(design_dim)))
        if scale <= 1 and not shade:
            return frame

        def load(data_uri: str) -> Image.Image:
            payload = data_uri.split(",", 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA")

        def dump(image: Image.Image) -> str:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

        def enlarge(data_uri: str, light: bool) -> str:
            image = load(data_uri)
            if light:
                image = Image.fromarray(apply_lighting(np.array(image), None), mode="RGBA")
            if scale > 1:
                image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
            return dump(image)

        out = dict(frame)
        out["image_base64"] = enlarge(frame["image_base64"], shade)
        out["passes"] = {k: enlarge(v, False) for k, v in (frame.get("passes") or {}).items()}
        out["width"], out["height"] = int(width), int(height)
        if frame.get("script_text"):
            out["script_text"] = scale_script(frame["script_text"], scale)
        out["pixel_count"] = int(frame.get("pixel_count", 0)) * scale * scale
        return out

    def _score_candidate(self, prog: PixelScriptProgram, palette: str, width: int = 32, height: int = 32) -> float:
        """Structural-only score. Kept for callers that do not want the colour/critic terms."""
        res = self._render_candidate(prog, palette, width, height)
        if res is None:
            return -100.0
        return self._heuristic_score(prog, res, width, height)

    def _rerank_score(
        self,
        prog: PixelScriptProgram,
        palette: str,
        width: int = 32,
        height: int = 32,
        category: str = "knight",
        colour: Optional[str] = None,
        prompt_text: str = "",
        use_critic: bool = True,
        use_colour: bool = True,
    ) -> float:
        """Structural score plus how much of the sprite is in a requested colour plus the critic's verdict.

        Both extra terms are computed from the rendered pixels, so a candidate is
        never rewarded for naming a colour it did not draw or for a critic score it
        did not earn.
        """
        comp = self._rerank_components(
            prog, palette, width, height,
            category=category, colour=colour, prompt_text=prompt_text,
            use_critic=use_critic, use_colour=use_colour,
        )
        if comp is None:
            return -100.0
        return self._score_components(comp)

    def _rerank_components(
        self,
        prog: PixelScriptProgram,
        palette: str,
        width: int = 32,
        height: int = 32,
        category: str = "knight",
        colour: Optional[str] = None,
        prompt_text: str = "",
        use_critic: bool = True,
        use_colour: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """The measured parts of a candidate, so ranking weights can be varied without re-rendering."""
        res = self._render_candidate(prog, palette, width, height)
        if res is None:
            return None
        comp: Dict[str, Any] = {
            "render": res,
            "heuristic": self._heuristic_score(prog, res, width, height),
            "leak": self._palette_leak_share(res.composite_array, palette) or 0.0,
            "colour_score": None,
            "coverage": None,
            "critic": None,
            "critic_defects": None,
        }
        requested = (colour or prompt_text) if use_colour else ""
        if requested:
            faith = colour_faithfulness(requested, res.composite_array)
            if faith is not None:
                comp["colour_score"] = float(faith["score"])
                comp["coverage"] = float(faith.get("matched_share") or 0.0)
        if use_critic:
            report = self._run_critic(res, prog, category=category, palette_name=palette)
            quality = report.get("quality_score")
            if isinstance(quality, (int, float)):
                comp["critic"] = float(quality)
            defects = report.get("defects")
            comp["critic_defects"] = int(report.get("defects_count", len(defects) if defects else 0))
        return comp

    def _score_components(self, comp: Dict[str, Any]) -> float:
        """Ranks an already-measured candidate; the only place ranking weights are applied."""
        score = float(comp["heuristic"]) - PALETTE_LEAK_PENALTY * float(comp["leak"])
        if comp["colour_score"] is not None:
            score += COLOUR_RERANK_WEIGHT * (float(comp["colour_score"]) - COLOUR_RERANK_REFERENCE)
            if self.coverage_rerank_weight > 0.0 and comp["coverage"] is not None:
                shortfall = max(0.0, self.coverage_rerank_target - float(comp["coverage"]))
                score -= self.coverage_rerank_weight * shortfall
        if comp["critic"] is not None:
            score += CRITIC_RERANK_WEIGHT * (float(comp["critic"]) - CRITIC_RERANK_REFERENCE)
        return score

    def _palette_leak_share(self, rgba: np.ndarray, palette: str) -> Optional[float]:
        """Share of opaque pixels whose RGB is not in the palette; None for an unknown palette."""
        colours = PALETTES.get(palette)
        if colours is None:
            return None
        opaque = rgba[:, :, 3] > 0
        total = int(opaque.sum())
        if total == 0:
            return 0.0
        allowed = {tuple(int(c) for c in rgb[:3]) for rgb in colours}
        pixels = rgba[opaque][:, :3]
        inside = sum(1 for pixel in pixels.tolist() if tuple(pixel) in allowed)
        return (total - inside) / float(total)

    def _heuristic_score(
        self, prog: PixelScriptProgram, res: RenderResult, width: int = 32, height: int = 32
    ) -> float:
        """Structural score over an already-rendered candidate."""
        from spritemind.pixelscript.validator import PixelScriptValidator

        if not prog.actions:
            return -100.0
        report = PixelScriptValidator.validate(prog, canvas_width=width, canvas_height=height)
        score = 0.0 if report.is_valid else -20.0
        alpha = res.composite_array[..., 3] > 0

        coverage = float(alpha.mean())
        if coverage < 0.06:
            score -= 30.0
        elif coverage > 0.60:
            score -= 10.0
        else:
            score += 20.0 * (1.0 - abs(coverage - 0.22) / 0.22)

        col_mass = alpha.sum(axis=0).astype(float)
        total = col_mass.sum()
        if total > 0:
            center = (col_mass * np.arange(len(col_mass))).sum() / total
            score -= abs(center - (width - 1) / 2.0) * 0.4
            mirrored = col_mass[::-1]
            symmetry = 1.0 - np.abs(col_mass - mirrored).sum() / (2.0 * total)
            score += 8.0 * symmetry

        parts = {a.params.get("name") for a in prog.actions if a.action_type.value == "PART"}
        score += min(len(parts), 8) * 1.5
        if not any(a.action_type.value == "END" for a in prog.actions):
            score -= 10.0
        return score

    def _palette_index(self, model: SVANModel, palette: str) -> int:
        """Index into the palette embedding, or 0 when the checkpoint never saw a palette label.

        The shipped SVAN checkpoints were trained with `palette_idx == 0` for every
        sample, so rows 1..N are untrained noise. Feeding them would change output
        without carrying information, so they stay unused until a checkpoint is
        trained on real palette labels and sets `palette_conditioning_trained`.
        """
        if not getattr(model, "palette_conditioning_trained", False):
            return 0
        names = list(PALETTES)
        if palette not in names:
            return 0
        limit = model.semantic_encoder.palette_embed.num_embeddings
        return min(names.index(palette), limit - 1)

    def _prompt_tensor(self, model: SVANModel, prompt_text: str) -> Optional[torch.Tensor]:
        """Character ids for the request, or None when the checkpoint has no text encoder."""
        if getattr(model, "prompt_encoder", None) is None:
            return None
        ids = encode_prompt(prompt_text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _conditioning_kwargs(self, colour: Optional[str], element: Optional[str]) -> Dict[str, torch.Tensor]:
        """Colour/element ids for the semantic encoder; skipped when the realm is unknown."""
        kwargs: Dict[str, torch.Tensor] = {}
        if colour:
            kwargs["colour_idx"] = torch.tensor([colour_id(colour)], device=self.device)
        if element:
            kwargs["element_idx"] = torch.tensor([element_id(element)], device=self.device)
        return kwargs

    def _sample_tokens(
        self,
        model: SVANModel,
        category: str,
        style: str,
        pose: str,
        max_actions: int,
        seed: int,
        temperature: float = 0.9,
        top_k: int = 0,
        prompt_text: str = "",
        colour: Optional[str] = None,
        element: Optional[str] = None,
        palette: str = "pico8",
        colour_steer: float = 0.0,
    ) -> List[int]:
        """Sampling pass for one candidate program."""
        torch.manual_seed(seed)
        cat_idx = torch.tensor([CATEGORIES.index(category) if category in CATEGORIES else 0], device=self.device)
        style_idx = torch.tensor([STYLES.index(style) if style in STYLES else 0], device=self.device)
        pose_idx = torch.tensor([POSES.index(pose) if pose in POSES else 0], device=self.device)
        pal_idx = torch.tensor([self._palette_index(model, palette)], device=self.device)

        with torch.inference_mode():
            memory = model.memory_tokens(
                cat_idx,
                style_idx,
                pose_idx,
                pal_idx,
                self._prompt_tensor(model, prompt_text),
                **self._conditioning_kwargs(colour, element),
            )

            curr_tokens = [self.tokenizer.sos_id]
            grammar_state = self.tokenizer.grammar_start_state()
            transformer = model.action_transformer
            max_tokens = min(transformer.max_seq_len - 1, max_actions * 8)
            use_cache = hasattr(transformer, "forward_step")
            cache = transformer.new_cache(max_len=max_tokens + 1) if use_cache else None

            steer_ids = self._colour_steer_ids(palette, colour) if colour_steer > 0.0 else []
            for step in range(max_tokens):
                if use_cache:
                    token_tensor = torch.tensor([[curr_tokens[-1]]], dtype=torch.long, device=self.device)
                    logits = transformer.forward_step(token_tensor, memory, cache, step)
                else:
                    tgt_tensor = torch.tensor([curr_tokens], dtype=torch.long, device=self.device)
                    logits = transformer(tgt_tensor, memory)
                next_token_logits = logits[0, -1, :]

                # Grammar mask: only tokens legal for the current action position.
                allowed = self.tokenizer.allowed_from_state(grammar_state)
                if allowed is not None:
                    legal = [t for t in allowed if t < next_token_logits.numel()]
                    masked = torch.full_like(next_token_logits, float("-inf"))
                    masked[legal] = 0.0
                    next_token_logits = next_token_logits + masked
                if steer_ids:
                    next_token_logits[steer_ids] += colour_steer

                # Sample next token
                probs = torch.softmax(next_token_logits / max(1e-5, temperature), dim=-1)
                if top_k > 0:
                    top = torch.topk(probs, min(top_k, probs.numel()))
                    keep = torch.zeros_like(probs)
                    keep[top.indices] = probs[top.indices]
                    probs = keep
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_tokens.append(next_tok)
                grammar_state = self.tokenizer.advance_grammar(grammar_state, next_tok)

                if next_tok == self.tokenizer.eos_id:
                    break

        return curr_tokens

    def _sample_batch(
        self,
        model: SVANModel,
        category: str,
        style: str,
        pose: str,
        max_actions: int,
        seed: int,
        temperature: float,
        num_candidates: int,
        top_k: int = 0,
        prompt_text: str = "",
        colour: Optional[str] = None,
        element: Optional[str] = None,
        palette: str = "pico8",
        colour_steer: float = 0.0,
    ) -> List[List[int]]:
        """Samples several candidate programs in one batched decode pass.

        All candidates share the same conditioning memory, so running them as a
        batch costs barely more than a single pass and replaces N sequential
        sampling loops.
        """
        torch.manual_seed(seed)
        cat_idx = torch.tensor([CATEGORIES.index(category) if category in CATEGORIES else 0], device=self.device)
        style_idx = torch.tensor([STYLES.index(style) if style in STYLES else 0], device=self.device)
        pose_idx = torch.tensor([POSES.index(pose) if pose in POSES else 0], device=self.device)
        pal_idx = torch.tensor([self._palette_index(model, palette)], device=self.device)

        transformer = model.action_transformer
        max_tokens = min(transformer.max_seq_len - 1, max_actions * 8)
        vocab_size = transformer.head.out_features
        sos, eos, pad = self.tokenizer.sos_id, self.tokenizer.eos_id, self.tokenizer.pad_id

        with torch.inference_mode():
            sem_cond = model.semantic_encoder(
                cat_idx, style_idx, pose_idx, pal_idx,
                colour_idx=torch.tensor([colour_id(colour)], device=self.device) if colour else None,
                element_idx=torch.tensor([element_id(element)], device=self.device) if element else None,
            )
            memory = torch.stack(
                [sem_cond, torch.zeros_like(sem_cond), torch.zeros_like(sem_cond)]
                + ([model.prompt_encoder(self._prompt_tensor(model, prompt_text))] if model.prompt_encoder is not None else []),
                dim=1
            ).expand(num_candidates, -1, -1)
            cache = transformer.new_cache(max_len=max_tokens + 1, batch_size=num_candidates)

            tokens: List[List[int]] = [[sos] for _ in range(num_candidates)]
            finished = [False] * num_candidates
            states = [self.tokenizer.grammar_start_state() for _ in range(num_candidates)]

            steer_ids = self._colour_steer_ids(palette, colour) if colour_steer > 0.0 else []
            for step in range(max_tokens):
                if all(finished):
                    break
                last = torch.tensor(
                    [[seq[-1]] for seq in tokens], dtype=torch.long, device=self.device
                )
                logits = transformer.forward_step(last, memory, cache, step)[:, -1, :]

                mask = torch.zeros_like(logits)
                for i, _seq in enumerate(tokens):
                    if finished[i]:
                        continue
                    allowed = self.tokenizer.allowed_from_state(states[i])
                    if allowed is not None:
                        legal = [t for t in allowed if t < vocab_size]
                        row = torch.full((vocab_size,), float("-inf"), device=self.device)
                        row[torch.tensor(legal, device=self.device)] = 0.0
                        mask[i] = row
                adjusted = logits + mask
                if steer_ids:
                    adjusted[:, steer_ids] += colour_steer
                probs = torch.softmax(adjusted / max(1e-5, temperature), dim=-1)
                if top_k > 0:
                    cutoff = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1).values[:, -1:]
                    probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1).tolist()

                for i, tok in enumerate(next_tokens):
                    if finished[i]:
                        tokens[i].append(pad)
                        continue
                    tokens[i].append(tok)
                    states[i] = self.tokenizer.advance_grammar(states[i], tok)
                    if tok == eos:
                        finished[i] = True

        return [seq for seq in tokens]

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
