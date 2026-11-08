"""Golden-image regression helpers for the renderer and grammar.

One reference sprite per category is rendered through the procedural pipeline;
its perceptual hash lives beside the PNG. Aggregate metrics can stay flat while
the pixels change, so this is the check that notices.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from spritemind.dataset.generator import CATEGORIES
from spritemind.ml.pipeline import SpriteMindPipeline

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "backend" / "tests" / "goldens"
HASH_FILE = GOLDEN_DIR / "golden_hashes.json"
SEED = 20240101
STYLE = "classic_16bit"
POSE = "idle"
PALETTE = "pico8"
HASH_BITS = 64
HAMMING_TOLERANCE = 0
PIXEL_TOLERANCE = 0


def perceptual_hash(rgba: np.ndarray) -> str:
    """64-bit dHash over luminance, flattened onto white so transparency counts."""
    image = Image.fromarray(rgba, mode="RGBA")
    flat = Image.new("RGB", image.size, (255, 255, 255))
    flat.paste(image, mask=image.split()[3])
    grey = flat.convert("L").resize((9, 8), Image.Resampling.BOX)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return "".join("1" if bit else "0" for bit in bits)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def decode_png(image_base64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[-1]))).convert("RGBA"))


def render(category: str, prompt: str, pipeline: SpriteMindPipeline | None = None) -> np.ndarray:
    pipeline = pipeline or SpriteMindPipeline()
    result = pipeline.generate(
        prompt=prompt,
        category=category,
        style=STYLE,
        pose=POSE,
        palette_name=PALETTE,
        engine_mode="procedural_baseline",
        shade=False,
        seed=SEED,
    )
    return decode_png(result["frames"][0]["image_base64"])


def golden_prompt(category: str) -> str:
    # Deliberately free of colour/element words: conditioning words (e.g. "golden")
    # would tint the render and turn this regression check into a moving target.
    return f"reference sprite for {category.replace('_', ' ')}"


def generate_all() -> Dict[str, dict]:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = SpriteMindPipeline()
    hashes: Dict[str, dict] = {}
    for category in CATEGORIES:
        prompt = golden_prompt(category)
        rgba = render(category, prompt, pipeline)
        Image.fromarray(rgba, mode="RGBA").save(GOLDEN_DIR / f"{category}.png")
        hashes[category] = {
            "hash": perceptual_hash(rgba),
            "seed": SEED,
            "style": STYLE,
            "pose": POSE,
            "palette": PALETTE,
            "prompt": prompt,
        }
    HASH_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def differing_pixels(a: np.ndarray, b: np.ndarray) -> int:
    """Visible pixels that differ. A 1px hole is a defect, so this is exact."""
    if a.shape != b.shape:
        return -1
    visible = (a[:, :, 3] > 0) | (b[:, :, 3] > 0)
    return int(((a != b).any(axis=2) & visible).sum())


def verify() -> List[str]:
    if not HASH_FILE.exists():
        return ["no golden hashes committed; run python scripts/golden_images.py --update"]
    expected = json.loads(HASH_FILE.read_text())
    pipeline = SpriteMindPipeline()
    failures: List[str] = []
    for category in CATEGORIES:
        want = expected.get(category)
        if want is None:
            failures.append(f"{category}: no golden recorded")
            continue
        rgba = render(category, want["prompt"], pipeline)
        stored_path = GOLDEN_DIR / f"{category}.png"
        distance = hamming(perceptual_hash(rgba), want["hash"])
        if not stored_path.exists():
            failures.append(f"{category}: golden PNG missing; run python scripts/golden_images.py --update")
            continue
        stored = np.array(Image.open(stored_path).convert("RGBA"))
        changed = differing_pixels(stored, rgba)
        if changed == 0:
            continue
        if changed < 0:
            failures.append(f"{category}: render size changed {stored.shape} -> {rgba.shape}")
        else:
            failures.append(f"{category}: {changed} px differ ({distance} of {HASH_BITS} hash bits)")
    return failures
