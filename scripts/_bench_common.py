"""Shared helpers for the SpriteMind benchmark scripts under `scripts/`.

Holds the reference silhouettes, image decoding and JSON/markdown writing so
each benchmark script stays focused on its own arms and metrics.
"""
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.generator import CATEGORIES, CharacterGenerator  # noqa: E402
from spritemind.ml.metrics import alpha_mask  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402

BENCH_DIR = ROOT / "data" / "benchmarks"
PROMPTS_PATH = BENCH_DIR / "generalization_prompts.json"

_REFERENCE_CACHE: Dict[str, np.ndarray] = {}


def decode_rgba(image_base64: str) -> Optional[np.ndarray]:
    if not image_base64:
        return None
    payload = image_base64.split(",", 1)[-1]
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA"))
    except Exception:
        return None


def canonical_reference(category: str, palette_name: str = "pico8") -> np.ndarray:
    """Silhouette of the category's canonical idle template.

    This is the IoU reference used by every benchmark that reports
    `silhouette_iou`: the same template at proportional size `standard`, pose
    `idle`, seed 0. It is a *fixed* reference, so a higher IoU means the
    generated sprite is closer to the canonical anatomy of its category, not to
    the procedural candidate it is being compared with.
    """
    key = f"{category}:{palette_name}"
    if key not in _REFERENCE_CACHE:
        prog = CharacterGenerator.generate_character(
            category=category, proportion="standard", pose="idle", palette_name=palette_name, seed=0
        )
        script = PixelScriptSerializer.serialize_program(prog)
        res = PixelScriptEngine.execute(script, width=32, height=32, palette_name=palette_name)
        _REFERENCE_CACHE[key] = alpha_mask(res.composite_array)
    return _REFERENCE_CACHE[key]


def load_prompts(limit: Optional[int] = None) -> list:
    prompts = json.loads(PROMPTS_PATH.read_text())
    return prompts[:limit] if limit else prompts


def categories_present(prompts: list) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in prompts:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def markdown_table(rows: list, columns: list) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


assert len(CATEGORIES) == 10
