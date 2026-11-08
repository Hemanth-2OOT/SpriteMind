"""Measures the learned rig-parameter head against the analytic offsets.

python scripts/eval_rig_params.py

Reports three things and nothing else:
  * held-out error of the head against the authored targets;
  * how much the same action actually changes across prompts;
  * whether frames built from the learned offsets keep the sprite connected.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from spritemind.dataset.generator import CharacterGenerator  # noqa: E402
from spritemind.ml import rig_params  # noqa: E402
from spritemind.pixelscript.coherence import build_frame_script  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402

PROMPTS = [
    "a heavy armored knight with a sword",
    "an agile rogue in a hood",
    "a floating ghost",
    "a zombie warrior",
    "a wizard casting fireballs",
    "a nimble assassin",
]


def islands(rgba: np.ndarray) -> int:
    opaque = rgba[:, :, 3] > 0
    h, w = opaque.shape
    seen = np.zeros_like(opaque)
    count = 0
    for y in range(h):
        for x in range(w):
            if not opaque[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count


def frame_islands(base_script: str, offsets: dict, size: int = 32) -> int:
    script = build_frame_script(base_script, offsets, width=size, height=size, palette_name="pico8")
    res = PixelScriptEngine.execute(script, width=size, height=size, palette_name="pico8")
    return islands(res.composite_array)


def main() -> None:
    model, metrics = rig_params.train(steps=600, samples=4096, seed=0)
    print(f"head parameters: {model.count_parameters()}")
    print(
        f"train mse {metrics['train_mse']:.3f} | held-out mse {metrics['held_out_mse']:.3f} "
        f"| held-out mae {metrics['held_out_mae']:.3f}"
    )

    action, phase = "walk", 0.25
    learned = [rig_params.predict(model, action, phase, "knight", p) for p in PROMPTS]
    analytic = [
        rig_params.modulate(rig_params.base_offsets(action, phase, "knight"), rig_params.tags_from_prompt(p))
        for p in PROMPTS
    ]
    fixed = [rig_params.base_offsets(action, phase, "knight") for _ in PROMPTS]

    print(f"\nprompt diversity at {action} phase {phase}:")
    print(f"  analytic mapping       {rig_params.offset_diversity(analytic):.3f}")
    print(f"  learned head           {rig_params.offset_diversity(learned):.3f}")
    print(f"  fixed template (today) {rig_params.offset_diversity(fixed):.3f}")

    base = PixelScriptSerializer.serialize_program(
        CharacterGenerator.generate_character(category="knight", seed=7, width=32, height=32)
    )
    static_islands = frame_islands(base, {})
    checks = []
    for prompt in PROMPTS:
        for act in ("walk", "run", "cast", "dash"):
            for ph in (0.0, 0.2, 0.4, 0.6, 0.8):
                offs = rig_params.predict(model, act, ph, "knight", prompt)
                checks.append((prompt, act, ph, frame_islands(base, offs)))
    torn = [c for c in checks if c[3] > static_islands]
    print(
        f"\nconnectivity: {len(checks) - len(torn)}/{len(checks)} learned frames keep "
        f"{static_islands} island(s); {len(torn)} torn"
    )
    for c in torn[:5]:
        print(f"  torn: {c[0]!r} {c[1]} phase {c[2]} -> {c[3]} islands")

    Path("/tmp/rig_params_eval.json").write_text(
        json.dumps(
            {
                "parameters": model.count_parameters(),
                "metrics": metrics,
                "diversity": {
                    "analytic": rig_params.offset_diversity(analytic),
                    "learned": rig_params.offset_diversity(learned),
                    "fixed_template": rig_params.offset_diversity(fixed),
                },
                "connectivity": {
                    "checks": len(checks),
                    "torn": len(torn),
                    "static_islands": static_islands,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
