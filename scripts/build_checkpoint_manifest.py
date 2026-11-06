"""Writes models/checkpoints/manifest.json with shape, taxonomy and hash facts.

Run after training, or whenever a checkpoint file is replaced:

    python scripts/build_checkpoint_manifest.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import torch  # noqa: E402

from spritemind.ml.checkpoint_manifest import describe_file, read_manifest, record, write_manifest  # noqa: E402

DEFAULT_DIR = Path("models/checkpoints")


def describe(path: Path) -> dict:
    entry = describe_file(path)
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - corrupt checkpoint
        entry["error"] = f"unreadable: {exc}"
        return entry

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if isinstance(state, dict):
        entry["parameter_count"] = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
        anim = state.get("animation_embed.weight")
        if anim is not None:
            entry["trained_animations"] = int(anim.shape[0])
        vocab = state.get("token_embed.weight")
        if vocab is not None:
            entry["vocab_size"] = int(vocab.shape[0])
    for key in ("epoch", "best_val_loss", "val_accuracy", "taxonomy", "model_config"):
        if key in metadata:
            value = metadata[key]
            entry[key] = value if not hasattr(value, "tolist") else value.tolist()
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--prune", action="store_true", help="drop entries whose checkpoint file is gone")
    args = parser.parse_args()

    directory = Path(args.dir)
    files = sorted(directory.glob("*.pt"))
    if not files:
        print(f"No checkpoints in {directory}")
        return 1

    for path in files:
        entry = describe(path)
        record(directory, entry)
        print(f"recorded {path.name}: {entry.get('bytes', 0)} bytes, {entry.get('parameter_count', 0)} params")
    if args.prune:
        present = {path.name for path in files}
        entries = read_manifest(directory)["checkpoints"]
        for name in sorted(set(entries) - present):
            del entries[name]
            print(f"pruned {name}")
        write_manifest(directory, entries)
    print(f"manifest written to {directory / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
