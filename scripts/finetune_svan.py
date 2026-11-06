"""Train SVAN on a dataset, with optional replay mixing and a critic-driven gate.

Training runs one epoch at a time so every epoch's weights can be scored by the
visual critic (share of sprites with zero defects). The best-clean epoch wins and
is only copied over the shipped checkpoint when it clears --min-clean, so a run
that trades silhouette sharpness for floating limbs never ships.

    python scripts/finetune_svan.py --dataset gv2_synthetic_25k \
        --replay generalization_detail_12k --from-scratch --epochs 3 --lr 0.0006
"""
import argparse
import json
import shutil
import sys
import time

from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from spritemind.ml.training import ModelTrainer  # noqa: E402

CONFIG = {
    "dataset_name": "generalization_detail_12k",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.0005,
    "max_seq_len": 448,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.05,
    "label_smoothing": 0.03,
    "threads": 8,
    "patience": 1,
    "save_checkpoint_name": "svan_finetuned.pt",
    "resume_from": "svan_checkpoint_best.pt",
}

CHECKPOINTS = ROOT / "models" / "checkpoints"
DATASETS = ROOT / "data" / "datasets"
PROMPTS_PATH = ROOT / "data" / "benchmarks" / "generalization_prompts.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=CONFIG["dataset_name"])
    parser.add_argument("--replay", nargs="*", default=[], help="extra datasets mixed into training")
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--save-name", default=CONFIG["save_checkpoint_name"])
    parser.add_argument("--resume-from", default=CONFIG["resume_from"])
    parser.add_argument("--from-scratch", action="store_true", help="ignore the shipped checkpoint")
    parser.add_argument("--history", default="data/finetune_history.json")
    parser.add_argument("--clean-samples", type=int, default=16)
    parser.add_argument("--min-clean", type=float, default=90.0, help="clean-rate floor for adoption")
    parser.add_argument("--clean-patience", type=int, default=1)
    return parser.parse_args()


def _count_lines(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def build_mixed_dataset(primary: str, replay: List[str], fraction: float) -> str:
    """Writes <primary>_mixed holding the primary records plus a slice of each replay set."""
    if not replay:
        return primary

    sources = [primary] + replay
    counts = {name: _count_lines(DATASETS / name / "records.jsonl") for name in sources}
    out_dir = DATASETS / f"{primary}_mixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as dst:
        for name in sources:
            src = DATASETS / name / "records.jsonl"
            limit = counts[name] if name == primary else int(round(counts[name] * fraction))
            with open(src, encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= limit:
                        break
                    if line.strip():
                        dst.write(line if line.endswith("\n") else line + "\n")
                        written += 1
    print(f"mixed dataset: {out_dir.name} ({written} records from {', '.join(sources)})", flush=True)
    return out_dir.name


def critic_clean_rate(checkpoint: str, limit: int) -> float:
    """Runs the pipeline on `limit` benchmark prompts and returns the share with zero defects."""
    from spritemind.ml.pipeline import SpriteMindPipeline

    pipeline = SpriteMindPipeline()
    pipeline.load_checkpoint(checkpoint)
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))[:limit]
    clean = 0
    for item in prompts:
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        result = pipeline.generate(prompt=prompt, engine_mode="trained_model", seed=42, best_of=1)
        if int(result.get("critic", {}).get("defects_count", 0)) == 0:
            clean += 1
    return round(100.0 * clean / max(1, len(prompts)), 1)


def run_epoch(config: dict, save_name: str, resume_from: Optional[str], history_path: str) -> dict:
    """Trains exactly one epoch and returns the trainer's final status."""
    trainer = ModelTrainer(checkpoints_dir=str(CHECKPOINTS), history_path=history_path)
    trainer.start_training(
        seed=42,
        resume=resume_from is not None,
        resume_from=resume_from,
        save_checkpoint_name=save_name,
        **{k: v for k, v in config.items() if k not in {"save_checkpoint_name", "resume_from"}},
    )
    while trainer.is_training:
        time.sleep(5)
    return trainer.get_status()


def main():
    args = parse_args()
    config = dict(CONFIG)
    config.update(
        epochs=1,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    dataset = build_mixed_dataset(args.dataset, args.replay, args.replay_fraction)
    print("config:", json.dumps({**config, "dataset_name": dataset}), flush=True)

    history = {"dataset": dataset, "replay": args.replay, "from_scratch": args.from_scratch, "epochs": []}
    # Keep earlier stages of the same run when resuming, so epoch numbering and
    # per-epoch scores stay in one file instead of the last stage overwriting.
    existing = Path(args.history)
    if existing.exists():
        try:
            prior = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        if prior.get("dataset") == dataset:
            history["epochs"] = prior.get("epochs", [])
    offset = len(history["epochs"])
    previous = None if args.from_scratch else args.resume_from
    best_rate, best_checkpoint = -1.0, None
    stale = 0

    for index in range(1, args.epochs + 1):
        save_name = f"{args.save_name}.ep{index}.pt"
        status = run_epoch({**config, "dataset_name": dataset}, save_name, previous, args.history)
        if status.get("state") != "completed":
            print("aborted:", status.get("state"), status.get("error_message", ""), flush=True)
            break

        rate = critic_clean_rate(save_name, args.clean_samples)
        entry = {
            "epoch": offset + index,
            "checkpoint": save_name,
            "val_loss": status.get("val_loss"),
            "token_accuracy": status.get("token_accuracy"),
            "val_token_accuracy": status.get("val_token_accuracy"),
            "clean_rate": rate,
            "elapsed_seconds": status.get("elapsed_seconds"),
        }
        history["epochs"].append(entry)
        print(
            f"[Epoch {index}/{args.epochs}] val {entry['val_loss']} val_acc {entry['val_token_accuracy']}% "
            f"clean {rate}% elapsed {entry['elapsed_seconds']}s",
            flush=True,
        )

        if rate > best_rate:
            best_rate, best_checkpoint, stale = rate, save_name, 0
        else:
            stale += 1
        Path(args.history).write_text(json.dumps({**history, "best_clean_rate": best_rate}, indent=2), encoding="utf-8")

        previous = save_name
        if rate < args.min_clean and stale >= args.clean_patience:
            print(f"early stop: clean rate {rate}% below {args.min_clean}%", flush=True)
            break

    adopted = best_checkpoint is not None and best_rate >= args.min_clean
    if adopted:
        shutil.copyfile(CHECKPOINTS / best_checkpoint, CHECKPOINTS / "svan_checkpoint_best.pt")
        print(f"adopted {best_checkpoint} (clean {best_rate}%) as svan_checkpoint_best.pt", flush=True)
    else:
        print(f"not adopted: best clean rate {best_rate}% < {args.min_clean}%", flush=True)

    for checkpoint in CHECKPOINTS.glob(f"{args.save_name}.ep*.pt"):
        if checkpoint.name != best_checkpoint:
            checkpoint.unlink()
    print("final:", json.dumps(history["epochs"][-1] if history["epochs"] else {}), flush=True)


if __name__ == "__main__":
    main()
