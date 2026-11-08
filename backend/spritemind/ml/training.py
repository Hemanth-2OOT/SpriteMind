"""PyTorch Training Pipeline for SVAN Action Transformer."""
import time
import os
import json
import random
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spritemind.ml.prompt_encoder import PROMPT_MAX_LEN, describe_record, encode_prompt
from spritemind.ml.svan import SVANModel
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.dataset.generator import CATEGORIES, STYLES, POSES
from spritemind.dataset.conditioning import colour_id, element_id, palette_id
from spritemind.pixelscript.ast import PixelScriptProgram, PixelAction


def _record_checkpoint(path, extra):
    """Best-effort manifest entry; training must never fail because of bookkeeping."""
    try:
        from spritemind.ml.checkpoint_manifest import describe_file, record

        record(Path(path).parent, describe_file(Path(path), **extra))
    except Exception:
        pass


class PixelScriptDataset(Dataset):
    """PyTorch Dataset loading generated PixelScript samples with pre-tokenized memory buffers."""

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: ActionTokenizer,
        max_seq_len: int = 128,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        input_list = []
        target_list = []
        cat_list = []
        style_list = []
        pose_list = []
        palette_list = []
        colour_list = []
        element_list = []
        prompt_list = []

        # Find records.jsonl either at dataset_path or its parent
        jsonl_candidates = [
            dataset_path / "records.jsonl",
            dataset_path.parent / "records.jsonl" if dataset_path.name == "samples" else None,
            Path("data/datasets") / dataset_path.name / "records.jsonl",
        ]
        jsonl_path = next((p for p in jsonl_candidates if p and p.exists()), None)

        def _add(record: Dict[str, Any]) -> None:
            """Encode one record and append every per-sample field together.

            All values are computed before any append, so a malformed record is skipped
            whole and can never leave the per-sample tensors out of step.
            """
            actions = [PixelAction.from_dict(a) for a in record.get("actions", [])]
            token_ids = self.tokenizer.encode_program(
                PixelScriptProgram(actions=actions), max_length=self.max_seq_len
            )
            c_str = record.get("category", "knight")
            s_str = record.get("style", "classic_16bit")
            p_str = record.get("pose", "idle")
            cat = CATEGORIES.index(c_str) if c_str in CATEGORIES else 0
            style = STYLES.index(s_str) if s_str in STYLES else 0
            pose = POSES.index(p_str) if p_str in POSES else 0
            # Records from legacy corpora carry no palette label; palette_id(None) == 0,
            # the same neutral row the shipped checkpoints were trained on.
            pal = palette_id(record.get("palette"))
            colour = colour_id(record.get("colour"))
            element = element_id(record.get("element"))
            prompt = encode_prompt(describe_record(record))

            input_list.append(token_ids[:-1])
            target_list.append(token_ids[1:])
            cat_list.append(cat)
            style_list.append(style)
            pose_list.append(pose)
            palette_list.append(pal)
            colour_list.append(colour)
            element_list.append(element)
            prompt_list.append(prompt)

        if jsonl_path:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if split != "all" and item.get("split") != split:
                        continue
                    _add(item)
                    if max_samples and len(input_list) >= max_samples:
                        break
        else:
            # Fallback to individual sample_*.json files
            samples_dir = dataset_path if dataset_path.name == "samples" else dataset_path / "samples"
            if samples_dir.exists():
                for p in sorted(samples_dir.glob("sample_*.json")):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if split != "all" and data.get("split", "train") != split:
                            continue
                        _add(data)
                        if max_samples and len(input_list) >= max_samples:
                            break
                    except Exception:
                        pass

        if len(input_list) > 0:
            self.input_ids = torch.tensor(input_list, dtype=torch.long)
            self.target_ids = torch.tensor(target_list, dtype=torch.long)
            self.category_idx = torch.tensor(cat_list, dtype=torch.long)
            self.style_idx = torch.tensor(style_list, dtype=torch.long)
            self.pose_idx = torch.tensor(pose_list, dtype=torch.long)
            self.palette_idx = torch.tensor(palette_list, dtype=torch.long)
            self.colour_idx = torch.tensor(colour_list, dtype=torch.long)
            self.element_idx = torch.tensor(element_list, dtype=torch.long)
            self.prompt_tokens = torch.tensor(prompt_list, dtype=torch.long)
        else:
            self.input_ids = torch.empty((0, max_seq_len - 1), dtype=torch.long)
            self.target_ids = torch.empty((0, max_seq_len - 1), dtype=torch.long)
            self.category_idx = torch.empty(0, dtype=torch.long)
            self.style_idx = torch.empty(0, dtype=torch.long)
            self.pose_idx = torch.empty(0, dtype=torch.long)
            self.palette_idx = torch.empty(0, dtype=torch.long)
            self.colour_idx = torch.empty(0, dtype=torch.long)
            self.element_idx = torch.empty(0, dtype=torch.long)
            self.prompt_tokens = torch.empty((0, PROMPT_MAX_LEN), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
            "category_idx": self.category_idx[idx],
            "style_idx": self.style_idx[idx],
            "pose_idx": self.pose_idx[idx],
            "palette_idx": self.palette_idx[idx],
            "colour_idx": self.colour_idx[idx],
            "element_idx": self.element_idx[idx],
            "prompt_tokens": self.prompt_tokens[idx],
        }


def colour_token_weights(tokenizer, weight: float, device=None) -> "torch.Tensor":
    """Per-token CE multiplier: `weight` on COL_* slots, 1.0 everywhere else."""
    import torch as _torch

    weights = _torch.ones(tokenizer.vocab_size, device=device)
    for token, tid in tokenizer.token_to_id.items():
        if token.startswith("COL_"):
            weights[tid] = weight
    return weights


class ModelTrainer:
    """Background asynchronous trainer for SVAN models with full training rigor."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints", history_path: str = "data/training_history.json"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = ActionTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_training: bool = False
        self.stop_requested: bool = False
        self._thread: Optional[threading.Thread] = None

        self.current_run_status: Dict[str, Any] = {
            "state": "idle",  # "idle" | "training" | "completed" | "stopped" | "error"
            "dataset_name": "",
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "token_accuracy": 0.0,
            "val_token_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "elapsed_seconds": 0.0,
            "learning_rate": 0.001,
            "device": str(self.device),
            "loss_history": [],
            "val_loss_history": [],
            "token_accuracy_history": [],
        }

        # Load existing history if available
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.current_run_status.update({
                        "loss_history": saved.get("loss_history", []),
                        "val_loss_history": saved.get("val_loss_history", []),
                        "token_accuracy_history": saved.get("token_accuracy_history", []),
                        "best_val_loss": saved.get("best_val_loss", float("inf")),
                        "best_epoch": saved.get("best_epoch", 0),
                        "dataset_name": saved.get("dataset_name", ""),
                    })
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        return dict(self.current_run_status)

    def start_training(
        self,
        dataset_name: str = "generalization_10k",
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer_name: str = "adamw",
        save_checkpoint_name: str = "svan_checkpoint_best.pt",
        resume: bool = False,
        resume_from: Optional[str] = None,
        patience: int = 4,
        seed: int = 42,
        max_seq_len: int = 448,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        threads: int = 0,
        prompt_text: bool = False,
        max_samples: Optional[int] = None,
        palette_conditioning: bool = False,
        colour_token_weight: float = 1.0,
    ) -> Dict[str, Any]:
        if self.is_training:
            return {"status": "error", "message": "Training already in progress"}

        self.stop_requested = False
        self.is_training = True
        self._thread = threading.Thread(
            target=self._run_training_loop,
            args=(dataset_name, epochs, batch_size, learning_rate, optimizer_name, save_checkpoint_name, resume, resume_from, patience, seed,
                  max_seq_len, d_model, nhead, num_layers, dim_feedforward, dropout, label_smoothing, threads,
                  prompt_text, max_samples, palette_conditioning, colour_token_weight),
            daemon=True,
        )
        self._thread.start()
        return {"status": "started", "message": f"Training initiated for {epochs} epochs on {self.device}"}

    def stop_training(self) -> Dict[str, Any]:
        if not self.is_training:
            return {"status": "noop", "message": "No training in progress"}
        self.stop_requested = True
        return {"status": "stopping", "message": "Stop signal sent"}

    def _save_history(self, history_data: Dict[str, Any]):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2)
        except Exception:
            pass

    def _run_training_loop(
        self,
        dataset_name: str,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer_name: str,
        save_name: str,
        resume: bool = False,
        resume_from: Optional[str] = None,
        patience: int = 4,
        seed: int = 42,
        max_seq_len: int = 448,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        threads: int = 0,
        prompt_text: bool = False,
        max_samples: Optional[int] = None,
        palette_conditioning: bool = False,
        colour_token_weight: float = 1.0,
    ):
        start_time = time.time()
        if threads > 0:
            torch.set_num_threads(threads)
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            dataset_dir = Path("data/datasets") / dataset_name
            train_dataset = PixelScriptDataset(
                dataset_dir, self.tokenizer, max_seq_len=max_seq_len, split="train", max_samples=max_samples
            )
            val_dataset = PixelScriptDataset(
                dataset_dir, self.tokenizer, max_seq_len=max_seq_len, split="val", max_samples=max_samples
            )

            if len(train_dataset) == 0:
                self.current_run_status["state"] = "error"
                self.current_run_status["error_message"] = f"No training samples found in dataset {dataset_name}"
                self.is_training = False
                return

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

            model = SVANModel(
                vocab_size=self.tokenizer.vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                max_seq_len=max_seq_len,
                dropout=dropout,
                prompt_text=prompt_text,
                palette_conditioning=palette_conditioning,
            ).to(self.device)

            # Optional resume from previous checkpoint
            start_epoch = 1
            if resume:
                resume_file = self.checkpoints_dir / (resume_from or save_name)
                if resume_file.exists():
                    loaded = torch.load(resume_file, map_location=self.device)
                    state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
                    if prompt_text:
                        # Older checkpoints predate the text encoder, so its weights
                        # start fresh while everything else resumes.
                        state = {k: v for k, v in state.items() if not k.startswith("prompt_encoder.")}
                        model.load_state_dict(state, strict=False)
                    else:
                        model.load_state_dict(state)
                    if isinstance(loaded, dict):
                        start_epoch = loaded.get("metadata", {}).get("epoch", 0) + 1

            if optimizer_name.lower() == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id, label_smoothing=label_smoothing)
            # R11: upweight the colour slots so a rare wrong hue costs more than a
            # wrong coordinate. 1.0 leaves the objective exactly as it was.
            token_weights = None
            if colour_token_weight != 1.0:
                token_weights = colour_token_weights(self.tokenizer, colour_token_weight, self.device)

            total_steps = epochs * len(train_loader)
            loss_history: List[Dict[str, Any]] = []
            val_loss_history: List[Dict[str, Any]] = []
            token_acc_history: List[Dict[str, Any]] = []

            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0

            self.current_run_status.update({
                "state": "training",
                "dataset_name": dataset_name,
                "total_epochs": epochs,
                "total_steps": total_steps,
                "learning_rate": lr,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "loss_history": loss_history,
                "val_loss_history": val_loss_history,
                "token_accuracy_history": token_acc_history,
                "device": str(self.device),
            })

            global_step = 0

            for epoch in range(start_epoch, start_epoch + epochs):
                if self.stop_requested:
                    break

                model.train()
                running_loss = 0.0
                correct_tokens = 0
                total_tokens = 0

                for batch in train_loader:
                    if self.stop_requested:
                        break

                    inp = batch["input_ids"].to(self.device)
                    tgt = batch["target_ids"].to(self.device)
                    # Trim padded tail per batch to the longest real sequence
                    real_len = int((tgt != self.tokenizer.pad_id).sum(dim=1).max().item())
                    inp, tgt = inp[:, :real_len].contiguous(), tgt[:, :real_len].contiguous()
                    c_idx = batch["category_idx"].to(self.device)
                    s_idx = batch["style_idx"].to(self.device)
                    p_idx = batch["pose_idx"].to(self.device)
                    pal_idx = batch["palette_idx"].to(self.device)
                    col_idx = batch["colour_idx"].to(self.device)
                    el_idx = batch["element_idx"].to(self.device)

                    # Padding key mask for transformer decoder
                    padding_mask = (inp == self.tokenizer.pad_id)
                    prompt_tokens = batch["prompt_tokens"].to(self.device) if model.prompt_encoder is not None else None

                    optimizer.zero_grad()
                    logits = model(
                        tgt_tokens=inp,
                        category_idx=c_idx,
                        style_idx=s_idx,
                        pose_idx=p_idx,
                        palette_idx=pal_idx,
                        tgt_key_padding_mask=padding_mask,
                        prompt_tokens=prompt_tokens,
                        colour_idx=col_idx,
                        element_idx=el_idx,
                    )

                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_tgt = tgt.view(-1)
                    if token_weights is None:
                        loss = criterion(flat_logits, flat_tgt)
                    else:
                        per_token = nn.functional.cross_entropy(
                            flat_logits,
                            flat_tgt,
                            ignore_index=self.tokenizer.pad_id,
                            reduction="none",
                            label_smoothing=label_smoothing,
                        )
                        weights = token_weights[flat_tgt.clamp(min=0)]
                        weights = weights * (flat_tgt != self.tokenizer.pad_id).float()
                        loss = (per_token * weights).sum() / weights.sum().clamp(min=1.0)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    running_loss += loss.item()
                    global_step += 1

                    if global_step % 200 == 0:
                        print(
                            f"epoch {epoch} step {global_step}: loss {running_loss / global_step:.4f}",
                            flush=True,
                        )

                    # Training Token Accuracy
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=-1)
                        mask = tgt != self.tokenizer.pad_id
                        correct_tokens += ((preds == tgt) & mask).sum().item()
                        total_tokens += mask.sum().item()

                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                epoch_loss = running_loss / max(1, len(train_loader))
                train_acc = (correct_tokens / max(1, total_tokens)) * 100.0

                # Validation Loop
                val_loss = epoch_loss
                val_acc = train_acc
                if val_loader:
                    model.eval()
                    val_running_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            v_inp = val_batch["input_ids"].to(self.device)
                            v_tgt = val_batch["target_ids"].to(self.device)
                            v_real_len = int((v_tgt != self.tokenizer.pad_id).sum(dim=1).max().item())
                            v_inp, v_tgt = v_inp[:, :v_real_len].contiguous(), v_tgt[:, :v_real_len].contiguous()
                            v_mask = (v_inp == self.tokenizer.pad_id)
                            v_logits = model(
                                tgt_tokens=v_inp,
                                category_idx=val_batch["category_idx"].to(self.device),
                                style_idx=val_batch["style_idx"].to(self.device),
                                pose_idx=val_batch["pose_idx"].to(self.device),
                                palette_idx=val_batch["palette_idx"].to(self.device),
                                colour_idx=val_batch["colour_idx"].to(self.device),
                                element_idx=val_batch["element_idx"].to(self.device),
                                tgt_key_padding_mask=v_mask,
                                prompt_tokens=(
                                    val_batch["prompt_tokens"].to(self.device)
                                    if model.prompt_encoder is not None
                                    else None
                                ),
                            )
                            v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_tgt.view(-1))
                            val_running_loss += v_loss.item()

                            v_preds = torch.argmax(v_logits, dim=-1)
                            v_valid_mask = v_tgt != self.tokenizer.pad_id
                            val_correct += ((v_preds == v_tgt) & v_valid_mask).sum().item()
                            val_total += v_valid_mask.sum().item()

                    val_loss = val_running_loss / max(1, len(val_loader))
                    val_acc = (val_correct / max(1, val_total)) * 100.0

                elapsed = time.time() - start_time

                loss_history.append({"epoch": epoch, "loss": round(epoch_loss, 4)})
                val_loss_history.append({"epoch": epoch, "val_loss": round(val_loss, 4)})
                token_acc_history.append({
                    "epoch": epoch,
                    "train_accuracy": round(train_acc, 2),
                    "val_accuracy": round(val_acc, 2),
                })

                # Check if best validation loss
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best checkpoint with comprehensive metadata
                    best_ckpt_path = self.checkpoints_dir / save_name
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "metadata": {
                            "epoch": epoch,
                            "train_loss": round(epoch_loss, 4),
                            "val_loss": round(val_loss, 4),
                            "token_accuracy": round(val_acc, 2),
                            "best_val_loss": round(best_val_loss, 4),
                            "vocab_size": self.tokenizer.vocab_size,
                            "model_config": model.model_config,
                            "parameters": model.count_parameters(),
                            "dataset_name": dataset_name,
                            "device": str(self.device),
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    }, best_ckpt_path)
                    _record_checkpoint(best_ckpt_path, {
                        "parameter_count": model.count_parameters(),
                        "vocab_size": self.tokenizer.vocab_size,
                        "best_val_loss": round(best_val_loss, 4),
                    })
                else:
                    patience_counter += 1

                self.current_run_status.update({
                    "epoch": epoch,
                    "step": global_step,
                    "loss": round(epoch_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "token_accuracy": round(train_acc, 2),
                    "val_token_accuracy": round(val_acc, 2),
                    "best_val_loss": round(best_val_loss, 4),
                    "best_epoch": best_epoch,
                    "learning_rate": round(current_lr, 6),
                    "elapsed_seconds": round(elapsed, 1),
                    "loss_history": loss_history,
                    "val_loss_history": val_loss_history,
                    "token_accuracy_history": token_acc_history,
                })

                # Write training history to file
                self._save_history({
                    "dataset_name": dataset_name,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "total_epochs": epochs,
                    "completed_epochs": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": round(best_val_loss, 4),
                    "device": str(self.device),
                    "parameters": model.count_parameters(),
                    "vocab_size": self.tokenizer.vocab_size,
                    "learning_rate": lr,
                    "elapsed_seconds": round(elapsed, 1),
                    "loss_history": loss_history,
                    "val_loss_history": val_loss_history,
                    "token_accuracy_history": token_acc_history,
                })

                # Early stopping check
                if patience_counter >= patience:
                    break

            if not self.stop_requested:
                self.current_run_status["state"] = "completed"
                self.current_run_status["saved_checkpoint"] = save_name
            else:
                self.current_run_status["state"] = "stopped"

        except Exception as e:
            self.current_run_status["state"] = "error"
            self.current_run_status["error_message"] = str(e)
        finally:
            self.is_training = False
