"""Unit tests for Generalization Benchmark Suite."""
import pytest
from pathlib import Path
from spritemind.ml.benchmark import BENCHMARK_PROMPTS, GeneralizationBenchmark
from spritemind.ml.training import PixelScriptDataset
from spritemind.ml.tokenizer import ActionTokenizer


def test_benchmark_prompts_structure():
    assert len(BENCHMARK_PROMPTS) >= 100
    for p in BENCHMARK_PROMPTS[:20]:
        assert "id" in p
        assert "prompt" in p
        assert "category" in p
        assert "style" in p
        assert "pose" in p
        assert "palette" in p


def test_dataset_jsonl_loading():
    tok = ActionTokenizer()
    ds_path = Path("data/datasets/generalization_10k")
    if ds_path.exists():
        train_ds = PixelScriptDataset(ds_path, tok, split="train", max_samples=50)
        assert len(train_ds) > 0
        sample = train_ds[0]
        assert "input_ids" in sample
        assert "target_ids" in sample
        assert sample["input_ids"].shape[0] == 127


def test_benchmark_suite_is_300_prompts():
    from spritemind.ml.benchmark import BENCHMARK_SUITE_SIZE

    assert BENCHMARK_SUITE_SIZE == 300
    assert len(BENCHMARK_PROMPTS) == 300
    ids = [p["id"] for p in BENCHMARK_PROMPTS]
    assert len(set(ids)) == 300
    element_prompts = [p for p in BENCHMARK_PROMPTS[100:] if "-crackling" in p["prompt"]]
    assert element_prompts, "generated prompts should exercise the element/aura path"


def test_generated_prompts_are_deterministic():
    from spritemind.ml.benchmark import generated_prompts

    first = generated_prompts(5)
    second = generated_prompts(5)
    assert first == second
    assert all(p["id"].startswith("bench_") for p in first)
