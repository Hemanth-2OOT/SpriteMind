"""Prompt history store: newest first, capped, re-seedable."""
import sqlite3

from spritemind.server.history import (
    MAX_ROWS,
    clear_runs,
    delete_run,
    list_runs,
    record_run,
    replay_params,
)


def _params(**overrides):
    payload = {
        "prompt": "a purple zombie knight",
        "category": "zombie",
        "style": "dark_fantasy",
        "pose": "idle",
        "palette": "pico8",
        "engine": "trained_model",
        "seed": 42,
        "num_frames": 2,
        "best_of": 4,
        "temperature": 0.9,
        "top_k": 0,
    }
    payload.update(overrides)
    return payload


def _result(**overrides):
    result = {"engine_used": "trained_model", "inference_time_ms": 321.5, "frames": [{}, {}]}
    result.update(overrides)
    return result


def test_recorded_runs_come_back_newest_first(tmp_path):
    db = tmp_path / "history.sqlite"
    first = record_run(_params(prompt="first"), _result(), path=db)
    second = record_run(_params(prompt="second"), _result(), path=db)
    runs = list_runs(path=db)
    assert [run["id"] for run in runs] == [second, first]
    assert runs[0]["prompt"] == "second"
    assert runs[0]["frame_count"] == 2
    assert runs[0]["latency_ms"] == 321.5


def test_listing_a_missing_database_is_empty(tmp_path):
    assert list_runs(path=tmp_path / "nope.sqlite") == []


def test_limit_is_applied_and_clamped(tmp_path):
    db = tmp_path / "history.sqlite"
    for index in range(5):
        record_run(_params(prompt=f"p{index}"), _result(), path=db)
    assert len(list_runs(limit=2, path=db)) == 2
    assert len(list_runs(limit=0, path=db)) == 1


def test_old_rows_are_evicted_beyond_the_cap(tmp_path):
    db = tmp_path / "history.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, prompt TEXT NOT NULL,
               category TEXT, style TEXT, pose TEXT, palette TEXT, engine TEXT, seed INTEGER, num_frames INTEGER,
               best_of INTEGER, temperature REAL, top_k INTEGER, latency_ms REAL, frame_count INTEGER)"""
        )
    for index in range(MAX_ROWS + 3):
        record_run(_params(prompt=f"p{index}"), _result(), path=db)
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert count == MAX_ROWS
    assert list_runs(limit=1, path=db)[0]["prompt"] == f"p{MAX_ROWS + 2}"


def test_delete_and_clear(tmp_path):
    db = tmp_path / "history.sqlite"
    run_id = record_run(_params(), _result(), path=db)
    record_run(_params(), _result(), path=db)
    assert delete_run(run_id, path=db) is True
    assert delete_run(run_id, path=db) is False
    assert clear_runs(path=db) == 1
    assert list_runs(path=db) == []


def test_replay_params_round_trips_and_can_reseed(tmp_path):
    db = tmp_path / "history.sqlite"
    record_run(_params(), _result(), path=db)
    run = list_runs(path=db)[0]
    same = replay_params(run)
    assert same["seed"] == 42
    assert same["prompt"] == "a purple zombie knight"
    assert same["num_frames"] == 2
    assert replay_params(run, seed=7)["seed"] == 7
