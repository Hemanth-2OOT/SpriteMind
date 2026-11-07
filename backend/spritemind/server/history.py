"""SQLite-backed prompt history, so a run can be found and re-seeded later.

Only the request that produced a run is stored (never the rendered frames), which
keeps the file small enough to keep in the repo's data directory and lets a seed be
replayed against whatever model is loaded now.
"""
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PATH = Path(os.environ.get("SPRITEMIND_HISTORY_DB", "data/history.sqlite"))
MAX_ROWS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    prompt TEXT NOT NULL,
    category TEXT,
    style TEXT,
    pose TEXT,
    palette TEXT,
    engine TEXT,
    seed INTEGER,
    num_frames INTEGER,
    best_of INTEGER,
    temperature REAL,
    top_k INTEGER,
    latency_ms REAL,
    frame_count INTEGER
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def record_run(params: Dict[str, Any], result: Optional[Dict[str, Any]] = None, path: Optional[Path] = None) -> int:
    """Stores one generation request; returns the new row id."""
    db = Path(path or DEFAULT_PATH)
    result = result or {}
    frames = result.get("frames") or []
    with _connect(db) as conn:
        cursor = conn.execute(
            """INSERT INTO runs (created_at, prompt, category, style, pose, palette, engine, seed,
                                 num_frames, best_of, temperature, top_k, latency_ms, frame_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                (params.get("prompt") or "").strip(),
                params.get("category"),
                params.get("style"),
                params.get("pose"),
                params.get("palette"),
                result.get("engine_used") or params.get("engine"),
                params.get("seed"),
                params.get("num_frames"),
                params.get("best_of"),
                params.get("temperature"),
                params.get("top_k"),
                result.get("inference_time_ms"),
                len(frames),
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.execute(
            "DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY id DESC LIMIT ?)",
            (MAX_ROWS,),
        )
    return run_id


def list_runs(limit: int = 50, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    db = Path(path or DEFAULT_PATH)
    if not db.exists():
        return []
    limit = max(1, min(int(limit), MAX_ROWS))
    with _connect(db) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def delete_run(run_id: int, path: Optional[Path] = None) -> bool:
    db = Path(path or DEFAULT_PATH)
    if not db.exists():
        return False
    with _connect(db) as conn:
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0


def clear_runs(path: Optional[Path] = None) -> int:
    db = Path(path or DEFAULT_PATH)
    if not db.exists():
        return 0
    with _connect(db) as conn:
        cursor = conn.execute("DELETE FROM runs")
        return cursor.rowcount


def replay_params(run: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Turns a stored row back into a /api/generate payload, optionally with a new seed."""
    return {
        "prompt": run.get("prompt") or "",
        "category": run.get("category") or "knight",
        "style": run.get("style") or "classic_16bit",
        "pose": run.get("pose") or "idle",
        "palette": run.get("palette") or "pico8",
        "engine": run.get("engine") or "trained_model",
        "seed": run.get("seed") if seed is None else int(seed),
        "num_frames": run.get("num_frames") or 1,
        "best_of": run.get("best_of") or 4,
        "temperature": run.get("temperature") if run.get("temperature") is not None else 0.9,
        "top_k": run.get("top_k") or 0,
    }


def to_json(runs: List[Dict[str, Any]]) -> str:
    return json.dumps(runs, indent=2)
