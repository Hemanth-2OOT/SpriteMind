"""Asset Library manager for saving, retrieving, and organizing game-ready sprites."""
import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class AssetManager:
    """Manages saved sprites, animations, and assets in data/assets."""

    def __init__(self, storage_dir: str = "data/assets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_assets(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        assets = []
        for p in sorted(self.storage_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if query:
                        q = query.lower()
                        name = data.get("name", "").lower()
                        cat = data.get("category", "").lower()
                        if q not in name and q not in cat:
                            continue
                    assets.append(data)
            except Exception:
                pass
        return assets

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        target = self.storage_dir / f"{asset_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_asset(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = asset_data.get("id") or f"asset_{uuid.uuid4().hex[:8]}"
        asset_data["id"] = asset_id
        asset_data["created_at"] = asset_data.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")

        target = self.storage_dir / f"{asset_id}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(asset_data, f, indent=2)

        return asset_data

    def delete_asset(self, asset_id: str) -> bool:
        target = self.storage_dir / f"{asset_id}.json"
        if target.exists():
            target.unlink()
            return True
        return False
