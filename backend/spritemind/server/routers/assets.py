"""Asset catalog, composition, packaging, library and storage lifecycle."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException

from pathlib import Path
from spritemind.assets.composer import AssetComposer
from spritemind.assets.library import ASSET_CATEGORIES
from spritemind.assets.library import AssetLibrary
from spritemind.assets.pack import AssetPack
from spritemind.server.schemas import AssetPackExportRequest
from spritemind.server.schemas import AssetSaveRequest
from spritemind.server.schemas import BackupCreateRequest
from spritemind.server.schemas import CleanupExecuteRequest
from spritemind.server.schemas import CompositionAnimationRequest
from spritemind.server.schemas import CompositionRequest
from spritemind.server.schemas import EmptyTrashRequest
from spritemind.server.schemas import PermanentDeleteRequest
from spritemind.server.schemas import RestoreAssetRequest
from spritemind.server.schemas import TrashAssetRequest
from spritemind.storage.classification import ProtectedResourceException
import base64
import json

from spritemind.server.runtime import asset_mgr, backup_mgr, cleaner, lifecycle_mgr, storage_db
from spritemind.paths import resolve_data_dir
router = APIRouter()

@router.get("/api/asset_catalog")
def get_asset_catalog(category: Optional[str] = None) -> List[Dict[str, Any]]:
    assets = AssetLibrary.list_all(category=category)
    return [a.to_dict() for a in assets]

@router.post("/api/composition/compose")
def compose_asset(req: CompositionRequest) -> Dict[str, Any]:
    try:
        res = AssetComposer.compose(
            character_asset_or_script=req.character_script or req.character_id or "character_player_soldier",
            weapon_id=req.weapon_id,
            helmet_id=req.helmet_id,
            backpack_id=req.backpack_id,
            fx_id=req.fx_id,
            engine_mode=req.engine,
            palette_name=req.palette,
            width=req.width,
            height=req.height,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Composition failed: {str(e)}")

@router.post("/api/composition/animate")
def animate_composition(req: CompositionAnimationRequest) -> Dict[str, Any]:
    try:
        res = AssetComposer.compose_animation(
            character_asset_or_script=req.character_script or req.character_id or "character_zombie_walker",
            animation_category=req.animation_category,
            num_frames=req.num_frames,
            fps=req.fps,
            weapon_id=req.weapon_id,
            helmet_id=req.helmet_id,
            backpack_id=req.backpack_id,
            fx_id=req.fx_id,
            engine_mode=req.engine,
            palette_name=req.palette,
            width=req.width,
            height=req.height,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Animated composition failed: {str(e)}")

@router.post("/api/asset_pack/export_zip")
def export_asset_pack(req: AssetPackExportRequest) -> Dict[str, Any]:
    try:
        if req.pack_type == "deadzone":
            pack = AssetPack.create_deadzone_starter_pack()
        else:
            pack = AssetPack(
                pack_id="pack_custom",
                name=req.pack_name or "Custom Asset Pack",
            )
            for aid in req.asset_ids:
                a = AssetLibrary.get(aid)
                if a:
                    pack.add_asset(a)

        zip_bytes = pack.export_zip()
        b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
        filename = f"{pack.pack_id}.zip"

        return {
            "status": "success",
            "filename": filename,
            "pack_name": pack.name,
            "total_assets": len(pack.assets),
            "zip_base64": f"data:application/zip;base64,{b64_zip}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Asset pack export failed: {str(e)}")

@router.get("/api/composition/metrics")
def get_composition_metrics() -> Dict[str, Any]:
    res = {
        "status": "ready",
        "model_architecture": "SocketPredictor (Residual2D + Sigmoid Regressor)",
        "parameter_count": 619432,
        "supported_categories": ASSET_CATEGORIES,
    }
    metrics_path = resolve_data_dir() / "composition_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                res["training_metrics"] = json.load(f)
        except Exception:
            pass

    benchmarks_path = resolve_data_dir() / "benchmarks/composition_benchmark_results.json"
    if benchmarks_path.exists():
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                res["benchmark_results"] = json.load(f)
        except Exception:
            pass

    return res

@router.get("/api/assets")
def list_assets(query: Optional[str] = None) -> List[Dict[str, Any]]:
    return asset_mgr.list_assets(query=query)

@router.get("/api/assets/{asset_id}")
def get_asset(asset_id: str) -> Dict[str, Any]:
    asset = asset_mgr.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.post("/api/assets")
def save_asset(req: AssetSaveRequest) -> Dict[str, Any]:
    return asset_mgr.save_asset(req.model_dump())

@router.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str) -> Dict[str, Any]:
    try:
        res = lifecycle_mgr.trash_asset(asset_id)
        return {"status": "trashed", "id": asset_id, "trash_id": res["trash_id"]}
    except FileNotFoundError:
        success = asset_mgr.delete_asset(asset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Asset not found")
        return {"status": "deleted", "id": asset_id}
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/api/storage/dashboard")
def get_storage_dashboard() -> Dict[str, Any]:
    """Returns storage dashboard statistics, classification breakdown, and health."""
    return storage_db.get_storage_dashboard()

@router.get("/api/storage/records")
def get_storage_records(table: str = "storage_registry", query: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Inspects records in the database with optional search query."""
    allowed_tables = ["storage_registry", "assets", "checkpoints", "datasets", "benchmarks", "trash", "audit_log"]
    if table not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Table must be one of {allowed_tables}")
    records = storage_db.list_table_records(table_name=table, search_query=query, limit=limit)
    return {
        "table": table,
        "count": len(records),
        "records": records,
    }

@router.get("/api/storage/orphans")
def get_storage_orphans() -> Dict[str, Any]:
    """Identifies missing files and unindexed files on disk."""
    return storage_db.find_orphans_and_missing()

@router.post("/api/storage/sync")
def sync_storage_registry() -> Dict[str, Any]:
    """Rescans filesystem and synchronizes SQLite storage catalog."""
    count = storage_db.sync_from_disk()
    return {"status": "synced", "files_indexed": count}

@router.get("/api/storage/trash")
def list_trash() -> List[Dict[str, Any]]:
    """Lists all items currently in the trash bin."""
    return lifecycle_mgr.list_trash()

@router.post("/api/storage/trash")
def trash_asset_endpoint(req: TrashAssetRequest) -> Dict[str, Any]:
    """Safely moves a user-generated asset to trash. Rejects protected resources."""
    try:
        return lifecycle_mgr.trash_asset(asset_id=req.asset_id, reason=req.reason)
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/api/storage/restore")
def restore_asset_endpoint(req: RestoreAssetRequest) -> Dict[str, Any]:
    """Restores an item from trash back to active storage."""
    try:
        return lifecycle_mgr.restore_asset(trash_id=req.trash_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/api/storage/permanent_delete")
def permanent_delete_endpoint(req: PermanentDeleteRequest) -> Dict[str, Any]:
    """Permanently deletes an item from trash. Strictly requires confirm=True."""
    try:
        return lifecycle_mgr.permanent_delete(trash_id=req.trash_id, confirm=req.confirm)
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/storage/empty_trash")
def empty_trash_endpoint(req: EmptyTrashRequest) -> Dict[str, Any]:
    """Empties all eligible items from trash bin. Requires confirm=True."""
    try:
        return lifecycle_mgr.empty_trash(confirm=req.confirm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/storage/cleanup_scan")
def storage_cleanup_scan() -> Dict[str, Any]:
    """Performs a non-destructive dry-run scan for disposable temporary files."""
    return cleaner.scan()

@router.post("/api/storage/cleanup_execute")
def storage_cleanup_execute(req: CleanupExecuteRequest) -> Dict[str, Any]:
    """Executes safe cleanup of verified temporary files. Requires confirm=True."""
    try:
        return cleaner.execute(confirm=req.confirm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/storage/backup")
def create_backup_endpoint(req: BackupCreateRequest) -> Dict[str, Any]:
    """Creates a verified ZIP backup containing database catalog and metadata."""
    return backup_mgr.create_backup(
        backup_name=req.backup_name,
        include_db=req.include_db,
        include_benchmarks=req.include_benchmarks,
        include_assets_meta=req.include_assets_meta,
    )

@router.get("/api/storage/backups")
def list_backups_endpoint() -> List[Dict[str, Any]]:
    """Lists all available ZIP backups."""
    return backup_mgr.list_backups()
