"""Pydantic request and response schemas for FastAPI endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateRequest(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    category: Optional[str] = "knight"
    style: Optional[str] = "classic_16bit"
    pose: Optional[str] = "idle"
    palette: Optional[str] = "pico8"
    width: int = Field(default=32, ge=16, le=64)
    height: int = Field(default=32, ge=16, le=64)
    num_frames: int = Field(default=1, ge=1, le=16)
    engine: str = Field(default="procedural_baseline")  # procedural_baseline | trained_model | untrained_model
    seed: int = 42
    max_actions: int = Field(default=64, ge=8, le=256)


class ActionSetRequest(BaseModel):
    prompt: str = ""
    category: Optional[str] = None
    style: Optional[str] = None
    palette: Optional[str] = None
    width: int = Field(default=32, ge=16, le=64)
    height: int = Field(default=32, ge=16, le=64)
    engine: str = Field(default="procedural_baseline")
    seed: int = 42
    actions: Optional[List[str]] = None
    max_actions: int = Field(default=12, ge=1, le=18)
    include_gif: bool = True
    facing: str = Field(default="right", pattern="^(right|left)$")


class RefineRequest(BaseModel):
    script_text: str
    suggested_fix: Optional[str] = None
    width: int = 32
    height: int = 32
    palette: str = "pico8"


class RenderRequest(BaseModel):
    script_text: str
    width: int = 32
    height: int = 32
    palette: str = "pico8"
    scale: int = 1


class ExportRequest(BaseModel):
    frames_scripts: List[str]
    width: int = 32
    height: int = 32
    palette: str = "pico8"
    format: str = "png"  # "png" | "spritesheet" | "gif" | "json"
    scale: int = 1
    fps: int = 8
    cols: Optional[int] = None


class DatasetGenerateRequest(BaseModel):
    name: str = "default_synth"
    total_samples: int = Field(default=200, ge=10, le=10000)
    defect_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = 42


class TrainingStartRequest(BaseModel):
    dataset_name: str = "default_synth"
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=0.001, gt=0.0)
    optimizer: str = "adamw"
    checkpoint_name: str = "svan_model_v1.pt"


class CheckpointLoadRequest(BaseModel):
    checkpoint_name: str


class AssetSaveRequest(BaseModel):
    id: Optional[str] = None
    name: str
    category: str
    tags: List[str] = []
    frames: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}


class InspectRequest(BaseModel):
    script_text: str
    category: str = "knight"
    palette: str = "pico8"
    palette_name: Optional[str] = None
    width: int = 32
    height: int = 32


class CriticRepairRequest(BaseModel):
    script_text: str
    action_patch: Optional[str] = None
    auto_refine: bool = False
    max_iterations: int = 3
    category: str = "knight"
    palette: str = "pico8"
    palette_name: Optional[str] = None
    width: int = 32
    height: int = 32


class AnimationGenerateRequest(BaseModel):
    prompt: Optional[str] = ""
    base_script: Optional[str] = ""
    category: Optional[str] = "knight"
    style: Optional[str] = "classic_16bit"
    animation_category: str = "walk"
    palette: Optional[str] = "pico8"
    width: int = Field(default=32, ge=16, le=64)
    height: int = Field(default=32, ge=16, le=64)
    num_frames: int = Field(default=6, ge=2, le=16)
    fps: int = Field(default=8, ge=1, le=30)
    engine: str = Field(default="trained_model")  # trained_model | untrained_model | procedural_baseline
    seed: int = 42


class RegenerateFrameRequest(BaseModel):
    base_script: str
    frame_index: int = Field(default=0, ge=0)
    total_frames: int = Field(default=6, ge=2, le=16)
    animation_category: str = "walk"
    category: Optional[str] = "knight"
    palette: Optional[str] = "pico8"
    engine: str = Field(default="trained_model")
    seed: int = 42


class CompositionRequest(BaseModel):
    character_id: Optional[str] = "character_player_soldier"
    character_script: Optional[str] = None
    weapon_id: Optional[str] = "weapon_pump_shotgun"
    helmet_id: Optional[str] = None
    backpack_id: Optional[str] = None
    fx_id: Optional[str] = None
    engine: str = Field(default="trained_model")  # trained_model | untrained_model | procedural_baseline | hybrid
    palette: str = "pico8"
    width: int = 32
    height: int = 32


class CompositionAnimationRequest(BaseModel):
    character_id: Optional[str] = "character_zombie_walker"
    character_script: Optional[str] = None
    animation_category: str = "walk"
    num_frames: int = Field(default=6, ge=2, le=16)
    fps: int = Field(default=8, ge=1, le=30)
    weapon_id: Optional[str] = "weapon_pump_shotgun"
    helmet_id: Optional[str] = None
    backpack_id: Optional[str] = None
    fx_id: Optional[str] = None
    engine: str = Field(default="trained_model")
    palette: str = "pico8"
    width: int = 32
    height: int = 32


class AssetPackExportRequest(BaseModel):
    pack_type: str = "deadzone"  # "deadzone" | "custom"
    pack_name: Optional[str] = "SpriteMind Asset Pack"
    asset_ids: List[str] = []
    include_atlas: bool = True
    padding: int = 2


class TrashAssetRequest(BaseModel):
    asset_id: str
    reason: Optional[str] = "User deletion request"


class RestoreAssetRequest(BaseModel):
    trash_id: str


class PermanentDeleteRequest(BaseModel):
    trash_id: str
    confirm: bool = False


class EmptyTrashRequest(BaseModel):
    confirm: bool = False


class CleanupExecuteRequest(BaseModel):
    candidate_paths: Optional[List[str]] = None
    confirm: bool = False


class BackupCreateRequest(BaseModel):
    backup_name: Optional[str] = None
    include_db: bool = True
    include_benchmarks: bool = True
    include_assets_meta: bool = True



