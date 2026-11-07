"""Process-wide singletons shared by the route modules.

Everything here is stateful and long-lived (a loaded SVAN checkpoint, an open SQLite
handle, a training thread), so it is constructed once at import time and referenced by
the routers rather than passed around.
"""
from spritemind.actions import ActionSetBuilder, ActionSetPlanner
from spritemind.assets.composer import AssetComposer
from spritemind.assets.library import AssetLibrary, ASSET_CATEGORIES
from spritemind.assets.pack import AssetPack
from spritemind.critic.repair_engine import AutonomousRepairEngine
from spritemind.critic.model import DEFECT_CLASSES
from spritemind.dataset.assets import AssetManager
from spritemind.dataset.storage import DatasetManager
from spritemind.ml.evaluation import ModelEvaluator
from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.ml.temporal import TemporalAnimator, ANIMATION_CATEGORIES, CATEGORIES
from spritemind.ml.training import ModelTrainer
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.storage.backup import BackupManager
from spritemind.storage.cleanup import StorageCleaner
from spritemind.storage.database import StorageDatabase
from spritemind.server.metrics import LATENCY
from spritemind.storage.lifecycle import AssetLifecycleManager

pipeline = SpriteMindPipeline()
trainer = ModelTrainer()
dataset_mgr = DatasetManager()
asset_mgr = AssetManager()
repair_engine = AutonomousRepairEngine()
storage_db = StorageDatabase()
lifecycle_mgr = AssetLifecycleManager(db=storage_db)
cleaner = StorageCleaner(db=storage_db)
backup_mgr = BackupManager(db=storage_db)
