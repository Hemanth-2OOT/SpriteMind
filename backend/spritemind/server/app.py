"""FastAPI application wiring: middleware, router mounting and static hosting.

Route handlers live in :mod:`spritemind.server.routers`; shared state lives in
:mod:`spritemind.server.runtime`. The handler names are re-exported here because the
test-suite and a few scripts import them from this module.
"""
import hmac
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from spritemind.server import runtime
from spritemind.server.routers import (
    animation as routers_animation,
    assets as routers_assets,
    core as routers_core,
    critic as routers_critic,
    sprites as routers_sprites,
    training as routers_training,
)

from spritemind.server.routers.core import (  # noqa: F401  (re-exported for tests)
    get_health,
    get_models,
    load_checkpoint,
    get_metrics,
    generate,
    get_history,
    delete_history_run,
    clear_history,
)
from spritemind.server.routers.sprites import (  # noqa: F401  (re-exported for tests)
    action_set_plan,
    action_set,
    refine,
    render,
    export_assets,
)
from spritemind.server.routers.training import (  # noqa: F401  (re-exported for tests)
    get_dataset_stats,
    get_dataset_samples,
    generate_dataset,
    start_training,
    stop_training,
    get_training_status,
    get_training_metrics,
    get_training_history,
    get_benchmark_gallery,
    get_benchmark_report_card,
    run_benchmark_endpoint,
    get_evaluation_results,
)
from spritemind.server.routers.critic import (  # noqa: F401  (re-exported for tests)
    inspect_sprite,
    repair_sprite,
    get_critic_metrics,
)
from spritemind.server.routers.animation import (  # noqa: F401  (re-exported for tests)
    generate_animation,
    regenerate_frame,
    get_animation_metrics,
)
from spritemind.server.routers.assets import (  # noqa: F401  (re-exported for tests)
    get_asset_catalog,
    compose_asset,
    animate_composition,
    export_asset_pack,
    get_composition_metrics,
    list_assets,
    get_asset,
    save_asset,
    delete_asset,
    get_storage_dashboard,
    get_storage_records,
    get_storage_orphans,
    sync_storage_registry,
    list_trash,
    trash_asset_endpoint,
    restore_asset_endpoint,
    permanent_delete_endpoint,
    empty_trash_endpoint,
    storage_cleanup_scan,
    storage_cleanup_execute,
    create_backup_endpoint,
    list_backups_endpoint,
)

app = FastAPI(
    title="SpriteMind API",
    description="Local AI Pixel Art Generator & Research Platform",
    version="0.1.0",
)

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:7860,http://127.0.0.1:7860,"
    "http://localhost:7861,http://127.0.0.1:7861"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("SPRITEMIND_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

API_KEY_ENV = "SPRITEMIND_API_KEY"
API_KEY_EXEMPT = {"/api/health", "/docs", "/openapi.json", "/redoc"}

# The routers and the tests both reach for these names directly.
pipeline = runtime.pipeline
trainer = runtime.trainer
dataset_mgr = runtime.dataset_mgr
asset_mgr = runtime.asset_mgr
repair_engine = runtime.repair_engine
storage_db = runtime.storage_db
lifecycle_mgr = runtime.lifecycle_mgr
cleaner = runtime.cleaner
backup_mgr = runtime.backup_mgr


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Optional shared-secret guard for deployments that are not localhost-only."""
    expected = os.environ.get(API_KEY_ENV, "").strip()
    path = request.url.path
    if expected and path.startswith("/api") and path not in API_KEY_EXEMPT:
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            header = request.headers.get("authorization", "")
            if header.lower().startswith("bearer "):
                supplied = header[7:].strip()
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    routers_core.router,
    routers_sprites.router,
    routers_training.router,
    routers_critic.router,
    routers_animation.router,
    routers_assets.router,
):
    app.include_router(router)

# Mount frontend build if directory exists
possible_dist_paths = [
    Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist",
    Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
    Path.cwd() / "frontend" / "dist",
]

for dist_path in possible_dist_paths:
    if dist_path.exists():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")
        break
else:
    @app.get("/", include_in_schema=False)
    def frontend_not_built() -> HTMLResponse:
        return HTMLResponse(
            "<h1>SpriteMind API is running</h1>"
            "<p>The web studio has not been built yet. Run:</p>"
            "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>API docs: <a href='/docs'>/docs</a></p>",
            status_code=200,
        )
