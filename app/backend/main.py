# main.py — Palm Biometric API v0.6.0
# Includes dual-model support via ModelRegistry.

from contextlib import asynccontextmanager
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import get_settings
from db.database import create_tables, SessionLocal
from ml.cache import EmbeddingCache

# ─── TAMBAH validate_frame DI SINI ───────────────────────────────────────────
from api import health, users, identification, demos, demo_logs, debug, seed, validate_frame, models as models_api
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("palm-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    create_tables()
    app.state.settings = settings

    from ml.detection import HandDetector
    from ml.recognizer import PalmRecognizer
    from ml.registry import ModelRegistry

    # ── Hand Detector ──────────────────────────────────────────────────────────
    detector_ok = os.path.exists(settings.hand_landmarker_path)
    if not detector_ok:
        logger.warning("⚠️  hand_landmarker.task not found at '%s'.", settings.hand_landmarker_path)
    app.state.detector = HandDetector(settings.hand_landmarker_path)

    # ── Model Registry (dual-model support) ───────────────────────────────────
    registry = ModelRegistry(settings.models_dir)
    n_loaded = registry.discover()
    app.state.registry = registry
    logger.info("✓ ModelRegistry: %d model(s) loaded from '%s'", n_loaded, settings.models_dir)
    for m in registry.list_available():
        logger.info("   model: %s (%s) threshold=%.4f", m['id'], m['name'], m['threshold'])

    # ── Backward-compat single recognizer ────────────────────────────────────
    # Keep for services that still use app_state.recognizer directly.
    # Points to the default model via registry if available, else flat model.pt
    default_id = settings.default_model_id
    if registry.is_available(default_id):
        app.state.recognizer = registry.get(default_id)
        app.state.active_model_id = default_id
    else:
        # Fallback to legacy flat model.pt
        recognizer_ok = os.path.exists(settings.recognizer_model_path)
        if not recognizer_ok:
            logger.warning("⚠️  palm_recognizer.pt not found at '%s'.", settings.recognizer_model_path)
        app.state.recognizer = PalmRecognizer(settings.recognizer_model_path)
        app.state.active_model_id = "mobilefacenet-pretrained"

    # ── Embedding Cache ───────────────────────────────────────────────────────
    app.state.cache = EmbeddingCache()
    db = SessionLocal()
    try:
        app.state.cache.warm_up(db)
        logger.info("✓ Cache warmed up — %d users.", app.state.cache.user_count)
    except Exception as exc:
        logger.error("Cache warm-up failed: %s", exc)
    finally:
        db.close()

    logger.info("✓ Backend started (active model: %s).", app.state.active_model_id)
    yield
    logger.info("Backend shutdown.")



settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Palm Biometric identification API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request tidak valid.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Terjadi kesalahan pada server.",
        },
    )


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = int((time.time() - start) * 1000)
    logger.info("%s %s → %s (%dms)", request.method, request.url.path, response.status_code, latency_ms)
    response.headers["X-Latency-Ms"] = str(latency_ms)
    return response


app.include_router(health.router,          prefix="/health",          tags=["health"])
app.include_router(users.router,           prefix="/users",           tags=["users"])
app.include_router(identification.router,                             tags=["identification"])
app.include_router(demo_logs.router,       prefix="/demo-logs",       tags=["demo-logs"])
app.include_router(demos.router,           prefix="/demos",           tags=["demos"])
app.include_router(debug.router,           prefix="/debug",           tags=["debug"])
app.include_router(seed.router,                                       tags=["seed"])
app.include_router(validate_frame.router,                             tags=["validation"])
app.include_router(models_api.router,                                 tags=["models"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)