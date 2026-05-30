# ── PATCH untuk main.py ──────────────────────────────────────────────────────
#
# Tambahkan dua baris ini ke main.py bestehende:
#
# 1. Di bagian import (setelah baris `from api import health, users, ...`):
#    from api import validate_frame
#
# 2. Di bagian router registration (setelah baris `app.include_router(seed.router, ...)`):
#    app.include_router(validate_frame.router, tags=["validation"])
#
# ── VERSI LENGKAP main.py SETELAH PATCH ─────────────────────────────────────

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
from api import health, users, identification, demos, demo_logs, debug, seed, validate_frame
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

    detector_ok = os.path.exists(settings.hand_landmarker_path)
    recognizer_ok = os.path.exists(settings.recognizer_model_path)

    if not detector_ok:
        logger.warning("⚠️  hand_landmarker.task not found at '%s'.", settings.hand_landmarker_path)
    if not recognizer_ok:
        logger.warning("⚠️  palm_recognizer.pt not found at '%s'.", settings.recognizer_model_path)

    app.state.detector = HandDetector(settings.hand_landmarker_path)
    app.state.recognizer = PalmRecognizer(settings.recognizer_model_path)

    app.state.cache = EmbeddingCache()
    db = SessionLocal()
    try:
        app.state.cache.warm_up(db)
        logger.info("✓ Cache warmed up — %d users.", app.state.cache.user_count)
    except Exception as exc:
        logger.error("Cache warm-up failed: %s", exc)
    finally:
        db.close()

    logger.info("✓ Backend started.")
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

# ─── TAMBAH BARIS INI ────────────────────────────────────────────────────────
app.include_router(validate_frame.router,                             tags=["validation"])
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)