from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from db.database import create_tables
from services.identification import get_palm_service
from api import health, users, identification, demos, demo_logs, debug, seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Startup
    create_tables()
    app.state.settings = settings
    app.state.palm_service = get_palm_service(settings)
    app.state.cache = None

    print("Backend started. DB ready.")
    yield

    # Shutdown
    print("Backend shutdown.")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Palm recognition API for enrollment, identification, and demo modules.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:5500', '*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(identification.router, tags=["identification"])
app.include_router(demo_logs.router, prefix="/demo-logs", tags=["demo-logs"])
app.include_router(demos.router, prefix="/demos", tags=["demos"])
app.include_router(debug.router, prefix="/debug", tags=["debug"])
app.include_router(seed.router, tags=["seed"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
