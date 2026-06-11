from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from config import get_settings

settings = get_settings()

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        "timeout": 30.0,
    }

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    # Pool settings: satu koneksi per thread, tidak ada pool overhead
    pool_pre_ping=True,
)

# ── SQLite WAL mode + tuning ───────────────────────────────────────────────
# WAL (Write-Ahead Logging) memungkinkan concurrent read saat ada write,
# sehingga 5 upload template berurutan tidak saling block.
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")       # concurrent read+write
        cursor.execute("PRAGMA synchronous=NORMAL")     # balance safety vs speed
        cursor.execute("PRAGMA busy_timeout=30000")     # 30 s SQLite-level wait
        cursor.execute("PRAGMA cache_size=-16000")      # 16 MB page cache
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from db import models
    Base.metadata.create_all(bind=engine)
