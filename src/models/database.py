import os
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

Base = declarative_base()


class KYCRecord(Base):
    __tablename__ = "kyc_records"

    id = Column(Integer, primary_key=True, index=True)

    # customer_id is intentionally NOT unique: a customer can have multiple
    # verification attempts (rejected -> resubmitted -> verified), and a
    # KYC system needs that full history for audit/dispute/fraud review,
    # not just the latest attempt overwriting the record.
    customer_id = Column(String, index=True)

    full_name = Column(String)
    document_type = Column(String, nullable=True)
    document_number = Column(String, nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    address = Column(String, nullable=True)

    verification_status = Column(String, default="pending", index=True)
    risk_level = Column(String, nullable=True)
    face_match_score = Column(Float, nullable=True)

    document_path = Column(String, nullable=True)
    selfie_path = Column(String, nullable=True)

    # Full AI-service response payload (vision checks, template matches,
    # face-match details, etc.) kept for audit trail / manual review.
    ai_response = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Configurable via env var so this can point at Postgres/MySQL in
# staging/production without code changes; falls back to local SQLite
# for development.
DATABASE_URL = os.environ.get("KYC_DATABASE_URL", "sqlite:///./kyc.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")

# QueuePool + pool_size/max_overflow are meaningless for SQLite (it
# doesn't pool connections the same way and file-based SQLite is
# single-writer regardless), so only apply them for real DB backends.
_engine_kwargs = {}
if not _is_sqlite:
    _engine_kwargs.update(
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
    )
else:
    # SQLite requires this when the same connection may be used across
    # threads, which happens here since FastAPI runs sync dependencies
    # in a thread pool.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)

if _is_sqlite:
    # Ensure the directory for the SQLite file exists (e.g. "./data/kyc.db").
    sqlite_path = DATABASE_URL.replace("sqlite:///", "", 1)
    sqlite_dir = os.path.dirname(sqlite_path)
    if sqlite_dir:
        os.makedirs(sqlite_dir, exist_ok=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

Base.metadata.create_all(bind=engine)

# Standard per-request session factory. Deliberately NOT a scoped_session:
# FastAPI's dependency-injection lifecycle (via `get_db` below) already
# gives each request its own Session with a clear open/close boundary,
# and scoped_session's thread-local caching can leak session state across
# unrelated requests when threads get reused from FastAPI's thread pool.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: yields a fresh Session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()