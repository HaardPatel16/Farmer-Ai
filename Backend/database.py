"""
Sets up the PostgreSQL connection and defines every database table as a
SQLAlchemy model. Run this file directly (python database.py) to create
all tables in your Postgres database.

Other files import from here like:
    from database import SessionLocal, Chat, Feedback, KnowledgeChunk
"""

from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Date, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL


def utcnow_naive() -> datetime:
    """Naive UTC timestamp. Replaces `datetime.utcnow()`, which is
    deprecated in Python 3.12+ and emits a DeprecationWarning on every
    call. We keep the stored value naive (no tzinfo) so it matches the
    existing schema (`DateTime` columns without `timezone=True`) and so
    comparisons against legacy rows continue to work.

    Use this helper everywhere the codebase used to call
    `datetime.utcnow()` directly — both for column defaults below and
    for cache-freshness comparisons in services.py.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

# --- Connection setup ---

# The engine manages the actual connection pool to Postgres.
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory for creating database sessions — each request
# in main.py will open one session, use it, then close it.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the parent class every table model below inherits from.
Base = declarative_base()


# --- Table models ---

class Chat(Base):
    """Stores every question asked and the answer given, regardless of feedback."""
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)          # identifies a user's browser session
    query = Column(Text, nullable=False)              # the farmer's question
    response = Column(Text, nullable=False)           # the bot's answer
    language = Column(String, nullable=False)         # 'en' or 'gu'
    source_type = Column(String, nullable=True)       # 'knowledge_base' / 'llm_reasoning'
    confidence_score = Column(Float, nullable=True)   # how confident the KB match was
    district = Column(String, nullable=True, index=True)  # detected district, if any (e.g. 'bhavnagar')
    created_at = Column(DateTime, default=utcnow_naive)


class Feedback(Base):
    """Stores thumbs up/down on a specific chat response."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    score = Column(Integer, nullable=False)           # 1 = like, -1 = dislike
    reason = Column(String, nullable=True)             # only set for dislikes, e.g. 'wrong_info'
    created_at = Column(DateTime, default=utcnow_naive)


class KnowledgeChunk(Base):
    """Stores processed pieces of your uploaded agricultural documents."""
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source_filename = Column(String, nullable=False)   # which document this came from
    chunk_text = Column(Text, nullable=False)            # the actual text content
    keywords = Column(Text, nullable=True)               # simple keywords for search
    districts = Column(Text, nullable=True, index=True)  # comma-separated district keys this chunk covers, e.g. "rajkot,junagadh,bhavnagar" (NULL = applies Gujarat-wide)
    created_at = Column(DateTime, default=utcnow_naive)


class WeatherCache(Base):
    """Stores recent weather lookups so we don't call the API on every request."""
    __tablename__ = "weather_cache"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String, nullable=False, index=True)
    data_json = Column(Text, nullable=False)            # raw API response as JSON text
    fetched_at = Column(DateTime, default=utcnow_naive)


class MarketPriceSnapshot(Base):
    """One row per (commodity, market, variety, district) for today's IST
    date. Anything older is deleted on every refresh — only the current
    day is ever retained."""
    __tablename__ = "market_price_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)   # IST calendar date this snapshot belongs to
    commodity = Column(String, nullable=False)
    market = Column(String, nullable=True)
    district = Column(String, nullable=True)
    variety = Column(String, nullable=True)
    grade = Column(String, nullable=True)
    arrival_date = Column(String, nullable=True)               # as returned by data.gov.in (string DD/MM/YYYY)
    min_price = Column(String, nullable=True)
    max_price = Column(String, nullable=True)
    modal_price = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)

    __table_args__ = (
        Index(
            "ix_market_snapshot_lookup",
            "snapshot_date", "commodity", "market", "variety", "district",
        ),
    )


# --- Helper ---

def get_db():
    """
    Used by FastAPI routes to get a database session per request,
    and automatically close it afterward even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Run this file directly to create all tables above in your Postgres database.
    # Safe to run multiple times — it only creates tables that don't already exist.
    print("Connecting to database and creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Tables created: chats, feedback, knowledge_chunks, weather_cache, market_price_snapshots")