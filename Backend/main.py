"""
FastAPI app for Farmer AI.
Defines all routes and ties together database.py and services.py.

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to test all endpoints via Swagger UI.
"""

import asyncio
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, Chat, Feedback, SessionLocal
from services import (
    ask_groq, ask_groq_disease_remedy, search_knowledge_base, get_weather,
    get_all_weather_concurrent, get_all_market_prices, detect_district,
    kb_answer_coverage, answer_has_llm_hedge,
    is_farming_question, offtopic_refusal,
    refresh_market_snapshots_async, _seed_last_refresh_from_db,
    MARKET_REFRESH_MINUTES,
)
from ml_model import predict_top_k, warm_bg_remover, warm_classifier, MODEL_PATH
from embeddings import get_status as embeddings_get_status, warm_index_in_background
from config import MARKET_API_KEY

# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed the last-refresh marker so a restart inside a fresh poll window
    # doesn't immediately re-fetch market data.
    db = SessionLocal()
    try:
        _seed_last_refresh_from_db(db)
    finally:
        db.close()

    # Keep a strong reference to the poller task on app.state so it isn't
    # eligible for garbage collection while running, and so the shutdown
    # path below can cancel it cleanly.
    poller_task = asyncio.create_task(_market_price_poller(), name="market-price-poller")
    app.state.poller_task = poller_task

    # Warm the leaf-diagnosis stack (rembg U²-Net + ConvNeXt-Small) only
    # when the .pth model file is present — no point downloading 170 MB
    # of rembg weights or paying torch.load cost if /diagnose will return
    # 503 anyway.
    if os.path.exists(MODEL_PATH):
        def _warm_diagnosis_stack():
            try:
                warm_bg_remover()
                print("[startup] rembg session ready")
            except Exception as e:
                print(f"[startup] rembg warmup failed (diagnose will fall back to raw image): {type(e).__name__}: {e}")
            try:
                warm_classifier()
                print("[startup] ConvNeXt classifier loaded")
            except Exception as e:
                print(f"[startup] classifier warmup failed: {type(e).__name__}: {e}")
        threading.Thread(target=_warm_diagnosis_stack, name="diagnose-warm", daemon=True).start()
    else:
        print(f"[startup] ML model not found at '{MODEL_PATH}' — diagnosis warmup skipped")

    # Warm the sentence-transformer + chunk index; chat works keyword-only
    # until this finishes (~2-5 min on first boot).
    try:
        warm_index_in_background(SessionLocal)
        print("[startup] embeddings warmup started in background thread")
    except Exception as e:
        print(f"[startup] could not start embeddings warmup: {e}")

    try:
        yield
    finally:
        # Cancel the background poller cleanly so uvicorn shutdown doesn't
        # emit "Task was destroyed but it is pending!" warnings, and so the
        # task isn't killed mid-DB-write.
        poller_task.cancel()
        try:
            await poller_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Farmer AI",
    description="AI assistant for farmers in Gujarat, India.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the plain HTML/JS frontend (opened as a local file or served from
# the same machine) to call this backend without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this before any public deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    query: str
    language: str = "en"   # 'en' or 'gu'

class ChatResponse(BaseModel):
    chat_id: int
    response: str
    source_type: str        # 'knowledge_base' or 'llm_reasoning'
    language: str

class FeedbackRequest(BaseModel):
    chat_id: int
    score: int              # 1 = like, -1 = dislike
    reason: str | None = None   # optional, for dislikes only


# ---------------------------------------------------------------------------
# Weather intent detection (used inside /chat)
# ---------------------------------------------------------------------------

WEATHER_KEYWORDS = ["weather", "temperature", "rain", "rainfall", "forecast", "humidity", "climate", "precipitation"]

# Max bytes accepted by /diagnose. 8 MB is generous for any phone-camera
# JPEG/PNG; anything larger is almost certainly an upload mistake or abuse.
# Enforced after the bytes are read; combined with PIL's MAX_IMAGE_PIXELS
# cap in ml_model.py this bounds both compressed and decoded memory cost.
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def detect_weather_request(query: str):
    """
    Returns the matched district name if the query looks like a weather
    question AND mentions a known Gujarat district. Returns None otherwise.
    """
    query_lower = query.lower()

    if not any(keyword in query_lower for keyword in WEATHER_KEYWORDS):
        return None

    return detect_district(query)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Background market-price poller. Runs every MARKET_REFRESH_MINUTES, fetches
# data.gov.in, writes today's snapshot rows, prunes anything older than today.
# Endpoint handlers read straight from the snapshot table.
# ---------------------------------------------------------------------------

async def _market_price_poller():
    # First refresh fires immediately so the snapshot table is populated
    # before any frontend request arrives. HTTP calls are concurrent via
    # the async refresh path; the only sync part is the DB write.
    # CancelledError must propagate so the lifespan shutdown can join the
    # task; everything else is logged and the loop continues.
    while True:
        db = SessionLocal()
        try:
            await refresh_market_snapshots_async(db, MARKET_API_KEY)
        except asyncio.CancelledError:
            db.close()
            raise
        except Exception as e:
            print(f"[market/poller] refresh failed: {type(e).__name__}: {e}")
        finally:
            db.close()
        try:
            await asyncio.sleep(MARKET_REFRESH_MINUTES * 60)
        except asyncio.CancelledError:
            raise


@app.get("/embeddings/status")
def embeddings_status():
    """Tells you whether the semantic-search warmup has finished.
    state: not_started | warming | ready | failed | idle_empty_kb."""
    try:
        return embeddings_get_status()
    except Exception as e:
        return {"state": "unknown", "error": str(e)}


@app.get("/")
def root():
    """Health check — confirms the server is running."""
    return {"status": "ok", "message": "Farmer AI is running."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint.
    1. Searches the knowledge base for relevant chunks.
    2. Passes any matches as context to Groq.
    3. Saves the question + answer to the chats table.
    4. Returns the answer and whether it came from the KB or LLM.
    """
    if request.language not in ("en", "gu"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'gu'.")

    # Step 0 (cheapest): local off-topic filter. If the query mentions
    # nothing farming-related — math, sports, jokes, trivia, etc. — we
    # skip Groq entirely and return a canned refusal. Zero tokens spent,
    # zero quota burned. Groq's compact in-prompt SCOPE line is just a
    # backstop for borderline queries that pass the whitelist.
    if not is_farming_question(request.query):
        refusal = offtopic_refusal(language=request.language)
        chat_row = Chat(
            session_id=request.session_id,
            query=request.query,
            response=refusal,
            language=request.language,
            source_type="llm_reasoning",
            confidence_score=None,
            district=None,
        )
        db.add(chat_row)
        db.commit()
        db.refresh(chat_row)
        return ChatResponse(
            chat_id=chat_row.id,
            response=refusal,
            source_type="llm_reasoning",
            language=request.language,
        )

    # Step 0a: weather intent check — if the query asks about weather in a
    # known district, fetch real live data and hand it to Groq as context
    # instead of letting it guess.
    weather_context = None
    weather_district = detect_weather_request(request.query)
    if weather_district:
        try:
            weather_data = get_weather(db, weather_district)
            if "error" not in weather_data:
                weather_context = (
                    f"Live weather data for {weather_district.title()}: "
                    f"temperature is {weather_data['temperature_c']}°C, "
                    f"humidity is {weather_data['humidity_percent']}%, "
                    f"current rainfall is {weather_data['rainfall_now_mm']}mm, "
                    f"total rainfall so far today is {weather_data['rainfall_today_mm']}mm. "
                    f"Tomorrow's forecast: rainfall {weather_data.get('rainfall_tomorrow_mm', 0)}mm, "
                    f"max temperature {weather_data.get('temp_max_tomorrow_c', '?')}°C, "
                    f"min temperature {weather_data.get('temp_min_tomorrow_c', '?')}°C. "
                    "Use these exact figures in your answer; do not say you "
                    "lack real-time access or forecast data."
                )
        except Exception as e:
            # weather_context stays None (its initial value above), so the
            # rest of /chat proceeds as if the weather lookup didn't fire.
            print(f"[weather] get_weather() failed for '{weather_district}': {e}")

    # Step 0b: general district detection — unlike weather_district above,
    # this isn't gated on weather keywords, so it catches district mentions
    # in ANY kind of farming question (soil, crops, schemes, pests, etc).
    # Reuses weather_district if already found, to avoid detecting twice.
    district = weather_district or detect_district(request.query)

    # Step 1: knowledge base search
    try:
        kb_chunks = search_knowledge_base(db, request.query, district=district)
    except Exception:
        kb_chunks = []   # don't crash the whole chat if KB search fails

    context_parts = []
    if weather_context:
        context_parts.append(weather_context)
    context_parts.extend(kb_chunks)

    context = "\n\n".join(context_parts)

    # Step 2: call Groq — pass district through so it tailors advice to
    # local soil/climate/crop conditions, combined with any KB context.
    try:
        answer = ask_groq(
            question=request.query,
            language=request.language,
            context=context,
            district=district,
        )
    except Exception as e:
        # Log the underlying error so 503s are diagnosable from the uvicorn
        # terminal — without this print, a Groq 429 (rate limit), an
        # invalid API key, and a network outage all look identical to the
        # frontend, with no clue in the backend either.
        print(f"[chat] ask_groq failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )

    # Decide source_type AFTER the answer is generated, by measuring how
    # much of the answer's vocabulary overlaps with the retrieved chunks.
    # Three-tier so a partial-KB answer doesn't get the same badge as a
    # purely KB-derived one:
    #   coverage < 0.20  -> "llm_reasoning"   (KB had little/no influence)
    #   0.20 <= cov < 0.70 -> "mixed"           (some KB, mostly LLM filled gaps)
    #   coverage >= 0.70 -> "knowledge_base"  (answer is paraphrasing chunks)
    # Weather always wins when present, since the live-data injection
    # is the whole reason that path exists.
    kb_coverage = kb_answer_coverage(answer, kb_chunks)
    # A "knowledge_base" badge implies the answer came essentially from
    # your documents. The LLM hedging language ("the reference doesn't
    # mention X, however based on my general knowledge…") is a direct
    # admission it's mixing — demote to "mixed" even if coverage looks
    # high, since the user-visible truth is that part of the answer
    # came from training data.
    llm_hedged = bool(kb_chunks) and answer_has_llm_hedge(answer)
    if weather_context:
        source_type = "weather_api"
    elif kb_coverage >= 0.70 and not llm_hedged:
        source_type = "knowledge_base"
    elif kb_coverage >= 0.20 or llm_hedged:
        source_type = "mixed"
    else:
        source_type = "llm_reasoning"

    # Store the actual coverage ratio (0-1) so it's introspectable in
    # pgAdmin — much more meaningful than the old "count of chunks"
    # placeholder, and lets you sanity-check threshold calibration.
    confidence_score = round(kb_coverage, 3) if kb_chunks else None

    # Step 3: save to DB
    chat_row = Chat(
        session_id=request.session_id,
        query=request.query,
        response=answer,
        language=request.language,
        source_type=source_type,
        confidence_score=confidence_score,
        district=district,
    )
    db.add(chat_row)
    db.commit()
    db.refresh(chat_row)

    return ChatResponse(
        chat_id=chat_row.id,
        response=answer,
        source_type=source_type,
        language=request.language,
    )


@app.post("/diagnose")
async def diagnose_leaf(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    top_k: int = Form(3),
    language: str = Form("en"),
    remove_bg: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Accepts a photo of a plant leaf, runs it through the ConvNeXt-Small leaf
    disease classifier (with optional rembg background removal), asks Groq
    for remedies, persists the exchange to the chats table (so feedback can
    reference it), and returns a response shaped like /chat so the
    frontend's chat-rendering path can reuse the same fields (chat_id,
    response, source_type).
    """
    if language not in ("en", "gu"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'gu'.")
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10.")
    # Require a present, image/* Content-Type. The previous version
    # short-circuited when the header was missing, letting a client bypass
    # the check by simply omitting it; PIL would still reject it later but
    # the error path leaked decoder internals.
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image with a valid image/* Content-Type.")

    # Async read so an 8 MB upload doesn't pin the event loop the way the
    # previous sync `image.file.read()` did.
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 8 MB).")

    # predict_top_k is CPU-bound (rembg ONNX + ConvNeXt forward) — offload
    # to a worker thread so concurrent requests aren't serialized on the
    # event loop. Same treatment for the Groq call below.
    try:
        predictions = await asyncio.to_thread(predict_top_k, image_bytes, top_k, remove_bg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # Raised by _open_raw_rgb on un-decodable bytes — message is
        # safe to surface (no PIL internals).
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[diagnose] predict_top_k failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=400, detail="Could not process image.")

    try:
        remedy = await asyncio.to_thread(ask_groq_disease_remedy, predictions, language)
    except Exception as e:
        print(f"[diagnose] ask_groq_disease_remedy failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )

    # Synthesize a short "user message" describing what the model saw, so
    # the persisted Chat row and the chat-history sidebar make sense
    # (otherwise this conversation has an AI bubble with no preceding user
    # bubble, and the History preview would be blank).
    # predict_top_k() is contracted to return at least one item (k is
    # clamped to min(k, NUM_CLASSES) and NUM_CLASSES is 78), so no empty-
    # list guard is needed here.
    top_pred = predictions[0]
    query_label = (
        "🌿 Leaf diagnosis — most likely "
        f"{top_pred['label']} ({top_pred['confidence'] * 100:.1f}% confidence)"
    )

    chat_row = Chat(
        session_id=session_id,
        query=query_label,
        response=remedy,
        language=language,
        source_type="leaf_diagnosis",
        confidence_score=top_pred["confidence"],
        district=None,
    )
    db.add(chat_row)
    db.commit()
    db.refresh(chat_row)

    return {
        "chat_id": chat_row.id,
        "response": remedy,
        "source_type": "leaf_diagnosis",
        "language": language,
        "predictions": predictions,
    }


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Stores a thumbs up (score=1) or thumbs down (score=-1) for a chat response.
    Dislikes can optionally include a reason: 'wrong_info', 'wrong_language', 'irrelevant'.
    """
    if request.score not in (1, -1):
        raise HTTPException(status_code=400, detail="score must be 1 (like) or -1 (dislike).")

    # Confirm the chat_id actually exists
    chat_row = db.query(Chat).filter(Chat.id == request.chat_id).first()
    if not chat_row:
        raise HTTPException(status_code=404, detail=f"chat_id {request.chat_id} not found.")

    feedback_row = Feedback(
        chat_id=request.chat_id,
        score=request.score,
        reason=request.reason if request.score == -1 else None,
    )
    db.add(feedback_row)
    db.commit()

    return {"status": "ok", "chat_id": request.chat_id, "score": request.score}


@app.get("/weather")
def weather(
    district: str = Query(..., description="Gujarat district name, e.g. 'Ahmedabad'"),
    db: Session = Depends(get_db),
):
    """
    Returns current weather for a Gujarat district.
    Backed by the same WEATHER_CACHE_MINUTES cache as /weather/all — won't
    call Open-Meteo on every request.
    """
    try:
        result = get_weather(db, district)
    except Exception as e:
        print(f"[weather] get_weather failed for '{district}': {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Weather service is temporarily unavailable. Please try again shortly.",
        )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/weather/all")
async def weather_all(db: Session = Depends(get_db)):
    """
    Returns current weather for every known Gujarat district in one call.
    Powers the weather dashboard modal on the frontend.

    Open-Meteo calls for cache-miss districts run concurrently via
    httpx.AsyncClient + asyncio.gather, so the cold-cache path (server
    just restarted, all 33 entries expired at once) takes ~1s instead of
    ~10s. Cache-hit districts are served from a single batched DB query,
    so the warm path stays effectively instant. Same WEATHER_CACHE_MINUTES
    cache as the single-district /weather endpoint.
    """
    try:
        results = await get_all_weather_concurrent(db)
    except Exception as e:
        print(f"[weather/all] batch failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Weather service is temporarily unavailable. Please try again shortly.",
        )

    return {"districts": results}


@app.get("/market-price/all")
def market_price_all(
    district: str = Query(None, description="Optional: filter by Gujarat district"),
    db: Session = Depends(get_db),
):
    """Returns every snapshot row stored for today across all of Gujarat.
    Powers the single flat table on the frontend's market screen."""
    if not MARKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Market price data is not yet configured. "
                   "Register at data.gov.in to obtain an API key, "
                   "then set MARKET_API_KEY in your .env file.",
        )
    try:
        result = get_all_market_prices(db, api_key=MARKET_API_KEY, district=district)
    except Exception as e:
        print(f"[market-price/all] fetch failed: {e}")
        raise HTTPException(status_code=503, detail="Market price service is temporarily unavailable.")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/chat/history")
def chat_history(
    session_id: str = Query(..., description="Session ID to retrieve history for"),
    db: Session = Depends(get_db),
):
    """
    Returns all previous chat messages for a given session_id.
    Useful for the frontend to restore conversation history on page reload.
    """
    rows = (
        db.query(Chat)
        .filter(Chat.session_id == session_id)
        .order_by(Chat.created_at.asc())
        .all()
    )

    return [
        {
            "chat_id": row.id,
            "query": row.query,
            "response": row.response,
            "language": row.language,
            "source_type": row.source_type,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@app.get("/chat/sessions")
def chat_sessions(db: Session = Depends(get_db)):
    """
    Returns one entry per distinct conversation (session_id), newest first,
    for the History list in the sidebar. Each entry includes the FIRST
    query asked in that session (used as a short preview/title) and the
    timestamp of the most recent message in it.

    Note: this currently returns sessions across ALL users, since there's
    no per-user account system yet — every session_id stored in the chats
    table shows up here. Fine for a single-farmer-at-a-time use case today;
    would need a user_id column to scope this properly once there's auth.
    """
    # Single SQL pass instead of one query per session: aggregate min/max
    # created_at per session_id, then join back to Chat to pull the actual
    # first message's query text. Previously this looped over every session
    # and ran a separate SELECT for the first message — O(N) database
    # round-trips per sidebar load.
    agg = (
        db.query(
            Chat.session_id.label("session_id"),
            func.min(Chat.created_at).label("first_at"),
            func.max(Chat.created_at).label("last_active"),
        )
        .group_by(Chat.session_id)
        .subquery()
    )

    rows = (
        db.query(Chat.session_id, Chat.query, agg.c.last_active)
        .join(
            agg,
            (Chat.session_id == agg.c.session_id) & (Chat.created_at == agg.c.first_at),
        )
        .order_by(agg.c.last_active.desc())
        .all()
    )

    return [
        {
            "session_id": session_id,
            "preview": query_text,
            "last_active": last_active.isoformat(),
        }
        for session_id, query_text, last_active in rows
    ]


@app.delete("/chat/session/{session_id}")
def delete_chat_session(session_id: str, db: Session = Depends(get_db)):
    """
    Permanently deletes every chat message belonging to a session_id, along
    with any feedback (👍/👎) left on those messages.
    This is a HARD delete — rows are removed from the database, not just
    hidden — so this cannot be undone.

    Feedback rows must be deleted first: Feedback.chat_id has a foreign key
    to Chat.id, so Postgres rejects deleting a chat that still has feedback
    pointing at it (ForeignKeyViolation) unless the feedback goes first.
    """
    chat_ids = [
        row.id for row in
        db.query(Chat.id).filter(Chat.session_id == session_id).all()
    ]

    if not chat_ids:
        raise HTTPException(status_code=404, detail=f"No chat history found for session '{session_id}'.")

    db.query(Feedback).filter(Feedback.chat_id.in_(chat_ids)).delete(synchronize_session=False)

    deleted_count = (
        db.query(Chat)
        .filter(Chat.session_id == session_id)
        .delete()
    )
    db.commit()

    return {"deleted": True, "session_id": session_id, "rows_deleted": deleted_count}