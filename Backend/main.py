"""
FastAPI app for Farmer AI.
Defines all routes and ties together database.py and services.py.

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to test all endpoints via Swagger UI.
"""

from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db, Chat, Feedback
from services import (
    ask_groq, ask_groq_disease_remedy, search_knowledge_base, get_weather,
    get_all_weather_concurrent, get_market_price, get_market_prices_for_category,
    get_all_market_prices, CROP_CATEGORIES, detect_district,
)
from ml_model import predict_top_k
from config import MARKET_API_KEY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Farmer AI",
    description="AI assistant for farmers in Gujarat, India.",
    version="0.1.0",
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

    # Step 0: weather intent check — if the query asks about weather in a
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
                    "Use these exact figures in your answer; do not say you "
                    "lack real-time access."
                )
        except Exception as e:
            print(f"[weather] get_weather() failed for '{weather_district}': {e}")
            weather_context = None  # fall back to normal flow if the API call fails

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
    if weather_context:
        source_type = "weather_api"
    else:
        source_type = "knowledge_base" if kb_chunks else "llm_reasoning"
    confidence_score = float(len(kb_chunks)) if kb_chunks else None

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
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )

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


MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB — generous for phone photos, cheap to enforce


@app.post("/diagnose")
def diagnose_leaf(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    top_k: int = Form(3),
    language: str = Form("en"),
    db: Session = Depends(get_db),
):
    """
    Accepts a photo of a plant leaf, runs it through the ConvNeXt-Small leaf
    disease classifier, asks Groq for remedies, persists the exchange to the
    chats table (so feedback can reference it), and returns a response
    shaped like /chat so the frontend's chat-rendering path can reuse the
    same fields (chat_id, response, source_type).
    """
    if language not in ("en", "gu"):
        raise HTTPException(status_code=400, detail="language must be 'en' or 'gu'.")
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = image.file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 8 MB).")

    try:
        predictions = predict_top_k(image_bytes, k=top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    try:
        remedy = ask_groq_disease_remedy(predictions, language=language)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )

    # Synthesize a short "user message" describing what the model saw, so
    # the persisted Chat row and the chat-history sidebar make sense
    # (otherwise this conversation has an AI bubble with no preceding user
    # bubble, and the History preview would be blank).
    top_pred = predictions[0] if predictions else {"label": "unknown", "confidence": 0.0}
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
    Uses a 60-minute cache — won't call Open-Meteo on every request.
    """
    try:
        result = get_weather(db, district)
    except Exception as e:
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


@app.get("/market-price/categories")
def market_price_categories():
    """
    Returns the list of crop categories (cash crops, oilseeds, grains &
    cereals, pulses, spices, fruits, vegetables, others) with the crops
    in each — powers the category pills on the frontend's market screen.
    """
    return {
        "categories": [
            {"key": key, "label": cat["label"], "icon": cat["icon"], "crops": cat["crops"]}
            for key, cat in CROP_CATEGORIES.items()
        ]
    }


@app.get("/market-price/category/{category_key}")
def market_price_by_category(
    category_key: str,
    district: str = Query(None, description="Optional: filter by Gujarat district"),
    db: Session = Depends(get_db),
):
    """
    Returns live mandi prices for every crop in a category, e.g. 'oilseeds'.
    Each record has all 9 Agmarknet fields: state, district, market,
    commodity, variety, grade, arrival_date, min_price, max_price, modal_price.
    """
    if category_key not in CROP_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Unknown category '{category_key}'.")

    if not MARKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Market price data is not yet configured. "
                   "Register at data.gov.in to obtain an API key, "
                   "then set MARKET_API_KEY in your .env file.",
        )

    try:
        if category_key == "all":
            result = get_all_market_prices(db, api_key=MARKET_API_KEY, district=district)
        else:
            result = get_market_prices_for_category(db, category_key, api_key=MARKET_API_KEY, district=district)
    except Exception as e:
        print(f"[market-price] category fetch failed for '{category_key}': {e}")
        raise HTTPException(
            status_code=503,
            detail="Market price service is temporarily unavailable. Please try again shortly.",
        )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/market-price")
def market_price(
    commodity: str = Query(..., description="Crop/commodity name, e.g. 'Cotton'"),
    district: str = Query(None, description="Optional: filter by district"),
    market: str = Query(None, description="Optional: filter by mandi/market name"),
    db: Session = Depends(get_db),
):
    """
    Returns live mandi price records for a specific commodity in Gujarat.
    Each record has all 9 Agmarknet fields: state, district, market,
    commodity, variety, grade, arrival_date, min_price, max_price, modal_price.
    """
    if not MARKET_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Market price data is not yet configured. "
                   "Register at data.gov.in to obtain an API key, "
                   "then set MARKET_API_KEY in your .env file.",
        )

    try:
        result = get_market_price(db, commodity, api_key=MARKET_API_KEY, market=market, district=district)
    except Exception as e:
        print(f"[market-price] fetch failed for '{commodity}': {e}")
        raise HTTPException(
            status_code=503,
            detail="Market price service is temporarily unavailable. Please try again shortly.",
        )

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