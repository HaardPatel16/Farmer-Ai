"""
FastAPI app for Farmer AI.
Defines all routes and ties together database.py and services.py.

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs to test all endpoints via Swagger UI.
"""

import asyncio
import hmac
import os
import sys
import threading
from contextlib import asynccontextmanager

# Force stdout/stderr to UTF-8 before any other module's import-time prints
# can fire. The default Windows console is cp1252 ("charmap"), which raises
# UnicodeEncodeError on common diagnostic characters (Gujarati script,
# arrows, check-marks, em-dashes). A failed print inside a background
# thread — like the embeddings warmup — kills that thread silently and
# leaves the rest of the app pretending the work is still in progress.
# This is a pure-output fix; it does not affect file I/O or DB I/O.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
    MARKET_REFRESH_MINUTES, CHAT_HISTORY_MAX_TURNS,
)
from ml_model import predict_top_k, warm_bg_remover, warm_classifier, MODEL_PATH
from embeddings import get_status as embeddings_get_status, warm_index_in_background
from config import MARKET_API_KEY, ALLOWED_ORIGINS, ADMIN_TOKEN

# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One-line GPU summary up front — the individual warmup threads below
    # (embeddings, classifier, rembg) each log their own device once they
    # actually load, but that happens seconds to minutes later on separate
    # daemon threads. Printing the raw CUDA availability here immediately
    # tells an operator whether any GPU accel is even possible on this
    # machine, without waiting on those threads.
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[startup] CUDA GPU detected: {torch.cuda.get_device_name(0)} — ML workloads will use it")
        else:
            print("[startup] no CUDA GPU detected — embeddings/classifier/rembg will run on CPU")
    except Exception as e:
        print(f"[startup] GPU check failed ({type(e).__name__}: {e}) — assuming CPU")

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
# the same machine) to call this backend without CORS errors. Origins,
# methods, and headers are all narrowed: a public deployment should set
# ALLOWED_ORIGINS in .env, and the method/header allow-lists block stray
# preflight surprises like a malicious site sending TRACE or arbitrary
# X-* headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
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
    # source_type is one of: 'knowledge_base' (>=70% KB coverage),
    # 'mixed' (20-70% coverage or LLM hedge detected), 'weather_api'
    # (live Open-Meteo data injected as context), or 'llm_reasoning'
    # (no real KB grounding). /diagnose returns 'leaf_diagnosis' via
    # its own ad-hoc dict, not this schema.
    source_type: str
    language: str

class FeedbackRequest(BaseModel):
    chat_id: int
    score: int              # 1 = like, -1 = dislike
    reason: str | None = None   # optional, for dislikes only


# ---------------------------------------------------------------------------
# Weather intent detection (used inside /chat)
# ---------------------------------------------------------------------------

# English + Gujarati weather vocabulary. Gujarati farmers often type their
# questions in Gujarati script ("શું વરસાદ આવશે?", "હવામાન કેવું છે?") and
# the previous English-only list never triggered the live-weather injection
# for them, so Groq had to guess. Each term is matched as a plain substring,
# so root forms cover the inflected variants ("વરસાદ"/"વરસાદી", "તાપમાન"/
# "તાપમાનની"). Casing only matters for the English half — Gujarati script
# has no case, so query_lower doesn't affect those.
WEATHER_KEYWORDS = [
    # English
    "weather", "temperature", "rain", "rainfall", "forecast",
    "humidity", "climate", "precipitation",
    # Gujarati: hawaman (weather), varsad (rain), tapaman (temperature),
    # aagahi (forecast), bharaaye (humidity), vaataavaran (climate)
    "હવામાન", "વરસાદ", "તાપમાન", "આગાહી", "ભેજ", "વાતાવરણ",
]

# Max bytes accepted by /diagnose. 8 MB is generous for any phone-camera
# JPEG/PNG; anything larger is almost certainly an upload mistake or abuse.
# Enforced after the bytes are read; combined with PIL's MAX_IMAGE_PIXELS
# cap in ml_model.py this bounds both compressed and decoded memory cost.
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# Source types that should NOT be replayed as prior turns to Groq:
#  - "llm_reasoning" from the local off-topic refusal path never called
#    Groq, and its canned refusal text would just confuse a follow-up
#    (we handle that here rather than tag those rows specifically).
#  - "leaf_diagnosis" replies are image-triggered and reference labels
#    the follow-up turn has no image for.
# Everything else (knowledge_base / mixed / weather_api / genuine
# llm_reasoning that DID call Groq) is fair game.
_HISTORY_EXCLUDED_SOURCES = {"leaf_diagnosis"}


def load_chat_history(db: Session, session_id: str) -> list[dict]:
    """Return the last CHAT_HISTORY_MAX_TURNS (user, assistant) pairs for
    this session, in chronological order, in Groq's messages shape. Skips
    leaf-diagnosis rows since their "user message" is a synthetic label
    and their answer references an image the follow-up doesn't have.
    Off-topic refusals are also skipped: their query text is real but the
    canned refusal reply would derail any follow-up if replayed as
    prior context."""
    rows = (
        db.query(Chat)
        .filter(Chat.session_id == session_id)
        .filter(~Chat.source_type.in_(_HISTORY_EXCLUDED_SOURCES))
        .order_by(Chat.created_at.desc())
        .limit(CHAT_HISTORY_MAX_TURNS * 3)  # over-fetch, then filter refusals below
        .all()
    )
    # Drop off-topic refusals (they share source_type "llm_reasoning" with
    # legitimate LLM answers, so we can't filter them out in SQL — the
    # cheap tell is `chunks_sent_count is None AND confidence_score is None`
    # AND source_type=='llm_reasoning', which is exactly how the refusal
    # branch persists rows).
    kept = [
        r for r in rows
        if not (
            r.source_type == "llm_reasoning"
            and r.chunks_sent_count is None
            and r.confidence_score is None
        )
    ]
    kept = kept[:CHAT_HISTORY_MAX_TURNS]
    kept.reverse()  # oldest → newest, matching conversational order

    history: list[dict] = []
    for r in kept:
        history.append({"role": "user", "content": r.query})
        history.append({"role": "assistant", "content": r.response})
    return history


# Anaphoric pronouns that signal "this query refers back to a previous
# turn". Matched as whole words so "italy" doesn't trip on "it".
_FOLLOWUP_PRONOUNS = {
    "it", "its", "it's", "this", "that", "these", "those", "them", "they",
    "there", "he", "she", "him", "her", "his", "hers", "their", "theirs",
}

# Short-query threshold (words after stripping punctuation). A 1-3 word
# question like "and irrigation?" or "what about fertilizer" clearly
# depends on prior turns; longer questions usually stand on their own.
_FOLLOWUP_SHORT_WORDS = 3


def _looks_like_followup(query: str) -> bool:
    """True if the query looks like it depends on earlier turns — either
    it's very short or it contains an anaphoric pronoun. Used to decide
    whether KB retrieval should be augmented with prior user turns."""
    words = [w.strip(".,!?:;\"'()[]{}<>").lower() for w in query.split()]
    words = [w for w in words if w]
    if len(words) <= _FOLLOWUP_SHORT_WORDS:
        return True
    if any(w in _FOLLOWUP_PRONOUNS for w in words):
        return True
    return False


def _build_retrieval_query(current: str, history: list[dict]) -> str:
    """Return the query string used for KB search + district detection.
    Concatenates prior user turns onto `current` only when `current`
    looks like a follow-up per _looks_like_followup(); otherwise returns
    the raw current query so a fresh topic isn't polluted by earlier
    conversation."""
    if not history or not _looks_like_followup(current):
        return current
    prev_user_turns = " ".join(
        m["content"] for m in history if m.get("role") == "user"
    ).strip()
    return f"{prev_user_turns} {current}".strip() if prev_user_turns else current


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
    state: not_started | warming | ready | failed | idle_empty_kb.
    Also returns `warm_error` (str or null) when state == "failed" so
    operators can see the actual exception message in the response."""
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
    #
    # Exception: mid-conversation follow-ups. If the session already has
    # at least one on-topic exchange AND the current query is short or
    # pronoun-heavy ("about its fetrilizers", "and irrigation?"), treat
    # it as a continuation instead of refusing. Rejecting these was the
    # single biggest usability hit: typos and pronouns will never match
    # the vocabulary whitelist, but they're almost never off-topic in an
    # active farming chat. We DO peek at history here (an extra DB read
    # on the refusal path) — cheap compared to the wrong refusal.
    is_followup_context = False
    if not is_farming_question(request.query):
        try:
            _peek_history = load_chat_history(db, request.session_id)
        except Exception:
            _peek_history = []
        if _peek_history and _looks_like_followup(request.query):
            is_followup_context = True

    if not is_followup_context and not is_farming_question(request.query):
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

    # Step 0b: pull recent conversation for this session so follow-ups
    # like "and what fertilizer for that?" have context. Bounded by
    # CHAT_HISTORY_MAX_TURNS and further trimmed by ask_groq to
    # CHAT_HISTORY_MAX_CHARS (~600 words) as a safety net. Loaded BEFORE
    # KB search so we can also feed the previous user turn into keyword
    # retrieval — otherwise "tell me about its irrigation" as a follow-up
    # to a cotton question retrieves generic irrigation chunks (no "cotton"
    # token in the query) and Groq's answer scores 0% KB coverage.
    try:
        history = load_chat_history(db, request.session_id)
    except Exception as e:
        print(f"[chat] load_chat_history failed (proceeding without): {e}")
        history = []

    # Build a retrieval-only query. Only augment with previous user turns
    # when the CURRENT query looks like a follow-up (short, or contains
    # anaphoric pronouns like "it/its/that/those"). A self-contained
    # question that names a real topic — "tell me about bajra in gujarat"
    # — must NOT be diluted with earlier cotton/irrigation words, or
    # search_knowledge_base returns a mix of unrelated chunks and coverage
    # drops below the KB threshold. Groq's messages array carries full
    # history either way, so context isn't lost for the LLM.
    retrieval_query = _build_retrieval_query(request.query, history)

    # Step 0c: general district detection — unlike weather_district above,
    # this isn't gated on weather keywords, so it catches district mentions
    # in ANY kind of farming question (soil, crops, schemes, pests, etc).
    # Runs against retrieval_query so "what about Bhavnagar?" following an
    # earlier cotton question still detects the district; falls back to
    # current-query-only if that yields nothing.
    district = weather_district or detect_district(retrieval_query)

    # Step 1: knowledge base search — using the augmented retrieval_query
    # so a follow-up like "tell me about its irrigation" carries the
    # earlier "cotton" token into keyword scoring and semantic search.
    # The exception path LOGS the failure now (previously it silently
    # swallowed, which is exactly how the "0 chunks with no [search_kb]
    # trace" symptom was going undiagnosed — the function was crashing
    # and we couldn't tell from the outside).
    print(f"[chat] calling search_knowledge_base retrieval={retrieval_query!r}", flush=True)
    try:
        kb_chunks = search_knowledge_base(db, retrieval_query, district=district)
    except Exception as e:
        print(f"[chat] search_knowledge_base RAISED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        kb_chunks = []

    # Terminal-side visibility so retrieval can be debugged live while
    # tailing uvicorn output, not just after the fact from pgAdmin.
    print(
        f"[chat] sent {len(kb_chunks)} chunks to Groq | "
        f"query='{request.query[:80]}' | retrieval='{retrieval_query[:120]}' | district={district}"
    )

    context_parts = []
    if weather_context:
        context_parts.append(weather_context)
    context_parts.extend(kb_chunks)

    context = "\n\n".join(context_parts)

    # Step 2: call Groq — pass district through so it tailors advice to
    # local soil/climate/crop conditions, combined with any KB context.
    try:
        answer, usage = ask_groq(
            question=request.query,
            language=request.language,
            context=context,
            district=district,
            history=history,
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
    # Persist the chunks that fed Groq so retrieval failures are
    # diagnosable from pgAdmin. Join with a clear separator so SELECTs
    # stay readable; cap total length at ~6 KB so a runaway retrieval
    # can't blow up the row size.
    if kb_chunks:
        chunks_joined = "\n\n---\n\n".join(kb_chunks)
        if len(chunks_joined) > 6000:
            chunks_joined = chunks_joined[:6000] + "…[truncated]"
    else:
        chunks_joined = None

    # Terminal-side token visibility so quota spend can be tailed live.
    # Also stored on the row so cumulative per-session cost is queryable
    # from pgAdmin without re-parsing logs.
    print(
        f"[chat] tokens: prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} | history_turns={len(history) // 2}"
    )

    chat_row = Chat(
        session_id=request.session_id,
        query=request.query,
        response=answer,
        language=request.language,
        source_type=source_type,
        confidence_score=confidence_score,
        district=district,
        chunks_sent_count=len(kb_chunks),
        chunks_sent=chunks_joined,
        prompt_tokens=usage["prompt_tokens"] or None,
        completion_tokens=usage["completion_tokens"] or None,
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

    # Cheap pre-check on the Content-Length header (when present) so we
    # reject a 500 MB body *before* allocating 500 MB of RAM to read it.
    # Curl, every browser, and the project's frontend all set this header
    # on file uploads. A malicious client could omit it — but then the
    # post-read length check below still catches them; this is just the
    # first line of defense, not the only one.
    declared_size = getattr(image, "size", None)
    if declared_size is not None and declared_size > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 8 MB).")

    # Async read so an 8 MB upload doesn't pin the event loop the way the
    # previous sync `image.file.read()` did.
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 8 MB).")

    # predict_top_k is CPU-bound (rembg ONNX + ConvNeXt forward) — offload
    # to a worker thread so concurrent requests aren't serialized on the
    # event loop. Same treatment for the Groq call below.
    try:
        diagnosis = await asyncio.to_thread(predict_top_k, image_bytes, top_k, remove_bg)
        predictions = diagnosis["predictions"]
        processed_image_b64 = diagnosis["processed_image_b64"]
        bg_removed = diagnosis["bg_removed"]
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
        # Small WebP preview of the image the classifier actually saw — the
        # frontend crossfades the user's upload thumbnail to this so the
        # farmer sees exactly what the model was looking at. None if rembg
        # was disabled and we skipped the encode, or if encoding failed.
        "processed_image": (
            f"data:image/webp;base64,{processed_image_b64}"
            if processed_image_b64 else None
        ),
        "bg_removed": bg_removed,
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


@app.get("/stats")
def stats(
    token: str = Query(None, description="Admin token; required only if ADMIN_TOKEN is set in .env"),
    db: Session = Depends(get_db),
):
    """
    Operator-only dashboard data: usage, feedback satisfaction, and a list of
    recent dislikes with the retrieval diagnostics already stored on each
    Chat row (source_type, KB coverage, and the exact chunks_sent to Groq).
    Powers the hidden Statistics view in the frontend's Settings panel.

    Auth: possession-of-token, same model as DELETE /chat/session. If
    ADMIN_TOKEN is configured in .env we require a constant-time match;
    if it's unset (local dev) the endpoint is open. Constant-time compare
    (hmac.compare_digest) avoids leaking token length/prefix via response
    timing.
    """
    if ADMIN_TOKEN:
        if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
            raise HTTPException(status_code=401, detail="Invalid or missing admin token.")

    total_chats = db.query(func.count(Chat.id)).scalar() or 0

    # Feedback can accumulate MORE THAN ONE row per chat: the chat UI
    # re-enables its vote buttons after a page reload (history re-renders
    # them fresh), so a farmer can rate the same answer again. Counting
    # every row double-counts those votes and inflates every satisfaction
    # number. Restrict ALL feedback aggregation below to the LATEST vote per
    # chat — the highest Feedback.id for each chat_id. Reused as a subquery
    # in each .in_() filter.
    latest_feedback_ids = (
        db.query(func.max(Feedback.id)).group_by(Feedback.chat_id)
    )

    likes = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.score == 1, Feedback.id.in_(latest_feedback_ids))
        .scalar()
    ) or 0
    dislikes = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.score == -1, Feedback.id.in_(latest_feedback_ids))
        .scalar()
    ) or 0
    total_feedback = likes + dislikes
    like_pct = round(100.0 * likes / total_feedback, 1) if total_feedback else None

    # Satisfaction by source_type: total chats per type, plus how many of
    # those got liked / disliked. This is the headline "does the KB pipeline
    # actually help?" metric — compare like/dislike ratios across
    # knowledge_base vs mixed vs llm_reasoning.
    by_source: dict[str, dict] = {}
    for st, cnt in (
        db.query(Chat.source_type, func.count(Chat.id))
        .group_by(Chat.source_type)
        .all()
    ):
        by_source[st or "unknown"] = {"chats": cnt, "likes": 0, "dislikes": 0}
    for st, score, cnt in (
        db.query(Chat.source_type, Feedback.score, func.count(Feedback.id))
        .join(Feedback, Feedback.chat_id == Chat.id)
        .filter(Feedback.id.in_(latest_feedback_ids))
        .group_by(Chat.source_type, Feedback.score)
        .all()
    ):
        bucket = by_source.setdefault(st or "unknown", {"chats": 0, "likes": 0, "dislikes": 0})
        if score == 1:
            bucket["likes"] = cnt
        elif score == -1:
            bucket["dislikes"] = cnt
    by_source_type = [
        {"source_type": k, **v}
        for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]["chats"])
    ]

    # Dislikes grouped by the reason the farmer picked (wrong_info /
    # wrong_language / irrelevant / other), most common first.
    dislikes_by_reason = [
        {"reason": r or "unspecified", "count": c}
        for r, c in sorted(
            db.query(Feedback.reason, func.count(Feedback.id))
            .filter(Feedback.score == -1, Feedback.id.in_(latest_feedback_ids))
            .group_by(Feedback.reason)
            .all(),
            key=lambda x: -x[1],
        )
    ]

    # Dislikes grouped by district — surfaces regions the KB covers poorly.
    dislikes_by_district = [
        {"district": d or "none", "count": c}
        for d, c in sorted(
            db.query(Chat.district, func.count(Feedback.id))
            .join(Feedback, Feedback.chat_id == Chat.id)
            .filter(Feedback.score == -1, Feedback.id.in_(latest_feedback_ids))
            .group_by(Chat.district)
            .all(),
            key=lambda x: -x[1],
        )
    ][:15]

    # Usage + token spend per day, last 14 days (newest first). func.date()
    # collapses the timestamp to a calendar day; token sum coalesces NULLs
    # (refusal rows never called Groq) to 0 so the sum stays numeric.
    usage_by_day = [
        {"day": str(day), "chats": c, "tokens": int(tok or 0)}
        for day, c, tok in (
            db.query(
                func.date(Chat.created_at),
                func.count(Chat.id),
                func.coalesce(
                    func.sum(
                        func.coalesce(Chat.prompt_tokens, 0)
                        + func.coalesce(Chat.completion_tokens, 0)
                    ),
                    0,
                ),
            )
            .group_by(func.date(Chat.created_at))
            .order_by(func.date(Chat.created_at).desc())
            .limit(14)
            .all()
        )
    ]

    # The actionable list: the 50 most recent dislikes, each with the
    # diagnostics needed to triage it without opening pgAdmin. response and
    # chunks_sent are the raw stored values (chunks_sent already capped at
    # ~6 KB by /chat); response is trimmed here to keep the payload light.
    recent_dislikes = [
        {
            "chat_id": chat_row.id,
            "created_at": chat_row.created_at.isoformat() if chat_row.created_at else None,
            "query": chat_row.query,
            "response": (chat_row.response or "")[:600],
            "reason": fb.reason,
            "source_type": chat_row.source_type,
            "confidence_score": chat_row.confidence_score,
            "district": chat_row.district,
            "language": chat_row.language,
            "chunks_sent": chat_row.chunks_sent,
        }
        for chat_row, fb in (
            db.query(Chat, Feedback)
            .join(Feedback, Feedback.chat_id == Chat.id)
            .filter(Feedback.score == -1, Feedback.id.in_(latest_feedback_ids))
            .order_by(Feedback.id.desc())
            .limit(50)
            .all()
        )
    ]

    return {
        "totals": {
            "total_chats": total_chats,
            "total_feedback": total_feedback,
            "likes": likes,
            "dislikes": dislikes,
            "like_pct": like_pct,
        },
        "by_source_type": by_source_type,
        "dislikes_by_reason": dislikes_by_reason,
        "dislikes_by_district": dislikes_by_district,
        "usage_by_day": usage_by_day,
        "recent_dislikes": recent_dislikes,
    }


@app.get("/weather")
async def weather(
    district: str = Query(..., description="Gujarat district name, e.g. 'Ahmedabad'"),
    db: Session = Depends(get_db),
):
    """
    Returns current weather for a Gujarat district.
    Backed by the same WEATHER_CACHE_MINUTES cache as /weather/all — won't
    call Open-Meteo on every request.

    get_weather() is sync (it uses requests + SQLAlchemy's sync session),
    so we offload it to a worker thread; that keeps the event loop free
    to serve other requests during the up-to-10s Open-Meteo round-trip on
    a cache miss, instead of blocking a precious threadpool slot for the
    whole call the way the previous sync route did.
    """
    try:
        result = await asyncio.to_thread(get_weather, db, district)
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
def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
):
    """
    Permanently deletes every chat message belonging to a session_id, along
    with any feedback (👍/👎) left on those messages.
    This is a HARD delete — rows are removed from the database, not just
    hidden — so this cannot be undone.

    Auth model: knowing the session_id is itself the capability. The
    session_id is an unguessable client-generated UUID, never put in URLs
    or shared, and the same value already grants full *read* access via
    /chat/history?session_id=… — so requiring a separate admin token for
    *delete* added no real security (an attacker who knows the UUID can
    already exfiltrate everything) while breaking the in-app delete UI
    for the legitimate owner. The two operations now share one auth
    surface: possession of the UUID.

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