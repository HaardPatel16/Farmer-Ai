# Farmer AI — How everything works

This document is the engineering companion to `Start.md`. Start.md tells you *how to run it*; this one tells you *what each piece does and why*. It reflects the state of the project as of July 2026 (post-"Final Before Testing").

The app is a FastAPI backend + a vanilla HTML/CSS/JS frontend talking to PostgreSQL, three external services (Groq, Open-Meteo, data.gov.in), and two local ML models (a ConvNeXt-Small leaf-disease classifier and a sentence-transformer for semantic search). There's no framework on the frontend on purpose — every screen is rendered by `app.js` directly.

---

## High-level architecture

```
┌────────────────────────┐         ┌──────────────────────────┐
│  Frontend (vanilla JS) │ ──────► │  Backend  (FastAPI)      │
│  http://127.0.0.1:5500 │         │  http://127.0.0.1:8000   │
└────────────────────────┘         └────────────┬─────────────┘
                                                │
   ┌──────────┬──────────────┬──────────────────┼─────────────────┬──────────────┐
   ▼          ▼              ▼                  ▼                 ▼              ▼
Groq API   Open-Meteo    data.gov.in     ConvNeXt-Small     MiniLM-L12-v2   PostgreSQL
(LLM chat  (free weather, (Agmarknet      + rembg U²-Net    (embeddings,    (chats, feedback,
 & remedies) pull on      polled every    (.pth + ONNX,      in-memory       KB chunks, weather
             demand)      30 min)         GPU if available)  index)          cache, market
                                                                             snapshots)
```

Two timing patterns to keep in mind:

- **Weather = pull-on-demand.** Frontend dashboard auto-refreshes every 10 min; backend hits Open-Meteo only when the cache is stale.
- **Market = push-on-schedule.** A background coroutine in FastAPI polls data.gov.in every 30 min and writes snapshot rows; user requests just SELECT from the table.

All heavy ML (embeddings, ConvNeXt, rembg) runs on **CUDA GPU when available, CPU otherwise** — device selection is automatic and logged at startup so an operator can see which path was taken.

---

## Backend modules at a glance

| File | Role |
|---|---|
| `Backend/main.py` | FastAPI routes, request/response schemas, lifespan startup/shutdown: market-price poller, diagnosis-stack warmup (rembg + classifier), embeddings warmup, GPU detection print, UTF-8 console fix. Also the chat-history loader and follow-up detection helpers. |
| `Backend/services.py` | All the *doing work* logic: Groq calls (with multi-turn history), hybrid KB retrieval (keyword + semantic, RRF fusion, MMR dedup), the cached KB corpus index, weather fetch + cache, market snapshot read/write, async market refresh, district detection, the off-topic filter, coverage/hedge post-hoc labelling. |
| `Backend/database.py` | SQLAlchemy models + the `get_db` session dependency. `pool_pre_ping` recovers dropped idle connections. Run directly to create tables. |
| `Backend/config.py` | Loads `.env` from the project root into module constants. `DATABASE_URL` and `GROQ_API_KEY` are required (fail-fast); `MARKET_API_KEY`, `ADMIN_TOKEN`, and `ALLOWED_ORIGINS` are optional with graceful degradation. |
| `Backend/ml_model.py` | Loads the ConvNeXt-Small leaf-disease classifier (78 classes) and the rembg background-removal session; exposes `predict_top_k`, `warm_classifier`, `warm_bg_remover`. |
| `Backend/embeddings.py` | Semantic search over KB chunks via a vectorized in-memory index (single matmul per query). Warm-on-startup so user chats are never blocked. Exposes `get_status()` for the `/embeddings/status` endpoint. |
| `Knowledge_base/ingest.py` | One-shot script: split source `.txt`/`.pdf` files into ~200-word paragraph-aware chunks, auto-tag districts (from text AND folder name), write to `knowledge_chunks`. Safe to re-run — keyed by relative path. |

### API surface

| Endpoint | Purpose |
|---|---|
| `GET /` | Health check. |
| `POST /chat` | Main Q&A pipeline (see Pipeline 1). |
| `POST /diagnose` | Leaf-photo disease diagnosis (see Pipeline 4). |
| `POST /feedback` | 👍/👎 (+optional reason) on a chat_id. |
| `GET /stats` | Operator dashboard data — gated by `ADMIN_TOKEN` when set (see Pipeline 7). |
| `GET /weather?district=` | Single-district weather (cached). |
| `GET /weather/all` | All 33 districts, concurrent fan-out. |
| `GET /market-price/all?district=` | Today's market snapshot rows. |
| `GET /chat/history?session_id=` | Full transcript for a session (page-reload restore). |
| `GET /chat/sessions` | One entry per conversation for the History sidebar (single aggregated SQL query, no N+1). |
| `DELETE /chat/session/{id}` | Hard-deletes a session's chats + feedback. Auth = possession of the unguessable session UUID. |
| `GET /embeddings/status` | Semantic-search warmup state. |

### Security hardening (accumulated)

- **CORS narrowed**: `ALLOWED_ORIGINS` from `.env` (defaults to localhost dev ports + `null` for `file://`); methods and headers allow-listed, no `*`.
- **`/stats` auth**: constant-time token compare (`hmac.compare_digest`) when `ADMIN_TOKEN` is set; open in local dev when unset.
- **Upload limits**: 8 MB body cap checked both via Content-Length pre-read and post-read; `PIL.Image.MAX_IMAGE_PIXELS = 25 MP` blocks decompression bombs; Content-Type must be `image/*`.
- **`torch.load(weights_only=True)`** refuses pickle code execution when loading the `.pth`; falls back with a loud warning only if the format demands it.
- **No secret leakage**: `config.py`'s self-test prints only yes/no + driver name, never key prefixes.
- **Prompt-injection resistance**: the Groq system prompt explicitly refuses role changes, "forget your instructions", and fake-emergency jailbreaks; the primary off-topic block happens locally before Groq anyway.

---

## Database tables

| Table | Purpose | Lifecycle |
|---|---|---|
| `chats` | Every Q&A: query, response, language, source_type, district, confidence_score (KB coverage 0–1), **chunks_sent_count, chunks_sent (the exact chunks fed to Groq, capped ~6 KB), prompt_tokens, completion_tokens**. Composite index on (session_id, created_at). | Append-only; one row per `/chat` or `/diagnose` call (including local refusals, which have NULL token/chunk fields). |
| `feedback` | 👍/👎 with optional reason (`wrong_info` / `wrong_language` / `irrelevant`). FK to `chats.id`, indexed. Can hold multiple votes per chat (UI re-enables buttons after reload); `/stats` counts only the latest vote per chat. | Append-only. |
| `knowledge_chunks` | Chunked & district-tagged content from `Knowledge_base/sources/` (~14,000+ chunks from ~1,400 source files). `source_filename` is the path relative to `sources/`. | Rebuilt per-file by re-running `ingest.py`. |
| `weather_cache` | One JSON row per Open-Meteo response, keyed by district. | Bounded at ~33 rows — every fetch prunes everything older than the 10 min TTL. |
| `market_price_snapshots` | The *only* source of truth for market prices. One row per `(commodity, market, variety, district)` for today's IST date. | Today-only — every refresh deletes `snapshot_date < today`. |

Debug columns on `chats` exist so retrieval failures are diagnosable from pgAdmin ("was it 0 chunks, or low coverage?") and token spend is queryable per session without re-parsing logs.

---

## The knowledge base corpus

`Knowledge_base/sources/` holds **~1,426 curated `.txt` documents** organized by topic folder:

| Folder | Files | Folder | Files |
|---|---|---|---|
| crops | 328 | market | 65 |
| schemes | 160 | apiculture | 56 |
| niche | 141 | water | 50 |
| livestock | 137 | soil_agronomy | 45 |
| advisory | 130 | districts | 39 |
| digital | 130 | soils | 36 |
| post_harvest | 82 | finance | 14 |
| — | — | regions 10, equipment 3 | |

Content is Gujarat-specific: crop cultivation guides, district profiles, government schemes (PM-KISAN, PMFBY, i-Khedut…), pest/disease management, climate advisories, monthly crop calendars, market/finance primers, and traditional-knowledge documentation.

**Ingest** (`python Knowledge_base/ingest.py` from project root):

1. Walk `sources/` recursively for `.txt` and `.pdf`.
2. Per file: read, split into ~200-word chunks along paragraph boundaries (never mid-paragraph), extract top-15 frequent keywords, run `detect_all_districts()` so each chunk carries every district it mentions (comma-separated, NULL = Gujarat-wide). If the file sits inside a folder named after a district, that district is added too.
3. Delete-then-reinsert that relative path's rows — safe to re-run, no duplicates, other files untouched.

---

## Pipeline 1 — `/chat`

The most complex path. Lives in `main.py:chat` and `services.py`.

```
request {session_id, query, language}
  │
  ▼
is_farming_question(query)? ── No ──► follow-up exception check:
  │ Yes                              session has history AND query is short/
  │                                  pronoun-heavy? → treat as continuation.
  │                                  Otherwise → canned refusal (0 Groq tokens)
  ▼
detect_weather_request → weather kw + district → get_weather() live JSON
  │
  ▼
load_chat_history(session_id)          ← last 2 exchanges, skipping
  │                                      leaf-diagnosis rows and refusals
  ▼
_build_retrieval_query()               ← if the query looks like a follow-up
  │                                      ("and irrigation?", "about its pests"),
  │                                      prepend previous user turns so "cotton"
  │                                      carries into retrieval; otherwise raw query
  ▼
detect_district(retrieval_query)
  │
  ▼
search_knowledge_base(retrieval_query, district)   ← hybrid, see below
  │
  ▼
ask_groq(question, language, context, district, history)
  │    system prompt = scope lock + anti-hallucination + phone formatting
  │      + language instruction + district instruction (+ KB reference)
  │    messages = system + trimmed history (≤3000 chars) + fresh question
  │    returns (answer, {prompt_tokens, completion_tokens})
  ▼
post-hoc source_type label from the actual answer:
    coverage = kb_answer_coverage(answer, chunks)
    hedged   = answer_has_llm_hedge(answer)      ("the reference doesn't mention…")
    weather_context               → "weather_api"
    coverage ≥ 0.70 and !hedged   → "knowledge_base"
    coverage ≥ 0.20 or hedged     → "mixed"
    else                          → "llm_reasoning"
  │
  ▼
INSERT INTO chats (incl. chunks_sent, token usage), return
    {chat_id, response, source_type, language}
```

**Multi-turn memory.** Every session carries up to `CHAT_HISTORY_MAX_TURNS = 2` previous exchanges into the Groq messages array, hard-capped at `CHAT_HISTORY_MAX_CHARS = 3000` (~600 words) — oldest turns dropped first. Leaf-diagnosis rows are excluded (their answer references an image the follow-up doesn't have), and local off-topic refusals are excluded via their fingerprint (`source_type=='llm_reasoning'` with NULL chunk/confidence fields).

**Follow-up handling** was the single biggest usability fix. Two mechanisms:
1. *Off-topic filter exception* — a short or pronoun-heavy query ("and irrigation?") in a session with prior on-topic history bypasses the whitelist refusal, because typos and pronouns never match keywords but are almost never off-topic mid-conversation.
2. *Retrieval augmentation* — the same "looks like a follow-up" test (≤3 words, or contains an anaphoric pronoun like it/its/that/those) decides whether previous user turns get concatenated into the retrieval query. A self-contained new question is deliberately NOT diluted with earlier topics, or coverage drops and the KB badge is lost.

**Why the post-hoc label.** The retriever is generous on purpose — it'd rather over-return than under-return. Labelling a response "knowledge_base" *just because chunks were retrieved* lies to the user when the LLM ignored them and answered from training data anyway. Comparing the answer's distinct content words (4+ chars) against the chunks' content words catches this cheaply, and the hedge-phrase check demotes obviously-mixed answers even when overlap looks high. The actual coverage ratio is stored in `confidence_score` for threshold calibration.

**Off-topic filter (`is_farming_question`).** Keyword whitelist (~200 farming terms) + Gujarati-script catch-all (any character in U+0A80–U+0AFF is accepted — a Gujarati-typing user of a Gujarat farming bot is almost certainly on-topic) + greeting patterns (exact/prefix match only) + district name/alias match. Runs before Groq so off-topic queries cost zero tokens; Groq's system prompt is the jailbreak-resistant backstop.

**Error visibility.** Groq failures are logged with type + message before raising 503; token usage (`prompt/completion`) and chunk counts are printed per request so quota spend and retrieval behaviour can be tailed live in the uvicorn terminal.

---

## Pipeline 2 — Hybrid knowledge-base retrieval

`services.search_knowledge_base` runs **both retrievers on every query** and fuses them. This was the "feat/hybrid-retrieval" branch (RRF fusion + MMR dedup).

### Stage 0 — the cached corpus index

Scoring 14k+ chunks per request used to re-read the whole table and recompute derived strings every time (~200–500 ms/query). `_get_kb_index()` now caches, per corpus:
- primitive per-chunk dicts (id, text, filename, lowercased text+keywords, word count, filename tokens, districts set) — primitives, not ORM rows, so no `DetachedInstanceError` across request sessions;
- each chunk's heading (first non-divider line);
- heading-word document frequencies (for the rarity check).

Invalidation is keyed on `max(id)` of `knowledge_chunks` — re-running ingest bumps it and the next request rebuilds under a lock.

### Stage 1 — keyword scorer

- Tokenize the query, strip punctuation, drop <3-char tokens and stopwords. If nothing technical remains ("hi", "namaste"), return `[]` — the caller labels the answer `llm_reasoning`.
- Per chunk (skipping chunks tagged for *other* districts — district isolation):
  - **Density score**: matches per 100 words, so a short chunk that's *about* the topic beats a long chunk mentioning it once.
  - **+80** if a rare query word (appears in <5% of corpus headings, and not in a structural-noise list like "guide"/"management") is in the heading's first ~4 words — a dedicated section.
  - **+15** if it appears elsewhere in the heading.
  - **+100** if the *filename* names a query word — the strongest signal; covers files whose chunks have generic headings ("Land Preparation") that never repeat the topic word. Filename-matched chunks are kept even with 0 body matches.
  - **+20** if the chunk is tagged with the asked-about district.
- Score floor 1.0 drops stray substring hits.
- **Filename force-include**: the top chunk of every filename that names a query word is guaranteed a slot in the final answer, whatever fusion does.

### Stage 2 — semantic candidates

`embeddings.semantic_search()` always runs too (pool of `max(top_k*5, 20)`), returning text + filename + cosine similarity + the chunk's L2-normalized index vector (reused later, nothing re-encoded). It applies the same district isolation. Returns `None` while warming or if the model failed to load — the pipeline then degrades to keyword-only with Jaccard dedup.

### Stage 3 — Reciprocal Rank Fusion (k=60)

Density scores and cosine similarities aren't on the same scale, so fusion is by **rank**: each list contributes `1/(60 + rank)` per chunk, summed. Chunks both retrievers like rise to the top; a chunk found by only one still ranks on its own merit. With semantic unavailable this collapses to the keyword ranking unchanged.

### Stage 4 — MMR de-duplication (λ=0.7)

The fused list often contains near-duplicate chunks. MMR greedily re-ranks: pick the candidate maximizing `0.7·relevance − 0.3·max_similarity(candidate, already_picked)`, where relevance is the normalized RRF score and similarity is cosine over the reused index embeddings (falling back to Jaccard word overlap when either side lacks a vector).

### Stage 5 — final assembly

Force-included filename representatives first, then MMR order, never repeating a filename, capped at `top_k = 4` chunks (kept small deliberately for Groq-context efficiency).

---

## Pipeline 3 — Weather (`/weather`, `/weather/all`)

Backed by Open-Meteo's free `/v1/forecast` (no key needed), `forecast_days=2`: current temperature/humidity/weather-code/precipitation **plus tomorrow's rainfall and max/min temperature**, so "will it rain tomorrow?" gets a real predicted value.

**Single district** (`get_weather`, offloaded to a worker thread by the async route): check `weather_cache` for a row newer than 10 minutes *and* containing all expected schema keys (`_WEATHER_REQUIRED_KEYS` — older partial rows are auto-refetched). On miss, call Open-Meteo, normalize, insert, prune rows older than the TTL in the same commit.

**All districts** (`get_all_weather_concurrent`): three phases to keep the sync SQLAlchemy session on the main thread:

1. **Sync:** one query pulls every cached row, latest per district, classifies fresh vs stale (same schema guard).
2. **Async:** `httpx.AsyncClient` + `asyncio.gather` fan out only the cache-miss fetches; a failed district becomes `None` and doesn't sink the batch.
3. **Sync:** write new rows in one commit, then prune `< cutoff`.

Cold-cache `/weather/all` (33 districts) takes ~1 s. Steady state: the table holds at most one row per district — every cache read is instant.

When `/chat` detects a weather question (English **or Gujarati** keyword list: હવામાન, વરસાદ, તાપમાન, …) plus a district, the live figures are injected into Groq's context with an instruction to use the exact numbers.

---

## Pipeline 4 — Market prices

The **background poller writes a snapshot table; endpoints just read.** No request-time API calls.

### The poller

`main.py:_market_price_poller()` runs as an asyncio task created in the lifespan hook (strong reference kept on `app.state`, cancelled cleanly at shutdown so uvicorn doesn't warn about pending tasks). First refresh fires immediately at boot; then every `MARKET_REFRESH_MINUTES = 30`. On startup, `_seed_last_refresh_from_db()` seeds the last-refresh marker from the newest snapshot row so a restart inside a fresh window doesn't re-fetch.

`refresh_market_snapshots_async(db, api_key)`:

1. Build the request list — one async fetch per crop in every category in `CROP_CATEGORIES` (cash crops, oilseeds, grains, pulses, spices, fruits, vegetables, others), plus a single generic `state=Gujarat` discovery call.
2. `httpx.AsyncClient` + `asyncio.gather` fans the batch out concurrently: ~3–5 s per cycle vs ~40–80 s sequential.
3. Dedupe responses by `(commodity, market, variety, district, arrival_date)`.
4. `_store_records(db, collected)`:
   - `DELETE … WHERE snapshot_date < today_IST` — today-only retention.
   - Upsert by `(commodity, market, variety, district)` — **arrival_date deliberately excluded from the key**, because data.gov.in shifts the reported date by a day between polls; with it included, duplicates would balloon over the day. Existing rows get prices/grade/arrival_date overwritten in place.
   - The log line reports new/updated/total-today, since today's snapshot is the *union* of every mandi seen across the day's polls.
5. Update `_last_market_refresh_at`. An atomic check-and-set under `_market_refresh_lock` stops a second refresh from piling up.

### The endpoint

Only one market endpoint exists: **`GET /market-price/all`** (optional `?district=`, case-insensitive):

```python
ensure_fresh_market_data(db, api_key)    # fire-and-forget bg refresh if stale
return _read_snapshot(db, district)      # plain SELECT, today's IST date
```

`ensure_fresh_market_data` never blocks — if data is stale and no refresh is in flight (both flags read under the lock), it spawns a daemon thread and returns immediately. The user request always serves whatever's already in the snapshot table.

### Why IST

`arrival_date` strings from data.gov.in are `DD/MM/YYYY` in India time. `snapshot_date` is set by `_ist_today() = datetime.now(UTC+5:30).date()` — UTC would roll over 5.5 hours late and mis-date late-evening fetches.

### Why the User-Agent header

data.gov.in stalls forever on `python-requests/x.y` (no error — just hangs to timeout). A browser-style UA gets sub-second responses. Baked into every market call.

### Retention guarantee

At most one `snapshot_date` value exists at any moment: today. The first refresh past IST midnight drops yesterday's rows. No cron, no separate cleanup job.

---

## Pipeline 5 — `/diagnose` (leaf disease)

```
multipart upload (image, session_id, top_k, language, remove_bg)
  │  Content-Type must be image/* • Content-Length pre-check (413 before
  │  reading) • async read • 8 MB post-read cap • 25 MP decode cap
  ▼
asyncio.to_thread(predict_top_k)          ← CPU/GPU-bound work off the event loop
  │
  │  ← rembg (U²-Net) strips the background and composites the leaf onto
  │    white, so the classifier doesn't waste capacity on soil/hands/desk
  │    pixels. Falls back to the raw image if rembg errors (bg_removed=False).
  ▼
ConvNeXt-Small, 78 classes, forward pass (CUDA if available)
  │  → predictions + a ≤512px WebP preview of the exact image the model saw
  ▼
asyncio.to_thread(ask_groq_disease_remedy(predictions, language))
  │  → symptoms, organic + chemical remedies, prevention; instructed to
  │    never invent dosages — "follow the label rate" instead
  ▼
INSERT INTO chats (synth query "🌿 Leaf diagnosis — most likely X (NN.N%)",
                   source_type "leaf_diagnosis")
  ▼
{chat_id, response, predictions, processed_image (data:image/webp;base64),
 bg_removed, source_type, language}
```

**Background removal** (`ml_model._remove_background`) uses [rembg](https://github.com/danielgatis/rembg) with the `u2net` model (~170 MB, auto-downloaded to `~/.u2net/` on first ever run). rembg returns RGBA; we composite onto solid white before returning RGB so the ImageNet 3-channel preprocess is unchanged. Send `remove_bg=false` to skip. The active ONNX execution provider (CUDA vs CPU) is printed at load.

**Warmup**: at server boot, *if* the `.pth` exists, a daemon thread pre-loads both the rembg session and the classifier so the first `/diagnose` request doesn't pay the download/load cost. If the model file is missing, warmup is skipped and `/diagnose` returns a clear 503.

**Preprocessing fidelity**: inference replicates the training notebook exactly — `Resize(256) → CenterCrop(224)` + ImageNet mean/std, not a direct stretch, to avoid a silent distribution shift.

**Class names.** `ml_model.py` loads `ML model/class_names.json` — a flat JSON array of 78 strings in the training-time ImageFolder order (this file now exists and is committed). If missing/malformed, inference still runs with placeholder labels (`class_0`…`class_77`) and a boot-time warning.

**Processed-image round trip.** The response includes a small WebP of the post-rembg image; the frontend **crossfades** the user's upload thumbnail to it and flashes a "Background removed" tag for ~1.8 s, so the farmer sees exactly what the model saw.

The synthetic chat row exists so the History sidebar and the feedback flow still work — feedback is FK'd to `chats.id`. Leaf-diagnosis rows are excluded from multi-turn history replay.

The front door is the **Diagnose a Leaf** home card, which offers **two buttons: upload from gallery and take a photo** (a `capture`-attributed camera input on mobile). Either starts a fresh chat session and posts to `/diagnose`.

---

## Pipeline 6 — Embeddings warm-up & semantic search

The model is `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB, 384-dim, handles English + Gujarati). First-ever use takes 2–5 min on CPU to download and embed every KB chunk; the index is then held in memory (~22 MB for 15k chunks) and each query costs ~1–5 ms.

**Warmup** is a daemon thread kicked off from the FastAPI lifespan (`warm_index_in_background(SessionLocal)`), so route registration isn't blocked. While `_warming` is true, `semantic_search()` returns `None` immediately and the keyword path serves alone — chat never blocks.

**The hot path is vectorized**: chunk embeddings live in one L2-normalized float32 matrix, so scoring is a single matmul; top-k uses `argpartition` (O(N)) instead of a full sort; district filtering masks scores to −inf.

**Windows gotcha that cost real debugging time**: the default Windows console is cp1252, and a `print()` containing Gujarati script or fancy arrows raised `UnicodeEncodeError` *inside the warmup thread*, silently killing it — semantic search then appeared "stuck warming" forever. Fixed two ways: `main.py` reconfigures stdout/stderr to UTF-8 before any imports print, and the warmup's own log lines are plain ASCII.

**Status endpoint** — `GET /embeddings/status`:

```json
{
  "state": "warming" | "ready" | "failed" | "not_started" | "idle_empty_kb",
  "chunks_indexed": 0, "chunks_total": 14000,
  "elapsed_seconds": 47.2, "model_name": "...",
  "model_load_failed": false,
  "warm_error": null
}
```

`warm_error` carries the actual exception message when the encode crashed *after* the model loaded — distinct from a model-load failure, and prevents a crash from masquerading as an empty KB.

---

## Pipeline 7 — Feedback & the `/stats` operator dashboard

`POST /feedback` stores 1/−1 (+optional reason) against a chat_id. Because the UI re-renders vote buttons after a reload, a chat can accumulate multiple votes — so **all `/stats` aggregation counts only the latest vote per chat** (max `Feedback.id` per `chat_id` subquery).

`GET /stats` (token-gated when `ADMIN_TOKEN` is set) returns:

- **Totals**: chats, feedback, likes/dislikes, like %.
- **Satisfaction by source_type** — the headline "does the KB pipeline actually help?" metric: compare like/dislike ratios across knowledge_base vs mixed vs llm_reasoning vs weather_api vs leaf_diagnosis.
- **Dislikes by reason** and **by district** (surfaces regions the KB covers poorly).
- **Usage + token spend per day** (last 14 days) — powered by the per-row `prompt_tokens`/`completion_tokens`.
- **50 most recent dislikes**, each with the full triage kit: query, response excerpt, reason, source_type, coverage score, district, language, and the *exact chunks that were sent to Groq* — so a bad answer can be diagnosed without opening pgAdmin.

The frontend renders this as a hidden **Statistics view inside the Settings panel**; the admin token is prompted for on a 401 and remembered in `localStorage`.

---

## Frontend layout

Vanilla JS, no bundler: one `app.js` (~1,800 lines) + `index.html` + `style.css` (~3,200 lines).

**API base auto-resolution** (no more hardcoded constant): `window.__API_BASE__` override → `file://` implies dev backend on `127.0.0.1:8000` → localhost dev server on any port also points at `:8000` → anything else uses same-origin (reverse-proxy-ready).

Home screen structure:

```
┌──────────────────────────────────────────────────────────────┐
│                       Farmer AI                              │  ← animated wordmark
│        Your fields. Your language. Real answers.             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Ask Farmer AI                                          │  │  ← hero chat card
│  │ [type your question……………………………………………] [→]            │  │     w/ rainbow border
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────┐  ┌────────┐  ┌────────────────┐  ┌────────┐      │
│  │Weather │  │ Mandi  │  │    Diagnose    │  │ Scheme │      │  ← action shelf
│  │  blue  │  │ leaf   │  │ Upload | Camera│  │ wheat  │      │     (70ms stagger)
│  └────────┘  └────────┘  └────────────────┘  └────────┘      │
└──────────────────────────────────────────────────────────────┘
```

Responsive: 4 cards in a row on desktop, 2×2 on tablet, single column on mobile. A left nav rail carries New Chat, the History list (from `/chat/sessions`, with per-session delete), and Settings.

### Per-card accent identity

| Card | Accent | What it touches |
|---|---|---|
| Weather | sky-blue (`--weather-accent`) | icon, dot, glow, hover border, CTA color |
| Mandi Prices | leaf-green (`--leaf`) | same set |
| Diagnose | teal `#2D8C7E` | same set |
| Scheme Finder | wheat-gold (`--wheat-dark`) | same set |
| Chat hero | always-on animated rainbow border via `::before` | signature element, not hover-tied |

### Action handlers

- **Weather** → dashboard modal: skeleton loaders, one card per district with icon/temp/humidity/rain + tomorrow's forecast, live search filter, 10-min auto-refresh timer (aligned with the backend cache TTL so each refresh actually pulls fresh data).
- **Mandi** → single flat sortable table of today's snapshot with debounced search across commodity/market/district/variety and a record count.
- **Diagnose** → Upload (file picker) or Camera (capture input) → `startNewChat()` → `showChat()` → posts to `/diagnose`; renders the photo bubble, then crossfades to the background-removed cutout.
- **Scheme Finder** → Schemes modal (static seed in `app.js`, 6 schemes, fully bilingual via `_gu` fields). Each card: *Open official portal* (new tab) and *Ask AI about this* (auto-sends a comprehensive pre-filled question — eligibility, amount, application, documents, status, rejection reasons — in the current language).

### Chat rendering

- **Source badges** on every AI bubble map all five backend source_types: Knowledge Base / Weather Data / Leaf Diagnosis / **Mixed** / AI Reasoning — the whole point of the post-hoc labelling made visible.
- `formatAiText()` parses the Groq output conventions (`- ` bullets, `**bold**`, `**Heading:**` groups) into HTML; everything else is escaped.
- Feedback row per AI message (👍 direct, 👎 opens a reason modal).
- History restores on reload via `/chat/history`; sessions are switchable and hard-deletable from the sidebar.

### Type system

Two families across the whole app, set as CSS vars:

```css
--font-display: 'Fraunces', 'Noto Serif Gujarati', Georgia, serif;
--font-body:    'Inter Tight', 'Noto Sans Gujarati', system-ui, sans-serif;
```

Display family at 24px+ only — wordmark, hero subtitle, card titles, modal titles, weather temperatures. Body family for chat, paragraphs, labels, buttons (weight 600, no third family).

### Backgrounds & theming

- **Light mode** (`:root`): cream `#F6F1E7` substrate + a subtle `body::before` vertical gradient — cool pale cream fading to dusty wheat, leaf-pale glow lower-right. Reads as "sky above, earth below".
- **Dark mode** (`[data-theme="dark"]`): warm amber-coffee `#221C12` substrate, elevated cards `#2E2718`. Primary stays olive-green (brightened to `#7BB582`) so the brand's action identity carries across modes.

### i18n

Single `t()` lookup with `en` and `gu` tables covering the entire UI (home, modals, badges, settings, errors). `language` persists in `localStorage`; the toggle re-runs the language updater. District names have a Gujarati display map; schemes are bilingual per-record.

---

## Startup sequence (what happens on `uvicorn main:app`)

1. stdout/stderr forced to UTF-8 (Windows cp1252 fix).
2. `config.py` loads `.env` from the project root (explicit path — works from any CWD); fails fast if `DATABASE_URL`/`GROQ_API_KEY` missing.
3. Lifespan: one-line CUDA GPU summary.
4. Market last-refresh marker seeded from the DB; market poller task created (first poll immediate).
5. If the `.pth` exists: daemon thread warms rembg + ConvNeXt.
6. Daemon thread warms the embeddings index (2–5 min first boot; chat is keyword-only meanwhile).
7. On shutdown, the poller task is cancelled and awaited cleanly.

## Dependencies (`requirements.txt`)

fastapi, uvicorn, sqlalchemy, psycopg2-binary, python-dotenv, groq, pydantic, requests, httpx, pypdf, torch, torchvision, pillow, python-multipart, sentence-transformers, numpy, rembg, onnxruntime (swap for `onnxruntime-gpu` on an NVIDIA machine so rembg uses CUDA).

Groq model in use: `llama-3.3-70b-versatile`, temperature 0.3, max_tokens 1000 (chat) / 700 (remedies).

---

## Quick mental model

If anything ever stops working, run through this checklist:

1. **`.env` loaded?** `python Backend/config.py` prints yes/no per key (no secret material).
2. **DB reachable?** `python Backend/database.py` exits clean iff so.
3. **Tables current?** Compare `database.py` models vs. `\d` in pgAdmin. `create_all` won't add columns to existing tables — you'll need an ALTER (recent additions: `chats.chunks_sent_count/chunks_sent/prompt_tokens/completion_tokens`).
4. **Market table populating?** `SELECT snapshot_date, COUNT(*) FROM market_price_snapshots GROUP BY snapshot_date;` — ≥ 1 row within ~5 s of boot (needs `MARKET_API_KEY`).
5. **Embeddings ready?** `GET /embeddings/status`. `ready` = semantic active; `warming` is fine (keyword-only meanwhile); `failed` + `warm_error` tells you why.
6. **Retrieval behaving?** Tail uvicorn: every `/chat` prints the retrieval query, matched/after-floor/top-score, chunk count, and token usage. Or read `chunks_sent` on the chat row in pgAdmin.
7. **Weather cache pruned?** `SELECT COUNT(*) FROM weather_cache;` — always ≤ 33.
8. **Frontend talking to the right backend?** API base is auto-resolved (see Frontend section); set `window.__API_BASE__` in `index.html` to override.
9. **CORS?** Allowed origins come from `ALLOWED_ORIGINS` in `.env` (localhost dev defaults + `null`). Frontend served over `http://` or opened as `file://` both work in dev.
10. **`/stats` 401?** Set the same `ADMIN_TOKEN` in the frontend prompt (stored in `localStorage` as `farmerai_admin_token`).

That's the whole system.
