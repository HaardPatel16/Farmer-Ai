# Farmer AI — How everything works

This document is the engineering companion to `Start.md`. Start.md tells you *how to run it*; this one tells you *what each piece does and why*.

The app is a FastAPI backend + a vanilla HTML/CSS/JS frontend talking to PostgreSQL, three external services (Groq, Open-Meteo, data.gov.in), and one local ML model. There's no framework on the frontend on purpose — every screen is rendered by `app.js` directly.

---

## High-level architecture

```
┌────────────────────────┐         ┌──────────────────────────┐
│  Frontend (vanilla JS) │ ──────► │  Backend  (FastAPI)      │
│  http://127.0.0.1:5500 │         │  http://127.0.0.1:8000   │
└────────────────────────┘         └────────────┬─────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────┐
        ▼                ▼                      ▼                  ▼            ▼
   Groq API       Open-Meteo            data.gov.in        ConvNeXt-Small   PostgreSQL
   (LLM chat &    (free weather,        (Agmarknet         (.pth weights,   (chats,
    remedies)      pull on demand)       polled every       local CPU)       feedback,
                                         30 min)                              KB chunks,
                                                                              weather cache,
                                                                              market snapshots)
```

Two timing patterns to keep in mind:

- **Weather = pull-on-demand.** Frontend dashboard auto-refreshes every 10 min; backend hits Open-Meteo only when the cache is stale.
- **Market = push-on-schedule.** A background coroutine in FastAPI polls data.gov.in every 30 min and writes snapshot rows; user requests just SELECT from the table.

---

## Backend modules at a glance

| File | Role |
|---|---|
| `Backend/main.py` | FastAPI routes, request/response schemas, two startup tasks: the market-price poller and the embeddings warm-up. |
| `Backend/services.py` | All the *doing work* logic: Groq calls, KB search, weather fetch + cache, market snapshot read/write, async market refresh, district detection, the off-topic filter. |
| `Backend/database.py` | SQLAlchemy models + the `get_db` session dependency. Run directly to create tables. |
| `Backend/config.py` | Loads `.env` into module constants. Single source of truth for keys. |
| `Backend/ml_model.py` | Loads the ConvNeXt-Small leaf-disease classifier and exposes `predict_top_k`. |
| `Backend/embeddings.py` | Semantic search over KB chunks. Warm-on-startup so user chats are never blocked. Exposes `get_status()` for the `/embeddings/status` endpoint. |
| `Knowledge_base/ingest.py` | One-shot script: split source `.txt`/`.pdf` files into chunks, auto-tag districts, write to `knowledge_chunks`. |

---

## Database tables

| Table | Purpose | Lifecycle |
|---|---|---|
| `chats` | Every Q&A: query, response, language, source_type, district, confidence_score. | Append-only; one row per `/chat` or `/diagnose-leaf` call. |
| `feedback` | 👍/👎 with optional reason. FK to `chats.id`. | Append-only; bounded by chat count. |
| `knowledge_chunks` | Chunked & district-tagged content from `Knowledge_base/sources/`. | Rebuilt by re-running `ingest.py`. |
| `weather_cache` | One row per Open-Meteo response, keyed by district. | Bounded at ~33 rows — every fetch prunes everything older than the 10 min TTL. |
| `market_price_snapshots` | The *only* source of truth for market prices. One row per `(commodity, market, variety, district)` for today's IST date. | Today-only — every refresh deletes `snapshot_date < today`. |

`MarketCache` is gone. The market path is snapshot-only.

---

## Pipeline 1 — `/chat`

The most complex path. Lives in `main.py:chat` and `services.py`.

```
request
  │
  ▼
is_farming_question(query)? ──► No ──► canned refusal (0 Groq tokens)
  │ Yes
  ▼
detect_weather_request → if weather kw + district found, get_weather() returns live JSON
  │
  ▼
detect_district() (reused if weather already picked one)
  │
  ▼
search_knowledge_base(query, district)
    • keyword density score, +heading boost (rare words only),
      +filename boost, +district boost
    • district isolation: skip chunks tagged for *other* districts
    • merges with semantic_search() (skipped silently if embeddings
      still warming — see Pipeline 6)
  │
  ▼
ask_groq(question, language, context, district)
    • language_instruction: "reply in English/Gujarati"
    • scope_instruction: compact farming-only guard
    • district_instruction: tailor to that district
    • context: weather JSON (if any) + KB chunks
  │
  ▼
post-hoc source_type label, computed from the actual answer:
    coverage = kb_answer_coverage(answer, chunks)
    if weather_context        → "weather_api"
    elif coverage >= 0.70 and no hedge phrase → "knowledge_base"
    elif coverage >= 0.20 or hedge phrase      → "mixed"
    else                                        → "llm_reasoning"
  │
  ▼
INSERT INTO chats, return {chat_id, response, source_type, language}
```

**Why the post-hoc label.** The keyword retriever is generous on purpose — it'd rather over-return than under-return. Labelling a response "knowledge_base" *just because chunks were retrieved* lies to the user when the LLM ignored them and answered from training data anyway. Comparing the answer's distinct content words against the chunks' content words catches this cheaply, and the hedge-phrase check ("the reference doesn't mention…") demotes obviously-mixed answers even when overlap looks high.

**Off-topic filter (`is_farming_question`).** Keyword whitelist + Gujarati-script catch-all + greeting patterns + district name match. Runs before Groq so off-topic queries cost zero tokens; Groq's system prompt still has a one-line scope backstop for edge cases.

---

## Pipeline 2 — Weather (`/weather`, `/weather/all`)

Backed by Open-Meteo's free `/v1/forecast` (no key needed). Two-day forecast — current readings + tomorrow's max/min/precip.

**Single district** (`get_weather`): check `weather_cache` for a row newer than 10 minutes (and with the expected schema keys; older partial rows are auto-refetched). On miss, call Open-Meteo, normalize, insert, **prune rows older than the TTL in the same commit**.

**All districts** (`get_all_weather_concurrent`): three phases to keep the sync SQLAlchemy session on the main thread:

1. **Sync:** one query pulls every cached row, latest per district, classifies fresh vs stale.
2. **Async:** `httpx.AsyncClient` + `asyncio.gather` fan out only the cache-miss fetches.
3. **Sync:** write new rows in one commit, then prune `< cutoff`.

Cold-cache `/weather/all` (33 districts) takes ~1 s. Steady state: the table holds at most one row per district — every cache read is instant.

---

## Pipeline 3 — Market prices

This is the most heavily reworked subsystem. The **background poller writes a snapshot table; endpoints just read.** No request-time API calls.

### The poller

`main.py:_market_price_poller()`:

```python
while True:
    await refresh_market_snapshots_async(db, MARKET_API_KEY)
    await asyncio.sleep(MARKET_REFRESH_MINUTES * 60)   # 30 min
```

`refresh_market_snapshots_async(db, api_key)`:

1. Build the request list — one async fetch per crop in every category in `CROP_CATEGORIES`, plus a single generic `state=Gujarat` discovery call.
2. `httpx.AsyncClient` + `asyncio.gather` fans the whole batch out concurrently. ~3–5 s per cycle, instead of ~40–80 s if it were sequential.
3. Dedupe responses by `(commodity, market, variety, district, arrival_date)`.
4. Call `_store_records(db, collected)`:
   - `DELETE FROM market_price_snapshots WHERE snapshot_date < today_IST` — today-only retention.
   - Upsert by `(commodity, market, variety, district)` — new rows inserted, existing same-key rows have their `arrival_date / grade / prices` overwritten with the fresh values. Prevents bloat when data.gov.in shifts the reported `arrival_date` by a day between polls.
5. Update `_last_market_refresh_at`.

A `_market_refresh_in_progress` flag stops a second refresh from piling up while one is running.

### The endpoint

Only one market endpoint exists: **`GET /market-price/all`** (with optional `?district=`). It calls `get_all_market_prices`, which is a ~10-line read:

```python
ensure_fresh_market_data(db, api_key)    # fire-and-forget bg refresh if stale
return _read_snapshot(db, district)      # plain SELECT, today's IST date
```

`ensure_fresh_market_data` never blocks — if the data is stale and no refresh is in flight, it spawns a daemon thread to refresh and returns immediately. The user request always serves whatever's already in the snapshot table.

### Why IST

`arrival_date` strings from data.gov.in are `DD/MM/YYYY` in India time. The snapshot's `snapshot_date` is the IST calendar date when we stored the row — set by `_ist_today() = datetime.now(timezone(+5:30)).date()`. UTC would roll over 5.5 hours late.

### Why the User-Agent header

data.gov.in stalls forever on `python-requests/x.y` (no error, no response — just hangs to timeout). A browser-style UA gets sub-second responses. Baked into every market call.

### Retention guarantee

At most one `snapshot_date` value exists at any moment: today. The first refresh past IST midnight drops yesterday's rows. No cron, no separate cleanup job. Yesterday-tab concept removed entirely — the UI is a single flat table.

---

## Pipeline 4 — `/diagnose-leaf`

```
multipart upload (image) ──► size guard (8 MB)
                              │
                              ▼
                       predict_top_k(image_bytes, k)
                              │  ← ConvNeXt-Small, 78 classes, CPU forward pass
                              ▼
                       ask_groq_disease_remedy(predictions, language)
                              │
                              ▼
                       INSERT INTO chats (synth query "🌿 Leaf diagnosis — …")
                              │
                              ▼
                       {chat_id, response, predictions, …}
```

The synthetic chat row exists so the History sidebar and the feedback flow still work — feedback is FK'd to `chats.id`, and a leaf diagnosis with no chat row would have nowhere to attach a thumbs-up to.

The front door for this feature is now the **Diagnose a Leaf** home card. Clicking it starts a fresh chat session and pops the OS file picker. Same backend; just a more findable entry point than the old chat-sidebar Upload button.

Known limit: `CLASS_NAMES` in `ml_model.py` is still placeholders (`class_0`…`class_77`). The pipeline runs end-to-end; the labels read as generic until the original training class order is filled in.

---

## Pipeline 5 — Knowledge base

**Ingest** (`Knowledge_base/ingest.py`, run from project root):

1. Walk `Knowledge_base/sources/` for `.txt` and `.pdf`.
2. Per-file: read, split into chunks (sentence/paragraph aware), extract keywords, run `detect_all_districts()` so each chunk carries every district it mentions (comma-separated string, NULL = Gujarat-wide).
3. Wipe and reinsert that filename's rows — safe to re-run.

**Search** (`services.search_knowledge_base`):

- Tokenize query, drop stopwords. If nothing technical remains, return `[]` (caller hits Groq from general knowledge).
- For each chunk, compute density (matches per 100 words), then add boosts:
  - **+80** if a heading's first words contain a query word (only rare heading words count — rarity threshold 5% of corpus headings).
  - **+15** if a query word appears later in the heading.
  - **+100** if the *filename* contains a query word (covers files where chunks have generic section titles like "Land Preparation").
  - **+20** if the chunk is tagged with the asked-about district.
- District isolation: chunks tagged for *other* districts are excluded entirely.
- Score floor (1.0) drops stray substring hits.
- "Filename force-include" pass: ensure at least one chunk per filename whose name names a query word.
- Optional merge with `embeddings.semantic_search` — catches synonym hits the keyword path misses (e.g. "cold storage" ↔ "Cold Chain").

---

## Pipeline 6 — Embeddings warm-up

The semantic-search model (`paraphrase-multilingual-MiniLM-L12-v2`, ~120 MB) takes 2–5 min to download and embed every KB chunk on first use. Without intervention, the first user-facing `/chat` after a server restart would block for that whole window.

**The fix** is a daemon thread kicked off at server boot:

```python
# main.py — startup hook
threading.Thread(target=lambda: _ensure_indexed(SessionLocal()),
                 name="embeddings-warm", daemon=True).start()
```

Inside `embeddings.py`:

- `_warming` flag is True for the whole load.
- `semantic_search()` checks `_warming` *before* taking the lock. If warming, returns `None` immediately and the keyword path serves the answer.
- `services.search_knowledge_base` already treats `None` as "fall back to keyword only" — chat never blocks.

**Status endpoint** — `GET /embeddings/status` returns a snapshot:

```json
{
  "state": "warming" | "ready" | "failed" | "not_started" | "idle_empty_kb",
  "chunks_indexed": 0,
  "chunks_total": 847,
  "elapsed_seconds": 47.2,
  "model_name": "...",
  "model_load_failed": false
}
```

Use it to confirm warm-up is in progress without grepping the uvicorn log.

---

## Frontend layout

Vanilla JS, no bundler. The whole app is one `app.js` file plus `index.html` + `style.css`. Home screen structure:

```
┌──────────────────────────────────────────────────────────────┐
│                       Farmer AI                              │  ← animated wordmark
│        Your fields. Your language. Real answers.             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Ask Farmer AI                                          │  │  ← hero chat card
│  │ Crops, schemes, …                                      │  │     w/ rainbow border,
│  │ [type your question……………………………………………] [→]            │  │     input field
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐              │
│  │Weather │  │ Mandi  │  │Diagnose│  │ Scheme │              │  ← action shelf
│  │  blue  │  │ leaf   │  │  teal  │  │ wheat  │              │     (dealt in with
│  │View →  │  │Check → │  │Scan →  │  │Browse →│              │     a 70ms stagger)
│  └────────┘  └────────┘  └────────┘  └────────┘              │
└──────────────────────────────────────────────────────────────┘
```

Responsive: 4 cards in a row on desktop, 2×2 on tablet, single column on mobile.

### Per-card accent identity

| Card | Accent | What it touches |
|---|---|---|
| Weather | sky-blue (`--weather-accent`) | icon, dot, glow, hover border, CTA color |
| Mandi Prices | leaf-green (`--leaf`) | same set |
| Diagnose | teal `#2D8C7E` | same set |
| Scheme Finder | wheat-gold (`--wheat-dark`) | same set |
| Chat hero | always-on animated rainbow border via `::before` | signature element, not hover-tied |

### Action handlers

- **Weather / Mandi** → open their respective modals (live data from backend).
- **Diagnose** → `startNewChat()` → `showChat()` → triggers the hidden file input. Posts to `/diagnose-leaf`.
- **Scheme Finder** → opens the Schemes modal (no backend; static seed in `app.js`). Each scheme card has two actions:
  - *Open official portal* → `<a target="_blank">` to the real govt portal.
  - *Ask AI about this* → `startNewChat()` + `showChat(scheme.question)` — auto-sends a comprehensive pre-filled question (eligibility, amount, application, documents, status, rejection reasons). All 6 schemes have English + Gujarati versions; the right one is picked at click time.

### Type system

Two families across the whole app, set as CSS vars:

```css
--font-display: 'Fraunces', 'Noto Serif Gujarati', Georgia, serif;
--font-body:    'Inter Tight', 'Noto Sans Gujarati', system-ui, sans-serif;
```

- Display family used at 24px+ only — wordmark, hero subtitle, card titles, modal titles, weather temperature readings.
- Body family for chat, paragraphs, labels, buttons. Buttons just bump weight to 600 — no third family.

### Backgrounds

- **Light mode** (`:root`): cream `#F6F1E7` substrate plus a subtle `body::before` vertical gradient — cool pale cream at the top fading to dusty wheat at the bottom, plus a leaf-pale glow in the lower-right corner. Reads as "sky above, earth below".
- **Dark mode** (`[data-theme="dark"]`): warm amber-coffee `#221C12` substrate, elevated cards at `#2E2718`. Body gradient is hidden in dark — the warm dark substrate carries the mood on its own. Primary stays olive-green (just brightened to `#7BB582` for dark legibility) so the brand's action identity carries across modes — wheat/leaf/weather each stay in their own hue family at dark-appropriate luminance.

### i18n

Single `t()` lookup with `en` and `gu` tables. `language` is a module-level variable; the toggle just flips it and re-runs the language updater. Schemes (modal chrome + per-scheme title/tag/desc/question) are bilingual via `_gu` suffix fields on each scheme record.

---

## Quick mental model

If anything ever stops working, run through this checklist:

1. **`.env` loaded?** `config.py` prints the keys at startup.
2. **DB reachable?** `python Backend/database.py` exits clean iff so.
3. **Tables current?** Compare `database.py` models vs. `\d` in pgAdmin. If you added a column after first run, `create_all` won't add it — you'll need an ALTER.
4. **Market table populating?** `SELECT snapshot_date, COUNT(*) FROM market_price_snapshots GROUP BY snapshot_date;` — should have ≥ 1 row within ~5 s of uvicorn boot.
5. **Embeddings ready?** Hit `GET /embeddings/status`. `state: ready` means semantic search is active. `warming` is fine — chat still works (keyword-only) until it finishes.
6. **Weather cache pruned?** `SELECT COUNT(*) FROM weather_cache;` — should always be ≤ 33.
7. **Frontend talking to the right backend?** `API_BASE` at the top of `app.js`.
8. **CORS?** Backend allows `*`. Frontend must be served over `http://`, never `file://`.

That's the whole system.
