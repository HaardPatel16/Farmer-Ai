# 🌾 Farmer AI — Setup & run

AI assistant for Gujarat farmers — chat (English/Gujarati), live weather, mandi prices, leaf-disease diagnosis, government-scheme finder.

This guide assumes a fresh machine with nothing installed. Run through it top-to-bottom once, then use **[Daily run](#daily-run)** every other day.

For *how the system works internally*, see `explanation.md`.

---

## What you need

| Tool | Why | Get it |
|---|---|---|
| **Python 3.10+** | Runs the backend | [python.org/downloads](https://www.python.org/downloads/) |
| **PostgreSQL** | Stores chats, KB chunks, caches, snapshots | [postgresql.org/download](https://www.postgresql.org/download/) |
| **pgAdmin** | Database GUI | Bundled with PostgreSQL on Windows/Mac, or [pgadmin.org](https://www.pgadmin.org/download/) |
| **Groq API key** (free) | Powers the chatbot | [console.groq.com](https://console.groq.com/) → API Keys → Create |
| **data.gov.in API key** (free, optional) | Powers mandi-price card. App runs without it; only that one card breaks. | [data.gov.in](https://www.data.gov.in/) → My Account → API key |

> **Windows users:** tick **"Add python.exe to PATH"** on the first installer screen — most `'python' is not recognized` errors trace back to skipping this.

---

## One-time setup

### 1. Create + activate a virtual environment

In the project root:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell or cmd)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Prompt should now start with `(venv)`. **Do this every time** you open a new terminal for this project.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Pulls `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`, `groq`, `httpx`, `requests`, `pypdf`, `torch`, `torchvision`, `pillow`, `python-multipart`, `sentence-transformers`.

> `torch` + `torchvision` are large (~150 MB) — they power the leaf classifier. First install takes a few minutes on slow connections.

### 3. Create the database

1. Open **pgAdmin**, connect to your local PostgreSQL server.
2. Right-click **Databases → Create → Database…**
3. Name: exactly `farmer_ai`. Save.

You don't need to create any tables by hand — step 5 does it.

### 4. Create your `.env` file

In the **project root** (same level as `Backend/`, `Frontend/`, `Knowledge_base/` — **not** inside any of them), create a file called exactly `.env`:

```dotenv
DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5432/farmer_ai
GROQ_API_KEY=your_groq_api_key_here
MARKET_API_KEY=your_data_gov_in_api_key_here
```

- Replace `YOUR_POSTGRES_PASSWORD` with your local Postgres password.
- Leave `MARKET_API_KEY=` empty if you don't have a key yet — everything except the mandi-prices card will still work.
- If Postgres runs on a non-default port, replace `5432`.

> Never commit `.env` — it's in `.gitignore`.

### 5. Create the database tables

```bash
cd Backend
python database.py
```

Expected output:

```
Connecting to database and creating tables...
Done. Tables created: chats, feedback, knowledge_chunks, weather_cache, market_price_snapshots
```

If you see a connection error, double-check `DATABASE_URL` — see [Troubleshooting](#troubleshooting).

### 6. (Optional) Load knowledge-base documents

Drop your `.txt` / `.pdf` source documents into `Knowledge_base/sources/` (create the folder if it doesn't exist), then **from the project root**:

```bash
cd ..                              # back to project root if you're still in Backend/
python Knowledge_base/ingest.py
```

The script chunks each file, auto-tags any Gujarat districts it mentions, and inserts rows into `knowledge_chunks`. Safe to re-run — re-ingesting a file replaces that file's chunks, doesn't duplicate them.

Setup is done.

---

## Daily run

You need **two terminals running at once**.

### Terminal 1 — Backend

```bash
cd path/to/Farmer-Ai-main
venv\Scripts\activate              # macOS/Linux: source venv/bin/activate
cd Backend
uvicorn main:app --reload
```

Expected boot output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
[market/refresh] fetched N records, inserted N new rows
[startup] embeddings warmup started in background thread
[embeddings] ▶ warmup started — loading model + indexing chunks…
```

Two background things kick in on boot:

- **Market poller** — first fetch fires immediately, then every 30 min.
- **Embeddings warm-up** — runs in a daemon thread (~2–5 min on first boot, then cached forever). Chat works during warm-up (keyword search only); once `[embeddings] ✓ warmup complete` appears, semantic search is active too.

Leave this terminal running.

### Terminal 2 — Frontend

The frontend **must** be served over `http://`, not opened as a file. Double-clicking `index.html` will silently break chat, weather, and market features.

```bash
cd path/to/Farmer-Ai-main/Frontend
python -m http.server 5500
```

Then open:

```
http://127.0.0.1:5500
```

### Stopping

`Ctrl + C` in each terminal.

---

## Quick test checklist

1. Type any farming question — *"What crops grow well in summer?"* — you should get a reply within seconds.
2. Mention a district — *"What soil suits Bhavnagar?"* — reply should be district-specific.
3. Switch to **ગુ** in Settings, ask in Gujarati — replies in Gujarati, district detection works in Gujarati script too (e.g. ભાવનગર).
4. Click 👍 / 👎 on a reply — feedback row goes into the `feedback` table.
5. Open **Weather** — all 33 districts load within ~1 s.
6. Open **Mandi Prices** — flat table sorted by commodity → variety → district.
7. Open **Diagnose a Leaf** — file picker opens, upload a plant photo, get a ranked diagnosis + remedy.
8. Open **Scheme Finder** — 6 schemes listed; click *Ask AI about this* on any → starts a fresh chat and auto-sends a comprehensive question (eligibility, amount, how to apply, documents, status, rejection reasons).

---

## Where to look when verifying things

| What | Where |
|---|---|
| Swagger UI (test endpoints) | `http://127.0.0.1:8000/docs` |
| Embeddings warm-up state | `http://127.0.0.1:8000/embeddings/status` |
| Was a chat answer KB-grounded? | `SELECT id, query, source_type, confidence_score FROM chats ORDER BY id DESC LIMIT 1;` — `source_type` is `knowledge_base` / `mixed` / `llm_reasoning` / `weather_api` / `leaf_diagnosis` |
| Market table populated? | `SELECT snapshot_date, COUNT(*) FROM market_price_snapshots GROUP BY snapshot_date;` (one row expected: today's IST date) |
| Weather cache bounded? | `SELECT COUNT(*) FROM weather_cache;` (≤ 33 always) |
| KB contents | `SELECT id, source_filename, districts FROM knowledge_chunks ORDER BY id;` |

### Reading `source_type`

| Value | What it means |
|---|---|
| `knowledge_base` | Answer is paraphrasing your ingested documents — trustworthy. |
| `mixed` | KB chunks helped but the LLM filled gaps with general knowledge. |
| `llm_reasoning` | No relevant KB chunk — answer is pure LLM training data. Plausible, not verified. |
| `weather_api` | Live Open-Meteo data was injected into the answer. |
| `leaf_diagnosis` | Came from `/diagnose-leaf` — ConvNeXt prediction + Groq remedy. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH" ticked, or use `py` on Windows. |
| `ModuleNotFoundError: No module named 'fastapi'` (or similar) | Virtual environment isn't active — run `venv\Scripts\activate` (or `source venv/bin/activate`), then `pip install -r requirements.txt`. |
| `DATABASE_URL is missing. Check your .env file.` | `.env` is missing, misnamed, or not in the project root. Must be one level above `Backend/`. |
| `ModuleNotFoundError: No module named 'database'` (or `services`, `config`) | You're not in `Backend/` — `cd Backend` first; `main.py`, `services.py`, `database.py`, `config.py` all import each other directly. |
| `connection to server … failed` when running `python database.py` | PostgreSQL isn't running, or password/port in `DATABASE_URL` is wrong. Check pgAdmin shows the server as connected. |
| `database "farmer_ai" does not exist` | You haven't created it in pgAdmin — see step 3 of [setup](#one-time-setup). |
| `column "districts" of relation "knowledge_chunks" does not exist` | You added the `districts` column after running `database.py` once — `create_all` doesn't add columns to existing tables. Run once in pgAdmin's Query Tool: `ALTER TABLE knowledge_chunks ADD COLUMN districts VARCHAR;` |
| Frontend loads but clicking anything does nothing | You opened `index.html` directly (`file://…`) instead of via `http://127.0.0.1:5500`. Must serve via `python -m http.server`. |
| Browser console shows CORS or "Unsafe attempt to load URL" | Same as above — must use `http://`, not `file://`. |
| Mandi-prices card stays empty | `MARKET_API_KEY` missing or wrong in `.env`. **Restart uvicorn after editing `.env`** — it only reads the file at startup. Verify with `GET /market-price/all` in Swagger. |
| Mandi prices time out / never load even with valid key | data.gov.in is being slow — the request is already User-Agent-spoofed to dodge their default-UA stall. Wait one poll cycle (30 min) and re-check. |
| First chat after restart hangs for 2–5 min | Embeddings warm-up is loading the multilingual model. Check `GET /embeddings/status` — `state: warming` means it's working. Subsequent chats are instant. Chat *should not* actually block in this state — if it does, the warmup thread errored; see the uvicorn log. |
| First chat says `Warning: You are sending unauthenticated requests to the HF Hub` | Harmless — sentence-transformers is downloading the model anonymously. The model caches to `~/.cache/huggingface/` after first download. To silence: add `HF_TOKEN=hf_…` to `.env`. |
| Groq errors / chatbot won't reply (after warm-up is done) | Check `GROQ_API_KEY` in `.env` and that your Groq free quota isn't exhausted. The uvicorn terminal logs the actual Groq error. |
| Port 8000 in use | Run `uvicorn main:app --reload --port 8001`, then change `API_BASE` at the top of `Frontend/app.js` to `http://127.0.0.1:8001`. |
| Port 5500 in use | Run `python -m http.server 5600` instead, open `http://127.0.0.1:5600`. |
| KB question keeps returning `source_type: llm_reasoning` when you expected `knowledge_base` | Run the `knowledge_chunks` SELECT above — likely the doc was never ingested, or the chunk's `districts` tag doesn't include the district you asked about. |

---

## Push to GitHub

### First-time setup (once per machine)

1. **Install Git** — [git-scm.com/downloads](https://git-scm.com/downloads).

2. **Tell Git who you are** (one-time, global):

   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   ```

3. **Create an empty repo on GitHub** — go to [github.com/new](https://github.com/new). Name it (e.g. `farmer-ai`). **Do NOT** tick "Initialize with README / .gitignore / license" — this project already has its own. Click *Create repository*.

4. **Link your local repo to GitHub** — GitHub will show a `…or push an existing repository` block. From the project root:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/farmer-ai.git
   git branch -M main
   git push -u origin main
   ```

   First push will prompt for a **GitHub personal access token** as the password (regular passwords were retired). Create one at [github.com/settings/tokens](https://github.com/settings/tokens) → *Generate new token (classic)* → tick the `repo` scope. Paste it when prompted. Git Credential Manager remembers it after the first time.

### Day-to-day (every time you change code)

```bash
git status                          # see what changed
git add Backend/services.py Frontend/app.js   # stage specific files
# (or `git add .` to stage everything, but check `git status` first)
git commit -m "short message describing the change"
git push
```

### What never to commit

The repo already has a `.gitignore` covering `.env`, `venv/`, `__pycache__/`, and `ML model/*.pth`. If you ever run anything that creates a `Frontend/node_modules/` directory, add it to `.gitignore` too — that folder is large and unnecessary.

If you accidentally committed a secret, **rotate the key immediately** (revoke + create a new one on Groq/data.gov.in). Removing it from git history is more work than rotating, and the leaked one is already public.

### Useful checks before pushing

```bash
git diff                            # see line-by-line changes vs last commit
git log --oneline -5                # last 5 commits
git remote -v                       # confirm the remote URL
```

---

## Project structure

```
Farmer-Ai-main/
├── Backend/
│   ├── main.py            ← FastAPI routes + startup tasks (market poller, embeddings warm)
│   ├── services.py        ← chat, weather, market, KB search, district detection
│   ├── database.py        ← SQLAlchemy models; run once to create tables
│   ├── config.py          ← loads .env into module constants
│   ├── embeddings.py      ← semantic search; warms on startup
│   └── ml_model.py        ← ConvNeXt-Small leaf classifier
├── ML model/
│   └── Latest_Plant_Model_.pth    ← trained weights (NOT in repo — see "Model weights" below)
├── Frontend/
│   ├── index.html         ← serve via http.server, not file://
│   ├── style.css
│   └── app.js
├── Knowledge_base/
│   ├── sources/           ← drop .txt / .pdf files here, then run ingest.py
│   └── ingest.py          ← run from PROJECT ROOT, not from inside Backend/ or Knowledge_base/
├── venv/                  ← created in setup step 1
├── .env                   ← created in setup step 4 — never commit
└── requirements.txt
```

## Model weights

The trained `.pth` file isn't in this repo (too large for GitHub). Download `Latest_Plant_Model_.pth` from `<PASTE_YOUR_DRIVE_LINK_HERE>` and place it in `ML model/`.
