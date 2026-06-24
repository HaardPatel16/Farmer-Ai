# 🌾 Farmer AI

An AI-powered farming assistant for Gujarat — chat about crops, schemes, soil, and pests in English or Gujarati, plus live weather and mandi (market) price dashboards.

This guide assumes you've just downloaded/cloned this project and have **nothing installed yet**. Follow it top to bottom once, then jump to [Running the project (every time after)](#running-the-project-every-time-after) for daily use.

---

## What you'll need installed

| Tool | Why | Get it |
|---|---|---|
| **Python 3.10+** | Runs the backend | [python.org/downloads](https://www.python.org/downloads/) |
| **PostgreSQL** | Stores chats, feedback, cached weather/market data | [postgresql.org/download](https://www.postgresql.org/download/) |
| **pgAdmin** | GUI to manage the database (usually installed together with PostgreSQL on Windows/Mac) | bundled with the PostgreSQL installer, or [pgadmin.org/download](https://www.pgadmin.org/download/) |
| **A Groq API key** (free) | Powers the chatbot's AI responses | [console.groq.com](https://console.groq.com/) → sign up → API Keys → Create |
| **A data.gov.in API key** (free, optional) | Powers live mandi/crop prices — the app runs fine without it, just without that one feature | [data.gov.in](https://www.data.gov.in/) → sign up → "My Account" → API key |

> **Windows users:** when installing Python, tick **"Add python.exe to PATH"** on the first installer screen — easy to miss, and most `'python' is not recognized` errors trace back to this.

---

## One-time setup

### 1. Open a terminal in the project folder

Unzip the project, then open a terminal there.

### 2. Create and activate a virtual environment

This keeps this project's Python packages separate from everything else on your machine.

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell or cmd)
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

You'll know it worked when your terminal prompt starts with `(venv)`. **Do this every time** you open a new terminal for this project (see the [daily run guide](#running-the-project-every-time-after)).

### 3. Install all Python dependencies (pip)

With `(venv)` active:

```bash
pip install -r requirements.txt
```

This installs everything the backend needs: `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`, `groq`, `pydantic`, `requests`, `pypdf`, `torch`, `torchvision`, `pillow`, `python-multipart`.

> The `torch`/`torchvision` install is large (100MB+) and powers the leaf disease classifier — it can take a few minutes on a slow connection.

### 4. Set up PostgreSQL + pgAdmin

1. Open **pgAdmin**.
2. In the left sidebar, you should see a server (often called `PostgreSQL 16` or similar) — click to connect, enter the master password you set during installation.
3. Right-click **Databases** → **Create** → **Database...**
4. Name it exactly: `farmer_ai`
5. Click **Save**.

That's it for pgAdmin — you don't need to create any tables by hand, the app does that for you in step 6.

### 5. Create your `.env` file

In the project root (the top-level `Farmer-Ai-main` folder, alongside `Backend/`, `Frontend/`, and `Knowledge_base/` — NOT inside any of those subfolders), create a new file named exactly `.env` (no filename before the dot) with this content:

```dotenv
DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5432/farmer_ai
GROQ_API_KEY=your_groq_api_key_here
MARKET_API_KEY=your_data_gov_in_api_key_here
```

- Replace `YOUR_POSTGRES_PASSWORD` with the password you set for the `postgres` user during PostgreSQL installation.
- Replace `your_groq_api_key_here` with the key from [console.groq.com](https://console.groq.com/).
- `MARKET_API_KEY` is optional — leave it blank (`MARKET_API_KEY=`) if you don't have one yet; everything except live crop prices will still work.
- If your PostgreSQL runs on a different port, change `5432` to match.

> ⚠️ Never commit or share this file — it contains your private API keys. It's already listed in `.gitignore`.

### 6. Create the database tables

Still in your terminal with `(venv)` active:

```bash
cd Backend
python database.py
```

You should see:

```
Connecting to database and creating tables...
Done. Tables created: chats, feedback, knowledge_chunks, weather_cache, market_cache
```

If you see a connection error here, double check your `DATABASE_URL` in `.env` (at the project root) — see [Troubleshooting](#troubleshooting) below.

### 7. (Optional) Load knowledge base documents

If you have source documents to ground the chatbot's answers (soil reports, scheme PDFs, crop guides, etc.), put them in `Knowledge_base/sources/` (create the folder if it doesn't exist), then run this from the **project root** (not from inside `Backend/` or `Knowledge_base/`):

```bash
cd ..                              # back to the project root, if you're still in Backend/
python Knowledge_base/ingest.py
```

This reads every `.txt` and `.pdf` file in that folder, splits each into chunks, automatically tags each chunk with any Gujarat districts it mentions, and stores them in the `knowledge_chunks` table. You'll see output like:

```
Processing soil_types_gujarat.txt...
  Chunk 1: 313 words — districts: anand, kheda, vadodara, dahod, panchmahals
  ...
Done — inserted 4 chunk(s) from soil_types_gujarat.txt.
```

Safe to re-run any time — adding a new file or editing an existing one and running this again won't create duplicates, it just refreshes that file's chunks. See [How to check whether an answer used the knowledge base](#how-to-check-whether-an-answer-used-the-knowledge-base) below to confirm it's actually being used.

Setup is done. Now start the project — see the next section.

---

## Running the project (every time after)

Every time you want to use the app, you need **two terminals running at once**: one for the backend, one for the frontend.

### Terminal 1 — Backend

```bash
cd path/to/Farmer-Ai-main
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
cd Backend
uvicorn main:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Leave this terminal open and running.

### Terminal 2 — Frontend

The frontend **must** be served over `http://`, not opened as a file directly (double-clicking `index.html` will silently break the chat, weather, and market features — browsers block this for security reasons).

```bash
cd path/to/Farmer-Ai-main/Frontend
python -m http.server 5500
```


You won't see much printed — that's normal, it's just waiting.

### Open the app

In your browser, go to:

```
http://127.0.0.1:5500
```

You should see the Farmer AI home screen. Try asking a question, checking the weather dashboard, or browsing crop market prices.

### Stopping everything

In each terminal, press `Ctrl + C`.

---

## Quick test checklist

1. Type a question, e.g. *"What crops grow well in Gujarat in summer?"* — you should see a typing indicator, then a reply.
2. Ask something mentioning a district by name, e.g. *"What soil type is good for farming in Bhavnagar?"* — the reply should be tailored to that district specifically, not generic Gujarat-wide advice.
3. Click 👍 or 👎 on a response to test feedback (👎 opens a reason picker).
4. Switch to **ગુ** in Settings and ask another question in Gujarati — district detection works in Gujarati script too (e.g. ભાવનગર is recognized the same as "Bhavnagar").
5. Open the **Weather** card from the home screen — district data should load.
6. Open the **Crop Market Prices** card, click a category pill — a price table should load.
7. Upload a photo of a plant leaf — you should get back a ranked list of predicted diseases plus an AI-generated remedy (see [Leaf disease diagnosis](#leaf-disease-diagnosis-diagnose-leaf) below).

---

## Leaf disease diagnosis (`/diagnose-leaf`)

The backend can identify plant leaf diseases from a photo using a custom-trained image classifier, then ask Groq for remedies based on the result.

**How it works:**

1. `ML model/Latest_Plant_Model_.pth` holds the trained weights for a **ConvNeXt-Small** classifier (78 output classes). `Backend/ml_model.py` loads this architecture, loads the weights into it, and exposes `predict_top_k()` for inference on a single image.
2. The `POST /diagnose-leaf` endpoint (in `Backend/main.py`) accepts an uploaded leaf photo, runs it through the classifier to get the top-k most likely diseases with confidence scores, then calls `ask_groq_disease_remedy()` (in `Backend/services.py`) to get practical remedies (organic + chemical treatment options, plus prevention) tailored to those predictions.
3. Try it via the Swagger UI at `http://127.0.0.1:8000/docs` → `/diagnose-leaf` → "Try it out", upload an image, and set `top_k` (default 3) and `language` (`en` or `gu`).

**Known limitation:** the `.pth` checkpoint only contains model weights — it doesn't embed the actual disease names. `Backend/ml_model.py` currently has placeholder class names (`class_0`, `class_1`, ...) in `CLASS_NAMES`. The classifier and remedy pipeline both work end-to-end already, but until the real ordered list of 78 disease names (matching the original training order) is filled in, predictions will show as generic placeholders instead of real disease names.

---

## How district-aware answers and the knowledge base work

Two features work together to make answers more specific to Gujarat farming, instead of generic AI knowledge:

**District detection.** Every question is scanned for a Gujarat district name — in English (`Bhavnagar`, including common spelling variants like `Navasari`/`navsari`) or Gujarati script (`ભાવનગર`). If one is found, the chatbot is explicitly told to tailor its answer to that district's soil, climate, and crop conditions.

**Knowledge base.** Documents you put in `Knowledge_base/sources/` and run through `ingest.py` (see step 7 above) get split into chunks and tagged with whichever districts they mention. When a question mentions a district, matching chunks for that district are boosted, and chunks tagged for *other* districts only are excluded — so a question about Bhavnagar won't accidentally get answered using a chunk about Kheda just because the wording overlaps.

If no relevant chunk exists in the knowledge base yet, the chatbot still answers using the AI model's own general knowledge — it just means that particular answer isn't grounded in a document you've verified. The next section shows you how to tell which one happened.

---

## How to check whether an answer used the knowledge base

This is the most useful thing to check when you're not sure if an answer is trustworthy — it tells you definitively whether the chatbot pulled from your documents or just used its own general training knowledge.

### Steps

1. Open **pgAdmin** → expand your server → expand **Databases** → click **farmer_ai**.
2. Go to **Tools → Query Tool** (or right-click `farmer_ai` → **Query Tool**).
3. Paste in and run (F5, or the ▶ button) this query, which shows your most recent chat:

```sql
SELECT id, query, source_type, confidence_score, district, response
FROM chats
ORDER BY id DESC
LIMIT 1;
```

4. Look at the `source_type` column:

| `source_type` value | What it means |
|---|---|
| `knowledge_base` | The answer used real content from a document you ingested. `confidence_score` shows how many chunks matched (higher = more matching chunks found). Trustworthy — grounded in what you fed it. |
| `llm_reasoning` | No matching document was found. The answer came entirely from the AI model's own general knowledge. Plausible, but not verified against anything you've checked yourself. |
| `weather_api` | The answer used live weather data fetched in real time, not the knowledge base. |

5. The `district` column shows which district (if any) was detected in the question — useful for confirming district detection worked even when `source_type` is `llm_reasoning`.

### Checking what's actually in your knowledge base

To see every chunk currently stored, and which districts each one is tagged with:

```sql
SELECT id, source_filename, districts, keywords
FROM knowledge_chunks
ORDER BY id;
```

If a district-specific question keeps coming back as `llm_reasoning` when you expected `knowledge_base`, this query tells you why — either no document covering that district has been ingested yet, or the `districts` tag on the relevant chunk doesn't include the district you asked about.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick "Add to PATH", or use `py` instead of `python` on Windows |
| `ModuleNotFoundError: No module named 'fastapi'` (etc.) | Your virtual environment isn't active — run the `venv\Scripts\activate` step again, then `pip install -r requirements.txt` |
| `DATABASE_URL is missing. Check your .env file.` | Your `.env` file is missing, misnamed, or not in the project root (one level above `Backend/`) |
| `ModuleNotFoundError: No module named 'database'` (or `'services'`, `'config'`) when running `uvicorn` or `python main.py` | You're not in the `Backend/` folder — `cd Backend` first, since `main.py`, `database.py`, `services.py`, and `config.py` all live there together and import each other directly |
| `connection to server ... failed` when running `python database.py` | PostgreSQL isn't running, or your password/port in `DATABASE_URL` is wrong — check pgAdmin shows the server as connected |
| `database "farmer_ai" does not exist` | You haven't created it in pgAdmin yet — see step 4 above |
| Frontend loads but nothing happens when you click anything | You opened `index.html` directly (`file://...`) instead of through `http://127.0.0.1:5500` — see Terminal 2 above |
| Weather/market data won't load, browser console shows a CORS or "Unsafe attempt to load URL" error | Same as above — must use `python -m http.server`, not double-clicking the file |
| Crop prices show "Unable to load prices" | Check `MARKET_API_KEY` is set correctly in `.env`, and that you **restarted uvicorn** after editing `.env` — it only reads the file once, at startup |
| Crop prices time out / never load even with a valid key | Known quirk with `api.data.gov.in` — already handled in `services.py` (it sends a browser-style User-Agent header), but if it recurs, it's the government API being slow, not your setup |
| Groq API errors / chatbot won't reply | Check `GROQ_API_KEY` in `.env`, and that your Groq account has remaining free quota |
| `ModuleNotFoundError: No module named 'groq'` (or similar) when running `python Knowledge_base/ingest.py` | Your virtual environment isn't active in this terminal — run `venv\Scripts\activate` (Mac/Linux: `source venv/bin/activate`) first, every new terminal needs this. Note `ingest.py` is run from the **project root**, not from inside `Backend/` or `Knowledge_base/` — it reaches into `Backend/` on its own to import `database`/`services` |
| Knowledge base answers always show `source_type: llm_reasoning` even for a district you've ingested data for | Run the `SELECT ... FROM knowledge_chunks` query in [How to check whether an answer used the knowledge base](#how-to-check-whether-an-answer-used-the-knowledge-base) — most likely `ingest.py` was never run, or the `districts` column doesn't exist yet (see next row) |
| `column "districts" of relation "knowledge_chunks" does not exist` | You added the `districts` column to `database.py` after already running `python database.py` once — `create_all()` only creates new tables, not new columns on existing ones. Run this once in pgAdmin's Query Tool: `ALTER TABLE knowledge_chunks ADD COLUMN districts VARCHAR;` |
| Port 8000 already in use | Run `uvicorn main:app --reload --port 8001` instead, and change `API_BASE` at the top of `Frontend/app.js` to match |
| Port 5500 already in use | Run `python -m http.server 5600` (or any free port) instead, and open that port in your browser |

---

## Project structure

```
Farmer-Ai-main/
├── Backend/
│   ├── config.py          ← loads .env (from the project root) variables
│   ├── database.py        ← run once to create tables
│   ├── main.py             ← FastAPI backend, defines all routes
│   ├── services.py         ← chatbot, weather, market price, and disease remedy logic
│   └── ml_model.py         ← loads the leaf disease classifier, runs predictions
├── ML model/
│   └── Latest_Plant_Model_.pth   ← trained ConvNeXt-Small weights for leaf disease detection
├── Frontend/
│   ├── index.html         ← serve this folder with http.server, don't open directly
│   ├── style.css
│   └── app.js
├── Knowledge_base/
│   ├── sources/            ← drop your .txt / .pdf documents here, then run ingest.py
│   └── ingest.py            ← run from the PROJECT ROOT, not from inside Knowledge_base/ or Backend/
├── venv/                  ← created by you in step 2, not in the repo
├── .env                   ← created by you in step 5, never share this — stays at the project root
└── requirements.txt
```

---

## API docs (optional, for development)

While the backend is running, you can explore and test every endpoint directly at:

```
http://127.0.0.1:8000/docs
```

This is FastAPI's built-in interactive Swagger UI — useful for testing `/chat`, `/weather`, `/market-price`, or `/diagnose-leaf` without going through the frontend at all.

## Model weights
The trained model file is NOT in this repository (too large for GitHub).
Download `Latest_Plant_Model_.pth` from <PASTE_YOUR_DRIVE_LINK_HERE>
and place it in `ML model/`.