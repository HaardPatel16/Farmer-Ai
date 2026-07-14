# Farmer AI — Presentation Content (12 slides)

Part 1 is the slide content itself — fuller bullets you can paste onto slides nearly as-is.
Part 2 (at the end) keeps the speaker notes aside for rehearsal.

---

# PART 1 — SLIDE CONTENT

## Slide 1 — Title

> **Farmer AI — Gujarat**
> A bilingual AI assistant for farmers
>
> Haard Patel · 2024B3A71015G
> Varun Chaudhary · 2024B4A11122G
> Devarsh Shah · 2024B3A71468G
> Sheil Maniar · 2024A7PS0498G
> Albin Abraham · 2024A7PS0623G
>
> Practice School I — EmergingFive, Science City
> Industry Mentor: Mr. Maulik Patel · PS Faculty: Prof. Amit Dua

**Visual:** home-screen screenshot faded behind the title.

---

## Slide 2 — The Problem

> **One farmer. Four questions. Four different apps. None in Gujarati.**
>
> - "What's wrong with this leaf?" → a disease-identifier app
> - "What's the onion price at the mandi today?" → a government price portal
> - "Will it rain tomorrow?" → a weather app
> - "Am I eligible for PM-KISAN?" → a scheme website
>
> - Extension officers and agronomists are scarce in rural Gujarat
> - Existing tools solve one narrow problem each — and rarely speak the farmer's language
> - **Result: no single, fast, trustworthy place to ask**

**Visual:** four disconnected app icons on the left vs. one phone running Farmer AI on the right.

---

## Slide 3 — From AgroBot to Farmer AI

> **Midsemester — AgroBot:** a standalone leaf-disease classifier
> - MobileNetV2 · Flask backend · React frontend · 13 classes (potato & tomato only)
>
> **Mentor feedback:** a farmer with a sick leaf also asks about weather, prices, and compensation — *"farmers never have just one question."*
>
> **Second half — full re-architecture:**
> - Classifier upgraded, not discarded: MobileNetV2 → **ConvNeXt-Small**, 13 → **78 classes**
> - Flask → **FastAPI** · React → **vanilla JS** (built for low-end rural devices)
> - Disease diagnosis became **one feature of five** inside a conversational assistant

**Visual:** before/after boxes joined by an arrow labeled "mentor feedback."

---

## Slide 4 — What Farmer AI Does

> 1. **Chat** — farming Q&A grounded on ~1,400 curated Gujarat documents, with conversation memory for follow-ups
> 2. **Diagnose** — leaf disease from a photo (gallery or camera), 78 classes, remedies included
> 3. **Weather** — all 33 Gujarat districts, live readings + tomorrow's forecast
> 4. **Mandi prices** — auto-refreshed every 30 min in the background, served instantly from a local snapshot
> 5. **Scheme Finder** — PM-KISAN, PMFBY, i-Khedut & more: official portal links + one-tap "Ask AI about this"
>
> **Every screen, every answer: English / ગુજરાતી · light & dark theme · runs in any phone browser**

**Visual:** five icon cards mirroring the app's home screen, in its accent colors.

---

## Slide 5 — Architecture

> **Three layers:**
> - **Frontend** — framework-free HTML/CSS/JS; session & language kept in the browser
> - **Backend** — FastAPI + Uvicorn; owns validation, orchestration, caching, background jobs
> - **Data** — PostgreSQL: chats, feedback, 14k knowledge chunks, weather cache, price snapshots
>
> **External services:** Groq API (Llama-3.3-70B) · Open-Meteo (weather) · data.gov.in (mandi prices)
> **Local models:** ConvNeXt-Small classifier · multilingual sentence-embedding model
> *(both auto-select GPU when present, fall back to CPU)*
>
> **Two data patterns:**
> - Weather = **pull on demand** — fetch only if the 10-min cache is stale
> - Market = **push on schedule** — background poll every 30 min; user requests read locally, never wait

**Visual:** the architecture diagram — frontend → backend → DB, external services above, local models below, color-coded external vs. local.

---

## Slide 6 — The Knowledge Base

> **~1,400 documents → ~14,000 searchable chunks**
>
> - 16 topic folders, all Gujarat-specific:
>   crops (330+ cultivation guides) · schemes (160) · livestock (137) · advisories (130) · digital · post-harvest · market · soils · water · districts · monthly crop calendars…
> - Ingestion pipeline: read each document → split into **~200-word chunks along paragraph boundaries** → tag with keywords → tag with **every district mentioned**
> - No district tag = the chunk applies state-wide
> - Safe re-runs: editing a document and re-ingesting replaces only that file's chunks
>
> **The retrieval problem: find the right 4 chunks out of 14,000 — next slide**

**Visual:** funnel graphic — documents (folder labels) → ingest script → chunk rows; the numbers 1,400 and 14,000 in large type.

---

## Slide 7 — Hybrid Retrieval (core technical slide)

> **Two retrievers run in parallel on every query:**
>
> **1. Keyword scorer**
> - Match *density* (hits per 100 words) — a chunk *about* cotton beats a list that mentions it once
> - Boosts: topic word in the chunk's **heading** (+80) · in the source **filename** (+100, strongest signal) · **district** match (+20)
> - District isolation: chunks tagged for *other* districts are excluded outright
>
> **2. Semantic search**
> - Multilingual embeddings (MiniLM-L12-v2): meaning → 384-dim vectors, Gujarati & English in one space
> - Cosine similarity finds synonyms & cross-language matches ("cold storage" → *Cold Chain* guide)
>
> **Fusion & cleanup:**
> - **Reciprocal Rank Fusion** (k=60) — fuse by rank, since the two score scales aren't comparable
> - **MMR de-duplication** (λ=0.7) — relevance vs. redundancy, removes near-identical chunks
> - Final context capped at **4 chunks** → small prompts, low token cost
> - Semantic index not ready? → **graceful fallback to keyword-only**, chat never blocks

**Visual:** left-to-right flow: query → two parallel boxes → RRF funnel → MMR filter → 4 chunk cards → LLM.

---

## Slide 8 — Honest Answers: the Source Label

> **Most RAG systems label an answer "from the knowledge base" just because chunks were retrieved — even when the LLM ignored them.**
>
> Farmer AI verifies **after** the answer is written:
> - **Coverage score** = fraction of the answer's content words found in the retrieved chunks
> - **Hedge detection** = phrases like *"the reference doesn't mention…"* (the model admitting it went beyond our documents)
>
> | Condition | Badge |
> |---|---|
> | coverage ≥ 70%, no hedging | 🟢 Knowledge Base |
> | 20–70%, or hedged | 🟡 Mixed |
> | live weather injected | 🔵 Weather Data |
> | everything else | ⚪ AI Reasoning |
>
> **The farmer sees this badge on every answer. The score is stored per answer for auditing.**

**Visual:** real chat-bubble screenshot with its badge, beside the threshold table.

---

## Slide 9 — Leaf Diagnosis Pipeline

> **photo → background removal → classification → remedy**
>
> - **rembg / U²-Net** cuts the leaf out and places it on a white canvas — field photos are full of soil, hands, shadows; the classifier shouldn't waste capacity on them
> - **ConvNeXt-Small**, fine-tuned via transfer learning from ImageNet — **78 crop & condition classes** (vs. 13 at midsemester)
> - Preprocessing replicates training *exactly* (resize 256 → center-crop 224) — any mismatch silently destroys accuracy
> - **Groq writes the remedy**: symptoms, organic + chemical options, prevention — in the farmer's language
> - Safety rails: **never invents dosages** ("follow the label rate") · rembg failure → classify the raw image instead of failing
> - Transparency: the app crossfades the farmer's photo into **the exact cutout the model analyzed**

**Visual:** before/after leaf pair (cluttered photo → white-background cutout) with the pipeline stages as arrows beneath. The deck's best image.

---

## Slide 10 — Engineering for the Real World

> **Two debugging stories:**
>
> | Symptom | Actual cause | Fix |
> |---|---|---|
> | Market requests hang forever — no error, no timeout | data.gov.in silently drops non-browser clients | one browser-style **User-Agent** header → sub-second responses |
> | Semantic search "stuck warming" forever | Gujarati text crashed the Windows console encoding (cp1252), silently killing the warm-up thread | force **UTF-8** output at startup |
>
> **The rule that came out of it — every failure degrades, never dies:**
> - Embedding index still warming → chat answers keyword-only
> - Background removal fails → classify the raw image
> - Class-names file missing → placeholder labels + warning, request still succeeds
>
> **A farmer on a shaky network never sees a dead app.**

**Visual:** the symptom→cause→fix table; degradation rule as three short lines below it.

---

## Slide 11 — Results & the Feedback Loop

> **Answer quality**
> - KB-grounded answers: noticeably more specific, fewer invented figures than ungrounded ones
> - Off-topic queries blocked *before* the LLM → **0 tokens spent**
> - Follow-ups work: "and what about irrigation?" retrieves crop-specific chunks instead of being refused
>
> **Performance**
> - Weather, all 33 districts, cold cache: **~1 second** (concurrent fetch vs. ~10s sequential)
> - Full market refresh: **a few seconds** (parallel) vs. tens of seconds (sequential)
> - Retrieval scoring across 14k chunks: **milliseconds** (cached corpus index)
>
> **Self-measurement**
> - Every answer stores: exact chunks used · coverage score · token cost
> - 👍/👎 with reasons → dashboard: satisfaction **by answer source** & by district
> - *The system can show whether the knowledge base is actually helping — and where it's weak*

**Visual:** big stat callouts (0 tokens · ~1 s · 33 districts · ms); small stats-view screenshot if available.

---

## Slide 12 — Limitations, Future Work & Handoff

> **Honest limitations**
> - Knowledge base is hand-curated → needs manual refresh, can go stale
> - Colloquial / code-mixed Gujarati retrieval weaker than formal phrasing
> - 78 classes but uneven training data — original potato/tomato classes far better covered
> - Mandi prices depend on a single government source
> - **Not yet tested with real farmers**
>
> **Next steps**
> - Short field pilot via a Krishi Vigyan Kendra — *before* adding features
> - Find a live government data source for schemes
> - Persist the embedding index (skip the 2–5 min rebuild on restart)
>
> ### *Now — let us show you.*

**Visual:** two columns (Limitations / Next); the handoff line large and alone at the bottom.

---

## Timing guide

| Slides | Minutes |
|---|---|
| 1–4 (setup) | ~3.5 |
| 5–9 (technical core) | ~5.5 |
| 10–12 (reality, results, close) | ~3 |
| **Total before demo** | **~12** |

If forced to cut: merge 3 into 2, and 6 into 7. Never cut 7, 8, or 10.

---
---

# PART 2 — SPEAKER NOTES (kept aside, for rehearsal only)

**Slide 1.** "Good morning. We're presenting Farmer AI — Gujarat, a bilingual assistant that gives farmers one place to ask anything about their farm — in English or Gujarati." Introduce the team.

**Slide 2.** A farmer with a sick crop has four questions at once; today each needs a different tool and almost none work in Gujarati. The problem is fragmentation, not one missing feature.

**Slide 3.** Own the pivot: mentor feedback mid-internship reshaped the project. Emphasize the classifier was upgraded and embedded, not thrown away. Tease the React→vanilla decision ("I'll explain why later").

**Slide 4.** One line per feature; this slide is the map — "the next slides walk through how each piece works."

**Slide 5.** Walk the diagram top to bottom. Land the two data patterns: weather pull-on-demand, market push-on-schedule — "same problem, two designs, chosen per provider's behaviour."

**Slide 6.** We curated the corpus ourselves. Explain chunking in one sentence: you can't hand an LLM 1,400 documents, so you find the four most relevant 200-word pieces. District tagging = a Bhavnagar question never gets Kheda advice.

**Slide 7.** The rehearse-most slide. Keyword = precision on exact terms and filenames; semantic = synonyms and cross-language; RRF because scores aren't comparable so you fuse by rank; MMR because the fused list has near-duplicates; cap at 4 chunks for cost; degrade to keyword-only while the index warms.

**Slide 8.** The design decision we're most attached to. Retrieval happening ≠ retrieval being used — so verify against the written answer: coverage + hedge phrases. Only earned answers get the KB badge; farmer sees it on every reply; score stored for audit.

**Slide 9.** Three stages: rembg strips clutter (trained data is clean, fields aren't) → ConvNeXt classifies (transfer learning; preprocessing must match training exactly) → Groq writes remedies, forbidden from inventing dosages. Close with the crossfade transparency touch.

**Slide 10.** Tell both stories with their misdirection: hanging requests looked like async bugs, were a header; stuck warm-up looked like a slow model, was a console encoding crash in a background thread. Then the rule: every failure degrades, never dies — three examples.

**Slide 11.** Grounded answers measurably better; off-topic costs zero tokens; the two concurrency wins (~1s weather sweep, seconds-not-tens market refresh); and the system measures itself — satisfaction by answer source tells us if the KB is actually helping.

**Slide 12.** Read the limitations plainly — honesty scores points. Biggest one: no real-farmer testing yet, which is exactly why the first recommendation is a field pilot. End: "rather than tell you more — let us show you."

---

## Likely viva questions (one-line answers to rehearse)

- **What is RRF?** Rank-based fusion: each retriever's list contributes 1/(60+rank) per chunk; fuses incomparable score scales without calibration.
- **What is MMR?** Greedy re-ranking maximizing relevance minus similarity to already-picked chunks — kills near-duplicates.
- **What are embeddings?** Vectors placing similar meanings close together, across languages; similarity = cosine of the angle.
- **Why RAG instead of fine-tuning the LLM?** Cheaper, instantly updateable (edit a document vs. retrain), and traceable — we can show which document an answer came from.
- **Why vanilla JS over React?** Faster, more predictable loads on low-end rural devices; no build pipeline.
- **Why ConvNeXt over MobileNet?** Better robustness on cluttered field photos at acceptable latency.
- **How do you prevent hallucinated dosages?** Prompt forbids inventing figures; coverage label demotes ungrounded answers; remedies defer to label rates.
