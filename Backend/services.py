"""
All the 'doing work' logic lives here: talking to Groq, searching the
knowledge base, and calling the free weather/market price APIs.
main.py's routes stay thin and just call into these functions.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta

import httpx
import requests
from groq import Groq
from sqlalchemy.orm import Session

from config import GROQ_API_KEY
from database import KnowledgeChunk, WeatherCache, MarketCache

# --- Groq client setup ---

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"  # current general-purpose model on Groq


def ask_groq(question: str, language: str = "en", context: str = "", district: str | None = None) -> str:
    """
    Sends a question to Groq's LLM and returns the text response.

    `context` is optional — if we found relevant knowledge base chunks,
    we pass them in so Groq answers using that info instead of guessing.
    `language` tells the model whether to reply in English or Gujarati.
    Note: knowledge base documents are stored in English regardless of
    what language the farmer asks in, so when language="gu" we explicitly
    tell Groq to translate the reference content into Gujarati rather than
    leaving that implicit — otherwise the English reference text tends to
    pull the whole answer back into English despite the language instruction.
    `district` is optional — if the farmer mentioned a specific Gujarat
    district, we tell Groq explicitly so it tailors soil/climate/crop
    advice to that district instead of giving generic Gujarat-wide advice.
    Works alongside `context`: if both are present, the district
    instruction and the KB reference text are combined in the same prompt.
    """
    if language == "gu":
        language_instruction = (
            "Reply ONLY in Gujarati script. This applies even if the "
            "reference information below is written in English — translate "
            "any facts, figures, and terminology you use from it into "
            "Gujarati. Do not mix English sentences into your reply."
        )
    else:
        language_instruction = "Reply in English."

    district_instruction = ""
    if district:
        district_instruction = (
            f"\n\nThe farmer is asking specifically about {district.title()} "
            "district. Tailor your answer to that district's typical soil "
            "type, climate, and locally common crops where relevant, instead "
            "of giving generic Gujarat-wide advice. If the reference "
            "information below doesn't mention this district, rely on your "
            "own knowledge of its farming conditions, but keep the answer "
            "focused on it."
        )

    if context:
        system_prompt = (
            "You are Farmer AI, an assistant helping farmers in Gujarat, India. "
            f"{language_instruction} Use the following reference information to "
            "answer accurately. If the reference information doesn't fully answer "
            "the question, you may use your own knowledge, but make this clear."
            f"{district_instruction}\n\n"
            f"Reference information:\n{context}"
        )
    else:
        system_prompt = (
            "You are Farmer AI, an assistant helping farmers in Gujarat, India. "
            f"{language_instruction} Answer using your general knowledge, and keep "
            "answers practical and relevant to Gujarat's farming conditions."
            f"{district_instruction}"
        )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content


def ask_groq_disease_remedy(predictions: list[dict], language: str = "en") -> str:
    """
    Sends the leaf classifier's top_k predictions to Groq and asks for
    practical remedies. `predictions` is the output of
    ml_model.predict_top_k(): a list of {"label": str, "confidence": float}
    sorted by confidence descending.
    """
    if language == "gu":
        language_instruction = "Reply ONLY in Gujarati script."
    else:
        language_instruction = "Reply in English."

    predictions_text = "\n".join(
        f"{i + 1}. {p['label']} (confidence: {p['confidence'] * 100:.1f}%)"
        for i, p in enumerate(predictions)
    )

    system_prompt = (
        "You are Farmer AI, an assistant helping farmers in Gujarat, India "
        "diagnose plant leaf diseases. A leaf-disease image classifier has "
        "analyzed a photo of a crop leaf and produced the following ranked "
        "candidate diagnoses (most likely first):\n\n"
        f"{predictions_text}\n\n"
        f"{language_instruction} Identify the most likely disease, briefly "
        "explain the key symptoms that match, and give practical remedies "
        "(organic and chemical treatment options, plus preventive steps) "
        "suited to small-scale farming in Gujarat. If the top candidates "
        "are close in confidence, mention the next most likely option "
        "briefly as well."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is wrong with my plant and how do I treat it?"},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content


# --- Knowledge base search (simple keyword version for now) ---

# General-purpose English stopwords — common words that carry no
# topic-specific meaning on their own ("what", "the", "about", "tell").
# Used to filter query words down to the technical/topic terms that
# actually distinguish one chunk from another (e.g. "cotton", "soil",
# "bhavnagar"), both for density scoring and the heading-boost check.
# Deliberately broad and unmaintained-by-hand — anything NOT in this list
# is treated as a technical term, so new crops/pests/schemes added to the
# knowledge base automatically count as technical terms with no need to
# register them anywhere.
STOPWORDS_QUERY = {
    "tell", "all", "information", "about", "gujarat", "what", "how",
    "the", "and", "for", "are", "with", "this", "that", "from", "have",
    "does", "can", "you", "give", "please", "know", "need", "want",
    "is", "in", "of", "to", "it", "on", "at", "by", "or", "be", "as",
    "me", "my", "your", "yours", "i", "we", "us", "our", "they", "them",
    "their", "there", "here", "when", "where", "which", "who", "whom",
    "why", "will", "would", "could", "should", "shall", "may", "might",
    "must", "do", "did", "done", "had", "has", "but", "not", "no", "yes",
    "some", "any", "more", "most", "much", "many", "few", "good", "best",
    "better", "good", "also", "just", "like", "get", "got", "make",
    "made", "see", "look", "looking", "find", "explain", "describe",
    "list", "show", "say", "said", "going", "go", "want", "wants",
}

# Generic English words that describe document STRUCTURE rather than
# topic content (section labels, template boilerplate, vague descriptive
# words). Excluded from the heading boost even when they happen to be
# statistically rare in a given knowledge base's headings — rarity alone
# isn't enough, since a generic word like "options" or "use" can
# coincidentally appear in only a few headings by chance without being a
# meaningful topic signal. Used TOGETHER with the automatic rarity check
# below (a word must pass both): rarity catches knowledge-base-specific
# template noise this list can't anticipate (e.g. a recurring custom
# section title), while this list catches ordinary English words that
# rarity alone can't reliably flag.
HEADING_BOOST_NOISE_WORDS = {
    "guide", "guides", "cultivation", "manual", "complete", "section",
    "introduction", "overview", "profile", "practices", "package",
    "details", "detail", "information", "reference", "summary",
    "chapter", "part", "appendix", "annex", "document", "report",
    "module", "unit", "topic", "general", "basic", "basics",
    "options", "option", "use", "uses", "using", "management",
    "requirements", "requirement", "considerations", "consideration",
    "factors", "factor", "aspects", "aspect",
    "analysis", "method", "methods", "process", "procedure", "steps",
    "step", "notes", "note",
}


def search_knowledge_base(db: Session, query: str, district: str | None = None, limit: int = 3):
    """
    Keyword search over knowledge_chunks table, scored by match density
    rather than raw match count, so a long district crop-list chunk that
    merely mentions a crop once doesn't outscore a short, dense chunk that's
    actually ABOUT that crop. Returns a list of matching chunk texts, or an
    empty list if nothing matches or the knowledge base is empty.

    `district` is optional. When provided:
      - chunks tagged with that district get a score boost (more relevant)
      - chunks tagged with OTHER districts only (not this one) are skipped
        entirely, so e.g. a Kheda-specific chunk won't get pulled in when
        the farmer asked about Bhavnagar
      - chunks with no district tag at all (districts is NULL) are treated
        as Gujarat-wide and still considered normally

    This is intentionally basic — good enough for a handful of documents.
    Can be upgraded to vector/semantic search later without changing
    how it's called from main.py.
    """
    all_words = [w.lower() for w in query.split() if len(w) > 2]

    # Filter down to technical/topic terms — drops generic words like
    # "tell", "about", "what" so they don't dilute density scoring with
    # noise that matches almost every chunk equally. If filtering would
    # leave NOTHING (an entirely generic query with no topic words at
    # all), fall back to the unfiltered words rather than searching with
    # an empty list, since a vague query should still attempt a search.
    technical_words = [w for w in all_words if w not in STOPWORDS_QUERY]
    query_words = technical_words if technical_words else all_words

    if not query_words:
        return []

    all_chunks = db.query(KnowledgeChunk).all()
    if not all_chunks:
        return []

    # --- Pass 1: compute heading-word rarity across the WHOLE knowledge
    # base, so the heading boost below only trusts words that are
    # genuinely distinctive — not a fixed, hand-maintained noise list
    # (which can never anticipate every generic section label a large,
    # template-heavy knowledge base will contain, e.g. "Guide",
    # "Cultivation", or a lettered sub-heading like "C. Organic Nutrition
    # Options"). A word that appears in most chunks' headings carries
    # almost no information about which chunk is relevant; a word that
    # appears in only a handful of headings is a strong, trustworthy signal.
    total_chunks = len(all_chunks)
    heading_word_doc_count: dict[str, int] = {}
    chunk_headings: dict[int, str] = {}  # cache so pass 2 doesn't redo this work

    for chunk in all_chunks:
        content_lines = [
            line.strip() for line in chunk.chunk_text.strip().split("\n")
            if line.strip() and not re.match(r"^[-=_*#~]{3,}$", line.strip())
        ]
        heading_line = content_lines[0].lower() if content_lines else ""
        chunk_headings[chunk.id] = heading_line

        seen_in_this_heading = set(re.findall(r"[a-z]{3,}", heading_line))
        for word in seen_in_this_heading:
            heading_word_doc_count[word] = heading_word_doc_count.get(word, 0) + 1

    # A heading word is "rare enough to trust" if it appears in under 5%
    # of all chunk headings in the knowledge base. With ~800+ chunks this
    # comfortably excludes template boilerplate ("guide", "cultivation",
    # generic lettered section titles) while still allowing genuinely
    # topic-specific words (a crop name, a pest name) through, since those
    # only appear in the handful of chunks actually about that topic.
    RARITY_THRESHOLD = 0.05

    def is_rare_heading_word(word: str) -> bool:
        doc_count = heading_word_doc_count.get(word, 0)
        return doc_count > 0 and (doc_count / total_chunks) <= RARITY_THRESHOLD

    scored = []
    for chunk in all_chunks:
        chunk_districts = (
            [d.strip() for d in chunk.districts.split(",") if d.strip()]
            if chunk.districts else []
        )

        # Skip chunks that are tagged for specific districts that don't
        # include the one we're asking about — e.g. don't let a Kheda-only
        # chunk answer a Bhavnagar question just because words overlap.
        if chunk_districts and district and district not in chunk_districts:
            continue

        text_lower = (chunk.chunk_text + " " + (chunk.keywords or "")).lower()
        chunk_word_count = max(len(text_lower.split()), 1)
        match_count = sum(1 for word in query_words if word in text_lower)

        if match_count == 0:
            continue

        # Density score: matches per 100 words, so a short chunk that's
        # densely about the topic beats a long chunk that just mentions it
        # once in passing (e.g. a district's one-line crop list).
        score = (match_count / chunk_word_count) * 100

        # Strong boost when the chunk's own heading names the topic being
        # asked about — e.g. "4.1 COTTON (KAPAS) — Kharif, premier cash
        # crop of Gujarat" is the authoritative chunk for a cotton question,
        # well beyond what density alone would give it. Source documents
        # often wrap headings in dashed divider lines, so skip those and
        # look at the first real content line instead of just line one.
        heading_line = chunk_headings.get(chunk.id, "")

        # Only query words that pass ALL of these can trigger the heading
        # boost: not a generic question word (STOPWORDS_QUERY), not a
        # generic structural/template word (HEADING_BOOST_NOISE_WORDS),
        # AND rare across this knowledge base's actual headings. The
        # rarity check catches knowledge-base-specific template noise
        # automatically (no manual list could anticipate every recurring
        # custom section title); the noise list catches ordinary generic
        # English words that could coincidentally be statistically rare
        # in a given corpus by chance without being a real topic signal.
        meaningful_query_words = [
            w for w in query_words
            if w not in STOPWORDS_QUERY
            and w not in HEADING_BOOST_NOISE_WORDS
            and is_rare_heading_word(w)
        ]

        if meaningful_query_words:
            # A heading like "4.1 COTTON (KAPAS) — Kharif crop" is a
            # dedicated section ABOUT cotton — the matched word appears
            # within the first few words, right after any section number.
            # A heading like "MSP rates: Cotton, groundnut, paddy..." only
            # MENTIONS cotton among several other items — much weaker
            # signal that this chunk is the best answer for a cotton
            # question specifically, so it gets a smaller boost.
            heading_without_number = re.sub(r"^[\d.]+\s*", "", heading_line)
            heading_start = " ".join(heading_without_number.split()[:4])

            if any(word in heading_start for word in meaningful_query_words):
                score += 80  # dedicated section heading
            elif any(word in heading_line for word in meaningful_query_words):
                score += 15  # topic mentioned somewhere in the heading, but not the focus

        # Filename-based boost: covers files where the FILE is clearly
        # about one topic (e.g. "004_Bajra_Cultivation_Guide.txt") but
        # individual chunks inside use generic section titles that never
        # restate the topic ("LAND PREPARATION", "Zone 4") — so the
        # heading boost above can't fire even though this is exactly the
        # right file. Uses the same rarity-filtered word list as the
        # heading boost, so "guide"/"cultivation" can't trigger this via
        # the filename either. Smaller than a real heading match (+40,
        # not +80) since a filename hit is a weaker signal than the
        # chunk's own content actually naming the topic.
        if meaningful_query_words:
            filename_words = re.sub(r"[_\-./]", " ", chunk.source_filename.lower())
            filename_words = re.sub(r"\b\d+\b", " ", filename_words)  # drop leading numbers like "004"
            if any(word in filename_words for word in meaningful_query_words):
                score += 40

        # Boost chunks that are specifically tagged with the district asked
        # about, so they outrank Gujarat-wide chunks with similar word overlap.
        if district and district in chunk_districts:
            score += 20

        scored.append((score, chunk.chunk_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:limit]]


# --- Weather (Open-Meteo — free, no API key needed) ---

# Lat/lon for all 33 major Gujarat districts.
GUJARAT_DISTRICT_COORDS = {
    # --- Ahmedabad region ---
    "ahmedabad":    (23.0225, 72.5714),
    "gandhinagar":  (23.2156, 72.6369),
    "anand":        (22.5645, 72.9289),
    "kheda":        (22.7500, 72.6833),
    "mehsana":      (23.5880, 72.3693),
    "patan":        (23.8493, 72.1266),
    "sabarkantha":  (23.3714, 72.9720),
    "aravalli":     (23.5117, 73.1386),
    # --- Surat / South Gujarat ---
    "surat":        (21.1702, 72.8311),
    "tapi":         (21.1167, 73.4167),
    "navsari":      (20.9467, 72.9520),
    "valsad":       (20.5992, 72.9342),
    "dang":         (20.7500, 73.7167),
    "bharuch":      (21.7051, 72.9959),
    # --- Vadodara / Central Gujarat ---
    "vadodara":     (22.3072, 73.1812),
    "chhota udaipur": (22.3167, 74.0167),
    "dahod":        (22.8359, 74.2544),
    "panchmahals":  (22.7617, 73.6150),
    "mahisagar":    (23.1000, 73.5500),
    # --- Rajkot / Saurashtra ---
    "rajkot":       (22.3039, 70.8022),
    "jamnagar":     (22.4707, 70.0577),
    "morbi":        (22.8173, 70.8377),
    "surendranagar": (22.7272, 71.6490),
    "botad":        (22.1693, 71.6670),
    "amreli":       (21.6032, 71.2213),
    "bhavnagar":    (21.7645, 72.1519),
    # --- Junagadh / Sorath ---
    "junagadh":     (21.5222, 70.4579),
    "porbandar":    (21.6416, 69.6293),
    "gir somnath":  (20.9000, 70.3600),
    # --- Kutch / North Gujarat ---
    "kutch":        (23.7337, 69.8597),
    "banaskantha":  (24.1731, 72.4370),
    "narmada":      (21.8714, 73.4945),
    # --- Newly formed / smaller ---
    "devbhoomi dwarka": (22.3626, 69.0071),
}

# Gujarati-script spellings for each district, so users typing in Gujarati
# (language="gu") still get district detection. Keys must exactly match
# GUJARAT_DISTRICT_COORDS keys.
GUJARAT_DISTRICT_GUJARATI_NAMES = {
    "ahmedabad": "અમદાવાદ",
    "gandhinagar": "ગાંધીનગર",
    "anand": "આણંદ",
    "kheda": "ખેડા",
    "mehsana": "મહેસાણા",
    "patan": "પાટણ",
    "sabarkantha": "સાબરકાંઠા",
    "aravalli": "અરવલ્લી",
    "surat": "સુરત",
    "tapi": "તાપી",
    "navsari": "નવસારી",
    "valsad": "વલસાડ",
    "dang": "ડાંગ",
    "bharuch": "ભરૂચ",
    "vadodara": "વડોદરા",
    "chhota udaipur": "છોટા ઉદેપુર",
    "dahod": "દાહોદ",
    "panchmahals": "પંચમહાલ",
    "mahisagar": "મહીસાગર",
    "rajkot": "રાજકોટ",
    "jamnagar": "જામનગર",
    "morbi": "મોરબી",
    "surendranagar": "સુરેન્દ્રનગર",
    "botad": "બોટાદ",
    "amreli": "અમરેલી",
    "bhavnagar": "ભાવનગર",
    "junagadh": "જૂનાગઢ",
    "porbandar": "પોરબંદર",
    "gir somnath": "ગીર સોમનાથ",
    "kutch": "કચ્છ",
    "banaskantha": "બનાસકાંઠા",
    "narmada": "નર્મદા",
    "devbhoomi dwarka": "દેવભૂમિ દ્વારકા",
}

WEATHER_CACHE_MINUTES = 10  # don't re-fetch weather more often than this; aligned with frontend's WEATHER_REFRESH_MS so every dashboard auto-refresh actually pulls fresh data

# Common alternate English spellings for districts, mapping to the canonical
# key used in GUJARAT_DISTRICT_COORDS. Real-world documents and users don't
# always spell district names the same way we do (e.g. govt soil surveys
# write "Navasari"/"Valasad"/"Panchmahal", singular/extra-vowel variants),
# so this lets detection succeed without forcing every source to match our
# exact key spelling.
GUJARAT_DISTRICT_ALIASES = {
    "navasari": "navsari",
    "valasad": "valsad",
    "panchmahal": "panchmahals",
    "mahesana": "mehsana",
    "kachchh": "kutch",
    "kachh": "kutch",
    "surendra nagar": "surendranagar",
}


def detect_district(query: str) -> str | None:
    """
    Scans any user query (English or Gujarati script) for a mention of a
    known Gujarat district and returns its English key (matching
    GUJARAT_DISTRICT_COORDS), or None if no district is mentioned.

    This is intentionally generic — unlike the old weather-only district
    check, this has no keyword gating, so it works for any kind of
    farming question (soil, crops, schemes, pests, weather, etc).
    """
    query_lower = query.lower()

    for district in GUJARAT_DISTRICT_COORDS:
        if re.search(rf"\b{re.escape(district)}\b", query_lower):
            return district

    for alias, district in GUJARAT_DISTRICT_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", query_lower):
            return district

    # Gujarati script names aren't space-separated the same way English
    # words are, so a plain substring check is enough (no word-boundary regex).
    for district, gujarati_name in GUJARAT_DISTRICT_GUJARATI_NAMES.items():
        if gujarati_name in query:
            return district

    return None


def detect_all_districts(text: str) -> list[str]:
    """
    Like detect_district, but returns EVERY district mentioned in the text
    instead of just the first match. Used by ingest.py, since a single
    knowledge base chunk (e.g. a 'Saurashtra region' paragraph) often
    covers several districts at once, not just one.
    """
    text_lower = text.lower()
    found = []

    for district in GUJARAT_DISTRICT_COORDS:
        if re.search(rf"\b{re.escape(district)}\b", text_lower) and district not in found:
            found.append(district)

    for alias, district in GUJARAT_DISTRICT_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_lower) and district not in found:
            found.append(district)

    return found


def get_weather(db: Session, district: str) -> dict:
    """
    Returns current weather for a Gujarat district, using Open-Meteo.
    Checks the cache first; only calls the live API if the cached data
    is missing or older than WEATHER_CACHE_MINUTES.
    """
    district_key = district.strip().lower()

    cached = (
        db.query(WeatherCache)
        .filter(WeatherCache.district == district_key)
        .order_by(WeatherCache.fetched_at.desc())
        .first()
    )

    if cached and cached.fetched_at > datetime.utcnow() - timedelta(minutes=WEATHER_CACHE_MINUTES):
        data = json.loads(cached.data_json)
        # Guard against stale cache rows saved before rainfall fields existed.
        # If any expected key is missing, treat the cache as invalid and
        # fall through to a fresh API call instead of returning partial data.
        required_keys = {
            "temperature_c", "humidity_percent", "weather_code",
            "rainfall_now_mm", "rainfall_today_mm",
        }
        if required_keys.issubset(data.keys()):
            data["cached"] = True
            return data

    coords = GUJARAT_DISTRICT_COORDS.get(district_key)
    if not coords:
        return {"error": f"Unknown district '{district}'. Add it to GUJARAT_DISTRICT_COORDS."}

    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    result = {
        # Store the canonical lowercase key, not the user-supplied casing,
        # so cached and fresh responses use the same shape regardless of
        # whether the caller wrote "Ahmedabad", "ahmedabad", or "AHMEDABAD".
        "district": district_key,
        "temperature_c": raw["current"]["temperature_2m"],
        "humidity_percent": raw["current"]["relative_humidity_2m"],
        "weather_code": raw["current"]["weather_code"],
        "rainfall_now_mm": raw["current"]["precipitation"],
        "rainfall_today_mm": raw["daily"]["precipitation_sum"][0],
        "cached": False,
    }

    db.add(WeatherCache(district=district_key, data_json=json.dumps(result)))
    db.commit()

    return result


# Keys every cached weather row must have for the cache hit to be valid.
# Older cache rows written before these fields existed are treated as stale
# and re-fetched, instead of silently returning a partial result.
_WEATHER_REQUIRED_KEYS = {
    "temperature_c", "humidity_percent", "weather_code",
    "rainfall_now_mm", "rainfall_today_mm",
}


async def _fetch_one_weather_async(
    client: httpx.AsyncClient, district_key: str, coords: tuple[float, float]
) -> dict | None:
    """
    Fires a single Open-Meteo call for one district and returns the
    normalized record, or None if the call failed. Used by
    get_all_weather_concurrent() to fan out 33 districts in parallel.
    Errors are logged and converted to None so one slow/failed district
    doesn't sink the rest of the batch.
    """
    lat, lon = coords
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    try:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast", params=params, timeout=10
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as e:
        print(f"[weather/all] async fetch failed for '{district_key}': {e}")
        return None

    return {
        "district": district_key,
        "temperature_c": raw["current"]["temperature_2m"],
        "humidity_percent": raw["current"]["relative_humidity_2m"],
        "weather_code": raw["current"]["weather_code"],
        "rainfall_now_mm": raw["current"]["precipitation"],
        "rainfall_today_mm": raw["daily"]["precipitation_sum"][0],
        "cached": False,
    }


async def get_all_weather_concurrent(db: Session) -> list[dict]:
    """
    Returns current weather for every district in GUJARAT_DISTRICT_COORDS,
    fanning out cache-miss fetches concurrently via httpx + asyncio.gather
    instead of looping sequentially through get_weather().

    Three-phase design so SQLAlchemy's sync session stays on the main
    (sync) thread and only the slow HTTP calls actually run concurrently:
      1. Sync: read every cached row in one query, classify each district
         as fresh-cache-hit or needs-refetch.
      2. Async: fire all needs-refetch fetches concurrently and await all.
      3. Sync: write new cache rows for the successful fetches and commit
         once at the end (one round-trip instead of 33).
    """
    cutoff = datetime.utcnow() - timedelta(minutes=WEATHER_CACHE_MINUTES)

    # --- Phase 1: read existing cache rows in a single query ---
    # Pull the latest row per district by ordering newest-first then keeping
    # only the first row we see for each district key — avoids one query
    # per district, which would re-introduce the same N+1 pattern we're
    # trying to eliminate.
    cached_rows = (
        db.query(WeatherCache)
        .order_by(WeatherCache.fetched_at.desc())
        .all()
    )
    latest_per_district: dict[str, WeatherCache] = {}
    for row in cached_rows:
        latest_per_district.setdefault(row.district, row)

    results: list[dict] = []
    districts_to_fetch: list[tuple[str, tuple[float, float]]] = []

    for district_key, coords in GUJARAT_DISTRICT_COORDS.items():
        row = latest_per_district.get(district_key)
        if row and row.fetched_at > cutoff:
            try:
                data = json.loads(row.data_json)
            except Exception:
                data = {}
            if _WEATHER_REQUIRED_KEYS.issubset(data.keys()):
                data["cached"] = True
                results.append(data)
                continue
        # Cache miss / stale / schema-stale — needs a fresh fetch.
        districts_to_fetch.append((district_key, coords))

    # --- Phase 2: concurrent Open-Meteo fetches for cache misses ---
    if districts_to_fetch:
        async with httpx.AsyncClient() as client:
            fetched = await asyncio.gather(*[
                _fetch_one_weather_async(client, d_key, coords)
                for d_key, coords in districts_to_fetch
            ])

        # --- Phase 3: persist successful fetches in one commit ---
        new_rows = []
        for record in fetched:
            if record is None:
                continue
            results.append(record)
            new_rows.append(WeatherCache(
                district=record["district"],
                data_json=json.dumps(record),
            ))
        if new_rows:
            db.add_all(new_rows)
            db.commit()

    return results


# --- Market prices (data.gov.in Agmarknet — free, needs API key) ---

MARKET_API_BASE = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
MARKET_CACHE_HOURS = 6

# Categories shown as filter pills on the frontend, with the major Gujarat
# crops in each — used to power a "browse by category" market price screen
# without the farmer having to type an exact commodity name.
CROP_CATEGORIES = {
    "all": {
        "label": "All",
        "icon": "🌍",
        # No fixed crop list — this category queries data.gov.in with only
        # the Gujarat state filter, so it returns whatever's actually being
        # traded today, including crops not in any of the lists below.
        "crops": [],
    },
    "cash_crops": {
        "label": "Cash crops",
        "icon": "🌾",
        "crops": ["Cotton"],
    },
    "oilseeds": {
        "label": "Oilseeds",
        "icon": "💧",
        "crops": ["Groundnut", "Castor Seed", "Sesamum(Sesame,Til)", "Mustard", "Sunflower", "Soyabean", "Cotton Seed"],
    },
    "grains_cereals": {
        "label": "Grains & cereals",
        "icon": "✦",
        "crops": ["Wheat", "Bajra", "Jowar", "Maize", "Paddy(Dhan)(Common)", "Rice"],
    },
    "pulses": {
        "label": "Pulses",
        "icon": "⊙",
        "crops": ["Arhar (Tur/Red Gram)(Whole)", "Bengal Gram(Gram)(Whole)", "Green Gram (Moong)(Whole)", "Black Gram (Urd Beans)(Whole)", "Masur Dal"],
    },
    "spices": {
        "label": "Spices",
        "icon": "🌶",
        "crops": ["Cumin(Jeera)", "Fennel", "Coriander(Leaves)", "Ajwan"],
    },
    "fruits": {
        "label": "Fruits",
        "icon": "🍊",
        "crops": ["Banana", "Mango"],
    },
    "vegetables": {
        "label": "Vegetables",
        "icon": "🥬",
        "crops": ["Potato", "Onion", "Tomato"],
    },
    "others": {
        "label": "Others",
        "icon": "✦",
        "crops": ["Guar Seed(Cluster Beans Seed)", "Isabgul (Psyllium)", "Castor Oil"],
    },
}

# The 9 fields the data.gov.in resource returns per record, in display order.
MARKET_RECORD_FIELDS = [
    "state", "district", "market", "commodity", "variety",
    "grade", "arrival_date", "min_price", "max_price", "modal_price",
]


def _normalize_record(rec: dict) -> dict:
    """Pulls out exactly the 9 documented fields from a raw API record,
    so the frontend always gets a consistent shape regardless of any
    extra fields data.gov.in includes."""
    return {field: rec.get(field) for field in MARKET_RECORD_FIELDS}


def get_market_price(
    db: Session,
    commodity: str,
    api_key: str,
    market: str = None,
    district: str = None,
    limit: int = 20,
) -> dict:
    """
    Returns mandi price records for a commodity in Gujarat, using the
    data.gov.in Agmarknet dataset (resource 9ef84268-d588-465a-a308-a864a43d0070).
    Checks cache first; only calls the live API if the cache is missing or stale.

    Returns up to `limit` records (all 9 fields each: state, district, market,
    commodity, variety, grade, arrival_date, min_price, max_price, modal_price),
    not just the single cheapest/top one — so the frontend can show a full
    table of mandis instead of one row.
    """
    cache_key_commodity = commodity.lower()
    # district is folded into the same market-cache key field (no separate
    # column exists) — without this, a cache hit for one district's query
    # would be silently returned for a different district asked about
    # within the same MARKET_CACHE_HOURS window.
    cache_key_market = f"{(market or '').lower()}|{(district or '').lower()}"

    cached = (
        db.query(MarketCache)
        .filter(MarketCache.commodity == cache_key_commodity)
        .filter(MarketCache.market == cache_key_market)
        .order_by(MarketCache.fetched_at.desc())
        .first()
    )

    if cached and cached.fetched_at > datetime.utcnow() - timedelta(hours=MARKET_CACHE_HOURS):
        data = json.loads(cached.data_json)
        data["cached"] = True
        return data

    if not api_key:
        return {"error": "Market price API key is not configured. Add MARKET_API_KEY to your .env file."}

    params = {
        "api-key": api_key,
        "format": "json",
        "filters[state.keyword]": "Gujarat",
        "filters[commodity]": commodity,
        "limit": limit,
    }
    if market:
        params["filters[market]"] = market
    if district:
        params["filters[district]"] = district

    # data.gov.in silently stalls (no response, no error — just hangs until
    # timeout) on requests carrying the default "python-requests/x.x" User-
    # Agent string. A browser-style UA gets an instant response. Confirmed
    # via diagnostics: raw sockets and a Mozilla UA both return in <1s;
    # requests/httpx with their default UA time out 100% of the time.
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FarmerAI/1.0"}

    response = requests.get(MARKET_API_BASE, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    raw = response.json()

    records = raw.get("records", [])
    if not records:
        return {"error": f"No price data found for '{commodity}' in Gujarat."}

    normalized = [_normalize_record(r) for r in records]

    result = {
        "commodity": commodity,
        "count": len(normalized),
        "records": normalized,
        "cached": False,
    }

    db.add(MarketCache(
        commodity=cache_key_commodity,
        market=cache_key_market,
        data_json=json.dumps(result),
    ))
    db.commit()

    return result


def get_all_market_prices(db: Session, api_key: str, district: str = None, limit: int = 100) -> dict:
    """
    Returns mandi price records for EVERY commodity currently traded in
    Gujarat today — a true union of every crop across all of CROP_CATEGORIES
    (each looked up via get_market_price(), so it benefits from the same
    per-crop cache), PLUS a generic state-only discovery query to also pick
    up any crop currently trading that isn't in any of our category lists.

    Previously this only ran the generic discovery query and skipped the
    category crop lists entirely — since data.gov.in doesn't guarantee any
    particular ordering, that meant "All" often came back as a handful of
    records from a single mandi instead of actually covering cash crops,
    oilseeds, grains, etc. like the category pills suggest it would.

    Records are deduplicated by (commodity, market, variety, arrival_date),
    since the same record can otherwise show up once via its category fetch
    and again via the generic discovery query.
    """
    cache_key_commodity = "__all__"
    # district folded into the market-cache key field, same as
    # get_market_price() — otherwise this cache entry would be reused
    # across different district filters.
    cache_key_market = f"|{(district or '').lower()}"

    cached = (
        db.query(MarketCache)
        .filter(MarketCache.commodity == cache_key_commodity)
        .filter(MarketCache.market == cache_key_market)
        .order_by(MarketCache.fetched_at.desc())
        .first()
    )

    if cached and cached.fetched_at > datetime.utcnow() - timedelta(hours=MARKET_CACHE_HOURS):
        data = json.loads(cached.data_json)
        data["cached"] = True
        return data

    if not api_key:
        return {"error": "Market price API key is not configured. Add MARKET_API_KEY to your .env file."}

    all_records = []
    seen_keys = set()

    def add_record(rec: dict):
        key = (rec.get("commodity"), rec.get("market"), rec.get("variety"), rec.get("arrival_date"))
        if key in seen_keys:
            return
        seen_keys.add(key)
        all_records.append(rec)

    # 1. Union of every crop in every real category (each cached individually).
    for category_key, category in CROP_CATEGORIES.items():
        if category_key == "all":
            continue
        for crop_name in category["crops"]:
            try:
                crop_result = get_market_price(db, crop_name, api_key, district=district, limit=10)
            except Exception as e:
                print(f"[market/all] category-crop fetch failed for '{crop_name}': {e}")
                continue
            for rec in crop_result.get("records", []):
                add_record(rec)

    # 2. Generic discovery query, to also catch crops outside our category
    # lists — same approach the old implementation used on its own.
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[state.keyword]": "Gujarat",
        "limit": limit,
    }
    if district:
        params["filters[district]"] = district

    # Same fix as get_market_price(): data.gov.in stalls indefinitely on
    # the default python-requests User-Agent, so a browser-style one is
    # required for a normal response time.
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FarmerAI/1.0"}

    try:
        response = requests.get(MARKET_API_BASE, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        raw = response.json()
        for r in raw.get("records", []):
            add_record(_normalize_record(r))
    except Exception as e:
        print(f"[market/all] generic discovery fetch failed: {e}")

    if not all_records:
        return {"error": "No price data found for Gujarat today."}

    result = {
        "category": "all",
        "label": "All",
        "count": len(all_records),
        "records": all_records,
        "cached": False,
    }

    db.add(MarketCache(
        commodity=cache_key_commodity,
        market=cache_key_market,
        data_json=json.dumps(result),
    ))
    db.commit()

    return result


def get_market_prices_for_category(db: Session, category_key: str, api_key: str, district: str = None) -> dict:
    """
    Fetches market prices for every crop in a given category (e.g. 'oilseeds'),
    one commodity at a time, and merges the results into a single response.
    Crops with no data that day are skipped rather than failing the whole call.
    """
    category = CROP_CATEGORIES.get(category_key)
    if not category:
        return {"error": f"Unknown category '{category_key}'."}

    all_records = []
    for crop_name in category["crops"]:
        try:
            crop_result = get_market_price(db, crop_name, api_key, district=district, limit=10)
            if "records" in crop_result:
                all_records.extend(crop_result["records"])
        except Exception as e:
            print(f"[market] category fetch failed for '{crop_name}': {e}")
            continue

    return {
        "category": category_key,
        "label": category["label"],
        "count": len(all_records),
        "records": all_records,
    }


# --- Standalone tests: run `python services.py` to check each piece ---

if __name__ == "__main__":
    from database import SessionLocal

    db = SessionLocal()

    print("\n--- Testing ask_groq() ---")
    try:
        answer = ask_groq("What is the best season to grow cotton?", language="en")
        print("Groq response:", answer[:200], "...")
    except Exception as e:
        print("Groq test FAILED:", e)

    print("\n--- Testing get_weather() ---")
    try:
        weather = get_weather(db, "Ahmedabad")
        print("Weather response:", weather)
    except Exception as e:
        print("Weather test FAILED:", e)

    print("\n--- Testing search_knowledge_base() (expect empty list, nothing ingested yet) ---")
    try:
        results = search_knowledge_base(db, "cotton fertilizer")
        print("Knowledge base results:", results)
    except Exception as e:
        print("Knowledge base test FAILED:", e)

    print("\n--- Testing get_market_price() ---")
    try:
        from config import MARKET_API_KEY
        if not MARKET_API_KEY:
            print("Skipped: MARKET_API_KEY not set in .env.")
        else:
            market = get_market_price(db, "Cotton", api_key=MARKET_API_KEY)
            print("Market response:", market)
    except Exception as e:
        print("Market test FAILED:", e)

    db.close()