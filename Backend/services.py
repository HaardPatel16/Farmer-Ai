"""
All the 'doing work' logic lives here: talking to Groq, searching the
knowledge base, and calling the free weather/market price APIs.
main.py's routes stay thin and just call into these functions.
"""

import asyncio
import json
import re
import threading
from datetime import datetime, timedelta, timezone, date as date_cls

import httpx
import requests
from groq import Groq
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import GROQ_API_KEY
from database import KnowledgeChunk, WeatherCache, MarketPriceSnapshot, utcnow_naive

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

    # Compact scope guard (~30 tokens vs the ~150-token version we had
    # before). The PRIMARY off-topic block now happens in main.py BEFORE
    # we call Groq at all (see is_farming_question / OFFTOPIC_REFUSAL),
    # so by the time we get here the query has already passed a keyword
    # whitelist. This in-prompt line is just a backstop for edge cases
    # where the local filter let something through.
    scope_instruction = (
        "Stay strictly on farming topics. If asked about anything else, "
        "politely redirect to farming questions in one short sentence."
    )

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
            f"{scope_instruction} "
            f"{language_instruction} Use the following reference information to "
            "answer accurately. If the reference information doesn't fully answer "
            "the question, you may use your own knowledge, but make this clear."
            f"{district_instruction}\n\n"
            f"Reference information:\n{context}"
        )
    else:
        system_prompt = (
            "You are Farmer AI, an assistant helping farmers in Gujarat, India. "
            f"{scope_instruction} "
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
        max_tokens=700,
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
        max_tokens=700,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Local off-topic filter — runs BEFORE we call Groq, so genuinely off-
# topic queries (math, sports, jokes, trivia) cost zero Groq tokens. The
# system prompt still carries a backstop scope instruction in case a
# query slips through the whitelist, but in the common case Groq is
# never invoked for non-farming questions at all.
# ---------------------------------------------------------------------------

# Hardcoded refusal text — sent verbatim when is_farming_question() returns
# False. No Groq involvement, so cost is exactly 0 tokens.
OFFTOPIC_REFUSAL_EN = (
    "I'm Farmer AI — I can only help with farming-related questions "
    "(crops, livestock, soil, weather, pest control, agricultural schemes, "
    "and so on). Is there something agricultural I can help you with?"
)
OFFTOPIC_REFUSAL_GU = (
    "હું Farmer AI છું — હું માત્ર ખેતી સંબંધિત પ્રશ્નોમાં મદદ કરી શકું છું "
    "(પાક, પશુપાલન, માટી, હવામાન, જંતુ-નિયંત્રણ, કૃષિ યોજનાઓ વગેરે). "
    "શું હું તમને કોઈ ખેતી સંબંધિત બાબતમાં મદદ કરી શકું?"
)

# Vocabulary that signals a farming/agriculture question. Hand-curated;
# generous on inclusion — false positives (calling Groq for a borderline
# query) are much cheaper than false negatives (refusing a real farmer).
# Anything mentioning a Gujarat district or a Gujarati farming term is
# also caught by the script/alias loops below.
_FARMING_KEYWORDS = {
    # Crops / produce
    "crop", "crops", "cotton", "wheat", "bajra", "bajri", "rice", "paddy",
    "groundnut", "peanut", "castor", "cumin", "fennel", "coriander",
    "turmeric", "ginger", "garlic", "onion", "potato", "tomato", "brinjal",
    "okra", "cabbage", "cauliflower", "chilli", "chili", "pepper",
    "mango", "banana", "papaya", "guava", "lemon", "orange", "grape",
    "pomegranate", "watermelon", "muskmelon", "sapota", "amla", "ber",
    "sugarcane", "maize", "corn", "jowar", "barley", "millet", "millets",
    "pulse", "pulses", "lentil", "moong", "urd", "chana", "gram", "tur",
    "arhar", "soybean", "soyabean", "mustard", "sesame", "sunflower",
    "safflower", "tobacco", "betel", "coconut", "cashew", "areca", "rubber",
    "marigold", "rose", "jasmine", "gerbera", "carnation", "tuberose",
    "isabgul", "psyllium", "guar", "matki",
    # Livestock / dairy / fisheries
    "cow", "cows", "buffalo", "buffaloes", "goat", "sheep", "poultry",
    "chicken", "hen", "cattle", "livestock", "dairy", "milk", "butter",
    "ghee", "honey", "bee", "bees", "beekeeping", "apiculture", "fish",
    "fisheries", "shrimp", "fodder", "silage",
    # Farming activities / inputs
    "farm", "farms", "farmer", "farmers", "farming", "agriculture",
    "agricultural", "agri", "cultivation", "cultivate", "sowing", "sow",
    "harvest", "harvesting", "yield", "yields", "soil", "soils", "fertilizer",
    "fertiliser", "fertilizers", "manure", "compost", "vermicompost",
    "irrigation", "irrigate", "drip", "sprinkler", "pesticide", "pesticides",
    "insecticide", "fungicide", "herbicide", "pest", "pests", "disease",
    "diseases", "weed", "weeds", "bollworm", "aphid", "termite", "rust",
    "blight", "wilt", "mildew", "rot", "spot", "leaf", "leaves", "stem",
    "root", "seed", "seeds", "seedling", "sapling", "plant", "plants",
    "agronomy", "organic", "biofertilizer", "biopesticide", "biocontrol",
    "trichoderma", "pseudomonas", "neem", "panchagavya", "jeevamrut",
    "ipm", "intercropping", "rotation", "mulch", "mulching",
    # Climate / season
    "rain", "rainfall", "monsoon", "drought", "flood", "temperature",
    "humidity", "weather", "climate", "season", "kharif", "rabi", "zaid",
    "summer", "winter",
    # Soil / land
    "land", "field", "fields", "hectare", "hectares", "acre", "acres",
    "alluvial", "vertisol", "loam", "clay", "sandy", "saline", "alkaline",
    "acidic",
    # Schemes / finance / market
    "scheme", "schemes", "subsidy", "subsidies", "kisan", "pmkisan",
    "pmfby", "kcc", "msp", "mandi", "apmc", "fpo", "cooperative", "amul",
    "nabard", "ikhedut", "loan", "credit", "insurance", "premium",
    "procurement", "export", "import",
    # Geography (region-level; districts handled separately)
    "gujarat", "saurashtra", "kutch", "kachchh",
}

# Very-short whitelists for greetings and "who are you?" — these still
# go through Groq (so the LLM can introduce itself naturally), but the
# off-topic filter knows not to refuse them outright.
_GREETING_PATTERNS = (
    "hi", "hello", "hey", "namaste", "namaskar", "good morning",
    "good evening", "who are you", "what can you do", "what do you do",
    "help",
)


def _contains_gujarati_script(text: str) -> bool:
    """True if any character is in the Gujarati Unicode block (U+0A80-U+0AFF).
    A query typed in Gujarati on a Gujarat farming bot is almost certainly
    farming-related — including all greetings and casual phrasing the
    keyword whitelist can't cover. Accepting all Gujarati-script queries
    is a much better trade-off than building a Gujarati-vocabulary list."""
    return any(0x0A80 <= ord(ch) <= 0x0AFF for ch in text)


def is_farming_question(query: str) -> bool:
    """
    Returns True if the query looks farming-related (or is a greeting we
    should route through Groq normally). False means we can short-circuit
    with a canned refusal and skip Groq entirely.

    Conservative by design — false positives (calling Groq on a borderline
    query) are fine; false negatives (refusing a real farmer) are not.
    Any of these is sufficient:
      - the query contains Gujarati script
      - it matches an English greeting pattern
      - it contains an English farming keyword
      - it mentions a Gujarat district (English name, alias, or
        Gujarati-script name)
    """
    if not query:
        return False

    # Any Gujarati-script characters → treat as on-topic. The user is
    # interacting in their language with a Gujarat farming assistant —
    # very unlikely to be off-topic, and the keyword whitelist alone
    # can't cover Gujarati phrasing reliably.
    if _contains_gujarati_script(query):
        return True

    q_lower = query.lower()

    # English greetings → still go to Groq for a natural intro response.
    # We accept exact matches and prefix matches only ("hi there", "hello!")
    # — the previous version also did a bare `" " + g + " " in text` check
    # which substring-matched anywhere in the query and let arbitrary
    # non-farming text through to Groq just because it contained " hi ".
    q_stripped = q_lower.strip().rstrip("?.!,")
    if q_stripped in _GREETING_PATTERNS:
        return True
    if any(q_stripped.startswith(g + " ") for g in _GREETING_PATTERNS):
        return True

    # English farming vocabulary match.
    if any(kw in q_lower for kw in _FARMING_KEYWORDS):
        return True

    # Mentions any Gujarat district by canonical name or alias.
    if any(d in q_lower for d in GUJARAT_DISTRICT_COORDS):
        return True
    if any(a in q_lower for a in GUJARAT_DISTRICT_ALIASES):
        return True

    return False


def offtopic_refusal(language: str = "en") -> str:
    """Hardcoded refusal string used when is_farming_question() returns
    False. No Groq call, so this costs exactly 0 tokens."""
    return OFFTOPIC_REFUSAL_GU if language == "gu" else OFFTOPIC_REFUSAL_EN


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
    "does", "can", "you", "give", "please", "know", "need", "want", "wants",
    "is", "in", "of", "to", "it", "on", "at", "by", "or", "be", "as",
    "me", "my", "your", "yours", "i", "we", "us", "our", "they", "them",
    "their", "there", "here", "when", "where", "which", "who", "whom",
    "why", "will", "would", "could", "should", "shall", "may", "might",
    "must", "do", "did", "done", "had", "has", "but", "not", "no", "yes",
    "some", "any", "more", "most", "much", "many", "few", "good", "best",
    "better", "also", "just", "like", "get", "got", "make",
    "made", "see", "look", "looking", "find", "explain", "describe",
    "list", "show", "say", "said", "going", "go",
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

    Note: this returns chunks regardless of how strong the match is. The
    /chat route in main.py then runs kb_answer_coverage() post-hoc against
    the LLM's answer to decide whether to label the response as KB-grounded,
    mixed, or pure llm_reasoning. So a weak chunk being returned here
    doesn't automatically mean the answer gets the KB badge.
    """
    # Strip trailing/leading punctuation from each token so "PM-KISAN?",
    # "vermicompost?", or "groundnut," still match content that uses the
    # plain word — otherwise a single stray "?" silently kills the search
    # for any topic asked as a question (which is most of them).
    all_words = [
        w.strip(".,!?:;\"'()[]{}<>").lower()
        for w in query.split()
    ]
    all_words = [w for w in all_words if len(w) > 2]

    # Filter down to technical/topic terms — drops generic words like
    # "tell", "about", "what" so they don't dilute density scoring with
    # noise that matches almost every chunk equally. If filtering would
    # leave NOTHING (an entirely generic query with no topic words at
    # all), fall back to the unfiltered words rather than searching with
    # an empty list, since a vague query should still attempt a search.
    technical_words = [w for w in all_words if w not in STOPWORDS_QUERY]
    # Without at least one technical/topic word after stopword filtering,
    # the query is almost certainly a greeting ("hi", "namaste") or pure
    # chit-chat. Forcing a search via the unfiltered word list (the old
    # fallback) just produced stray false-positive chunks — e.g. "hi"
    # substring-matching "this" inside random text — so we now bail early
    # so the caller's post-hoc coverage check routes the response to
    # llm_reasoning instead of a false KB hit.
    if not technical_words:
        return []
    query_words = technical_words

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

    # `meaningful_query_words` — used for the HEADING boost. Requires the
    # word to be rare across the corpus's headings, since otherwise generic
    # terms light up the heading boost on unrelated chunks.
    meaningful_query_words = [
        w for w in query_words
        if w not in STOPWORDS_QUERY
        and w not in HEADING_BOOST_NOISE_WORDS
        and is_rare_heading_word(w)
    ]

    # `filename_match_words` — used for the FILENAME boost / force-include.
    # We deliberately DROP the heading-rarity requirement here: a topic
    # like "rodent" or "marigold" can be missing from every section
    # heading inside its dedicated file (because each chunk's heading is
    # generic like "1. INTRODUCTION", "2. KEY CONCEPTS"), making
    # is_rare_heading_word() return False — but the filename still
    # clearly names the topic. Only stopword + structural-noise filtering
    # applied here so we don't filename-match on "the"/"guide"/etc.
    filename_match_words = [
        w for w in query_words
        if w not in STOPWORDS_QUERY
        and w not in HEADING_BOOST_NOISE_WORDS
    ]

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

        # Does this chunk's filename name the topic? If so, we want it
        # in the candidate set even when its body section doesn't repeat
        # the topic word — that's the whole point of having dedicated
        # files per crop/scheme/topic. Critical for cases like
        # "409_Marigold_Cultivation_Gujarat.txt" where most chunks are
        # section bodies ("Planting", "Storage") that never repeat
        # "marigold" — under match_count==0 they'd be silently skipped.
        filename_match = False
        if filename_match_words:
            fn_tokens = re.sub(r"[_\-./]", " ", chunk.source_filename.lower())
            fn_tokens = re.sub(r"\b\d+\b", " ", fn_tokens)
            if any(w in fn_tokens for w in filename_match_words):
                filename_match = True

        if match_count == 0 and not filename_match:
            continue

        # Density score: matches per 100 words, so a short chunk that's
        # densely about the topic beats a long chunk that just mentions it
        # once in passing (e.g. a district's one-line crop list).
        # (Zero when match_count==0; relies on the filename/heading boosts
        # below to give filename-matched chunks a meaningful score.)
        score = (match_count / chunk_word_count) * 100

        # Strong boost when the chunk's own heading names the topic being
        # asked about — e.g. "4.1 COTTON (KAPAS) — Kharif, premier cash
        # crop of Gujarat" is the authoritative chunk for a cotton question,
        # well beyond what density alone would give it. Source documents
        # often wrap headings in dashed divider lines, so skip those and
        # look at the first real content line instead of just line one.
        heading_line = chunk_headings.get(chunk.id, "")

        # `meaningful_query_words` was hoisted out of this loop (computed
        # once above) — it's the rarity-filtered subset of query_words
        # used by both the heading boost (below) and the filename match
        # check (earlier in this iteration).
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
        # the filename either.
        # Boost is +100 (above even the +80 dedicated-heading boost)
        # because the PDF-test analysis showed that filename-matched
        # chunks are the single most reliable indicator of the right
        # file — and chunks inside the right file often score weakly
        # on density because the topic word appears only in the
        # filename + intro line, not throughout every chunk. Without
        # this strong boost the long banana-cultivation guide's
        # "Land Preparation" sub-chunk loses to a short paragraph in
        # another file that happens to mention "banana" twice in 50
        # words, and the right answer never surfaces.
        if filename_match:
            score += 100

        # Boost chunks that are specifically tagged with the district asked
        # about, so they outrank Gujarat-wide chunks with similar word overlap.
        if district and district in chunk_districts:
            score += 20

        # Track filename alongside score+text so the force-include pass
        # below can pull the top chunk per "expected" filename without
        # re-scanning the DB.
        scored.append((score, chunk.chunk_text, chunk.source_filename))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Absolute-junk floor: 1.0 means "the matched word makes up at least
    # 1% of the chunk's vocabulary" — anything less is essentially a
    # stray substring hit. Deliberately permissive because the real
    # "did Groq actually use these chunks?" verification happens post-hoc
    # in main.py via kb_answer_coverage(). Setting this floor higher
    # silently dropped legitimate hits where the topic word is hyphenated
    # (e.g. "PM-KISAN") and thus skips the rarity-based heading/filename
    # boosts that normally lift real hits to 80+.
    MIN_TOP_SCORE = 1.0
    scored = [(s, t, fn) for (s, t, fn) in scored if s >= MIN_TOP_SCORE]

    # ── Filename force-include pass ──────────────────────────────────
    # Goal: if the user asks about Trichoderma and we have a file named
    # "182_Trichoderma_Application_Guide.txt", that file's top chunk
    # must appear in the results — even if other files coincidentally
    # mention "trichoderma" more densely in passing.
    #
    # The +100 filename boost above already pushes such chunks high in
    # the ranking, but it's not a HARD guarantee — a very short heavily-
    # boosted chunk from a different file could still outrank a long
    # chunk from the right file with weak per-chunk density. This pass
    # adds the hard guarantee on top.
    #
    # Algorithm: find every distinct filename whose name matches a
    # meaningful query word, take that file's top-scoring chunk, and
    # ensure at least one chunk per matched file appears in the final
    # result (within the `limit`).
    final: list[tuple[float, str]] = []
    used_filenames: set[str] = set()

    if filename_match_words:
        # Build the set of "expected" filenames (those whose tokenized
        # filename contains at least one non-stopword query word).
        expected_filenames: set[str] = set()
        for _s, _t, fn in scored:
            fn_tokens = re.sub(r"[_\-./]", " ", fn.lower())
            fn_tokens = re.sub(r"\b\d+\b", " ", fn_tokens)
            if any(w in fn_tokens for w in filename_match_words):
                expected_filenames.add(fn)

        # First pass: pick the top-scoring chunk for each expected
        # filename (in score order, so the strongest-matching expected
        # file gets included before weaker ones if limit is tight).
        for s, t, fn in scored:
            if fn in expected_filenames and fn not in used_filenames:
                final.append((s, t))
                used_filenames.add(fn)
                if len(final) >= limit:
                    break

    # Second pass: fill remaining slots from the natural top-K ranking,
    # skipping any file we already represented above.
    for s, t, fn in scored:
        if len(final) >= limit:
            break
        if fn in used_filenames:
            continue
        final.append((s, t))
        used_filenames.add(fn)

    final_texts = [text for _, text in final]

    # ── Semantic-search augmentation (optional) ─────────────────────
    # Run semantic search in parallel with the keyword path and merge
    # results. This catches cases the keyword scorer misses due to
    # synonymy — e.g. "cold storage for mango" → the file is named
    # "Cold_Chain_Management", "chain" never appears in the query, so
    # keyword force-include doesn't fire. Embedding similarity does.
    #
    # Graceful: if sentence-transformers isn't installed or model load
    # fails, embeddings.semantic_search returns None and the keyword
    # results stand alone.
    try:
        from embeddings import semantic_search
        sem_results = semantic_search(db, query, top_k=limit, district=district)
    except Exception as e:
        print(f"[search_kb] semantic_search failed (falling back to keyword-only): {e}")
        sem_results = None

    if sem_results:
        # Merge: keep keyword results first (they have the district +
        # filename heuristics baked in), then append any semantic
        # results not already present. Cap at limit*2 — with limit=3
        # this gives Groq up to 6 chunks total (3 keyword + up to 3
        # unique semantic), keeping prompt size tight while still
        # letting semantic search contribute when it finds something
        # keyword missed.
        existing = set(final_texts)
        merged = list(final_texts)
        for r in sem_results:
            if r["text"] not in existing:
                merged.append(r["text"])
                existing.add(r["text"])
            if len(merged) >= limit * 2:
                break
        return merged

    return final_texts


# ---------------------------------------------------------------------------
# Source-of-truth verification for the /chat source_type label.
# ---------------------------------------------------------------------------
# The keyword scorer above is liberal — for queries like "tell me a joke
# about a tractor" it will surface the tractor-maintenance guide because
# the keyword match is technically there. Groq, given those irrelevant
# chunks, sees they don't fit the question and writes a joke from its own
# training data anyway. The /chat route used to label that response
# "Knowledge Base" because chunks were retrieved, even though none of the
# answer actually came from the KB. This post-hoc check fixes that by
# comparing the answer's content words to the chunks' content words: if
# the overlap is too low, the answer was effectively LLM-generated and
# gets labeled accordingly.

# Words this short are too noisy (mostly stopwords, conjunctions, etc.)
# to be useful for content overlap. 4+ char threshold keeps "soil",
# "crop", "wheat", "drip", "pest" while dropping "and", "the", "for".
_MIN_CONTENT_WORD_LEN = 4


def _content_words(text: str) -> set[str]:
    """Extract distinct lowercase content words (4+ chars, alpha-only)
    from arbitrary text. Used by kb_answer_coverage() to compute
    answer-vs-chunks overlap for source_type classification."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= _MIN_CONTENT_WORD_LEN}


# Phrases the LLM uses when it openly admits it's filling gaps with
# general knowledge instead of (or alongside) the retrieved chunks.
# Detecting these is a stronger "this is mixed" signal than coverage
# alone, because the LLM can have a high content-word overlap with the
# chunks while still being upfront that the chunks didn't fully answer
# the question. main.py demotes "knowledge_base" -> "mixed" when any of
# these phrases appears.
_LLM_HEDGE_PHRASES = (
    "based on my general knowledge",
    "based on general knowledge",
    "from my general knowledge",
    "the reference information does not",
    "the reference information doesn't",
    "the reference information provided does not",
    "the reference information provided doesn't",
    "the provided reference information does not",
    "the provided reference information doesn't",
    "the reference does not mention",
    "the reference doesn't mention",
    "doesn't specifically mention",
    "does not specifically mention",
    "not specifically mentioned",
    "however, i can provide",
    "however, based on my",
    "however, based on general",
    "i don't have specific",
    "i do not have specific",
)


def answer_has_llm_hedge(answer: str) -> bool:
    """True if the answer text contains one of the canonical phrases the
    LLM uses to signal it's drawing on general training data rather than
    (or in addition to) the retrieved chunks. Cheap substring check."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _LLM_HEDGE_PHRASES)


def kb_answer_coverage(answer: str, chunks: list[str]) -> float:
    """
    Returns the fraction (0.0-1.0) of the answer's distinct content
    words that also appear in the retrieved chunks. This is a measure
    of how much of the LLM's answer is actually grounded in the KB vs
    invented from its own general training.

    main.py thresholds this value into three source-type tiers:
      - < 0.20  -> answer is mostly LLM general knowledge
                   (tractor-joke vs tractor-maintenance guide: ~10-15%)
      - 0.20-0.70 -> mixed: KB context helped but LLM filled gaps
                   (Trichoderma-application answer that pulls some facts
                    from the guide but adds its own framing)
      - >= 0.70  -> answer is largely paraphrasing the chunks
                   (PM-KISAN definition, district profile lookups: ~60-90%)
    """
    if not chunks:
        return 0.0
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    chunk_words = _content_words(" ".join(chunks))
    return len(answer_words & chunk_words) / len(answer_words)


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


# Keys every cached weather row must have for the cache hit to be valid.
# Older cache rows written before these fields existed are treated as stale
# and re-fetched, instead of silently returning a partial result. Used by
# both get_weather() (single district) and get_all_weather_concurrent()
# (batch endpoint) so the schema-guard logic stays consistent.
# Includes the tomorrow-forecast fields (added for "will it rain tomorrow?"
# style questions) — any row missing them is auto-refetched.
_WEATHER_REQUIRED_KEYS = {
    "temperature_c", "humidity_percent", "weather_code",
    "rainfall_now_mm", "rainfall_today_mm",
    "rainfall_tomorrow_mm", "temp_max_tomorrow_c", "temp_min_tomorrow_c",
}


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

    if cached and cached.fetched_at > utcnow_naive() - timedelta(minutes=WEATHER_CACHE_MINUTES):
        data = json.loads(cached.data_json)
        # Guard against stale cache rows saved before rainfall fields existed:
        # if any expected key is missing, treat the cache as invalid and
        # fall through to a fresh API call instead of returning partial data.
        # Uses the same shape-check set as get_all_weather_concurrent() so
        # both code paths stay in lockstep when the schema evolves.
        if _WEATHER_REQUIRED_KEYS.issubset(data.keys()):
            data["cached"] = True
            return data

    coords = GUJARAT_DISTRICT_COORDS.get(district_key)
    if not coords:
        return {"error": f"Unknown district '{district}'. Add it to GUJARAT_DISTRICT_COORDS."}

    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    # `forecast_days=2` returns today + tomorrow in the daily arrays at
    # indices [0] and [1]. Adding temp_max/min and precipitation_sum to
    # the daily fields lets us answer "will it rain tomorrow?" style
    # questions with a real predicted value instead of admitting we
    # only have current readings.
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "forecast_days": 2,
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    daily = raw["daily"]
    result = {
        # Store the canonical lowercase key, not the user-supplied casing,
        # so cached and fresh responses use the same shape regardless of
        # whether the caller wrote "Ahmedabad", "ahmedabad", or "AHMEDABAD".
        "district": district_key,
        "temperature_c": raw["current"]["temperature_2m"],
        "humidity_percent": raw["current"]["relative_humidity_2m"],
        "weather_code": raw["current"]["weather_code"],
        "rainfall_now_mm": raw["current"]["precipitation"],
        "rainfall_today_mm": daily["precipitation_sum"][0],
        # Tomorrow's forecast (index [1] of the 2-day daily array).
        "rainfall_tomorrow_mm": daily["precipitation_sum"][1],
        "temp_max_tomorrow_c": daily["temperature_2m_max"][1],
        "temp_min_tomorrow_c": daily["temperature_2m_min"][1],
        "cached": False,
    }

    db.add(WeatherCache(district=district_key, data_json=json.dumps(result)))
    # Same retention as the batch path: anything older than the cache
    # TTL is dead weight, so prune in the same commit as the insert.
    db.query(WeatherCache).filter(
        WeatherCache.fetched_at < utcnow_naive() - timedelta(minutes=WEATHER_CACHE_MINUTES)
    ).delete(synchronize_session=False)
    db.commit()

    return result


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
    # Mirrors get_weather()'s params exactly so single-district and batch
    # paths produce the same response shape — same daily fields, same
    # forecast_days=2 covering today + tomorrow.
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "forecast_days": 2,
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

    daily = raw["daily"]
    return {
        "district": district_key,
        "temperature_c": raw["current"]["temperature_2m"],
        "humidity_percent": raw["current"]["relative_humidity_2m"],
        "weather_code": raw["current"]["weather_code"],
        "rainfall_now_mm": raw["current"]["precipitation"],
        "rainfall_today_mm": daily["precipitation_sum"][0],
        "rainfall_tomorrow_mm": daily["precipitation_sum"][1],
        "temp_max_tomorrow_c": daily["temperature_2m_max"][1],
        "temp_min_tomorrow_c": daily["temperature_2m_min"][1],
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
    cutoff = utcnow_naive() - timedelta(minutes=WEATHER_CACHE_MINUTES)

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

    # Prune anything older than the cache TTL. The reader only ever
    # consults rows newer than `cutoff` (= now - WEATHER_CACHE_MINUTES),
    # so older rows are pure disk weight. Keeping the table at ≤ 33
    # rows (one per district) means every cache read is instant.
    pruned = db.query(WeatherCache).filter(
        WeatherCache.fetched_at < cutoff
    ).delete(synchronize_session=False)
    if pruned:
        db.commit()

    return results


# --- Market prices (data.gov.in Agmarknet — free, needs API key) ---

MARKET_API_BASE = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

# Backend polls data.gov.in this often. Endpoint hits within this window
# return the snapshot table as-is — no synchronous fetch in the request path.
MARKET_REFRESH_MINUTES = 30

# Tracks the wall-clock time of the last successful refresh so the lazy-
# refresh fallback (endpoint hit) knows whether the background poller
# already covered it. Set by refresh_market_snapshots(); seeded on
# startup from the newest snapshot row's created_at, so a server restart
# inside a fresh 30-min window doesn't immediately trigger another fetch.
_last_market_refresh_at: datetime | None = None
# True while a refresh is in flight (background or lazy). Stops endpoint
# handlers from queuing a second refresh on top of the first. Mutated only
# under _market_refresh_lock so the background poller and the lazy
# fallback thread can't both pass the "is a refresh running?" check at the
# same moment and double-fetch.
_market_refresh_in_progress: bool = False
_market_refresh_lock = threading.Lock()

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

# Fields the data.gov.in resource returns per record that we actually
# store/display. `state` is dropped — it's always "Gujarat" (the only
# state we query) and nothing reads it downstream.
MARKET_RECORD_FIELDS = [
    "district", "market", "commodity", "variety",
    "grade", "arrival_date", "min_price", "max_price", "modal_price",
]


# Agmarknet data is Indian, so the snapshot calendar must be IST — UTC
# would roll over 5.5 hours late and briefly tag late-evening IST fetches
# with the previous day.
_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_today() -> date_cls:
    return datetime.now(_IST).date()


def _snapshot_key(rec: dict) -> tuple:
    """Identity used to dedupe a record within a single day's snapshot."""
    return (
        rec.get("commodity") or "",
        rec.get("market") or "",
        rec.get("variety") or "",
        rec.get("district") or "",
    )


def _normalize_record(rec: dict) -> dict:
    """Pulls out exactly the 9 documented fields from a raw API record,
    so the frontend always gets a consistent shape regardless of any
    extra fields data.gov.in includes."""
    return {field: rec.get(field) for field in MARKET_RECORD_FIELDS}


# ---------------------------------------------------------------------------
# Snapshot storage: market_price_snapshots is the SOLE source of truth for
# /market-price/all. The background poller (started in main.py) refreshes
# the table every MARKET_REFRESH_MINUTES; the endpoint just reads from it.
# ---------------------------------------------------------------------------

_MARKET_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FarmerAI/1.0"}


def _build_params(api_key: str, commodity: str | None = None, limit: int = 100) -> dict:
    p = {
        "api-key": api_key,
        "format": "json",
        "filters[state.keyword]": "Gujarat",
        "limit": limit,
    }
    if commodity:
        p["filters[commodity]"] = commodity
    return p


async def _fetch_from_api_async(
    client: "httpx.AsyncClient",
    api_key: str,
    commodity: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """One async HTTP call to data.gov.in, returning normalized records
    or [] on failure. data.gov.in stalls indefinitely on the default UA,
    so a browser-style header is required."""
    try:
        r = await client.get(
            MARKET_API_BASE,
            params=_build_params(api_key, commodity=commodity, limit=limit),
            headers=_MARKET_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"[market] fetch failed (commodity={commodity!r}): {e}")
        return []
    return [_normalize_record(rec) for rec in raw.get("records", [])]


def _store_records(db: Session, records: list[dict]) -> int:
    """Insert records into market_price_snapshots under today's IST date
    (deduped by commodity/market/variety/district), then delete every
    row whose snapshot_date < today. Returns the number of new rows."""
    today = _ist_today()

    # Retention: today only. The first refresh past IST midnight drops
    # everything from prior days.
    db.query(MarketPriceSnapshot).filter(
        MarketPriceSnapshot.snapshot_date < today
    ).delete(synchronize_session=False)

    if not records:
        db.commit()
        return 0

    # Dedupe key intentionally EXCLUDES arrival_date. data.gov.in often
    # shifts the same mandi's reported arrival_date by 1 day between polls
    # (e.g. Cotton@Rajkot at 10:00 shows 25/06, at 10:30 shows 26/06) — if
    # arrival_date was part of the key, both versions would survive as
    # separate rows and today's count would balloon over the day. With it
    # excluded, each (commodity, market, variety, district) gets exactly
    # ONE row per snapshot_date, updated in-place when fresher data lands.
    existing_today_rows = {
        (r.commodity or "", r.market or "", r.variety or "", r.district or ""): r
        for r in db.query(MarketPriceSnapshot)
        .filter(MarketPriceSnapshot.snapshot_date == today).all()
    }

    seen_in_batch: set[tuple] = set()
    new_rows = []
    updated = 0
    for rec in records:
        key = (
            rec.get("commodity") or "",
            rec.get("market") or "",
            rec.get("variety") or "",
            rec.get("district") or "",
        )
        if key in seen_in_batch:
            continue
        seen_in_batch.add(key)

        existing = existing_today_rows.get(key)
        if existing is not None:
            # Update in place — latest poll wins on price + arrival_date.
            existing.arrival_date = rec.get("arrival_date")
            existing.grade = rec.get("grade")
            existing.min_price = str(rec.get("min_price")) if rec.get("min_price") is not None else None
            existing.max_price = str(rec.get("max_price")) if rec.get("max_price") is not None else None
            existing.modal_price = str(rec.get("modal_price")) if rec.get("modal_price") is not None else None
            updated += 1
            continue

        new_rows.append(MarketPriceSnapshot(
            snapshot_date=today,
            commodity=rec.get("commodity"),
            market=rec.get("market"),
            district=rec.get("district"),
            variety=rec.get("variety"),
            grade=rec.get("grade"),
            arrival_date=rec.get("arrival_date"),
            min_price=str(rec.get("min_price")) if rec.get("min_price") is not None else None,
            max_price=str(rec.get("max_price")) if rec.get("max_price") is not None else None,
            modal_price=str(rec.get("modal_price")) if rec.get("modal_price") is not None else None,
        ))

    if new_rows:
        db.add_all(new_rows)
    db.commit()
    if updated:
        print(f"[market/store] {len(new_rows)} new rows, {updated} updated in place")
    return len(new_rows)


def _row_to_dict(r: MarketPriceSnapshot) -> dict:
    """Snapshot row → response shape (9 documented Agmarknet fields)."""
    return {
        "state": "Gujarat",
        "district": r.district,
        "market": r.market,
        "commodity": r.commodity,
        "variety": r.variety,
        "grade": r.grade,
        "arrival_date": r.arrival_date,
        "min_price": r.min_price,
        "max_price": r.max_price,
        "modal_price": r.modal_price,
    }


def _read_snapshot(db: Session, district: str | None = None) -> list[dict]:
    """Read today's snapshot rows, optionally filtered by district
    (case-insensitive). Only consumer is /market-price/all via
    get_all_market_prices — kept tiny on purpose."""
    today = _ist_today()
    q = db.query(MarketPriceSnapshot).filter(MarketPriceSnapshot.snapshot_date == today)
    if district:
        q = q.filter(func.lower(MarketPriceSnapshot.district) == district.lower())
    return [_row_to_dict(r) for r in q.all()]


async def refresh_market_snapshots_async(db: Session, api_key: str) -> int:
    """Full poll, concurrent: fan out every per-crop request via httpx +
    asyncio.gather, plus the generic discovery call. Writes to the
    snapshot table and prunes < today. Returns NEW rows inserted.
    ~3-5 s wall-clock vs ~40-80 s sequential."""
    global _last_market_refresh_at, _market_refresh_in_progress

    if not api_key:
        print("[market/refresh] skipped: MARKET_API_KEY not configured")
        return 0

    # Atomic check-and-set under the lock so two concurrent callers
    # (background poller + lazy fallback thread) can't both observe
    # in_progress=False and race into double-fetching.
    with _market_refresh_lock:
        if _market_refresh_in_progress:
            return 0
        _market_refresh_in_progress = True

    try:
        # Build the request list: one per crop in every category, plus a
        # final generic discovery call to catch anything not in our lists.
        crops = [
            crop
            for key, cat in CROP_CATEGORIES.items() if key != "all"
            for crop in cat["crops"]
        ]

        async with httpx.AsyncClient() as client:
            tasks = [
                _fetch_from_api_async(client, api_key, commodity=c, limit=20) for c in crops
            ]
            tasks.append(_fetch_from_api_async(client, api_key, limit=100))
            results = await asyncio.gather(*tasks)

        # Dedupe across all responses.
        seen: set[tuple] = set()
        collected: list[dict] = []
        for batch in results:
            for rec in batch:
                key = (
                    rec.get("commodity"), rec.get("market"),
                    rec.get("variety"), rec.get("district"), rec.get("arrival_date"),
                )
                if key in seen:
                    continue
                seen.add(key)
                collected.append(rec)

        inserted = _store_records(db, collected)
        _last_market_refresh_at = utcnow_naive()
        print(f"[market/refresh] fetched {len(collected)} records, inserted {inserted} new rows")
        return inserted
    finally:
        with _market_refresh_lock:
            _market_refresh_in_progress = False


def _refresh_is_stale() -> bool:
    """True if no refresh has run within MARKET_REFRESH_MINUTES — used by
    endpoint handlers as a lazy fallback when the background poller is
    behind (cold start, or it errored)."""
    if _last_market_refresh_at is None:
        return True
    return _last_market_refresh_at < utcnow_naive() - timedelta(minutes=MARKET_REFRESH_MINUTES)


def _seed_last_refresh_from_db(db: Session) -> None:
    """Called once on startup: seeds _last_market_refresh_at from the
    newest snapshot row's created_at, so a server restart inside a fresh
    30-min window doesn't immediately re-fetch."""
    global _last_market_refresh_at
    row = (
        db.query(MarketPriceSnapshot.created_at)
        .order_by(MarketPriceSnapshot.created_at.desc())
        .first()
    )
    if row and row[0]:
        _last_market_refresh_at = row[0]


def ensure_fresh_market_data(db: Session, api_key: str) -> None:
    """Fire-and-forget refresh trigger. If data is stale AND no refresh is
    already in flight, spawn a background thread to refresh — the current
    request returns whatever's in the snapshot table right now, instead
    of waiting up to ~5 s for ~40 HTTP calls.

    The first /market-price/all hit after server boot will get the
    snapshot rows already populated by the startup refresh; if for some
    reason none exist yet, the response is just empty and the next
    request (after the background refresh lands) gets the data."""
    # Cheap pre-check first (no lock). The authoritative in_progress
    # guard is inside refresh_market_snapshots_async itself, which holds
    # the lock — this short-circuit just avoids spawning a thread when
    # we already know there's nothing to do.
    if not _refresh_is_stale() or _market_refresh_in_progress:
        return

    def _bg():
        bg_db = None
        try:
            from database import SessionLocal
            bg_db = SessionLocal()
            asyncio.run(refresh_market_snapshots_async(bg_db, api_key))
        except Exception as e:
            print(f"[market] background lazy refresh failed: {e}")
        finally:
            if bg_db is not None:
                bg_db.close()

    threading.Thread(target=_bg, name="market-lazy-refresh", daemon=True).start()


def get_all_market_prices(db: Session, api_key: str, district: str = None) -> dict:
    """Returns every snapshot row stored for today across Gujarat, optionally
    district-filtered. Drives /market-price/all (the only market endpoint)."""
    ensure_fresh_market_data(db, api_key)

    records = _read_snapshot(db, district=district)
    if not records:
        return {"error": "No price data found for Gujarat."}

    return {
        "count": len(records),
        "records": records,
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

    print("\n--- Testing get_all_market_prices() ---")
    try:
        from config import MARKET_API_KEY
        if not MARKET_API_KEY:
            print("Skipped: MARKET_API_KEY not set in .env.")
        else:
            market = get_all_market_prices(db, api_key=MARKET_API_KEY)
            print("Market response keys:", list(market.keys()), "count:", market.get("count"))
    except Exception as e:
        print("Market test FAILED:", e)

    db.close()