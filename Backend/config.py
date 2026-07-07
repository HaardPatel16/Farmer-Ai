"""
Loads environment variables from .env and exposes them as simple constants.
Every other file should import settings from here instead of reading
os.environ directly — keeps secrets/config centralized in one place.
"""

import os
from dotenv import load_dotenv

# Explicitly point at the .env file in the PROJECT ROOT (one level up from
# this file, which lives in Backend/), rather than relying on
# load_dotenv()'s default behavior of searching the current working
# directory upward. That default only works by coincidence if you happen
# to launch uvicorn from the exact right folder — this way it works the
# same regardless of where you run `uvicorn` from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Market price API key (data.gov.in / Agmarknet). Optional — the
# /market-price endpoints degrade gracefully with a clear error message
# if this is missing, rather than the whole app refusing to start.
MARKET_API_KEY = os.getenv("MARKET_API_KEY")

# Secret token that gates the operator-only /stats endpoint. Optional:
#  - SET it (any long random string) to lock the stats dashboard so only
#    someone who knows the token can read usage/feedback data. Required
#    before any public deployment, since /stats exposes farmer questions,
#    token spend, and dislike diagnostics.
#  - LEAVE it unset for local/dev, where the endpoint stays open for
#    convenience. Same "possession of the secret = capability" model as the
#    session-delete route (no user accounts).
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# Comma-separated list of origins the browser is allowed to call the API
# from. Defaults to localhost dev ports + file:// (Origin: null) so the
# plain index.html opened from disk still works. Override with
# `ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com`
# before any public deployment. Use "*" only on a fully trusted LAN.
_default_origins = (
    "http://127.0.0.1:8000,http://localhost:8000,"
    "http://127.0.0.1:5500,http://localhost:5500,"
    "http://127.0.0.1:5173,http://localhost:5173,"
    "null"
)
# Treat an *empty* ALLOWED_ORIGINS env var the same as an unset one — a
# blank value would otherwise produce an empty allowlist and silently
# block every browser, with the failure showing up only as a generic CORS
# error in devtools (hard to attribute). `or` collapses ""/None to the
# default; an explicit list of origins still wins, including the literal
# "*" if the operator really wants a wide-open API on a trusted LAN.
ALLOWED_ORIGINS = [
    o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or _default_origins).split(",") if o.strip()
]

# Fail loudly and early if required variables are missing,
# rather than letting the app start and crash later with a
# confusing error somewhere deep in database.py or services.py.
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing. Check your .env file.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


if __name__ == "__main__":
    # Quick standalone check: confirms the .env is being read correctly
    # without printing any secret material. The previous version dumped
    # the first 8 chars of GROQ_API_KEY and the first 15 of DATABASE_URL,
    # which is enough to fingerprint the key on any pasted log.
    print("DATABASE_URL loaded:", "Yes" if DATABASE_URL else "No")
    print("GROQ_API_KEY loaded:", "Yes" if GROQ_API_KEY else "No")
    print("MARKET_API_KEY loaded:", "Yes" if MARKET_API_KEY else "No")
    if DATABASE_URL:
        # Show only the driver portion (e.g. "postgresql"), never the
        # host, user, password, or db name.
        driver = DATABASE_URL.split("://", 1)[0] if "://" in DATABASE_URL else "?"
        print("DATABASE_URL driver:", driver)