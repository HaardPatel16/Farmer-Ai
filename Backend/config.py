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

# Fail loudly and early if required variables are missing,
# rather than letting the app start and crash later with a
# confusing error somewhere deep in database.py or services.py.
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing. Check your .env file.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


if __name__ == "__main__":
    # Quick standalone check: run this file directly to confirm
    # your .env is being read correctly, without printing secrets.
    print("DATABASE_URL loaded:", "Yes" if DATABASE_URL else "No")
    print("GROQ_API_KEY loaded:", "Yes" if GROQ_API_KEY else "No")
    print("MARKET_API_KEY loaded:", "Yes" if MARKET_API_KEY else "No")
    print("DATABASE_URL starts with:", DATABASE_URL[:15] + "...")
    print("GROQ_API_KEY starts with:", GROQ_API_KEY[:8] + "...")