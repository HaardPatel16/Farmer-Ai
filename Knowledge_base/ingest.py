"""
Reads documents from Knowledge_base/sources/ (including subfolders), splits
them into chunks, tags each chunk with the Gujarat districts it mentions
(if any), and inserts them into the knowledge_chunks table.

Usage:
    python Knowledge_base/ingest.py

Safe to re-run: each run first deletes any existing chunks for a given
source file (identified by its path relative to sources/), then
re-inserts fresh ones. So if you edit a source file and run this again,
you won't end up with duplicates — you'll just get the updated version.
Other sources are untouched.

Supports .txt and .pdf files, and you can organize them into subfolders
for your own convenience, e.g.:
    Knowledge_base/sources/Soil/soil_types_gujarat.txt
    Knowledge_base/sources/Bhavnagar/pest_guide.txt

If a folder's name matches a Gujarat district (e.g. "Bhavnagar/" or
"Navasari/" — common spelling variants work too), every chunk from files
in that folder is automatically tagged with that district, IN ADDITION to
whatever districts are detected from the text itself. Folder names that
don't match a district (like "Soil/" or "Crops/") are just for your own
organization and have no effect on tagging.
"""

import os
import re
import sys

# database.py and services.py now live in Backend/, a SIBLING folder to
# Knowledge_base/ (both sit directly under the project root), not one
# level up from this file as before. Reach sideways into it explicitly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Backend"))

from database import SessionLocal, KnowledgeChunk, Base, engine
from services import detect_all_districts, detect_district

SOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")

# Roughly how many words per chunk. Paragraph boundaries are respected, so
# actual chunk size will vary a bit around this target.
TARGET_CHUNK_WORDS = 200

# A short list of generic words to leave out of `keywords` even though they
# pass the length filter — keeps stored keywords more meaningful.
STOPWORDS = {
    "the", "and", "for", "are", "with", "this", "that", "from", "have",
    "has", "had", "but", "not", "you", "your", "into", "their", "they",
    "where", "which", "also", "some", "soils", "soil",
}


def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(filepath: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        return read_pdf(filepath)
    return read_txt(filepath)


def split_into_chunks(text: str, target_words: int = TARGET_CHUNK_WORDS) -> list[str]:
    """
    Splits text into chunks along paragraph breaks, grouping consecutive
    paragraphs together until roughly target_words is reached. This keeps
    each chunk topically coherent (e.g. one Gujarat region per chunk in the
    soil document) instead of cutting paragraphs in half.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current_parts: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_word_count = len(para.split())

        # If a single paragraph already exceeds the target on its own,
        # flush what we have and let this paragraph stand as its own chunk
        # rather than splitting mid-paragraph and breaking its meaning.
        if current_word_count + para_word_count > target_words and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_word_count = 0

        current_parts.append(para)
        current_word_count += para_word_count

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def extract_keywords(chunk_text: str, max_keywords: int = 15) -> str:
    """
    Pulls out the most frequent meaningful words in the chunk as a simple
    keyword string, stored alongside chunk_text to help search_knowledge_base
    match queries that use different phrasing than the source document.
    """
    words = re.findall(r"[a-zA-Z]+", chunk_text.lower())
    counts: dict[str, int] = {}
    for w in words:
        if len(w) > 3 and w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1

    top_words = sorted(counts, key=lambda w: counts[w], reverse=True)[:max_keywords]
    return ", ".join(top_words)


def detect_district_from_path(relative_path: str) -> str | None:
    """
    Checks each folder name in a file's path (relative to SOURCES_DIR) for
    a Gujarat district match, e.g. "Bhavnagar/pest_guide.txt" -> "bhavnagar".
    Returns None if no folder in the path matches a known district —
    this is the normal case for files organized by topic instead of
    district (e.g. "Soil/soil_types_gujarat.txt").
    Checks the deepest folder first, so sources/Saurashtra/Bhavnagar/file.txt
    would tag "bhavnagar" (the more specific folder) rather than a region
    name that doesn't match any single district anyway.
    """
    folder_names = os.path.dirname(relative_path).split(os.sep)
    for folder in reversed(folder_names):
        if not folder:
            continue
        match = detect_district(folder)
        if match:
            return match
    return None


def ingest_file(db, filepath: str, relative_path: str) -> int:
    """Ingests a single file: chunks it, tags it, and inserts into the DB.
    `relative_path` is the file's path relative to SOURCES_DIR (e.g.
    "Bhavnagar/pest_guide.txt") — used both as the stored source identifier
    (so files with the same name in different folders don't collide) and
    to check whether the containing folder names a district.
    Returns the number of chunks inserted."""
    print(f"\nProcessing {relative_path}...")

    text = load_document(filepath)
    if not text.strip():
        print(f"  Skipped — no extractable text found in {relative_path}.")
        return 0

    folder_district = detect_district_from_path(relative_path)
    if folder_district:
        print(f"  Folder tag: every chunk will also be tagged '{folder_district}'")

    # Remove any chunks from a previous run of this same file, so re-running
    # ingest.py after editing a source doesn't create duplicates. Keyed by
    # the relative path (not just filename), so "Soil/intro.txt" and
    # "Schemes/intro.txt" are tracked as separate sources.
    deleted = db.query(KnowledgeChunk).filter(
        KnowledgeChunk.source_filename == relative_path
    ).delete()
    if deleted:
        print(f"  Removed {deleted} existing chunk(s) from a previous ingest.")

    chunks = split_into_chunks(text)
    inserted = 0

    for chunk_text in chunks:
        districts = detect_all_districts(chunk_text)

        # Folder-based district is ADDED to whatever the text itself
        # mentions, not a replacement — a file can live in a district
        # folder while its content still spans multiple districts.
        if folder_district and folder_district not in districts:
            districts = districts + [folder_district]

        keywords = extract_keywords(chunk_text)

        row = KnowledgeChunk(
            source_filename=relative_path,
            chunk_text=chunk_text,
            keywords=keywords,
            districts=",".join(districts) if districts else None,
        )
        db.add(row)
        inserted += 1

        district_label = ", ".join(districts) if districts else "Gujarat-wide"
        print(f"  Chunk {inserted}: {len(chunk_text.split())} words — districts: {district_label}")

    db.commit()
    print(f"  Done — inserted {inserted} chunk(s) from {relative_path}.")
    return inserted


def main():
    # Make sure the knowledge_chunks table (and any other new tables) exist
    # before we try to insert into it.
    Base.metadata.create_all(bind=engine)

    if not os.path.isdir(SOURCES_DIR):
        os.makedirs(SOURCES_DIR, exist_ok=True)
        print(f"Created {SOURCES_DIR} — drop your .txt or .pdf source documents there and run this script again.")
        return

    # Walk recursively so files in subfolders (e.g. sources/Soil/file.txt,
    # sources/Bhavnagar/file.txt) are picked up, not just files sitting
    # directly in sources/.
    source_files = []
    for dirpath, _dirnames, filenames in os.walk(SOURCES_DIR):
        for filename in filenames:
            if filename.lower().endswith((".txt", ".pdf")) and not filename.startswith("."):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, SOURCES_DIR)
                source_files.append(relative_path)

    if not source_files:
        print(f"No .txt or .pdf files found in {SOURCES_DIR}. Add some documents and run this script again.")
        return

    db = SessionLocal()
    total_inserted = 0
    try:
        for relative_path in sorted(source_files):
            filepath = os.path.join(SOURCES_DIR, relative_path)
            total_inserted += ingest_file(db, filepath, relative_path)
    finally:
        db.close()

    print(f"\nAll done. {total_inserted} chunk(s) total across {len(source_files)} file(s).")


if __name__ == "__main__":
    main()