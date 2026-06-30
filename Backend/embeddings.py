"""
Semantic retrieval over the knowledge_chunks table using sentence-transformer
embeddings + cosine similarity.

Designed as an opt-in COMPANION to the keyword-based search_knowledge_base()
in services.py — not a replacement. The keyword search remains the primary
retriever because it's:
  - fast (no model load)
  - district-aware (skips chunks tagged for other districts)
  - filename-aware (force-includes dedicated topic files)
and embeddings then provide a second candidate list for cases the keyword
scorer misses (semantic synonymy — e.g. "cold storage" matching the
"Cold_Chain_Management" file even though the user didn't type "chain").

Graceful fallback: if sentence-transformers isn't installed or model load
fails, semantic_search() returns None and main.py just uses keyword results.

Index lives in memory, lazy-built on first call. ~4000 chunks × 384-dim
float32 ≈ 6 MB — trivial. First call after server restart takes 2-5 min on
CPU to embed all chunks; subsequent queries cost ~10-50 ms.
"""

import threading
import time

# Sentence-transformer model identifier. paraphrase-multilingual-MiniLM-L12-v2
# is ~120 MB, handles English + 50 other languages including Gujarati,
# and produces 384-dim embeddings. Good speed/quality trade-off for CPU
# inference. Swap to "distiluse-base-multilingual-cased-v2" for better
# quality at 4x the size if you ever need it.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_chunks_index: list[dict] = []
_load_lock = threading.Lock()
_model_load_failed = False  # latched True if we tried and failed once
_warming = False  # True while the one-time model load + index build is running
_warm_started_at: float | None = None
_warm_finished_at: float | None = None
_warm_total_chunks: int = 0


def _try_load_model():
    """Lazy-load the sentence-transformer. Returns the model or None on
    failure. Failures are latched (won't retry every request) so a missing
    install doesn't keep wasting time on import attempts."""
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        return _model
    except Exception as e:
        print(f"[embeddings] could not load sentence-transformer model: {e}")
        print("[embeddings] semantic search disabled; keyword search remains active")
        _model_load_failed = True
        return None


def _ensure_indexed(db):
    """Build the in-memory embedding index on first call. Reads every
    KnowledgeChunk row, embeds the text, stashes embedding + metadata.
    Cheap no-op on subsequent calls (just returns)."""
    global _chunks_index, _warming, _warm_started_at, _warm_finished_at, _warm_total_chunks
    if _chunks_index:
        return
    with _load_lock:
        if _chunks_index:  # another thread won the race
            return
        _warming = True
        _warm_started_at = time.time()
        print("[embeddings] ▶ warmup started — loading model + indexing chunks…")
        try:
            model = _try_load_model()
            if model is None:
                print("[embeddings] ✗ warmup aborted — model failed to load")
                return
            print("[embeddings]   model loaded, reading chunks from DB…")
            from database import KnowledgeChunk
            rows = db.query(KnowledgeChunk).all()
            _warm_total_chunks = len(rows)
            if not rows:
                print("[embeddings] ✗ warmup aborted — knowledge_chunks table is empty")
                return
            texts = [r.chunk_text for r in rows]
            print(f"[embeddings]   encoding {len(texts)} chunks (one-time, ~2-5 min)…")
            emb = model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            import numpy as np
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
            _chunks_index = [
                {
                    "text": rows[i].chunk_text,
                    "filename": rows[i].source_filename,
                    "districts": rows[i].districts,
                    "embedding": emb[i],
                }
                for i in range(len(rows))
            ]
            elapsed = time.time() - _warm_started_at
            print(f"[embeddings] ✓ warmup complete — {len(_chunks_index)} chunks indexed in {elapsed:.1f}s. Semantic search ACTIVE.")
        finally:
            _warming = False
            _warm_finished_at = time.time()


def get_status() -> dict:
    """Snapshot of the warmup state — read by GET /embeddings/status."""
    if _chunks_index:
        state = "ready"
    elif _warming:
        state = "warming"
    elif _model_load_failed:
        state = "failed"
    elif _warm_started_at is None:
        state = "not_started"
    else:
        # Tried, finished, but no chunks (empty KB).
        state = "idle_empty_kb"

    elapsed = None
    if _warm_started_at is not None:
        end = _warm_finished_at if _warm_finished_at else time.time()
        elapsed = round(end - _warm_started_at, 1)

    return {
        "state": state,
        "chunks_indexed": len(_chunks_index),
        "chunks_total": _warm_total_chunks or len(_chunks_index),
        "elapsed_seconds": elapsed,
        "model_name": MODEL_NAME,
        "model_load_failed": _model_load_failed,
    }


def warm_index_in_background(db_factory):
    """Trigger the one-time model load + chunk indexing in a daemon thread.
    Called from main.py's startup hook so users never pay the 2-5 min cost
    on their first chat. db_factory is a callable returning a fresh Session
    (typically SessionLocal) — we open and close our own session so the
    background work doesn't depend on a request-scoped one."""
    def _run():
        db = db_factory()
        try:
            _ensure_indexed(db)
        except Exception as e:
            print(f"[embeddings] background warm failed: {e}")
        finally:
            db.close()
    threading.Thread(target=_run, name="embeddings-warm", daemon=True).start()


def semantic_search(db, query: str, top_k: int = 5, district: str | None = None):
    """
    Returns up to top_k chunks ranked by cosine similarity to the query.
    Each result is a dict with keys:
        "text"       — the chunk text
        "filename"   — its source filename
        "similarity" — cosine similarity to the query (pre-normalized dot)
        "embedding"  — the chunk's L2-normalized index vector
    The "embedding" is the SAME vector built once at index time — it is
    returned (not recomputed) so callers can reuse it for MMR
    de-duplication without re-encoding anything.

    Returns None if the model failed to load or the index couldn't be
    built — callers should treat None as "fall back to keyword search
    only". An empty list means the index is ready but produced no
    candidates (e.g. a degenerate zero-norm query).

    `district`, when given, filters out chunks tagged for OTHER districts
    (same semantics as services.py's keyword path), so a Bhavnagar-tagged
    chunk won't pollute a Kheda question.
    """
    # If the background warmup is still running, fall through to keyword-
    # only search instead of blocking this request for minutes. The first
    # few /chat calls after a cold start will be keyword-only; once warm
    # completes, every subsequent call uses semantic results too.
    if _warming and not _chunks_index:
        return None

    model = _try_load_model()
    if model is None:
        return None
    _ensure_indexed(db)
    if not _chunks_index:
        return None

    import numpy as np
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    qn = np.linalg.norm(q_emb)
    if qn == 0:
        return []
    q_emb = q_emb / qn

    scored = []
    for c in _chunks_index:
        # District filtering: skip chunks tagged for districts that
        # exclude the one we're asking about.
        if district and c["districts"]:
            chunk_districts = [d.strip() for d in c["districts"].split(",") if d.strip()]
            if chunk_districts and district not in chunk_districts:
                continue
        # Pre-normalized vectors → cosine sim is dot product.
        sim = float(q_emb @ c["embedding"])
        scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Carry the chunk's filename and its already-computed normalized
    # embedding through so the hybrid retriever in services.py can fuse
    # and MMR-dedup without re-encoding anything.
    return [
        {
            "text": c["text"],
            "filename": c["filename"],
            "similarity": s,
            "embedding": c["embedding"],
        }
        for s, c in scored[:top_k]
    ]


def rebuild_index(db_factory) -> None:
    """Drop the in-memory embedding index and rebuild from the current
    knowledge_chunks rows. Call this after re-running ingest.py so the
    semantic-search results pick up newly-added chunks without needing a
    server restart. Rebuild runs synchronously on the caller's thread; if
    you want it non-blocking, wrap in a daemon thread the same way
    warm_index_in_background does.
    """
    global _chunks_index, _warming, _warm_started_at, _warm_finished_at, _warm_total_chunks
    with _load_lock:
        _chunks_index = []
        _warming = False
        _warm_started_at = None
        _warm_finished_at = None
        _warm_total_chunks = 0
    db = db_factory()
    try:
        _ensure_indexed(db)
    finally:
        db.close()
