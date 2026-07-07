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

Index lives in memory, lazy-built on first call. ~15 000 chunks × 384-dim
float32 ≈ 22 MB — trivial. First call after server restart takes 2-5 min on
CPU to embed all chunks; subsequent queries cost ~1-5 ms (one matmul).
"""

import threading
import time

import numpy as np

# Sentence-transformer model identifier. paraphrase-multilingual-MiniLM-L12-v2
# is ~120 MB, handles English + 50 other languages including Gujarati,
# and produces 384-dim embeddings. Good speed/quality trade-off for CPU
# inference. Swap to "distiluse-base-multilingual-cased-v2" for better
# quality at 4x the size if you ever need it.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None

# Two parallel structures so the hot path can vectorize:
#   _chunks_meta[i]   — dict with text/filename/districts for chunk i
#   _embedding_matrix — float32 ndarray of shape (N, D), L2-normalized rows
# Scoring becomes a single matmul (_embedding_matrix @ q_emb) instead of a
# Python loop of N dot products. Both structures are written together
# inside _ensure_indexed() under _load_lock and never mutated afterward.
_chunks_meta: list[dict] = []
_embedding_matrix: np.ndarray | None = None

_load_lock = threading.Lock()
_model_load_failed = False  # latched True if we tried and failed once
_warming = False  # True while the one-time model load + index build is running
_warm_started_at: float | None = None
_warm_finished_at: float | None = None
_warm_total_chunks: int = 0
# Set when warmup raises an exception AFTER the model loaded — without this
# the status endpoint would fall through to "idle_empty_kb" and an operator
# tailing logs would assume the KB was empty rather than the encode crashed.
_warm_error: str | None = None


def _pick_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'.
    Encapsulated so both embeddings.py and ml_model.py can call the same
    logic — and so a future "force CPU" env var can land in one place."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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
        device = _pick_device()
        _model = SentenceTransformer(MODEL_NAME, device=device)
        # Print once at load time so the operator can see whether the
        # GPU was actually picked up — silent CPU fallback after a CUDA
        # install attempt is otherwise easy to miss.
        print(f"[embeddings] sentence-transformer running on {device.upper()}")
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
    global _chunks_meta, _embedding_matrix
    global _warming, _warm_started_at, _warm_finished_at, _warm_total_chunks, _warm_error
    if _embedding_matrix is not None:
        return
    with _load_lock:
        if _embedding_matrix is not None:  # another thread won the race
            return
        _warming = True
        _warm_started_at = time.time()
        _warm_error = None
        # Plain ASCII only in these prints — the Windows default console
        # (cp1252 / "charmap") raises UnicodeEncodeError on emoji like
        # the previous ▶ ✓ ✗ arrows, which silently killed the warmup
        # thread before a single chunk got encoded. That meant semantic
        # search appeared "stuck warming" forever, and /chat fell back
        # to keyword-only retrieval on every request.
        print("[embeddings] >> warmup started -- loading model + indexing chunks...")
        try:
            model = _try_load_model()
            if model is None:
                print("[embeddings] xx warmup aborted -- model failed to load")
                return
            print("[embeddings]    model loaded, reading chunks from DB...")
            from database import KnowledgeChunk
            rows = db.query(KnowledgeChunk).all()
            _warm_total_chunks = len(rows)
            if not rows:
                print("[embeddings] xx warmup aborted -- knowledge_chunks table is empty")
                return
            texts = [r.chunk_text for r in rows]
            print(f"[embeddings]    encoding {len(texts)} chunks (one-time, ~2-5 min)...")
            emb = model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            # Ensure float32 contiguous so the matmul in semantic_search is
            # fastest; SentenceTransformer.encode() already returns float32
            # in practice but we don't want to silently take a hit if a
            # future version changes that.
            emb = np.ascontiguousarray(emb, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb /= norms
            _chunks_meta = [
                {
                    "text": rows[i].chunk_text,
                    "filename": rows[i].source_filename,
                    "districts": rows[i].districts,
                }
                for i in range(len(rows))
            ]
            _embedding_matrix = emb
            elapsed = time.time() - _warm_started_at
            print(
                f"[embeddings] OK warmup complete -- {len(_chunks_meta)} chunks indexed "
                f"in {elapsed:.1f}s. Semantic search ACTIVE."
            )
        except Exception as e:
            # Surface the failure in get_status() instead of letting it
            # look like an empty KB. The daemon-thread caller in
            # warm_index_in_background also prints, but get_status is
            # what the frontend / operator polls.
            _warm_error = f"{type(e).__name__}: {e}"
            print(f"[embeddings] xx warmup failed -- {_warm_error}")
            raise
        finally:
            _warming = False
            _warm_finished_at = time.time()


def get_status() -> dict:
    """Snapshot of the warmup state — read by GET /embeddings/status."""
    if _embedding_matrix is not None:
        state = "ready"
    elif _warming:
        state = "warming"
    elif _model_load_failed:
        state = "failed"
    elif _warm_error:
        # The warmup ran far enough to load the model but the encode (or
        # something else after) raised. Distinct from "failed" which is
        # specifically a model-load failure.
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
        "chunks_indexed": len(_chunks_meta),
        "chunks_total": _warm_total_chunks or len(_chunks_meta),
        "elapsed_seconds": elapsed,
        "model_name": MODEL_NAME,
        "model_load_failed": _model_load_failed,
        "warm_error": _warm_error,
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
    if _warming and _embedding_matrix is None:
        return None

    model = _try_load_model()
    if model is None:
        return None
    _ensure_indexed(db)
    if _embedding_matrix is None:
        return None

    q_emb = model.encode([query], convert_to_numpy=True)[0]
    qn = np.linalg.norm(q_emb)
    if qn == 0:
        return []
    q_emb = (q_emb / qn).astype(np.float32, copy=False)

    # Single matmul replaces the Python loop of N dot products. On 14k
    # chunks this is ~1 ms vs ~50 ms loop-and-allocate. Cosine similarity
    # because both sides are already L2-normalized.
    sims = _embedding_matrix @ q_emb

    # District filter as a boolean mask — only do the splitting work for
    # chunks that actually have a districts tag, and bail early for the
    # common Gujarat-wide case (districts is None).
    if district:
        for i, m in enumerate(_chunks_meta):
            d_str = m["districts"]
            if not d_str:
                continue
            chunk_districts = [d.strip() for d in d_str.split(",") if d.strip()]
            if chunk_districts and district not in chunk_districts:
                # -inf so this chunk can't possibly land in the top_k
                sims[i] = -np.inf

    # Top-k without a full sort: argpartition gets the indices of the k
    # largest similarities in O(N), then we sort just those k entries.
    k = min(top_k, sims.shape[0])
    if k <= 0:
        return []
    # If every chunk got filtered out (all -inf), bail.
    if not np.isfinite(sims).any():
        return []
    top_unsorted = np.argpartition(-sims, k - 1)[:k]
    top_indices = top_unsorted[np.argsort(-sims[top_unsorted])]

    results = []
    for idx in top_indices:
        s = float(sims[idx])
        if not np.isfinite(s):
            # Filtered-out chunks made it into the partition only because
            # there were fewer than k surviving chunks.
            continue
        m = _chunks_meta[idx]
        results.append({
            "text": m["text"],
            "filename": m["filename"],
            "similarity": s,
            "embedding": _embedding_matrix[idx],
        })
    return results
