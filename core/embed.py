"""Memory that indexes meaning instead of words.

The existing `recall` is FTS5 — a keyword match. It is very good at the things
keywords are good at, and blind to everything else:

    "what did I say about the insurance letter"  -> works, "insurance" is there
    "what did we decide about the roof"          -> nothing, because what was
                                                    actually said was "we're
                                                    getting the shingles
                                                    replaced"

That second failure is silent. It does not say "no match on that word", it says
"nothing found", and you conclude the assistant forgot. This module fixes that
without giving up the first case, because keyword search is still the only
thing that reliably finds "INC-4412".

Three decisions worth defending:

  * LOCAL EMBEDDINGS. Every other heavy thing on this box runs locally —
    wake word, transcription, speech. Sending household transcripts and work
    case notes to a hosted embedding API to be indexed would be the one place
    the data leaves, and it would break recall whenever the internet did.
    ONNX Runtime is already a dependency here for openWakeWord, so the model
    is nearly free.

  * NUMPY, NOT A VECTOR DATABASE. Cosine over a few thousand 384-float vectors
    is about two milliseconds. sqlite-vec and friends earn their keep at a
    hundred thousand rows; below that they are a native extension to load, a
    binary to install, and a failure mode to debug, in exchange for nothing.
    Vectors live as BLOBs in the same SQLite file as everything else, so they
    survive a restart with no extra machinery.

  * RECIPROCAL RANK FUSION for the hybrid merge. BM25 scores and cosine
    similarities are not on the same scale and no amount of tuning makes them
    comparable. RRF throws the magnitudes away and uses only the RANK each
    result got in each list, which is the part that is actually meaningful.

If the model is missing or fails to load, everything degrades to the keyword
search that exists today. Semantic recall is an improvement, not a dependency.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("assistant.embed")

# Vectors are stored as float32. float64 would double the file for precision
# that cosine similarity cannot use.
DTYPE = np.float32

# The constant in reciprocal rank fusion. 60 is the value from the original
# paper and the one everyone uses; it controls how quickly the contribution of
# a result decays with rank. Not worth tuning without an evaluation set.
RRF_K = 60


class Embedder:
    """Lazy wrapper around a local ONNX sentence embedder.

    Lazy because loading costs a second or two and most restarts never recall
    anything. Locked because the backfill loop and a live question can arrive
    at the same time, and an ONNX session is not reentrant.
    """

    def __init__(self, cfg: dict):
        m = (cfg.get("memory", {}) or {}).get("semantic", {}) or {}
        self.enabled = bool(m.get("enabled", False))
        self.model_name = str(m.get("model", "BAAI/bge-small-en-v1.5"))
        self.cache_dir = str(m.get("cache_dir", "~/assistant/models/embed"))
        # Capped deliberately. Whisper wants the cores when someone is talking,
        # and an embedding that takes 40ms instead of 25ms is not noticeable
        # while a stuttering transcription very much is.
        self.threads = int(m.get("threads", 2))
        self._model = None
        self._lock = threading.Lock()
        self._broken = False
        self._dim: int | None = None

    @property
    def available(self) -> bool:
        return self.enabled and not self._broken

    def _load(self):
        if self._model is not None or self._broken:
            return self._model
        import os
        from pathlib import Path
        try:
            from fastembed import TextEmbedding
        except ImportError:
            log.warning("fastembed is not installed — semantic recall off")
            self._broken = True
            return None
        try:
            t0 = time.time()
            cache = str(Path(self.cache_dir).expanduser())
            os.makedirs(cache, exist_ok=True)
            self._model = TextEmbedding(self.model_name, cache_dir=cache,
                                        threads=self.threads)
            self._dim = int(self._model.embedding_size)
            log.info("embedder %s (dim %d) ready in %.1fs",
                     self.model_name, self._dim, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            # Almost always the first-run download with no network. Say so
            # once and fall back; do not retry on every question.
            log.error("could not load %s: %s — semantic recall off. "
                      "Run scripts/fetch_embed_model.sh",
                      self.model_name, exc)
            self._broken = True
        return self._model

    @property
    def dim(self) -> int | None:
        if self._dim is None and self.available:
            self._load()
        return self._dim

    def documents(self, texts: list[str]) -> np.ndarray | None:
        """Embed things being stored."""
        if not texts or not self.available:
            return None
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                return np.asarray(list(model.embed(texts)), dtype=DTYPE)
            except Exception as exc:  # noqa: BLE001
                log.error("embed failed: %s", exc)
                return None

    def query(self, text: str) -> np.ndarray | None:
        """Embed a question.

        Deliberately NOT the same call as `documents`. bge models are trained
        asymmetrically — queries get an instruction prefix that passages do
        not — and using the passage path for a question quietly costs you
        retrieval quality with no error to notice.
        """
        if not text.strip() or not self.available:
            return None
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                return np.asarray(next(iter(model.query_embed([text]))),
                                  dtype=DTYPE)
            except Exception as exc:  # noqa: BLE001
                log.error("query embed failed: %s", exc)
                return None


class SemanticIndex:
    """Vectors in SQLite, cosine in numpy, hybrid merge on top."""

    def __init__(self, store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._cache: dict[str, tuple[np.ndarray, list[int]]] = {}
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------

    def index(self, kind: str, rows: list[tuple[int, str]]) -> int:
        """Embed and store. rows is [(ref_id, text)]."""
        rows = [(i, t) for i, t in rows if t and t.strip()]
        if not rows or not self.embedder.available:
            return 0
        vecs = self.embedder.documents([t for _, t in rows])
        if vecs is None:
            return 0
        self.store.embeddings_put(
            kind, self.embedder.model_name,
            [(rid, v.astype(DTYPE).tobytes()) for (rid, _), v in zip(rows, vecs)])
        with self._lock:
            self._cache.pop(kind, None)
        return len(rows)

    # -- reading -----------------------------------------------------

    def _matrix(self, kind: str) -> tuple[np.ndarray, list[int]]:
        """All vectors of one kind as a matrix, cached until something writes.

        Rebuilding costs one query and one reshape. At a few thousand rows
        that is milliseconds, which is why there is no incremental update path
        here to get subtly wrong.
        """
        with self._lock:
            hit = self._cache.get(kind)
            if hit is not None:
                return hit
        rows = self.store.embeddings_all(kind, self.embedder.model_name)
        if not rows:
            out = (np.zeros((0, 1), dtype=DTYPE), [])
        else:
            ids = [r["ref_id"] for r in rows]
            mat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=DTYPE)
            mat = mat.reshape(len(ids), -1)
            out = (mat, ids)
        with self._lock:
            self._cache[kind] = out
        return out

    def similar(self, text: str, kind: str = "message",
                limit: int = 8, floor: float = 0.0) -> list[tuple[int, float]]:
        """[(ref_id, cosine)] most like `text`, best first."""
        qv = self.embedder.query(text)
        if qv is None:
            return []
        mat, ids = self._matrix(kind)
        if not ids or mat.shape[1] != qv.shape[0]:
            return []
        # Both sides are already L2-normalised by the model, so the dot
        # product IS the cosine. Re-normalising here would be wasted work.
        scores = mat @ qv
        order = np.argsort(-scores)[:limit]
        return [(ids[i], float(scores[i])) for i in order
                if float(scores[i]) >= floor]

    def duplicate_of(self, text: str, threshold: float = 0.92) -> int | None:
        """The memory this is a restatement of, if there is one.

        UNIQUE on the text column only catches character-identical facts, so
        "his daughter plays soccer" and "Brendan's daughter is on a soccer
        team" both get stored, and both then go into every future prompt
        forever. The threshold is deliberately high: a false positive here
        silently refuses to learn something new, which is worse than a
        duplicate.
        """
        hits = self.similar(text, kind="memory", limit=1, floor=threshold)
        return hits[0][0] if hits else None

    # -- the hybrid ---------------------------------------------------

    def hybrid(self, query: str, keyword_ids: list[int],
               limit: int = 8) -> list[int]:
        """Merge a keyword ranking and a semantic ranking by rank, not score.

        Keyword search is kept, not replaced. It is the only one of the two
        that reliably finds "INC-4412", a surname, or a part number — an
        embedding of a ticket number is close to every other ticket number.
        Semantic search is the only one that finds "roof" in a sentence about
        shingles. Neither subsumes the other, so both vote.
        """
        ranks: dict[int, float] = {}
        for pos, rid in enumerate(keyword_ids):
            ranks[rid] = ranks.get(rid, 0.0) + 1.0 / (RRF_K + pos + 1)
        for pos, (rid, _) in enumerate(self.similar(query, limit=limit * 2)):
            ranks[rid] = ranks.get(rid, 0.0) + 1.0 / (RRF_K + pos + 1)
        return [rid for rid, _ in
                sorted(ranks.items(), key=lambda kv: -kv[1])][:limit]

    # -- backfill ------------------------------------------------------

    def backfill(self, kind: str, batch: int = 64) -> int:
        """Embed one batch of rows that have no vector yet. Returns how many.

        Batched and driven from outside rather than looping here, so the
        caller decides how much of the machine this is allowed to have. On
        first run there may be thousands of messages, and doing them all at
        startup would mean the assistant is deaf for a minute after every
        update.
        """
        if not self.embedder.available:
            return 0
        rows = self.store.embeddings_missing(
            kind, self.embedder.model_name, batch)
        return self.index(kind, [(r["id"], r["text"]) for r in rows]) if rows else 0

    def stats(self) -> dict:
        return {
            "model": self.embedder.model_name,
            "available": self.embedder.available,
            "indexed": self.store.embeddings_counts(self.embedder.model_name),
            "pending": {
                k: len(self.store.embeddings_missing(
                    k, self.embedder.model_name, 100000))
                for k in ("message", "memory")
            },
        }
