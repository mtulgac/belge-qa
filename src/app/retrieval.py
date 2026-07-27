import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import (
    BM25_B,
    BM25_K1,
    BM25_TOKEN,
    DEFAULT_VARIANT,
    EMBEDDING_MODEL,
    RERANK_BASE,
    RERANK_DEPTH,
    RERANK_MODEL,
    RETRIEVER,
    RRF_DEPTH,
    RRF_K,
    RRF_WEIGHTS,
    prefixes,
)
from app.ingest import index_dir
from app.lexical import TOKEN_PATTERNS, Bm25

CHUNK_FIELDS = ("chunk_id", "dosya", "sayfa_baslangic", "sayfa_bitis", "yontem")


@dataclass
class Hit:
    skor: float               # the active retriever's own score
    chunk_id: str
    dosya: str
    sayfa_baslangic: int
    sayfa_bitis: int
    yontem: str
    metin: str
    # Diagnostics. Which retriever produced this hit, and where it stood in each
    # one — the hybrid's own score is an RRF value and says nothing about how
    # strong the match was. None means that retriever did not rank the chunk.
    dense_skor: float | None = None
    bm25_skor: float | None = None
    dense_sira: int | None = None
    bm25_sira: int | None = None
    # Set only by the reranker: its own score (which becomes `skor`) and the rank
    # this chunk held in the hybrid candidate list BEFORE reranking, so the
    # cross-encoder's reordering is visible against what fusion handed it.
    rerank_skor: float | None = None
    aday_sira: int | None = None


class Index:
    def __init__(self, path: Path):
        self.path = path
        self.vectors = np.load(path / "vectors.npy")
        self.chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.chunks)


@lru_cache(maxsize=4)
def load_index(variant: str = DEFAULT_VARIANT, model_name: str = EMBEDDING_MODEL) -> Index:
    path = index_dir(variant, model_name)
    if not (path / "vectors.npy").exists():
        raise FileNotFoundError(
            f"index missing: {path}\n"
            f"build it: .venv/bin/python -m app.ingest --variant {variant} --model {model_name}"
        )
    return Index(path)


@lru_cache(maxsize=4)
def load_bm25(
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    token: str = BM25_TOKEN,
) -> Bm25:
    
    index = load_index(variant, model_name)
    return Bm25(
        [c["metin"] for c in index.chunks],
        pattern=TOKEN_PATTERNS[token], k1=BM25_K1, b=BM25_B,
    )


# bge-m3 is ~2 GB resident and the sweeps walk models one at a time, so caching
# more than the model in use costs memory without ever paying it back.
@lru_cache(maxsize=1)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# Cached because a sweep asks the same 30 questions under every setting, and the
# embedding does not depend on any of them — without this the fusion sweep pays
# bge-m3's ~35 ms per query once per configuration instead of once. Callers treat
# the vector as read-only.
@lru_cache(maxsize=256)
def embed_query(question: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    query_prefix, _ = prefixes(model_name)
    return _encoder(model_name).encode(
        [query_prefix + question], convert_to_numpy=True, normalize_embeddings=True,
    )[0]


def _mask(scores: np.ndarray, index: Index, exclude: tuple[str, ...]) -> np.ndarray:
    """Drop the excluded documents (the golden set's `izole`) from contention."""
    if not exclude:
        return scores
    blocked = np.array([c["dosya"] in exclude for c in index.chunks])
    return np.where(blocked, -np.inf, scores)


def _hit(index: Index, i: int, skor: float, **extra) -> Hit:
    chunk = index.chunks[i]
    return Hit(
        skor=float(skor),
        metin=chunk["metin"],
        **{key: chunk[key] for key in CHUNK_FIELDS},
        **extra,
    )


def search_dense(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
) -> list[Hit]:
    index = load_index(variant, model_name)
    scores = _mask(index.vectors @ embed_query(question, model_name), index, exclude)
    top = np.argsort(-scores)[:k]
    return [
        _hit(index, i, scores[i], dense_skor=float(scores[i]), dense_sira=rank)
        for rank, i in enumerate(top, start=1)
    ]


def search_bm25(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
    token: str = BM25_TOKEN,
) -> list[Hit]:
    index = load_index(variant, model_name)
    scores = _mask(load_bm25(variant, model_name, token).score(question), index, exclude)
    top = np.argsort(-scores)[:k]
    # A query whose terms appear nowhere leaves every document at 0.0, and argsort
    # would still hand back k chunks in index order. Those are not matches.
    return [
        _hit(index, i, scores[i], bm25_skor=float(scores[i]), bm25_sira=rank)
        for rank, i in enumerate(top, start=1)
        if scores[i] > 0
    ]


def rrf(
    rankings: list[list[int]],
    k_const: int = RRF_K,
    weights: tuple[float, ...] | None = None,
) -> dict[int, float]:
    
    weights = weights or (1.0,) * len(rankings)
    fused: dict[int, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + weight / (k_const + rank)
    return fused


def search_hybrid(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
    depth: int = RRF_DEPTH,
    k_const: int = RRF_K,
    weights: tuple[float, float] = RRF_WEIGHTS,
    token: str = BM25_TOKEN,
) -> list[Hit]:
    index = load_index(variant, model_name)
    dense = search_dense(question, depth, variant, model_name, exclude)
    lexical = search_bm25(question, depth, variant, model_name, exclude, token)

    position = {c["chunk_id"]: i for i, c in enumerate(index.chunks)}
    fused = rrf(
        [[position[h.chunk_id] for h in dense], [position[h.chunk_id] for h in lexical]],
        k_const, weights,
    )

    by_dense = {h.chunk_id: h for h in dense}
    by_bm25 = {h.chunk_id: h for h in lexical}
    hits = []
    for i, skor in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
        chunk_id = index.chunks[i]["chunk_id"]
        d, b = by_dense.get(chunk_id), by_bm25.get(chunk_id)
        hits.append(_hit(
            index, i, skor,
            dense_skor=d.dense_skor if d else None,
            dense_sira=d.dense_sira if d else None,
            bm25_skor=b.bm25_skor if b else None,
            bm25_sira=b.bm25_sira if b else None,
        ))
    return hits


# The retrievers a reranker can draw candidates from. Both take (question, k,
# variant, model_name, exclude) positionally, so search_rerank calls either the
# same way. "hibrit" is the default; "dense" is the ablation that asks whether the
# reranker still needs BM25 in front of it.
BASE_RETRIEVERS = {"hibrit": search_hybrid, "dense": search_dense}


def search_rerank(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
    depth: int = RERANK_DEPTH,
    rerank_model: str = RERANK_MODEL,
    base: str = RERANK_BASE,
) -> list[Hit]:
    
    from app.models import rerank_scores

    if base not in BASE_RETRIEVERS:
        raise ValueError(f"unknown rerank base: {base} (options: {list(BASE_RETRIEVERS)})")
    candidates = BASE_RETRIEVERS[base](question, depth, variant, model_name, exclude)
    if not candidates:
        return []
    scores = rerank_scores(question, [h.metin for h in candidates], rerank_model)
    order = np.argsort(-scores)[:k]
    return [
        replace(
            candidates[i],
            skor=float(scores[i]),
            rerank_skor=float(scores[i]),
            aday_sira=i + 1,  # candidates are already in fused rank order
        )
        for i in order
    ]


RETRIEVERS = {
    "dense": search_dense,
    "bm25": search_bm25,
    "hibrit": search_hybrid,
    "rerank": search_rerank,
}


def search(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
    retriever: str = RETRIEVER,
) -> list[Hit]:
    
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever: {retriever} (options: {list(RETRIEVERS)})")
    return RETRIEVERS[retriever](question, k, variant, model_name, exclude)
