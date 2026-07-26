import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import DEFAULT_VARIANT, EMBEDDING_MODEL, prefixes
from app.ingest import index_dir


@dataclass
class Hit:
    skor: float
    chunk_id: str
    dosya: str
    sayfa_baslangic: int
    sayfa_bitis: int
    yontem: str
    dil: str
    metin: str


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
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_query(question: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    query_prefix, _ = prefixes(model_name)
    return _encoder(model_name).encode(
        [query_prefix + question], convert_to_numpy=True, normalize_embeddings=True,
    )[0]


def search(
    question: str,
    k: int = 5,
    variant: str = DEFAULT_VARIANT,
    model_name: str = EMBEDDING_MODEL,
    exclude: tuple[str, ...] = (),
) -> list[Hit]:
    index = load_index(variant, model_name)
    scores = index.vectors @ embed_query(question, model_name)
    if exclude:
        blocked = np.array([c["dosya"] in exclude for c in index.chunks])
        scores = np.where(blocked, -np.inf, scores)
    top = np.argsort(-scores)[:k]
    return [
        Hit(
            skor=float(scores[i]),
            metin=index.chunks[i]["metin"],
            **{
                key: index.chunks[i][key]
                for key in ("chunk_id", "dosya", "sayfa_baslangic", "sayfa_bitis", "yontem", "dil")
            },
        )
        for i in top
    ]
