import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.config import (
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    EMBEDDING_MODEL,
    RUNTIME_VARIANT,
    UPLOADS_DIR,
)
from app.ingest import chunk_document, embed, extract_pages, index_dir


def runtime_dir(model_name: str = EMBEDDING_MODEL) -> Path:
    return index_dir(RUNTIME_VARIANT, model_name)


def has_index(model_name: str = EMBEDDING_MODEL) -> bool:
    return (runtime_dir(model_name) / "vectors.npy").exists()


def invalidate() -> None:
    """Drop retrieval's cached Index/BM25 so the next search sees the new chunks.
    embed_query's cache is keyed on the question alone and stays valid.
    """
    from app.retrieval import load_bm25, load_index

    load_index.cache_clear()
    load_bm25.cache_clear()


def list_documents(model_name: str = EMBEDDING_MODEL) -> list[dict]:
    """Indexed documents in upload order: [{"dosya":..., "chunk": int}]."""
    path = runtime_dir(model_name) / "chunks.json"
    if not path.exists():
        return []
    counts: dict[str, int] = {}
    for c in json.loads(path.read_text(encoding="utf-8")):
        counts[c["dosya"]] = counts.get(c["dosya"], 0) + 1
    return [{"dosya": d, "chunk": n} for d, n in counts.items()]


def add_document(path: Path, model_name: str = EMBEDDING_MODEL) -> dict:
    """Extract, chunk and embed ONE document, then merge it into the runtime index.
    A file with the same name replaces its previous chunks (re-upload = update).
    """
    started = time.perf_counter()
    pages = extract_pages(path)
    chunks = chunk_document(path.name, pages)
    if not chunks:
        raise ValueError(f"{path.name}: no extractable text (empty or unreadable)")
    vectors = embed(chunks, model_name)

    out = runtime_dir(model_name)
    old_chunks: list[dict] = []
    old_vectors = np.empty((0, vectors.shape[1]), dtype=vectors.dtype)
    if (out / "vectors.npy").exists():
        old_vectors = np.load(out / "vectors.npy")
        old_chunks = json.loads((out / "chunks.json").read_text(encoding="utf-8"))

    keep = [i for i, c in enumerate(old_chunks) if c["dosya"] != path.name]
    replaced = len(keep) != len(old_chunks)
    all_chunks = [old_chunks[i] for i in keep] + [asdict(c) for c in chunks]
    all_vectors = np.concatenate([old_vectors[keep], vectors])

    documents = list(dict.fromkeys(c["dosya"] for c in all_chunks))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", all_vectors)
    (out / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    (out / "meta.json").write_text(
        json.dumps({
            "variant": RUNTIME_VARIANT,
            "documents": documents,
            "model": model_name,
            "dim": int(all_vectors.shape[1]),
            "chunks": len(all_chunks),
            "chunk_target": CHUNK_TARGET,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_min": CHUNK_MIN,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    invalidate()

    return {
        "dosya": path.name,
        "sayfa": len(pages),
        "ocr_sayfa": sum(1 for _, m in pages if m != "text_layer"),
        "chunk": len(chunks),
        "sure": time.perf_counter() - started,
        "guncellendi": replaced,
    }


def clear_index(model_name: str = EMBEDDING_MODEL) -> int:
    """Delete the runtime index and the stored uploads; returns the removed doc count.
    Touches only the web UI's own directories (runtime_dir + UPLOADS_DIR); the
    measured corpus indexes under other variants stay untouched.
    """
    removed = len(list_documents(model_name))
    shutil.rmtree(runtime_dir(model_name), ignore_errors=True)
    shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
    invalidate()
    return removed


def save_upload(name: str, data: bytes) -> Path:
    """Persist an uploaded file under UPLOADS_DIR and return its path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOADS_DIR / Path(name).name   # strip any client-side directory parts
    target.write_bytes(data)
    return target
