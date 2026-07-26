import argparse
import bisect
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from app.config import (
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    DEFAULT_VARIANT,
    DOC_LANG,
    EMBEDDING_MODEL,
    INDEX_DIR,
    SAMPLES,
    corpus,
    model_slug,
    prefixes,
)
from app.needs_ocr import needs_ocr
from app.ocr import is_photo, method_ocr_preprocessed, method_ocr_raw, page_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# A sentence end followed by the start of a new one. Turkish capitals are listed
# explicitly — [A-Z] alone would miss "Öğrenci", "İlgili", "Şu".
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9(])")


@dataclass
class Chunk:
    chunk_id: str
    dosya: str
    sayfa_baslangic: int      # 1-indexed, like the golden set's anchors
    sayfa_bitis: int
    yontem: str               # text_layer | ocr_raw | ocr_preprocessed | mixed
    dil: str
    metin: str


def normalize(text: str) -> str:
    """NFKC, de-hyphenate, collapse whitespace.

    NFKC rather than the bake-off's NFC: typographic ligatures survive the text
    layer (arXiv stores "ﬁlter" as U+FB01), so without the compatibility fold a
    search for "filter" never matches that chunk. evaluate.py stays on NFC —
    there the ligature is a real difference between two extractions, and folding
    it away would hide it.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n\s*", "", text)   # de-hyphenate word breaks
    text = re.sub(r"\s+", " ", text)        # collapse all whitespace
    return text.strip()


def extract_image(img) -> tuple[str, str]:
    """OCR one image. The photo gate decides whether preprocessing runs."""
    if is_photo(img):
        return method_ocr_preprocessed(img)[0], "ocr_preprocessed"
    return method_ocr_raw(img)[0], "ocr_raw"


def extract_pages(path: Path) -> list[tuple[str, str]]:
    """Text and extraction method per page, in page order."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        text, method = extract_image(cv2.imread(str(path)))
        return [(normalize(text), method)]

    pages = []
    doc = fitz.open(path)
    for page in doc:
        ocr, _ = needs_ocr(page)
        if ocr:
            text, method = extract_image(page_image(page))
        else:
            text, method = page.get_text("text"), "text_layer"
        pages.append((normalize(text), method))
    doc.close()
    return pages


def _stitch(pages: list[tuple[str, str]]) -> tuple[str, list[int], list[str]]:
    """Join the page texts into one stream, keeping each page's start offset.

    Chunking runs over the whole document rather than page by page, so a sentence
    that runs across a page break stays whole (Day 1 finding); the page number is
    recovered from the offset afterwards.
    """
    parts, starts, methods = [], [], []
    offset = 0
    for text, method in pages:
        starts.append(offset)
        methods.append(method)
        parts.append(text)
        offset += len(text) + 1          # +1 for the joining space
    return " ".join(parts), starts, methods


def _sentences(text: str) -> list[tuple[int, str]]:
    """(start offset, sentence) pairs, none longer than CHUNK_TARGET.

    Sentence granularity alone leaves pieces that are far too long: a definitions
    article carries no sentence end at all. Those are cut at word boundaries, so
    a chunk can still be bounded without splitting a word.
    """
    out, pos = [], 0
    for piece in SENTENCE_SPLIT.split(text):
        start = text.find(piece, pos)
        if start < 0:                    # the split never reorders, but stay safe
            start = pos
        for offset, part in _split_long(piece, start):
            out.append((offset, part))
        pos = start + len(piece)
    return out


def _split_long(sentence: str, start: int) -> list[tuple[int, str]]:
    if len(sentence) <= CHUNK_TARGET:
        return [(start, sentence)]
    parts, buffer, offset = [], [], start
    length = 0
    for word in sentence.split(" "):
        if length + len(word) + 1 > CHUNK_TARGET and buffer:
            body = " ".join(buffer)
            parts.append((offset, body))
            offset += len(body) + 1
            buffer, length = [], 0
        buffer.append(word)
        length += len(word) + 1
    if buffer:
        parts.append((offset, " ".join(buffer)))
    return parts


def chunk_document(name: str, pages: list[tuple[str, str]]) -> list[Chunk]:
    text, starts, methods = _stitch(pages)
    if not text:
        return []

    def page_of(offset: int) -> int:
        return bisect.bisect_right(starts, offset) - 1

    chunks: list[Chunk] = []
    buffer: list[tuple[int, str]] = []    # sentences waiting to become a chunk
    length = 0

    def flush() -> None:
        nonlocal buffer, length
        if not buffer:
            return
        body = " ".join(s for _, s in buffer)
        first, last = buffer[0][0], buffer[-1][0] + len(buffer[-1][1]) - 1
        p_start, p_end = page_of(first), page_of(last)
        used = sorted(set(methods[p_start:p_end + 1]))
        chunks.append(Chunk(
            chunk_id=f"{Path(name).stem}#{len(chunks):04d}",
            dosya=name,
            sayfa_baslangic=p_start + 1,
            sayfa_bitis=p_end + 1,
            yontem=used[0] if len(used) == 1 else "mixed",
            dil=DOC_LANG.get(name, "??"),
            metin=body,
        ))
        # Overlap: hand the tail sentences of this chunk to the next one, so a
        # fact sitting on a chunk boundary stays whole in at least one of them.
        tail, tail_len = [], 0
        for item in reversed(buffer):
            if tail_len >= CHUNK_OVERLAP:
                break
            tail.insert(0, item)
            tail_len += len(item[1]) + 1
        buffer = tail if len(tail) < len(buffer) else []
        length = sum(len(s) + 1 for _, s in buffer)

    for start, sentence in _sentences(text):
        if buffer and length + len(sentence) > CHUNK_MAX:
            flush()
        buffer.append((start, sentence))
        length += len(sentence) + 1
        if length >= CHUNK_TARGET:
            flush()

    if buffer:
        # A short remainder joins the previous chunk instead of standing alone.
        body = " ".join(s for _, s in buffer)
        if chunks and len(body) < CHUNK_MIN:
            last = chunks[-1]
            last.metin = f"{last.metin} {body}".strip()
            last.sayfa_bitis = page_of(buffer[-1][0]) + 1
        else:
            flush()
    return chunks


def embed(chunks: list[Chunk], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _, passage_prefix = prefixes(model_name)
    texts = [passage_prefix + c.metin for c in chunks]
    return model.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def index_dir(variant: str = DEFAULT_VARIANT, model_name: str = EMBEDDING_MODEL) -> Path:
    """One index per (variant, model) — comparing models must not overwrite."""
    return INDEX_DIR / Path(variant).stem / model_slug(model_name)


def ingest(variant: str = DEFAULT_VARIANT, model_name: str = EMBEDDING_MODEL) -> Path:
    files = corpus(variant)
    all_chunks: list[Chunk] = []
    started = time.perf_counter()

    print(f"{'document':26} {'pages':>6} {'ocr':>5} {'chunks':>7} {'time':>8}")
    print("-" * 58)
    for name in files:
        t0 = time.perf_counter()
        pages = extract_pages(SAMPLES / name)
        chunks = chunk_document(name, pages)
        all_chunks.extend(chunks)
        ocr_pages = sum(1 for _, m in pages if m != "text_layer")
        print(
            f"{name:26} {len(pages):6} {ocr_pages:5} {len(chunks):7} "
            f"{time.perf_counter() - t0:7.1f}s"
        )

    print(f"\nembedding {len(all_chunks)} chunks with {model_name} ...")
    t0 = time.perf_counter()
    vectors = embed(all_chunks, model_name)
    embed_time = time.perf_counter() - t0

    out = index_dir(variant, model_name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    (out / "chunks.json").write_text(
        json.dumps([asdict(c) for c in all_chunks], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out / "meta.json").write_text(
        json.dumps({
            "variant": variant,
            "documents": files,
            "model": model_name,
            "dim": int(vectors.shape[1]),
            "chunks": len(all_chunks),
            "chunk_target": CHUNK_TARGET,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_min": CHUNK_MIN,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(
        f"embedding {embed_time:.1f}s | total {time.perf_counter() - started:.1f}s\n"
        f"index: {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval index.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    args = parser.parse_args()
    ingest(args.variant, args.model)


if __name__ == "__main__":
    main()
