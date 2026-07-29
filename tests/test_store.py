import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import app.retrieval as R
import app.store as S
from app.config import RUNTIME_VARIANT

DIM = 8


def fake_embed(chunks, model_name=None):
    rng = np.random.default_rng(len(chunks))
    vectors = rng.normal(size=(len(chunks), DIM)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def fake_pages(n_pages, seed):
    # Distinct sentences so chunk_document produces at least one chunk per page.
    return [
        (f"Belge {seed} sayfa {p} cümlesi. " * 40, "text_layer")
        for p in range(n_pages)
    ]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ingest.INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr("app.store.UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr("app.store.embed", fake_embed)
    S.invalidate()  # earlier tests may have cached an index for this variant
    yield S
    S.invalidate()


def add(store, monkeypatch, name, n_pages, seed):
    monkeypatch.setattr("app.store.extract_pages", lambda path: fake_pages(n_pages, seed))
    return store.add_document(Path(name))


def test_create_append_replace(store, monkeypatch):
    assert not store.has_index()
    assert store.list_documents() == []

    info = add(store, monkeypatch, "a.pdf", n_pages=2, seed="A")
    assert store.has_index()
    assert (info["sayfa"], info["ocr_sayfa"], info["guncellendi"]) == (2, 0, False)
    assert info["chunk"] >= 1

    add(store, monkeypatch, "b.pdf", n_pages=1, seed="B")
    docs = store.list_documents()
    assert [d["dosya"] for d in docs] == ["a.pdf", "b.pdf"]

    # Re-upload of a.pdf replaces its chunks (and moves it to the end).
    info = add(store, monkeypatch, "a.pdf", n_pages=3, seed="A2")
    assert info["guncellendi"] is True
    docs = {d["dosya"]: d["chunk"] for d in store.list_documents()}
    assert set(docs) == {"a.pdf", "b.pdf"}

    # Vectors and chunks stay aligned through append + replace.
    index = R.load_index(RUNTIME_VARIANT)
    assert index.vectors.shape == (sum(docs.values()), DIM)
    assert len(index.chunks) == sum(docs.values())


def test_invalidate_lets_search_see_new_documents(store, monkeypatch):
    add(store, monkeypatch, "a.pdf", n_pages=1, seed="A")
    before = len(R.load_index(RUNTIME_VARIANT))

    # add_document calls invalidate(), so the cached Index must not survive.
    add(store, monkeypatch, "b.pdf", n_pages=1, seed="B")
    after = len(R.load_index(RUNTIME_VARIANT))
    assert after > before


def test_empty_document_is_rejected(store, monkeypatch):
    monkeypatch.setattr("app.store.extract_pages", lambda path: [("", "text_layer")])
    with pytest.raises(ValueError):
        store.add_document(Path("bos.pdf"))
    assert not store.has_index()


def test_clear_index_removes_docs_and_uploads(store, monkeypatch, tmp_path):
    add(store, monkeypatch, "a.pdf", n_pages=1, seed="A")
    store.save_upload("a.pdf", b"data")
    R.load_index(RUNTIME_VARIANT)  # warm the cache so clear must invalidate it

    assert store.clear_index() == 1
    assert not store.has_index()
    assert store.list_documents() == []
    assert not (tmp_path / "uploads").exists()
    with pytest.raises(FileNotFoundError):
        R.load_index(RUNTIME_VARIANT)


def test_save_upload_strips_directories(store, tmp_path):
    path = store.save_upload("../evil/../rapor.pdf", b"data")
    assert path.parent == tmp_path / "uploads"
    assert path.name == "rapor.pdf"
    assert path.read_bytes() == b"data"
