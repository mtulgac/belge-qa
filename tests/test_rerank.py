import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))   # the app package
sys.path.insert(0, str(_ROOT / "eval"))  # eval_retrieval / sweep_rerank under eval/

import app.models as M
import app.retrieval as R
from app.config import DEFAULT_VARIANT, EMBEDDING_MODEL

QUESTION = "lisans programına yerleşme koşulları nedir"


def _fake_embed(question, model_name=EMBEDDING_MODEL):
    # Deterministic per question: the real embed_query is pure, so the base call
    # inside search_rerank and any call we make must return the same vector.
    dim = R.load_index(DEFAULT_VARIANT, EMBEDDING_MODEL).vectors.shape[1]
    seed = abs(hash(question)) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def _fake_rerank(question, texts, model_name=M.RERANK_MODEL):
    # Score = chunk length: predictable, and unrelated to the base ordering, so a
    # real reordering is observable. Real CE logits are continuous and never tie.
    return np.array([float(len(t)) for t in texts], dtype=np.float64)


@contextmanager
def _patched():
    embed, rerank = R.embed_query, M.rerank_scores
    R.embed_query = _fake_embed
    M.rerank_scores = _fake_rerank
    try:
        yield
    finally:
        R.embed_query, M.rerank_scores = embed, rerank


def test_orders_by_rerank_score_and_reorders():
    with _patched():
        for base in ("hibrit", "dense"):
            cands = R.BASE_RETRIEVERS[base](QUESTION, 20, DEFAULT_VARIANT, EMBEDDING_MODEL)
            hits = R.search_rerank(QUESTION, 5, DEFAULT_VARIANT, EMBEDDING_MODEL,
                                   depth=20, base=base)
            assert len(hits) == 5, base
            scores = [h.rerank_skor for h in hits]
            assert scores == sorted(scores, reverse=True), (base, scores)
            assert all(h.skor == h.rerank_skor for h in hits), base
            # top-5 by the fake score (chunk length), tie-robust: compare multiset
            top5 = sorted((len(c.metin) for c in cands), reverse=True)[:5]
            assert sorted(scores, reverse=True) == top5, base
            assert [h.chunk_id for h in hits] != [c.chunk_id for c in cands][:5], base


def test_preserves_provenance_and_ablation_shape():
    with _patched():
        hyb = R.search_rerank(QUESTION, 5, DEFAULT_VARIANT, EMBEDDING_MODEL,
                              depth=20, base="hibrit")
        dns = R.search_rerank(QUESTION, 5, DEFAULT_VARIANT, EMBEDDING_MODEL,
                              depth=20, base="dense")
        for h in hyb:
            assert h.aday_sira is not None and 1 <= h.aday_sira <= 20
        # Dense candidates carry no BM25 provenance, the ablation source is visible.
        assert all(h.bm25_skor is None for h in dns)
        assert all(h.dense_skor is not None for h in dns)


def test_measure_record_roundtrip(tmp_path=None):
    from eval_retrieval import measure, record

    with _patched():
        m = measure(DEFAULT_VARIANT, EMBEDDING_MODEL, 5, "rerank",
                    search_fn=partial(R.search_rerank, depth=20))
        assert 0.0 <= m["recall_at"][5] <= 1.0
        assert 0.0 <= m["auc"] <= 1.0
        path = record(m)
        try:
            assert path.name == "rerank_k5.json", path.name
        finally:
            # This run used a FAKE embedder, must never leave a bogus canonical file.
            path.unlink()


def test_paired_abstention_math():
    from sweep_rerank import paired_abstention

    rows = ([{"id": f"y{i}", "beklenen": "yanitla"} for i in range(20)]
            + [{"id": f"n{j}", "beklenen": "reddet"} for j in range(10)])
    rerank = {**{f"y{i}": 0.8 + 0.01 * i for i in range(20)},
              **{f"n{j}": 0.2 + 0.01 * j for j in range(10)}}   # clean split
    cosine = {**{f"y{i}": 0.5 + 0.02 * (i % 5) for i in range(20)},
              **{f"n{j}": 0.45 + 0.02 * (j % 5) for j in range(10)}}  # noisy
    a = paired_abstention(rerank, cosine, rows, bootstrap=2000)
    assert a["auc_rerank"] == 1.0, a["auc_rerank"]
    assert abs((a["auc_rerank"] - a["auc_cosine"]) - a["delta"]) < 1e-9
    lo, hi = a["ci95"]
    assert lo <= a["delta"] <= hi
    # Reranker strictly dominates here: it wins discordant pairs, cosine wins none.
    assert a["rerank_only"] > 0 and a["cosine_only"] == 0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\n{len(tests)} rerank wiring checks passed")


if __name__ == "__main__":
    _main()
