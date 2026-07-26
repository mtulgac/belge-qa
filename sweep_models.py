import argparse
import time

from app.config import DEFAULT_VARIANT
from app.ingest import index_dir, ingest
from app.retrieval import embed_query
from eval_retrieval import _mean, anchor_rank, measure, record

# Candidates: multilingual, CPU-runnable, on-prem.
CANDIDATES = [
    "intfloat/multilingual-e5-small",    # 118M
    "intfloat/multilingual-e5-base",     # 278M
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 118M, different family
    "BAAI/bge-m3",                       # 568M, strong multilingual, slowest
]


def query_latency(rows: list[dict], model: str) -> float:
    """Milliseconds to embed one question — the cost the user waits for on every
    query, as opposed to the indexing cost they pay once per upload."""
    embed_query("ısınma", model)                      # warm up: model load excluded
    t0 = time.perf_counter()
    for row in rows:
        embed_query(row["soru"], model)
    return (time.perf_counter() - t0) / len(rows) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding models.")
    parser.add_argument("--models", nargs="+", default=CANDIDATES)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        path = index_dir(args.variant, model)
        t0 = time.perf_counter()
        if not (path / "vectors.npy").exists():
            print(f"\n### indexing with {model}")
            ingest(args.variant, model)
        else:
            print(f"\n### {model}: index exists, reusing")
        build_time = time.perf_counter() - t0

        try:
            m = measure(args.variant, model, args.k)
        except Exception as e:                # a broken model must not sink the sweep
            print(f"  ! {model} failed: {e}")
            continue

        cross = [r for r in m["answerable"] if r["kategori"] == "capraz_dil"]
        ranks = {
            r["id"]: anchor_rank(r, args.variant, model, len(m["index"])) for r in cross
        }
        latency = query_latency(m["rows"], model)
        # Only an index built in this run has a meaningful build time; a reused
        # one would report the cost of reading it back, not of embedding it.
        per_chunk = build_time / max(len(m["index"]), 1) if build_time > 1 else None
        record(m, extra={
            "ms_per_query": latency,
            "s_per_chunk": per_chunk,
            "capraz_dil_ranks": ranks,
        })
        rows.append({
            "model": model,
            "dim": m["index"].meta["dim"],
            "recall": m["recall_at"][args.k],
            "tr": m["recall_tr"],
            "en": m["recall_en"],
            "cross": _mean(m["scored"][r["id"]]["recall"] for r in cross),
            "mrr": m["mrr"],
            "auc": m["auc"],
            "overlap": m["overlap"],
            "latency": latency,
            "per_chunk": per_chunk,
            "ranks": ranks,
        })

    k = args.k
    print()
    print("=" * 100)
    print(f"EMBEDDING MODEL COMPARISON — variant {args.variant}, k={k}")
    print("=" * 100)
    print(
        f"{'model':46} {'dim':>5} {f'r@{k}':>7} {'TR':>7} {'EN':>7} "
        f"{'cross':>7} {'MRR':>6} {'AUC':>6} {'ovl':>5} {'ms/q':>7} {'s/chunk':>8}"
    )
    print("-" * 100)
    # Worst-case across languages: a model has to be usable in both, the same
    # anti-masking rule the OCR bake-off used.
    for r in sorted(rows, key=lambda r: -min(r["tr"], r["en"])):
        per_chunk = f"{r['per_chunk']:7.2f}s" if r["per_chunk"] else f"{'cached':>8}"
        print(
            f"{r['model']:46} {r['dim']:5} {r['recall'] * 100:6.1f}% {r['tr'] * 100:6.1f}% "
            f"{r['en'] * 100:6.1f}% {r['cross'] * 100:6.1f}% {r['mrr']:6.2f} "
            f"{r['auc']:6.3f} {r['overlap']:5} {r['latency']:6.0f}  {per_chunk}"
        )
    print()
    # Below the cut recall@k reports 0 for everyone; the rank says how far away
    # the correct chunk actually was. Printed from the same run as the table
    # above, so the two can never drift apart.
    ids = sorted({i for r in rows for i in r["ranks"]})
    print(f"CROSS-LINGUAL RANK OF THE CORRECT CHUNK  ({' / '.join(ids)}, k={k} is the cut)")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: -min(r["tr"], r["en"])):
        cells = " / ".join(
            f"{r['ranks'].get(i)}" if r["ranks"].get(i) else "yok" for i in ids
        )
        print(f"{r['model']:46} {cells}")
    print()
    print("cross   = recall on capraz_dil rows (TR question, EN document); n=3, so one")
    print("          question is 33 points — read it with the per-question ranks, not alone.")
    print("AUC     = P(a yanitla row outscores a reddet row), scale-free. The raw score gap")
    print("          is not comparable across models: e5 packs its scores into a narrow band.")
    print("ovl     = how many reddet rows sit inside the yanitla score range.")
    print("ms/q    = query embedding latency, paid on every question.")
    print("s/chunk = indexing cost, paid once per uploaded document — the same ingest-wait")
    print("          budget that ruled out VLM-based OCR on Day 1.")


if __name__ == "__main__":
    main()
