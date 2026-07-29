import argparse
import json
import statistics as stats
from time import perf_counter

import yaml

from app.config import (
    DEFAULT_VARIANT,
    EMBEDDING_MODEL,
    GOLDEN_SET,
    RERANK_CANDIDATES,
    RERANK_DEVICE,
    ROOT,
)
from app.models import _cross_encoder, rerank_scores
from app.retrieval import search_hybrid

HYBRID_MS = 34.0  # Day 4 median, so the delta a reranker adds is legible
OUT = ROOT / "out" / "retrieval" / "rerank_latency.json"


def build_candidates(variant: str, model_name: str, depth: int) -> list[tuple[str, list[str]]]:
    """(question, hybrid top-`depth` chunk texts) for every golden question."""
    rows = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        hits = search_hybrid(
            r["soru"], depth, variant, model_name, tuple(r.get("izole") or [])
        )
        out.append((r["soru"], [h.metin for h in hits]))
    return out


def time_query(question: str, texts: list[str], repeats: int) -> float:
    """Median wall-clock ms of one rerank call over these candidates."""
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        rerank_scores(question, texts)
        samples.append((perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def bench_model(
    model_name: str,
    candidates: list[tuple[str, list[str]]],
    depths: list[int],
    repeats: int,
) -> dict:
    # Warm up: the first call downloads/loads the model and pays a one-time JIT
    # cost that would otherwise land on whichever query ran first.
    _cross_encoder(model_name)
    q0, t0 = candidates[0]
    rerank_scores(q0, t0[: depths[0]] or ["x"])

    by_depth = {}
    for depth in depths:
        per_query, per_pair = [], []
        for question, texts in candidates:
            sliced = texts[:depth]
            if not sliced:
                continue  # a short candidate list (nothing to rerank)
            ms = time_query(question, sliced, repeats)
            per_query.append(ms)
            per_pair.append(ms / len(sliced))
        by_depth[depth] = {
            "median_ms": stats.median(per_query),
            "mean_ms": stats.mean(per_query),
            "min_ms": min(per_query),
            "max_ms": max(per_query),
            "per_pair_ms": stats.median(per_pair),
            "n_queries": len(per_query),
        }
    return by_depth


def main() -> None:
    p = argparse.ArgumentParser(description="Reranker CPU latency (step 1).")
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--model", default=EMBEDDING_MODEL, help="embedding model for candidates")
    p.add_argument("--depths", default="5,10,20")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--models", default=",".join(RERANK_CANDIDATES),
                   help="comma-separated reranker candidates")
    args = p.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"building real hybrid candidates (depth {max(depths)})…")
    candidates = build_candidates(args.variant, args.model, max(depths))

    results = {}
    for model_name in models:
        print(f"\n### {model_name}")
        try:
            by_depth = bench_model(model_name, candidates, depths, args.repeats)
        except Exception as e:
            # A candidate that will not load on this stack is a real elimination,
            # not a reason to lose the models that do load (jina-reranker-v2's
            # remote code is incompatible with the pinned transformers).
            print(f"  ATLANDI — yüklenemedi: {type(e).__name__}: {e}")
            results[model_name] = {"error": f"{type(e).__name__}: {e}"}
            _cross_encoder.cache_clear()
            continue
        results[model_name] = by_depth
        # Free the ~0.5-2 GB before the next candidate loads.
        _cross_encoder.cache_clear()
        for depth in depths:
            d = by_depth[depth]
            print(
                f"  depth {depth:>2}: rerank {d['median_ms']:7.1f} ms "
                f"(pair {d['per_pair_ms']:5.1f} ms) "
                f"-> query total ~{HYBRID_MS + d['median_ms']:7.1f} ms "
                f"(+{d['median_ms'] / HYBRID_MS * 100:5.0f}% over hybrid)"
            )

    print("\n" + "=" * 92)
    print(f"RERANKER LATENCY — device={RERANK_DEVICE}, {len(candidates)} golden "
          f"questions, real chunk lengths")
    print(f"(the +hybrid ~{HYBRID_MS:.0f} ms is the embedding cost, measured device-unpinned "
          f"— see the embedding-device caveat)")
    print("=" * 92)
    print(f"{'model':46} {'depth':>5} {'rerank ms':>10} {'+hybrid':>9} {'pair ms':>8}")
    print("-" * 92)
    for model_name, by_depth in results.items():
        if "error" in by_depth:
            print(f"{model_name:46}   —  yüklenemedi ({by_depth['error']})")
            continue
        for depth in depths:
            d = by_depth[depth]
            print(
                f"{model_name:46} {depth:5} {d['median_ms']:10.1f} "
                f"{HYBRID_MS + d['median_ms']:8.1f}  {d['per_pair_ms']:8.1f}"
            )
    print("\nrerank ms = added on top of the ~34 ms hybrid; +hybrid = full query latency.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "variant": args.variant, "embedding_model": args.model,
        "device": RERANK_DEVICE, "hybrid_ms": HYBRID_MS, "repeats": args.repeats,
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nrecorded: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
