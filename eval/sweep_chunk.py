import argparse
import json

import app.ingest as ingest_mod
from app import store
from app.config import DEFAULT_VARIANT, EMBEDDING_MODEL, ROOT, model_slug
from app.ingest import ingest
from eval_retrieval import measure

TARGETS = (600, 800, 1000, 1200)
OVERLAPS = (100, 150, 200)
BASELINE = (800, 150)   # the frozen config, the row every other row is judged against

RESULTS_DIR = ROOT / "out" / "chunk_sweep"


def capraz_recall(m: dict) -> float:
    """recall@k on the capraz_dil category alone: n=3, the most fragile column,
    where a single question moves the number 33 points."""
    rows = [r for r in m["answerable"] if r["kategori"] == "capraz_dil"]
    if not rows:
        return 0.0
    return sum(m["scored"][r["id"]]["recall"] for r in rows) / len(rows)


def run_combo(target: int, overlap: int, variant: str, model: str, k: int) -> dict:
    """Rebuild the index with (target, overlap) and measure dense retrieval.

    The chunk constants are module globals in app.ingest (bound by value at import),
    so they are patched THERE, not in app.config; patching config would not reach
    the functions that already captured the names.
    """
    ingest_mod.CHUNK_TARGET = target
    ingest_mod.CHUNK_OVERLAP = overlap
    ingest_mod.CHUNK_MAX = 2 * target
    print(f"\n### indexing {variant} — target {target} / overlap {overlap} / max {2 * target}")
    ingest(variant, model)                       # overwrites index_dir(variant, model)
    store.invalidate()
    m = measure(variant, model, k, retriever="dense")
    return {
        "target": target,
        "overlap": overlap,
        "chunks": len(m["index"]),
        "recall_at": {kk: m["recall_at"][kk] for kk in m["recall_at"]},
        "recall_tr": m["recall_tr"],
        "recall_en": m["recall_en"],
        "capraz": capraz_recall(m),
        "mrr": m["mrr"],
        "auc": m["auc"],
        # per-question recall, kept so the aggregate can be checked question by question
        "scored": {rid: s["recall"] for rid, s in m["scored"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep chunk target/overlap on the golden set (dense only)."
    )
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    original = (ingest_mod.CHUNK_TARGET, ingest_mod.CHUNK_OVERLAP, ingest_mod.CHUNK_MAX)

    combos = [(t, o) for t in TARGETS for o in OVERLAPS if o < t]
    runs = []
    try:
        for target, overlap in combos:
            runs.append(run_combo(target, overlap, args.variant, args.model, args.k))
    finally:
        ingest_mod.CHUNK_TARGET, ingest_mod.CHUNK_OVERLAP, ingest_mod.CHUNK_MAX = original
        print(f"\n### restoring frozen config {original[0]}/{original[1]} (max {original[2]}) "
              f"and rebuilding {args.variant}")
        ingest(args.variant, args.model)
        store.invalidate()

    k = args.k
    print()
    print("=" * 86)
    print(f"CHUNK SWEEP — model {args.model}, variant {args.variant}, DENSE, k={k}")
    print("=" * 86)
    print(f"{'target/overlap':>16} {'chunks':>7} {f'r@{k}':>7} {'TR':>7} {'EN':>7} "
          f"{'çapraz':>7} {'MRR':>6} {'AUC':>6}")
    print("-" * 86)
    for r in runs:
        mark = " *" if (r["target"], r["overlap"]) == BASELINE else "  "
        print(
            f"{r['target']:>7}/{r['overlap']:<4}{mark:>4} {r['chunks']:7} "
            f"{r['recall_at'][k] * 100:6.1f}% {r['recall_tr'] * 100:6.1f}% "
            f"{r['recall_en'] * 100:6.1f}% {r['capraz'] * 100:6.1f}% "
            f"{r['mrr']:6.2f} {r['auc']:6.3f}"
        )
    print("\n  * = frozen config (baseline)")

    # Per-question diff vs baseline: an unchanged r@5 can still hide a question that
    # broke and another that started working (the project's per-question rule).
    base = next((r for r in runs if (r["target"], r["overlap"]) == BASELINE), None)
    if base is not None:
        print()
        print(f"PER-QUESTION CHANGES vs baseline {BASELINE[0]}/{BASELINE[1]} (recall@{k})")
        print("-" * 86)
        any_change = False
        for r in runs:
            if (r["target"], r["overlap"]) == BASELINE:
                continue
            for rid, after in r["scored"].items():
                before = base["scored"][rid]
                if abs(before - after) > 1e-9:
                    any_change = True
                    print(f"  {r['target']:>4}/{r['overlap']:<3}  {rid}: "
                          f"{before * 100:.0f}% -> {after * 100:.0f}%")
        if not any_change:
            print("  none — every question keeps the same recall across all combos")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.variant.replace('.pdf', '')}_{model_slug(args.model)}.json"
    out.write_text(
        json.dumps({"model": args.model, "variant": args.variant, "k": k,
                    "baseline": list(BASELINE), "runs": runs},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nrecorded: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
