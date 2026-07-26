import argparse

from app.config import DEFAULT_VARIANT, EMBEDDING_MODEL, VARIANT_GROUP
from app.ingest import index_dir, ingest
from eval_retrieval import measure, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare format variants.")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    runs = {}
    for variant in VARIANT_GROUP:
        if not (index_dir(variant, args.model) / "vectors.npy").exists():
            print(f"\n### indexing {variant} with {args.model}")
            ingest(variant, args.model)
        runs[variant] = measure(variant, args.model, args.k)
        record(runs[variant])

    k = args.k
    print()
    print("=" * 88)
    print(f"FORMAT VARIANTS — model {args.model}, k={k}")
    print("=" * 88)
    print(
        f"{'variant':22} {'ocr chunks':>11} {f'r@{k}':>7} {'TR':>7} {'EN':>7} "
        f"{'MRR':>6} {'AUC':>6}"
    )
    print("-" * 88)
    for variant, m in runs.items():
        ocr_chunks = sum(1 for c in m["index"].chunks if c["yontem"] != "text_layer")
        print(
            f"{variant:22} {ocr_chunks:11} {m['recall_at'][k] * 100:6.1f}% "
            f"{m['recall_tr'] * 100:6.1f}% {m['recall_en'] * 100:6.1f}% "
            f"{m['mrr']:6.2f} {m['auc']:6.3f}"
        )
    print()

    # Per-question diff against the default variant: an unchanged aggregate can
    # still hide a question that broke and another that started working.
    base = runs[DEFAULT_VARIANT]
    print(f"PER-QUESTION CHANGES vs {DEFAULT_VARIANT}")
    print("-" * 88)
    changed = False
    for variant, m in runs.items():
        if variant == DEFAULT_VARIANT:
            continue
        for row in m["answerable"]:
            before = base["scored"][row["id"]]["recall"]
            after = m["scored"][row["id"]]["recall"]
            if abs(before - after) > 1e-9:
                changed = True
                print(
                    f"  {variant:20} {row['id']} [{row['dil']}/{row['kategori']}] "
                    f"recall {before * 100:.0f}% -> {after * 100:.0f}%"
                )
    if not changed:
        print("  none — every question keeps the same result in all three formats")
    print()


if __name__ == "__main__":
    main()
