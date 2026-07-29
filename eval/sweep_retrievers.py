import argparse
import json
from collections import defaultdict
from functools import partial

from app.config import (
    DEFAULT_VARIANT,
    EMBEDDING_MODEL,
    ROOT,
    model_slug,
)
from app.retrieval import search_bm25, search_dense, search_hybrid
from eval_retrieval import _mean, measure

CROSS_LINGUAL = ("q061", "q062", "q063")


def summary(m: dict) -> str:
    k = m["k"]
    return (
        f"{m['recall_at'][k] * 100:6.1f}% {m['recall_tr'] * 100:6.1f}% "
        f"{m['recall_en'] * 100:6.1f}% {m['mrr']:6.2f}"
    )


def cross_lingual_recall(m: dict) -> float:
    rows = [r for r in m["answerable"] if r["id"] in CROSS_LINGUAL]
    return _mean(m["scored"][r["id"]]["recall"] for r in rows)


def header(title: str, width: int = 92) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def table_head(label: str, width: int = 92) -> None:
    print(f"{label:34} {'r@5':>7} {'TR':>7} {'EN':>7} {'MRR':>6} {'çapraz':>8}")
    print("-" * width)


def row(label: str, m: dict) -> None:
    print(f"{label:34} {summary(m)} {cross_lingual_recall(m) * 100:7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrievers and fusion settings.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run = partial(measure, args.variant, args.model, args.k)
    results = {}

    # --- baselines ------------------------------------------------------------
    dense = run("dense", search_dense)
    results["dense"] = dense

    # --- axis 1: tokenizer, BM25 alone ---------------------------------------
    header("EKSEN 1 — BM25 TOKENIZER (BM25 tek başına; hibrit içinde fark maskelenir)")
    table_head("tokenizer")
    tokenizers = {}
    notes = {"split": r"düz \w+", "keep": "sayı/tanımlayıcı bütün"}
    for token in ("split", "keep"):
        m = run(f"bm25_{token}", partial(search_bm25, token=token))
        tokenizers[token] = m
        results[f"bm25_{token}"] = m
        row(f"{token}  ({notes[token]})", m)
    # A silent max() would hand back an arbitrary winner when the two tie, and on
    # this golden set they do: that tie is the finding, not a detail to smooth over.
    scores = {t: (m["recall_tr"], m["recall_en"], m["mrr"]) for t, m in tokenizers.items()}
    top = max(scores.values())
    tied = [t for t, s in scores.items() if s == top]
    best_token = tied[0]
    print(
        f"\nkazanan: {best_token}" if len(tied) == 1
        else f"\nBERABERE ({', '.join(tied)}) — golden set bu ekseni ayırt etmiyor; "
             f"{best_token} kullanılıyor"
    )

    # --- axis 2: fusion settings ---------------------------------------------
    header("EKSEN 2 — FÜZYON AYARLARI (RRF)")
    table_head("depth / k_const / ağırlık (d:b)")
    best, best_key = None, None
    for depth in (10, 20, 50):
        for k_const in (10, 60):
            for weights in ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0)):
                label = f"hibrit_d{depth}_k{k_const}_w{weights[0]:.0f}-{weights[1]:.0f}"
                m = run(label, partial(
                    search_hybrid, depth=depth, k_const=k_const,
                    weights=weights, token=best_token,
                ))
                results[label] = m
                row(f"{depth:>3} / {k_const:>3} / {weights[0]:.0f}:{weights[1]:.0f}", m)
                # Ranked on the worst-case language, the rule the project already
                # uses; the cross-lingual rows break ties because that is where
                # the hybrid is most at risk of losing ground dense already holds.
                key = (m["recall_tr"], cross_lingual_recall(m), m["mrr"])
                if best is None or key > best_key:
                    best, best_key = m, key

    # --- axis 3: head to head -------------------------------------------------
    header("EKSEN 3 — KARŞILAŞTIRMA")
    table_head("retriever")
    row("dense", dense)
    row(f"bm25 ({best_token})", tokenizers[best_token])
    row(best["retriever"], best)

    print(
        f"\nen iyi hibrit ayarı: {best['retriever']}\n"
        f"dense'e karşı: r@5 {(best['recall_at'][args.k] - dense['recall_at'][args.k]) * 100:+.1f} "
        f"puan, TR {(best['recall_tr'] - dense['recall_tr']) * 100:+.1f}, "
        f"EN {(best['recall_en'] - dense['recall_en']) * 100:+.1f}, "
        f"çapraz {(cross_lingual_recall(best) - cross_lingual_recall(dense)) * 100:+.1f}"
    )

    # --- paired per-question diff --------------------------------------------
    # The aggregate can stay flat while one question breaks and another starts
    # working; only the paired view shows that, and n=20 makes each question worth
    # 5 points, so a small aggregate win may be a single question of luck.
    header("SORU BAZLI FARK — en iyi hibrit vs dense")
    moved = defaultdict(list)
    for r in dense["answerable"]:
        before = dense["scored"][r["id"]]["recall"]
        after = best["scored"][r["id"]]["recall"]
        if abs(before - after) > 1e-9:
            direction = "kazanç" if after > before else "KAYIP"
            moved[direction].append(
                f"  {r['id']} [{r['dil']}/{r['kategori']}] "
                f"{before * 100:.0f}% -> {after * 100:.0f}%"
            )
    for direction in ("kazanç", "KAYIP"):
        print(f"{direction}: {len(moved[direction])}")
        for line in moved[direction]:
            print(line)
    if not moved:
        print("  hiçbir soru değişmedi")

    # --- cross-lingual regression watch --------------------------------------
    # Dense holds these at 100% and BM25 scores exactly 0 term overlap on them by
    # construction, so they are where fusion can only do damage.
    header("ÇAPRAZ DİL NÖBETİ — BM25'in yapısal olarak katkı veremediği satırlar")
    print(f"{'soru':8} {'dense':>8} {'bm25':>8} {'hibrit':>8}")
    print("-" * 92)
    for qid in CROSS_LINGUAL:
        cells = [
            f"{run_result['scored'][qid]['recall'] * 100:7.0f}%"
            if qid in run_result["scored"] else "      -"
            for run_result in (dense, tokenizers[best_token], best)
        ]
        print(f"{qid:8} " + " ".join(cells))

    # --- persist --------------------------------------------------------------
    # Only the sweep table. The per-retriever files under
    # out/retrieval/<variant>/<model>/ stay canonical and are written by
    # eval_retrieval.py with the frozen settings; recording the sweep's own labelled
    # variants beside them would leave two files claiming to be the same run.
    out = ROOT / "out" / "retrieval" / f"sweep_{model_slug(args.model)}.json"
    out.write_text(json.dumps({
        "variant": args.variant, "model": args.model, "k": args.k,
        "best_token": best_token, "best_hybrid": best["retriever"],
        "runs": {
            name: {
                "recall_at5": m["recall_at"][args.k],
                "recall_tr": m["recall_tr"],
                "recall_en": m["recall_en"],
                "mrr": m["mrr"],
                "capraz_dil": cross_lingual_recall(m),
                "auc": m["auc"],
                "auc_dense": m["auc_dense"],
            }
            for name, m in results.items()
        },
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nrecorded: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
