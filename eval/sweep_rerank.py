import argparse
import json
import random
import statistics as stats
from collections import defaultdict
from functools import partial

from app.config import (
    DEFAULT_VARIANT,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANK_DEPTH,
    ROOT,
    model_slug,
)
from app.models import _cross_encoder
from app.retrieval import search_dense, search_hybrid, search_rerank
from eval_retrieval import _mean, measure, separability

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


def top1_scores(m: dict) -> dict[str, float]:

    return {qid: (hits[0].skor if hits else 0.0) for qid, hits in m["hits"].items()}


def paired_abstention(
    rerank_score: dict, cosine_score: dict, rows: list, bootstrap: int, seed: int = 0
) -> dict:

    yes = [r["id"] for r in rows if r["beklenen"] == "yanitla"]
    no = [r["id"] for r in rows if r["beklenen"] == "reddet"]

    def auc(score: dict, ys: list, ns: list) -> float:
        return separability([score[q] for q in ys], [score[q] for q in ns])

    a_rerank = auc(rerank_score, yes, no)
    a_cosine = auc(cosine_score, yes, no)

    # Descriptive breakdown: pairs where exactly one signal orders yanitla above
    # reddet. Ties are vanishingly rare on continuous scores, so strict > is fine
    # here; the precise delta comes from the AUCs above, which do handle ties.
    rerank_only = cosine_only = 0
    discord = {"rerank": [], "cosine": []}
    for i in yes:
        for j in no:
            r_ok = rerank_score[i] > rerank_score[j]
            c_ok = cosine_score[i] > cosine_score[j]
            if r_ok and not c_ok:
                rerank_only += 1
                discord["rerank"].append((i, j))
            elif c_ok and not r_ok:
                cosine_only += 1
                discord["cosine"].append((i, j))

    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap):
        ys = [rng.choice(yes) for _ in yes]
        ns = [rng.choice(no) for _ in no]
        deltas.append(auc(rerank_score, ys, ns) - auc(cosine_score, ys, ns))
    deltas.sort()
    lo = deltas[int(0.025 * bootstrap)] if bootstrap else 0.0
    hi = deltas[min(int(0.975 * bootstrap), bootstrap - 1)] if bootstrap else 0.0

    return {
        "auc_rerank": a_rerank,
        "auc_cosine": a_cosine,
        "delta": a_rerank - a_cosine,
        "ci95": (lo, hi),
        "rerank_only": rerank_only,
        "cosine_only": cosine_only,
        "discord": discord,
        "yes_scores": {"rerank": [rerank_score[q] for q in yes],
                       "cosine": [cosine_score[q] for q in yes]},
        "no_scores": {"rerank": [rerank_score[q] for q in no],
                      "cosine": [cosine_score[q] for q in no]},
    }


def score_stats(values: list[float]) -> str:
    return (f"{min(values):8.3f} {stats.mean(values):8.3f} "
            f"{stats.median(values):8.3f} {max(values):8.3f}")


def threshold_confusion(yes: list[float], no: list[float]) -> dict:

    n_yes, n_no = len(yes), len(no)

    def confusion(t: float) -> tuple[int, int]:
        return sum(s >= t for s in no), sum(s < t for s in yes)  # false accept, reject

    fa50, fr50 = confusion(0.5)

    # Zero false-accept: threshold just above the highest reddet score.
    t_safe = max(no) + 1e-9
    kept = sum(s >= t_safe for s in yes)

    # Best in-sample accuracy over candidate thresholds at every observed score.
    best_t, best_acc = 0.0, 0.0
    for t in sorted(set(yes + no)):
        fa, fr = confusion(t)
        acc = (n_yes - fr + n_no - fa) / (n_yes + n_no)
        if acc > best_acc:
            best_t, best_acc = t, acc

    return {
        "at_0.5": {"false_accept": fa50, "false_reject": fr50},
        "zero_false_accept": {"threshold": t_safe, "answerable_kept": kept,
                              "answerable_total": n_yes},
        "best_accuracy": {"threshold": best_t, "accuracy": best_acc},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Reranker recall + abstention (steps 2-3).")
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--model", default=EMBEDDING_MODEL, help="embedding model")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--depth", type=int, default=RERANK_DEPTH)
    p.add_argument("--models", default=",".join(RERANK_CANDIDATES),
                   help="comma-separated reranker candidates")
    p.add_argument("--bases", default="hibrit,dense",
                   help="candidate sources to rerank (ablation axis)")
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()

    rerankers = [m.strip() for m in args.models.split(",") if m.strip()]
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    run = partial(measure, args.variant, args.model, args.k)

    # --- frozen baselines -----------------------------------------------------
    dense = run("dense", search_dense)
    hibrit = run("hibrit", search_hybrid)
    cosine_gate = top1_scores(dense)  # the Day-3 abstention signal

    width = 96
    print("=" * width)
    print(f"RERANK SWEEP — model {args.model}, variant {args.variant}, "
          f"depth {args.depth}, k={args.k}")
    print("=" * width)
    print(f"{'retriever':44} {'r@5':>7} {'TR':>7} {'EN':>7} {'MRR':>6} {'çapraz':>8}")
    print("-" * width)
    for label, m in (("dense (baz)", dense), ("hibrit (dondurulmuş)", hibrit)):
        print(f"{label:44} {summary(m)} {cross_lingual_recall(m) * 100:7.1f}%")

    # Keyed by (model, base). Loop model-outer so each ~0.5-2 GB model loads once
    # and is reused across bases before being evicted.
    results = {}
    skipped = {}
    for name in rerankers:
        try:
            for base in bases:
                m = run(f"rerank[{name}·{base}]",
                        partial(search_rerank, depth=args.depth, rerank_model=name, base=base))
                results[(name, base)] = m
                print(f"{(f'rerank {name} /{base}')[:44]:44} {summary(m)} "
                      f"{cross_lingual_recall(m) * 100:7.1f}%")
        except Exception as e:
            # One candidate failing to load must not lose the others (jina-reranker-v2
            # is incompatible with the pinned transformers). Drop any partial base
            # runs for this model so the tables never show a half-measured candidate.
            print(f"{('rerank ' + name)[:44]:44} ATLANDI — {type(e).__name__}: {e}")
            skipped[name] = f"{type(e).__name__}: {e}"
            results = {k: v for k, v in results.items() if k[0] != name}
        finally:
            _cross_encoder.cache_clear()  # free before the next candidate model loads

    # --- base ablation: does the reranker still need BM25 in front? ------------
    if "hibrit" in bases and "dense" in bases:
        print(f"\n{'-' * width}")
        print("BASE ABLATION — reranker'a hibrit vs dense aday vermek (BM25 gereksiz mi?)")
        for name in rerankers:
            if (name, "hibrit") not in results or (name, "dense") not in results:
                continue
            h, d = results[(name, "hibrit")], results[(name, "dense")]
            print(f"  {name[:40]:40} "
                  f"r@5 {h['recall_at'][args.k] * 100:5.1f}/{d['recall_at'][args.k] * 100:<5.1f} "
                  f"çapraz {cross_lingual_recall(h) * 100:5.1f}/{cross_lingual_recall(d) * 100:<5.1f} "
                  f"(hibrit/dense)")

    # --- step 2: paired per-question diff vs the frozen hybrid -----------------
    for (name, base), m in results.items():
        print(f"\n{'-' * width}")
        print(f"SORU BAZLI FARK — rerank[{name}·{base}] vs hibrit (dondurulmuş)")
        moved = defaultdict(list)
        for r in hibrit["answerable"]:
            before = hibrit["scored"][r["id"]]["recall"]
            after = m["scored"][r["id"]]["recall"]
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

    # --- step 3: reranker score as an abstention gate, PAIRED vs cosine --------
    abstention = {}
    for (name, base), m in results.items():
        rerank_gate = top1_scores(m)
        a = paired_abstention(rerank_gate, cosine_gate, m["rows"], args.bootstrap)
        abstention[(name, base)] = a

        print(f"\n{'=' * width}")
        print(f"REDDETME SİNYALİ — rerank[{name}·{base}] skoru vs kosinüs (dense top-1, Gün 3 kapısı)")
        print("=" * width)
        print(f"{'sinyal':10} {'':6}{'min':>8} {'mean':>8} {'median':>8} {'max':>8}")
        print(f"{'rerank':10} yanitla {score_stats(a['yes_scores']['rerank'])}")
        print(f"{'':10} reddet  {score_stats(a['no_scores']['rerank'])}")
        print(f"{'cosine':10} yanitla {score_stats(a['yes_scores']['cosine'])}")
        print(f"{'':10} reddet  {score_stats(a['no_scores']['cosine'])}")
        print()
        print(f"AUC(rerank)={a['auc_rerank']:.3f}   AUC(cosine)={a['auc_cosine']:.3f}   "
              f"— tek başlarına SE≈0.083, KARŞILAŞTIRMA İÇİN OKUMA")
        lo, hi = a["ci95"]
        verdict = ("reranker daha iyi" if lo > 0 else
                   "kosinüs daha iyi" if hi < 0 else
                   "ayırt edilemez (CI 0'ı kapsıyor)")
        print(f"EŞLEŞMELİ Δ (rerank − cosine) = {a['delta']:+.3f}  "
              f"%95 CI [{lo:+.3f}, {hi:+.3f}]  → {verdict}")
        print(f"uyumsuz çiftler: reranker ayırıyor/kosinüs ayıramıyor {a['rerank_only']}, "
              f"tersi {a['cosine_only']} (20×10=200 çift)")
        for i, j in a["discord"]["rerank"][:5]:
            print(f"    rerank kazandı: yanitla {i} > reddet {j}")
        for i, j in a["discord"]["cosine"][:5]:
            print(f"    cosine kazandı: yanitla {i} > reddet {j}")

        # Why a single threshold (e.g. 0.5) is not enough despite the split.
        thr = threshold_confusion(a["yes_scores"]["rerank"], a["no_scores"]["rerank"])
        a["threshold"] = thr
        n_yes = thr["zero_false_accept"]["answerable_total"]
        print(f"\nTEK EŞİK maliyeti (reranker skoru):")
        print(f"  t=0.5           : {thr['at_0.5']['false_accept']} uydurma (false-accept), "
              f"{thr['at_0.5']['false_reject']}/{n_yes} gereksiz reddetme")
        print(f"  0 false-accept  : eşik {thr['zero_false_accept']['threshold']:.3f} → "
              f"{n_yes - thr['zero_false_accept']['answerable_kept']}/{n_yes} yanıtlanabilir feda edilir")
        print(f"  en iyi doğruluk : eşik {thr['best_accuracy']['threshold']:.3f} → "
              f"{thr['best_accuracy']['accuracy'] * 100:.0f}% (in-sample, aşırı-uyumlu)")

    # --- persist (summary only; canonical file is eval_retrieval's) ------------
    out = ROOT / "out" / "retrieval" / f"sweep_rerank_{model_slug(args.model)}.json"
    out.write_text(json.dumps({
        "variant": args.variant, "embedding_model": args.model,
        "k": args.k, "depth": args.depth, "bootstrap": args.bootstrap,
        "skipped": skipped,
        "baselines": {
            r: {"recall_at5": m["recall_at"][args.k], "recall_tr": m["recall_tr"],
                "recall_en": m["recall_en"], "mrr": m["mrr"], "auc": m["auc"]}
            for r, m in (("dense", dense), ("hibrit", hibrit))
        },
        "rerank": {
            f"{name}·{base}": {
                "model": name, "base": base,
                "recall_at5": m["recall_at"][args.k],
                "recall_tr": m["recall_tr"], "recall_en": m["recall_en"],
                "mrr": m["mrr"], "capraz_dil": cross_lingual_recall(m),
                "abstention": {
                    "auc_rerank": abstention[(name, base)]["auc_rerank"],
                    "auc_cosine": abstention[(name, base)]["auc_cosine"],
                    "delta": abstention[(name, base)]["delta"],
                    "ci95": abstention[(name, base)]["ci95"],
                    "rerank_only": abstention[(name, base)]["rerank_only"],
                    "cosine_only": abstention[(name, base)]["cosine_only"],
                    "threshold": abstention[(name, base)]["threshold"],
                },
            }
            for (name, base), m in results.items()
        },
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nrecorded: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
