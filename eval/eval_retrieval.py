import argparse
import json
import statistics as stats
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.config import (
    DEFAULT_VARIANT,
    EMBEDDING_MODEL,
    GOLDEN_SET,
    RETRIEVER,
    ROOT,
    VARIANT_GROUP,
    model_slug,
)
from app.ingest import normalize
from app.retrieval import RETRIEVERS, load_index, search

K_VALUES = (1, 3, 5, 10)
RESULTS_DIR = ROOT / "out" / "retrieval"


def anchors_of(row: dict, variant: str) -> list[tuple[str, str]]:
    """(dosya, quote) pairs, with the variant group mapped to the active variant."""
    out = []
    for k in row.get("kaynak") or []:
        name = k.get("dosya")
        if name in VARIANT_GROUP:
            name = variant
        out.append((name, normalize(str(k.get("alinti", "")))))
    return out


def covers(hit, anchor: tuple[str, str]) -> bool:
    name, quote = anchor
    return hit.dosya == name and quote in normalize(hit.metin)


def evaluate_row(row: dict, hits: list, variant: str, k: int) -> dict:
    anchors = anchors_of(row, variant)
    top = hits[:k]
    found = [a for a in anchors if any(covers(h, a) for h in top)]
    relevant_hits = [h for h in top if any(covers(h, a) for a in anchors)]
    rank = next(
        (i + 1 for i, h in enumerate(top) if any(covers(h, a) for a in anchors)), None
    )
    return {
        "recall": len(found) / len(anchors) if anchors else 0.0,
        "hit": 1.0 if found else 0.0,
        "precision": len(relevant_hits) / k,
        "mrr": 1 / rank if rank else 0.0,
        "missing": [a for a in anchors if a not in found],
    }


def anchor_rank(row: dict, variant: str, model_name: str, depth: int,
                retriever: str = RETRIEVER) -> int | None:
    hits = search(
        row["soru"], k=depth, variant=variant, model_name=model_name,
        exclude=tuple(row.get("izole") or []), retriever=retriever,
    )
    anchors = anchors_of(row, variant)
    return next(
        (i + 1 for i, h in enumerate(hits) if any(covers(h, a) for a in anchors)), None
    )


def _mean(values) -> float:
    values = list(values)
    return stats.mean(values) if values else 0.0


def separability(yes: list[float], no: list[float]) -> float:
    if not yes or not no:
        return 0.0
    wins = sum((y > n) + 0.5 * (y == n) for y in yes for n in no)
    return wins / (len(yes) * len(no))


def _top1(hits: list) -> float:
    return float(hits[0].skor) if hits else 0.0


def _top1_dense(hits: list) -> float | None:
    if not hits:
        return None
    return None if hits[0].dense_skor is None else float(hits[0].dense_skor)


def measure(variant: str = DEFAULT_VARIANT, model_name: str = EMBEDDING_MODEL,
            k: int = 5, retriever: str = RETRIEVER, search_fn=None) -> dict:
    search_fn = search_fn or (lambda *a, **kw: search(*a, retriever=retriever, **kw))
    rows = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))
    index = load_index(variant, model_name)
    answerable = [r for r in rows if r.get("beklenen") == "yanitla"]
    rejectable = [r for r in rows if r.get("beklenen") == "reddet"]

    max_k = max(K_VALUES + (k,))
    hits = {
        r["id"]: search_fn(
            r["soru"], k=max_k, variant=variant, model_name=model_name,
            exclude=tuple(r.get("izole") or []),
        )
        for r in rows
    }
    scored = {r["id"]: evaluate_row(r, hits[r["id"]], variant, k) for r in answerable}

    yes = [_top1(hits[r["id"]]) for r in answerable]
    no = [_top1(hits[r["id"]]) for r in rejectable]
    yes_dense = [_top1_dense(hits[r["id"]]) for r in answerable]
    no_dense = [_top1_dense(hits[r["id"]]) for r in rejectable]
    # Only meaningful when every row actually has a cosine, otherwise the column
    # is absent, not zero.
    has_dense = all(s is not None for s in yes_dense + no_dense)

    return {
        "variant": variant,
        "model": model_name,
        "retriever": retriever,
        "k": k,
        "index": index,
        "rows": rows,
        "answerable": answerable,
        "rejectable": rejectable,
        "hits": hits,
        "scored": scored,
        "recall_at": {
            kk: _mean(evaluate_row(r, hits[r["id"]], variant, kk)["recall"] for r in answerable)
            for kk in K_VALUES
        },
        "recall_tr": _mean(scored[r["id"]]["recall"] for r in answerable if r["dil"] == "tr"),
        "recall_en": _mean(scored[r["id"]]["recall"] for r in answerable if r["dil"] == "en"),
        "precision": _mean(m["precision"] for m in scored.values()),
        "mrr": _mean(m["mrr"] for m in scored.values()),
        "yes_scores": yes,
        "no_scores": no,
        "yes_dense": yes_dense,
        "no_dense": no_dense,
        "overlap": sum(1 for s in no if s >= min(yes)) if yes else 0,
        "auc": separability(yes, no),
        "auc_dense": separability(yes_dense, no_dense) if has_dense else None,
    }


def record(m: dict, extra: dict | None = None) -> Path:
    # The retriever label is read off the measurement rather than passed in, so a
    # sweep cannot file a run under the wrong name.
    index, k, retriever = m["index"], m["k"], m["retriever"]

    by_category = defaultdict(list)
    for row in m["answerable"]:
        by_category[row["kategori"]].append(row)

    questions = []
    for row in m["rows"]:
        hits = m["hits"][row["id"]]
        entry = {
            "id": row["id"],
            "dil": row["dil"],
            "kategori": row["kategori"],
            "beklenen": row["beklenen"],
            "top1": round(hits[0].skor, 6) if hits else None,
            "top1_dense": round(hits[0].dense_skor, 6)
            if hits and hits[0].dense_skor is not None else None,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "dosya": h.dosya,
                    "sayfa": [h.sayfa_baslangic, h.sayfa_bitis],
                    "skor": round(h.skor, 6),
                    "dense": [h.dense_sira, round(h.dense_skor, 6)]
                    if h.dense_skor is not None else None,
                    "bm25": [h.bm25_sira, round(h.bm25_skor, 6)]
                    if h.bm25_skor is not None else None,
                }
                for h in hits[:k]
            ],
        }
        scored = m["scored"].get(row["id"])
        if scored:
            entry |= {
                "recall": scored["recall"],
                "precision": scored["precision"],
                "mrr": scored["mrr"],
                "missing": [list(a) for a in scored["missing"]],
            }
        questions.append(entry)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retriever": retriever,
        "variant": m["variant"],
        "model": m["model"],
        "k": k,
        "index": {
            "chunks": len(index),
            "dim": index.meta["dim"],
            "chunk_target": index.meta["chunk_target"],
            "chunk_overlap": index.meta["chunk_overlap"],
        },
        "aggregate": {
            "recall_at": {str(kk): v for kk, v in m["recall_at"].items()},
            "recall_tr": m["recall_tr"],
            "recall_en": m["recall_en"],
            "precision": m["precision"],
            "mrr": m["mrr"],
            "auc": m["auc"],              # on the retriever's own top-1 score
            "auc_dense": m["auc_dense"],  # on the top-1 hit's cosine, comparable across retrievers
            "overlap": m["overlap"],
        },
        "categories": {
            name: {
                "recall": _mean(m["scored"][r["id"]]["recall"] for r in subset),
                "precision": _mean(m["scored"][r["id"]]["precision"] for r in subset),
                "mrr": _mean(m["scored"][r["id"]]["mrr"] for r in subset),
                "n": len(subset),
            }
            for name, subset in sorted(by_category.items())
        },
        "scores": {
            "yanitla": [round(s, 6) for s in m["yes_scores"]],
            "reddet": [round(s, 6) for s in m["no_scores"]],
            "yanitla_dense": [None if s is None else round(s, 6) for s in m["yes_dense"]],
            "reddet_dense": [None if s is None else round(s, 6) for s in m["no_dense"]],
        },
        "questions": questions,
    }
    if extra:
        payload["extra"] = extra

    path = (
        RESULTS_DIR / Path(m["variant"]).stem / model_slug(m["model"])
        / f"{retriever}_k{k}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the golden set.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=5, help="k for the detail tables")
    parser.add_argument("--retriever", default=RETRIEVER, choices=sorted(RETRIEVERS))
    args = parser.parse_args()

    m = measure(args.variant, args.model, args.k, args.retriever)
    index, k = m["index"], m["k"]

    print("=" * 78)
    print(f"RETRIEVAL ({args.retriever}) — golden set")
    print("=" * 78)
    print(f"model    {args.model}")
    print(f"variant  {args.variant}  ({len(index)} chunks, dim {index.meta['dim']})")
    print(f"chunking target {index.meta['chunk_target']} / overlap {index.meta['chunk_overlap']}")
    print()

    # --- recall@k, TR and EN separately --------------------------------------
    print(f"{'':10} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9} {'recall@10':>10} {'n':>4}")
    print("-" * 78)
    for label, subset in (
        ("all", m["answerable"]),
        ("TR", [r for r in m["answerable"] if r["dil"] == "tr"]),
        ("EN", [r for r in m["answerable"] if r["dil"] == "en"]),
    ):
        cells = [
            _mean(evaluate_row(r, m["hits"][r["id"]], args.variant, kk)["recall"] for r in subset)
            for kk in K_VALUES
        ]
        print(
            f"{label:10} " + " ".join(f"{c * 100:8.1f}%" for c in cells[:3])
            + f" {cells[3] * 100:9.1f}% {len(subset):4}"
        )
    print()

    # --- per category ---------------------------------------------------------
    by_category = defaultdict(list)
    for row in m["answerable"]:
        by_category[row["kategori"]].append(row)

    print(f"category         {'recall@' + str(k):>10} {'precision':>10} {'MRR':>7} {'n':>4}")
    print("-" * 78)
    for name, subset in sorted(by_category.items()):
        scored = [m["scored"][r["id"]] for r in subset]
        print(
            f"{name:16} {_mean(s['recall'] for s in scored) * 100:9.1f}% "
            f"{_mean(s['precision'] for s in scored) * 100:9.1f}% "
            f"{_mean(s['mrr'] for s in scored):7.2f} {len(subset):4}"
        )
    print()

    # --- misses ---------------------------------------------------------------
    misses = [(r, m["scored"][r["id"]]["missing"]) for r in m["answerable"] if m["scored"][r["id"]]["missing"]]
    print(f"MISSED ANCHORS at k={k} ({sum(len(x) for _, x in misses)})")
    print("-" * 78)
    for row, missing in misses:
        for name, quote in missing:
            print(f"  {row['id']} [{row['dil']}/{row['kategori']}] {name} :: {quote[:52]!r}")
    if not misses:
        print("  none")
    print()

    # --- score separation: answerable vs unanswerable -------------------------
    print("TOP-1 SCORE — the signal the abstention threshold has to separate")
    print("-" * 78)
    print(f"{'':14} {'min':>8} {'mean':>8} {'median':>8} {'max':>8} {'n':>4}")
    for label, values in (("yanitla", m["yes_scores"]), ("reddet", m["no_scores"])):
        print(
            f"{label:14} {min(values):8.3f} {stats.mean(values):8.3f} "
            f"{stats.median(values):8.3f} {max(values):8.3f} {len(values):4}"
        )
    print(
        f"\noverlap: {m['overlap']}/{len(m['no_scores'])} reddet rows score at or above the "
        f"weakest yanitla row —\nthat overlap is what a single threshold cannot separate."
    )
    print(f"separability (AUC): {m['auc']:.3f}   (1.0 = clean split, 0.5 = coin flip)")
    if args.retriever != "dense":
        cosine = (
            f"{m['auc_dense']:.3f}; only that column is comparable with the dense run"
            if m["auc_dense"] is not None
            else "not available — these hits carry no cosine at all"
        )
        print(f"  ^ computed on the {args.retriever} score. On the top-1 hit's cosine: {cosine}.")
    if args.retriever == "hibrit":
        spread = max(m["yes_scores"] + m["no_scores"]) - min(m["yes_scores"] + m["no_scores"])
        print(
            f"    RRF scores span {spread:.4f} across all 30 questions — bounded by the fusion\n"
            f"    constant and full of ties, so this is not a candidate abstention signal."
        )
    print()
    print(f"recorded: {record(m).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
