import argparse
import json
import re
import statistics as stats
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from app.config import (
    DEFAULT_VARIANT,
    GEN_K,
    GOLDEN_SET,
    LLM_CANDIDATES,
    REJECT_THRESHOLD,
    RERANK_DEPTH,
    RETRIEVER,
    ROOT,
    VARIANT_GROUP,
    model_slug,
)
from app.ingest import normalize
from app.pipeline import answer as pipeline_answer

# Token-set match threshold for non-numeric anchors (see answer_covers). 85 tolerates
# paraphrase and word order without admitting an unrelated answer.
MATCH_THRESHOLD = 85
RESULTS_DIR = ROOT / "out" / "generation"


def expected_anchors(row: dict, variant: str) -> list[dict]:
    """Golden anchors with the variant group mapped to the active variant."""
    out = []
    for k in row.get("kaynak") or []:
        name = k.get("dosya")
        if name in VARIANT_GROUP:
            name = variant
        out.append({
            "dosya": name,
            "sayfa": int(k.get("sayfa")),
            "alinti": normalize(str(k.get("alinti", ""))),
        })
    return out


def _canon(t: str) -> str:
    """Lowercase, NFKC, and unify the decimal separator so 45,14 == 45.14."""
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"(?<=\d)[.,](?=\d)", ".", t)
    return re.sub(r"\s+", " ", t).strip()


def _numbers(t: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", _canon(t)))


def answer_covers(text: str, quote: str) -> bool:
    
    qnums = _numbers(quote)
    if qnums:
        return qnums.issubset(_numbers(text))
    return fuzz.token_set_ratio(_canon(quote), _canon(text)) >= MATCH_THRESHOLD


def citation_covers(kaynaklar: list[dict], anchor: dict) -> bool:
    return any(
        k.get("dosya") == anchor["dosya"] and int(k.get("sayfa", -1)) == anchor["sayfa"]
        for k in kaynaklar
    )


def verdict(row: dict, ans, variant: str) -> dict:
    """Classify one answered/refused row into the abstention matrix + correctness."""
    anchors = expected_anchors(row, variant)
    rejected_stage1 = bool(ans.hits) and ans.hits[0].skor < REJECT_THRESHOLD
    rejected_stage1 = rejected_stage1 or not ans.hits
    stage = "stage1" if ans.reddedildi and rejected_stage1 else ("stage2" if ans.reddedildi else None)

    if row["beklenen"] == "reddet":
        label = "dogru_red" if ans.reddedildi else "UYDURMA"
        return {"label": label, "stage": stage, "content_ok": None, "citation_ok": None,
                "anchors_found": None}

    # yanitla
    if ans.reddedildi:
        return {"label": "GEREKSIZ_RED", "stage": stage, "content_ok": False,
                "citation_ok": False, "anchors_found": 0}

    found = [a for a in anchors if answer_covers(ans.cevap, a["alinti"])]
    content_ok = len(found) == len(anchors) and bool(anchors)
    citation_ok = bool(anchors) and all(citation_covers(ans.kaynaklar, a) for a in anchors)
    label = "dogru" if content_ok else "YANLIS_CEVAP"
    return {"label": label, "stage": None, "content_ok": content_ok,
            "citation_ok": citation_ok, "anchors_found": len(found)}


def measure(model: str, rows: list[dict], variant: str, k: int) -> dict:
    results = []
    n = len(rows)
    print(f"[{model}] {n} soru başlıyor…", file=sys.stderr, flush=True)
    for i, row in enumerate(rows, start=1):
        t0 = time.perf_counter()
        ans = pipeline_answer(row["soru"], model=model, k=k, retriever=RETRIEVER)
        dt = time.perf_counter() - t0
        v = verdict(row, ans, variant)
        # Live progress to stderr (stdout stays the clean report). Long models
        # (gemma4:12b) otherwise look hung for tens of minutes.
        durum = "RED" if ans.reddedildi else v["label"]
        print(f"[{model}] {i:>2}/{n} {row['id']} [{row['dil']}/{row['kategori']}] "
              f"{dt:5.1f}s -> {durum}", file=sys.stderr, flush=True)
        results.append({
            "id": row["id"], "dil": row["dil"], "kategori": row["kategori"],
            "beklenen": row["beklenen"], "soru": row["soru"],
            "reddedildi": ans.reddedildi, "gerekce": ans.gerekce,
            "cevap": ans.cevap, "kaynaklar": ans.kaynaklar,
            "referans_cevap": row.get("referans_cevap", ""),
            "latency_s": round(dt, 3), **v,
        })
    return {"model": model, "variant": variant, "k": k, "results": results}


def _rate(rows: list[dict], pred) -> tuple[int, int]:
    hit = sum(1 for r in rows if pred(r))
    return hit, len(rows)


def _pct(hit: int, total: int) -> str:
    return f"{hit / total * 100:5.1f}% ({hit}/{total})" if total else "   n/a"


def summarize(m: dict) -> dict:
    res = m["results"]
    yanitla = [r for r in res if r["beklenen"] == "yanitla"]
    reddet = [r for r in res if r["beklenen"] == "reddet"]

    uydurma = [r for r in reddet if r["label"] == "UYDURMA"]
    gereksiz = [r for r in yanitla if r["label"] == "GEREKSIZ_RED"]
    dogru_cevap = [r for r in yanitla if r["label"] == "dogru"]
    yanlis = [r for r in yanitla if r["label"] == "YANLIS_CEVAP"]

    def by_lang(subset, dil):
        return [r for r in subset if r["dil"] == dil]

    return {
        "n_yanitla": len(yanitla), "n_reddet": len(reddet),
        "uydurma": len(uydurma), "uydurma_ids": [r["id"] for r in uydurma],
        "gereksiz_red": len(gereksiz), "gereksiz_ids": [r["id"] for r in gereksiz],
        "dogru_red": len(reddet) - len(uydurma),
        "dogru_cevap": len(dogru_cevap), "yanlis_cevap": len(yanlis),
        "content_ok_tr": _rate(by_lang(yanitla, "tr"), lambda r: r["content_ok"]),
        "content_ok_en": _rate(by_lang(yanitla, "en"), lambda r: r["content_ok"]),
        "citation_ok": _rate(yanitla, lambda r: r["citation_ok"]),
        "latency_med": stats.median(r["latency_s"] for r in res) if res else 0.0,
        "latency_max": max((r["latency_s"] for r in res), default=0.0),
    }


def record(m: dict, s: dict) -> Path:
    # Depth is in the filename so a depth-20 re-run does not overwrite the depth-10
    # results — the two are a comparison, not a replacement. RERANK_DEPTH is a config
    # constant, so it names the run the eval was actually produced under.
    path = RESULTS_DIR / f"{model_slug(m['model'])}_d{RERANK_DEPTH}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": m["model"], "variant": m["variant"], "k": m["k"],
        "retriever": RETRIEVER, "rerank_depth": RERANK_DEPTH,
        "reject_threshold": REJECT_THRESHOLD,
        "summary": s, "results": m["results"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def report(m: dict, s: dict) -> None:
    width = 78
    print("=" * width)
    print(f"GENERATION — {m['model']}  (variant {m['variant']}, retriever {RETRIEVER}, k={m['k']})")
    print("=" * width)

    print("ABSTENTION MATRISI")
    print("-" * width)
    print(f"  UYDURMA (reddet beklenen, cevap verildi)   : {s['uydurma']}/{s['n_reddet']}"
          + (f"  {s['uydurma_ids']}" if s['uydurma_ids'] else "  ← sıfır hedef"))
    print(f"  doğru red                                   : {s['dogru_red']}/{s['n_reddet']}")
    print(f"  GEREKSİZ RED (yanitla beklenen, reddedildi): {s['gereksiz_red']}/{s['n_yanitla']}"
          + (f"  {s['gereksiz_ids']}" if s['gereksiz_ids'] else ""))
    print()

    print("DOĞRULUK (yanitla — cevap beklenen anahtarı içeriyor mu)")
    print("-" * width)
    print(f"  doğru cevap        : {s['dogru_cevap']}/{s['n_yanitla']}")
    print(f"  yanlış cevap       : {s['yanlis_cevap']}/{s['n_yanitla']}  (cevap verdi ama anahtar tutmadı)")
    print(f"  içerik doğru  TR   : {_pct(*s['content_ok_tr'])}")
    print(f"  içerik doğru  EN   : {_pct(*s['content_ok_en'])}")
    print(f"  citation doğru     : {_pct(*s['citation_ok'])}")
    print()
    print(f"gecikme: medyan {s['latency_med']:.1f}s  max {s['latency_max']:.1f}s")
    print(f"kaydedildi: {record(m, s).relative_to(ROOT)}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Generation + abstention on the golden set.")
    p.add_argument("--models", default=",".join(LLM_CANDIDATES),
                   help="comma-separated Ollama models")
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--k", type=int, default=GEN_K)
    p.add_argument("--limit", type=int, default=0, help="first N golden rows (smoke)")
    args = p.parse_args()

    rows = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summaries = {}
    for model in models:
        try:
            m = measure(model, rows, args.variant, args.k)
        except Exception as e:  # a missing/unpulled model must not lose the others
            print(f"{model}: ATLANDI — {type(e).__name__}: {e}\n")
            continue
        s = summarize(m)
        summaries[model] = s
        report(m, s)

    if len(summaries) > 1:
        width = 78
        print("=" * width)
        print("KARŞILAŞTIRMA")
        print("-" * width)
        print(f"{'model':28} {'uydurma':>8} {'ger.red':>8} {'doğru':>8} {'med s':>7}")
        for model, s in summaries.items():
            print(f"{model[:28]:28} {s['uydurma']:>8} {s['gereksiz_red']:>8} "
                  f"{s['dogru_cevap']:>8} {s['latency_med']:>7.1f}")


if __name__ == "__main__":
    main()
