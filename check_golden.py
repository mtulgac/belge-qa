import json
import re
import sys
import unicodedata
from collections import Counter

import fitz
import yaml
from rapidfuzz import fuzz

from app.config import (
    DEFAULT_VARIANT,
    DOC_LANG,
    GOLDEN_SET,
    SAMPLES,
    VARIANT_GROUP,
    corpus,
)
from app.ingest import index_dir, normalize
from app.needs_ocr import needs_ocr
from app.ocr import method_ocr_raw

CATEGORIES = {
    "tek_belge", "cok_belge", "yanitlanamaz", "yanlis_oncul",
    "celiski", "tablo", "capraz_dil", "alan_disi",
}
REJECT_CATEGORIES = {"yanitlanamaz", "yanlis_oncul", "alan_disi"}
MULTI_DOC_CATEGORIES = {"cok_belge", "celiski"}

FUZZY_MIN = 85          # below this, the quote is not on the page in any form
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
EN_QUESTION_STARTS = ("what", "who", "how", "when", "where", "which", "why", "is", "does", "do")

_docs = {}
_pages = {}
_variants = {}          # dosya -> format variant used instead, e.g. the scanned copy


def tr_lower(text: str) -> str:
    """Turkish-aware lowercase: I->ı and İ->i before the ASCII fold."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def open_doc(name: str):
    if name not in _docs:
        path = SAMPLES / name
        if not path.exists():
            return None
        _docs[name] = fitz.open(path)
    return _docs[name]


def page_text(name: str, index: int) -> str:
    """Normalized text of one page, extracted the way ingest will extract it."""
    key = (name, index)
    if key not in _pages:
        page = open_doc(name)[index]
        ocr, _ = needs_ocr(page)
        text = method_ocr_raw(page)[0] if ocr else page.get_text("text")
        _pages[key] = normalize(text)
    return _pages[key]


def locate(name: str, alinti: str) -> dict:
    """Where the quote appears in the document, page by page (0-indexed)."""
    q = normalize(alinti)
    q_low = tr_lower(q)
    exact, loose, scores = [], [], {}
    for i in range(len(open_doc(name))):
        text = page_text(name, i)
        if q in text:
            exact.append((i, text.count(q)))
        elif q_low in tr_lower(text):
            loose.append(i)
        else:
            scores[i] = fuzz.partial_ratio(q_low, tr_lower(text))
    return {"exact": exact, "loose": loose, "scores": scores}


def check_anchor(kaynak: dict) -> tuple[str, str]:
    """Verify one dosya+sayfa+alinti anchor. Returns (status, detail)."""
    name = kaynak.get("dosya")
    sayfa = kaynak.get("sayfa")
    alinti = kaynak.get("alinti")
    if not name or sayfa is None or not alinti:
        return "SCHEMA", "missing dosya/sayfa/alinti"
    name = _variants.get(name, name)
    if open_doc(name) is None:
        return "NO_FILE", f"{name} not in data/samples"

    n_pages = len(open_doc(name))
    if not 1 <= sayfa <= n_pages:
        return "PAGE_RANGE", f"sayfa {sayfa}, document has {n_pages} pages"

    hit = locate(name, alinti)
    target = sayfa - 1
    on_target = [p for p, _ in hit["exact"] if p == target]
    total = sum(c for _, c in hit["exact"]) + len(hit["loose"])
    others = sorted({p for p, _ in hit["exact"] if p != target} | set(hit["loose"]))

    if on_target:
        note = f"{total}x in document" if total > 1 else ""
        if others:
            note = f"{note}, also p.{[p + 1 for p in others]}"
        return "OK", note.strip(", ")
    if target in hit["loose"]:
        return "CASE", f"matched case-insensitively only (p.{sayfa})"
    if others:
        return "WRONG_PAGE", f"not p.{sayfa}, found on p.{[p + 1 for p in others]}"

    best_page = max(hit["scores"], key=hit["scores"].get) if hit["scores"] else None
    best = hit["scores"].get(best_page, 0)
    if best >= FUZZY_MIN:
        return "FUZZY", f"no exact match; closest p.{best_page + 1} (similarity {best:.0f}%)"
    return "NOT_FOUND", f"not in document; best similarity {best:.0f}% (p.{best_page + 1 if best_page is not None else '?'})"


def check_labels(row: dict) -> list[str]:
    """Category / language / expectation consistency — no PDF needed."""
    issues = []
    rid = row.get("id", "?")
    kategori = row.get("kategori")
    dil = row.get("dil")
    beklenen = row.get("beklenen")
    kaynak = row.get("kaynak") or []
    soru = row.get("soru", "")

    for field in ("id", "soru", "kategori", "dil", "beklenen", "referans_cevap"):
        if not row.get(field):
            issues.append(f"{field} missing")
    if kategori not in CATEGORIES:
        issues.append(f"unknown kategori: {kategori}")
    if dil not in {"tr", "en"}:
        issues.append(f"unknown dil: {dil}")
    if beklenen not in {"yanitla", "reddet"}:
        issues.append(f"unknown beklenen: {beklenen}")

    if beklenen == "reddet" and kaynak:
        issues.append("beklenen=reddet but kaynak is not empty")
    if beklenen == "yanitla" and not kaynak:
        issues.append("beklenen=yanitla but kaynak is empty")
    if kategori in REJECT_CATEGORIES and beklenen != "reddet":
        issues.append(f"kategori={kategori} requires beklenen=reddet")

    files = {k.get("dosya") for k in kaynak}
    if kategori in MULTI_DOC_CATEGORIES and len(files) < 2:
        issues.append(f"kategori={kategori} requires >=2 distinct dosya (has {len(files)})")

    if kategori == "capraz_dil":
        doc_langs = {DOC_LANG.get(f, "??") for f in files if f}
        if doc_langs and all(lang.lower() == dil for lang in doc_langs):
            issues.append(f"kategori=capraz_dil but question and document share a language ({dil})")

    # izole drops documents from the index for this question only. It must not
    # drop the document the anchor points at, and format variants are handled by
    # VARIANT_GROUP instead, so naming one here means the two mechanisms disagree.
    for name in row.get("izole") or []:
        if name not in DOC_LANG:
            issues.append(f"izole: unknown document {name}")
        if name in files:
            issues.append(f"izole: {name} is also the anchor's dosya")
        if name in VARIANT_GROUP:
            issues.append(f"izole: {name} is a format variant — use VARIANT_GROUP")

    # A mislabeled dil silently corrupts the TR/EN metric split.
    lowered = tr_lower(soru).lstrip()
    if dil == "tr" and not (set(soru) & TR_CHARS) and lowered.startswith(EN_QUESTION_STARTS):
        issues.append("dil=tr but the question looks English")
    if dil == "en" and (set(soru) & TR_CHARS):
        issues.append("dil=en but the question contains Turkish characters")

    # Anchors are matched by substring, so a bare number can land in the wrong
    # place. Table questions are exempt: OCR flattens rows and columns, which
    # makes a longer quote the more brittle anchor there — the cell value is
    # deliberately the anchor.
    if kategori != "tablo":
        for k in kaynak:
            alinti = str(k.get("alinti", ""))
            if alinti and not any(c.isalpha() for c in alinti):
                issues.append(f"anchor is not distinctive (digits only): {alinti!r}")

    return [f"{rid}: {i}" for i in issues]


def coverage(rows: list[dict]) -> None:
    print("=" * 74)
    print("COVERAGE")
    print("=" * 74)

    for field in ("kategori", "dil", "beklenen"):
        counts = Counter(r.get(field) for r in rows)
        line = "  ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"{field:10} {line}")

    used = Counter(
        k.get("dosya") for r in rows for k in (r.get("kaynak") or []) if k.get("dosya")
    )
    print(f"\n{'document':28} {'rows':>6}  lang")
    print("-" * 74)
    # Only indexed documents can be a source. The variant group counts as one
    # slot (the same content in three formats), and anything outside the corpus
    # is an OCR-test asset, not an answer source — neither is "missing".
    for name in corpus():
        n = used.get(name, 0)
        note = "" if n else "   <- never used"
        if name in VARIANT_GROUP:
            note = f"   (variant group: {', '.join(VARIANT_GROUP)})"
        print(f"{name:28} {n:6}  {DOC_LANG.get(name, '??')}{note}")

    answerable = [r for r in rows if r.get("beklenen") == "yanitla"]
    en_rows = [r for r in rows if r.get("dil") == "en"]
    print(
        f"\ntotal {len(rows)} questions | yanitla {len(answerable)} / reddet {len(rows) - len(answerable)}"
        f" | EN {len(en_rows)} ({len(en_rows) / max(len(rows), 1) * 100:.0f}%)"
    )
    print()


def check_index(rows: list[dict], variant: str) -> list[str]:
    """Locate every anchor inside the built index, not just in the source PDF.

    Between the two lies chunking, and this is where it can go wrong: the quote
    can be cut across a chunk boundary (then no single chunk answers the
    question), or the chunk can carry the wrong page number (then the answer is
    cited to the wrong page). Both are silent failures for retrieval, so they
    get measured before the harness is built on top.
    """
    path = index_dir(variant) / "chunks.json"
    if not path.exists():
        return [f"index missing: {path} (run: python -m app.ingest --variant {variant})"]

    chunks = json.loads(path.read_text(encoding="utf-8"))
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(c["dosya"], []).append(c)

    print("=" * 74)
    print(f"INDEX CHECK — {len(chunks)} chunks, variant {variant}")
    print("=" * 74)
    print(f"{'id':6} {'dosya':26} {'p.':>3} {'status':11} chunk / note")
    print("-" * 74)

    problems, counts = [], Counter()
    for row in rows:
        for k in row.get("kaynak") or []:
            name, sayfa = k.get("dosya"), k.get("sayfa")
            # The anchors name the default document; a variant index holds the
            # same content under the variant's filename.
            if name in VARIANT_GROUP:
                name = variant
            quote = normalize(str(k.get("alinti", "")))
            hits = [
                c for c in by_file.get(name, [])
                if quote in normalize(c["metin"])
            ]
            if name not in by_file:
                status, detail = "NO_DOC", "document is not in this index"
            elif not hits:
                status, detail = "SPLIT", "no chunk holds the whole quote"
            else:
                on_page = [
                    c for c in hits
                    if c["sayfa_baslangic"] <= sayfa <= c["sayfa_bitis"]
                ]
                if on_page:
                    c = on_page[0]
                    status = "OK"
                    detail = f"{c['chunk_id']} p.{c['sayfa_baslangic']}-{c['sayfa_bitis']} ({c['yontem']})"
                else:
                    c = hits[0]
                    status = "PAGE_OFF"
                    detail = f"{c['chunk_id']} p.{c['sayfa_baslangic']}-{c['sayfa_bitis']}, anchor says p.{sayfa}"
            counts[status] += 1
            print(f"{row.get('id', '?'):6} {str(name):26} {str(sayfa):>3} {status:11} {detail}")
            if status != "OK":
                problems.append(f"{row.get('id')}: {status} — {name} p.{sayfa} ({detail})")

    print()
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    return problems


def main() -> None:
    index_variant = None
    for arg in sys.argv[1:]:
        if arg.startswith("--index"):
            _, _, value = arg.partition("=")
            index_variant = value or DEFAULT_VARIANT
            continue
        original, _, variant = arg.partition("=")
        if not variant:
            print("Usage: check_golden.py [original.pdf=variant.pdf ...] [--index[=variant.pdf]]")
            return 2
        _variants[original] = variant

    rows = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))

    if index_variant:
        problems = check_index(rows, index_variant)
        print("=" * 74)
        print(f"PROBLEMS ({len(problems)})")
        print("=" * 74)
        for p in problems:
            print(f"  - {p}")
        if not problems:
            print("  none")
        print()
        return 1 if problems else 0

    ids = Counter(r.get("id") for r in rows)
    problems = [f"duplicate id: {i}" for i, n in ids.items() if n > 1]

    print("=" * 74)
    print("ANCHOR CHECK — against page text extracted the way ingest routes it")
    if _variants:
        for original, variant in _variants.items():
            print(f"VARIANT: {original} -> {variant}")
    print("=" * 74)
    print(f"{'id':6} {'dosya':26} {'p.':>3} {'status':11} anchor / note")
    print("-" * 74)

    status_counts = Counter()
    for row in rows:
        problems += check_labels(row)
        kaynak = row.get("kaynak") or []
        if _variants:
            # Variant mode answers one question — do the anchors survive the
            # other format? — so only the swapped document's rows are shown.
            kaynak = [k for k in kaynak if k.get("dosya") in _variants]
        for k in kaynak:
            status, detail = check_anchor(k)
            status_counts[status] += 1
            alinti = str(k.get("alinti", ""))[:26]
            print(
                f"{row.get('id', '?'):6} {str(k.get('dosya', '?')):26} "
                f"{str(k.get('sayfa', '?')):>3} {status:11} {alinti!r} {detail}"
            )
            if status != "OK":
                problems.append(
                    f"{row.get('id')}: {status} — {k.get('dosya')} s.{k.get('sayfa')} "
                    f"{alinti!r} ({detail})"
                )

    print()
    print("  ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    print()

    coverage(rows)

    print("=" * 74)
    print(f"PROBLEMS ({len(problems)})")
    print("=" * 74)
    for p in problems:
        print(f"  - {p}")
    if not problems:
        print("  none")
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
