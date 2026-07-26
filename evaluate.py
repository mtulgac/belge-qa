import csv
import re
import unicodedata
from pathlib import Path

import fitz
from rapidfuzz.distance import Levenshtein

from app.config import DOC_LANG
from app.ocr import (
    method_easyocr,
    method_ocr_preprocessed,
    method_ocr_raw,
    method_rapidocr,
    method_text_layer,
)

SAMPLES = Path("data/samples")
PARSED = Path("out/parsed")
SUMMARY_CSV = PARSED / "_summary.csv"

# Manual ground truth: hand-typed transcription plus the page range it covers.
MANUAL_GT = {
    "resmi_gazete_1995": (Path("data/resmi_gazete_p1_ground_truth.txt"), (0, 1)),
}

# OCR methods that compete in the bake-off. text_layer is the reference path
OCR_METHODS = ["ocr_raw", "ocr_preprocessed", "easyocr", "rapidocr"]

# Reference overrides. Any doc not listed uses its own text layer, except
# MANUAL_GT docs, which are handled above (their own layer would be circular).
REFERENCES = {
    "tobb_taranmis": ("tobb_yonetmelik", None),      # Pair B, full document
    "yonetmelik_ss": ("tobb_yonetmelik", (0, 1)),    # image = regulation page 1
    "yonetmelik_foto": ("tobb_yonetmelik", (0, 1)),
}

# Language per doc. Metrics are reported TR/EN separately. The bake-off keys
# documents by stem (out/parsed/<stem>/), config keys them by filename.
LANG = {Path(name).stem: lang for name, lang in DOC_LANG.items()}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"-\s*\n\s*", "", text)   # de-hyphenate word breaks
    text = re.sub(r"\s+", " ", text)        # collapse all whitespace
    return text.strip()


def cer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp), normalize(ref)
    return Levenshtein.distance(h, r) / max(len(r), 1)


def pdf_text(stem: str, page_range) -> str:
    doc = fitz.open(SAMPLES / f"{stem}.pdf")
    pages = (
        [doc[i] for i in range(*page_range)] if page_range else list(doc)
    )
    text = "\n\n".join(p.get_text("text") for p in pages)
    doc.close()
    return text


def reference_text(stem: str):
    if stem in MANUAL_GT:
        path, _ = MANUAL_GT[stem]
        if not path.exists():
            return None, "ground truth missing — skipped"
        return path.read_text(encoding="utf-8"), None
    spec = REFERENCES.get(stem)
    if spec is None:
        return pdf_text(stem, None), None            # own text layer
    source, page_range = spec
    return pdf_text(source, page_range), None


LIVE_METHODS = {
    "text_layer": method_text_layer,
    "ocr_raw": method_ocr_raw,
    "ocr_preprocessed": method_ocr_preprocessed,
    "easyocr": method_easyocr,
    "rapidocr": method_rapidocr,
}


def live_hypotheses(stem: str, page_range) -> dict:
    doc = fitz.open(SAMPLES / f"{stem}.pdf")
    pages = [doc[i] for i in range(*page_range)]
    out = {}
    for name, method in LIVE_METHODS.items():
        try:
            out[name] = "\n\n".join(method(p)[0] for p in pages)
        except Exception as e:                  # keep the run alive; log and move on
            print(f"  ! {stem} {name} failed: {e}")
    doc.close()
    return out


def load_elapsed() -> dict:
    out = {}
    if not SUMMARY_CSV.exists():
        return out
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(Path(row["doc"]).stem, row["method"])] = float(row["elapsed"])
    return out


def main() -> None:
    elapsed = load_elapsed()
    doc_dirs = sorted(d for d in PARSED.iterdir() if d.is_dir())

    # Collect per (method, lang) CER lists and per-doc rows for the table.
    by_method = {}          # (method, lang) -> [cer, ...]
    time_by_method = {}     # method -> [elapsed, ...]
    table = []              # (doc, method, cer, ref_len, hyp_len)
    skipped = []

    for d in doc_dirs:
        stem = d.name
        ref, note = reference_text(stem)
        if ref is None:
            skipped.append((stem, note))
            continue
        lang = LANG.get(stem, "??")
        ref_len = len(normalize(ref))
        page_scoped = stem in MANUAL_GT
        if page_scoped:
            a, b = MANUAL_GT[stem][1]
            label = f"{stem} [p{a + 1}]" if b - a == 1 else f"{stem} [p{a + 1}-{b}]"
            hyps = sorted(live_hypotheses(stem, (a, b)).items())
        else:
            label = stem
            hyps = [
                (t.stem, t.read_text(encoding="utf-8"))
                for t in sorted(d.glob("*.txt"))
            ]
        for method, hyp in hyps:
            score = cer(hyp, ref)
            table.append((label, method, score, ref_len, len(normalize(hyp))))
            # Page-scoped docs stay out of the ranking: their numbers answer
            # the routing question, not the whole-document method comparison.
            if method in OCR_METHODS and not page_scoped:
                by_method.setdefault((method, lang), []).append(score)
                if (stem, method) in elapsed:
                    time_by_method.setdefault(method, []).append(elapsed[(stem, method)])

    _print_per_doc(table, skipped)
    _print_ranking(by_method, time_by_method)


def _print_per_doc(table, skipped) -> None:
    print("=" * 74)
    print("PER-DOCUMENT CER")
    print("=" * 74)
    print(f"{'doc':22} {'method':17} {'CER':>8} {'ref':>8} {'hyp':>8}")
    print("-" * 74)
    last = None
    for stem, method, score, ref_len, hyp_len in table:
        if last is not None and stem != last:
            print()
        last = stem
        print(f"{stem:22} {method:17} {score * 100:7.2f}% {ref_len:8} {hyp_len:8}")
    for stem, note in skipped:
        print(f"\n{stem:22} {note}")
    print()


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _print_ranking(by_method, time_by_method) -> None:
    print("=" * 74)
    print("METHOD RANKING — mean CER, TR and EN reported separately")
    print("=" * 74)
    print(f"{'method':17} {'CER TR':>9} {'CER EN':>9} {'avg s/doc':>11}")
    print("-" * 74)

    rows = []
    for method in OCR_METHODS:
        tr = _mean(by_method.get((method, "TR"), []))
        en = _mean(by_method.get((method, "EN"), []))
        secs = _mean(time_by_method.get(method, []))
        rows.append((method, tr, en, secs))

    def fmt(x, pct=True):
        if x is None:
            return f"{'—':>8}"
        return f"{x * 100:7.2f}%" if pct else f"{x:9.1f}"

    # Worst-case CER across languages: a method must be acceptable in BOTH.
    # This is the same anti-masking logic as the TR/EN split — a strong score in
    # one language must not hide a collapse in the other.
    def worst(r):
        vals = [v for v in (r[1], r[2]) if v is not None]
        return max(vals) if vals else None

    for method, tr, en, secs in sorted(rows, key=lambda r: (worst(r) is None, worst(r) or 0)):
        secs_str = f"{secs:10.1f}s" if secs is not None else f"{'—':>11}"
        flag = "  <- worst-case leader" if worst((method, tr, en, secs)) == min(
            (w for w in (worst(x) for x in rows) if w is not None), default=None
        ) else ""
        print(f"{method:17} {fmt(tr)} {fmt(en)} {secs_str}{flag}")

    print()
    scored = [(r, worst(r)) for r in rows if worst(r) is not None]
    if scored:
        win, win_worst = min(scored, key=lambda x: (x[1], x[0][3] or 0))
        print("WINNER")
        print("-" * 74)
        print(f"  Carry forward: {win[0]}")
        print(
            f"  Lowest worst-case CER across languages "
            f"(TR {fmt(win[1]).strip()}, EN {fmt(win[2]).strip()}, "
            f"{win[3]:.0f}s/doc)."
        )
        others = [r for r, _ in scored if r[0] != win[0]]
        for r in others:
            print(
                f"    vs {r[0]:16} worst-case {worst(r) * 100:5.2f}% "
                f"(TR {fmt(r[1]).strip()}, EN {fmt(r[2]).strip()})"
            )
    print()


if __name__ == "__main__":
    main()
