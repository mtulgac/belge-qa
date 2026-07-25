import csv
import sys
import time
from pathlib import Path

import cv2
import fitz
import pytesseract

from evaluate import LANG, MANUAL_GT, cer, reference_text
from parse import page_image

SAMPLES = Path("data/samples")
OUT = Path("out/sweep")
RESULTS_CSV = OUT / "results.csv"
BEST_DIR = "data/tessdata/best"

GRID = [
    ("fast_psm3_300", None, 3, 300),
    ("fast_psm4_300", None, 4, 300),
    ("fast_psm6_300", None, 6, 300),
    ("best_psm3_300", BEST_DIR, 3, 300),
    ("best_psm4_300", BEST_DIR, 4, 300),
    ("best_psm6_300", BEST_DIR, 6, 300),
    ("fast_psm3_200", None, 3, 200),
    ("fast_psm3_400", None, 3, 400),
]

FIELDS = ["config", "doc", "lang", "cer", "elapsed", "chars"]


def tess_config(tessdata_dir, psm) -> str:
    cfg = f"--psm {psm}"
    if tessdata_dir:
        cfg += f' --tessdata-dir "{tessdata_dir}"'
    return cfg


def extract(path: Path, config: str, dpi: int) -> tuple[str, float]:

    t0 = time.perf_counter()
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        parts = [
            pytesseract.image_to_string(
                page_image(page, dpi), lang="tur+eng", config=config
            )
            for page in doc
        ]
        doc.close()
        text = "\n\n".join(parts)
    else:
        img = cv2.imread(str(path))
        text = pytesseract.image_to_string(img, lang="tur+eng", config=config)
    return text.strip(), time.perf_counter() - t0


def load_done() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return {(r["config"], r["doc"]) for r in csv.DictReader(f)}


def append_row(row: dict) -> None:
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run(filter_str: str) -> None:
    files = sorted(
        p for p in SAMPLES.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
        and filter_str in p.name
        and p.stem not in MANUAL_GT
    )
    done = load_done()
    OUT.mkdir(parents=True, exist_ok=True)

    for label, tessdata_dir, psm, dpi in GRID:
        config = tess_config(tessdata_dir, psm)
        for path in files:
            stem = path.stem
            if (label, stem) in done:
                continue
            ref, note = reference_text(stem)
            if ref is None:
                continue
            text, elapsed = extract(path, config, dpi)
            folder = OUT / label
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{stem}.txt").write_text(text, encoding="utf-8")
            row = {
                "config": label, "doc": stem, "lang": LANG.get(stem, "??"),
                "cer": round(cer(text, ref), 4),
                "elapsed": round(elapsed, 1), "chars": len(text),
            }
            append_row(row)
            print(f"{label:16} {stem:22} CER {row['cer'] * 100:6.2f}%  {elapsed:5.1f}s")


def report() -> None:
    if not RESULTS_CSV.exists():
        return
    with RESULTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print()
    print("=" * 74)
    print("SWEEP RANKING — mean CER, TR and EN separate, worst-case decides")
    print("=" * 74)
    print(f"{'config':16} {'CER TR':>9} {'CER EN':>9} {'worst':>9} {'s/doc':>7} {'docs':>5}")
    print("-" * 74)

    stats = []
    for label, *_ in GRID:
        mine = [r for r in rows if r["config"] == label]
        if not mine:
            continue
        tr = [float(r["cer"]) for r in mine if r["lang"] == "TR"]
        en = [float(r["cer"]) for r in mine if r["lang"] == "EN"]
        mean_tr = sum(tr) / len(tr) if tr else None
        mean_en = sum(en) / len(en) if en else None
        worst = max(v for v in (mean_tr, mean_en) if v is not None)
        secs = sum(float(r["elapsed"]) for r in mine) / len(mine)
        stats.append((label, mean_tr, mean_en, worst, secs, len(mine)))

    def pct(x):
        return f"{x * 100:7.2f}%" if x is not None else f"{'—':>8}"

    for label, tr, en, worst, secs, n in sorted(stats, key=lambda s: s[3]):
        print(f"{label:16} {pct(tr)} {pct(en)} {pct(worst)} {secs:6.1f}s {n:5}")
    print()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")
    report()
