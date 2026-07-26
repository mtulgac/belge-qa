import csv
import sys
import time
from pathlib import Path

import cv2
import fitz

from app.needs_ocr import needs_ocr
from app.ocr import (
    method_easyocr,
    method_ocr_preprocessed,
    method_ocr_raw,
    method_rapidocr,
    method_text_layer,
)

SAMPLES = Path("data/samples")
OUT = Path("out/parsed")
SUMMARY_CSV = OUT / "_summary.csv"
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")

PDF_METHODS = {
    "text_layer": method_text_layer,
    "ocr_raw": method_ocr_raw,
    "ocr_preprocessed": method_ocr_preprocessed,
    "easyocr": method_easyocr,
    "rapidocr": method_rapidocr,
}
IMAGE_METHODS = {
    "ocr_raw": method_ocr_raw,
    "ocr_preprocessed": method_ocr_preprocessed,
    "easyocr": method_easyocr,
    "rapidocr": method_rapidocr,
}


def tr_ratio(text: str) -> float:
    return sum(1 for c in text if c in TR_CHARS) / len(text) if text else 0.0


def process_pdf(path: Path) -> list[dict]:
    doc = fitz.open(path)
    results = []

    for name, method in PDF_METHODS.items():
        parts, total_time = [], 0.0
        try:
            for page in doc:
                text, elapsed = method(page)
                parts.append(text)
                total_time += elapsed
        except Exception as e:                  # keep the run alive; log and move on
            print(f"  ! {path.name} {name} failed: {e}")
            continue
        full_text = "\n\n".join(parts)

        results.append({
            "doc": path.name,
            "method": name,
            "text": full_text,
            "elapsed": total_time,
            "pages": len(doc),
        })

    # The router's decision — informational, does not gate extraction
    decisions = [needs_ocr(p)[1] for p in doc]
    doc.close()
    for r in results:
        r["router"] = max(set(decisions), key=decisions.count)
    return results


def process_image(path: Path) -> list[dict]:
    img = cv2.imread(str(path))
    results = []
    for name, method in IMAGE_METHODS.items():
        try:
            text, elapsed = method(img)
        except Exception as e:
            print(f"  ! {path.name} {name} failed: {e}")
            continue
        results.append({
            "doc": path.name, "method": name, "text": text,
            "elapsed": elapsed, "pages": 1, "router": "image",
        })
    return results


def save(result: dict) -> None:
    folder = OUT / Path(result["doc"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{result['method']}.txt").write_text(result["text"], encoding="utf-8")


def main() -> None:
    filter_str = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(
        p for p in SAMPLES.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
        and filter_str in p.name
    )
    if not files:
        print("No documents found."); return

    print(f"{'doc':24} {'method':17} {'chars':>9} {'tr%':>6} {'time':>7}")
    print("-" * 68)

    rows = []
    start = time.perf_counter()
    for path in files:
        results = (
            process_pdf(path) if path.suffix.lower() == ".pdf" else process_image(path)
        )
        for r in results:
            save(r)
            ratio = tr_ratio(r["text"])
            rows.append({
                "doc": r["doc"], "method": r["method"],
                "chars": len(r["text"]), "tr_ratio": round(ratio, 4),
                "elapsed": round(r["elapsed"], 2),
            })
            print(
                f"{r['doc']:24} {r['method']:17} {len(r['text']):9} "
                f"{ratio * 100:5.1f}% {r['elapsed']:6.1f}s"
            )
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc", "method", "chars", "tr_ratio", "elapsed"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 68)
    print(f"Total {time.perf_counter() - start:.1f}s")
    print(f"Outputs: {OUT}/")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
