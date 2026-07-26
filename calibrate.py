from pathlib import Path

import fitz

from app.needs_ocr import (
    BENIGN_INVISIBLE,
    MOJIBAKE_TR,
    MOJIBAKE_UTF8,
    needs_ocr,
)

SAMPLES = Path("data/samples")


def ratios():

    print("=" * 78)
    print("CALIBRATION — set thresholds by looking at these values (per page)")
    print("=" * 78)
    print(
        f"{'doc':26} {'page':>4} {'chars':>6} {'mojibake':>9} "
        f"{'unprint':>8} {'benign':>7}"
    )
    print("-" * 78)

    for p in sorted(SAMPLES.glob("*.pdf")):
        for i, page in enumerate(fitz.open(p)):
            text = page.get_text("text").strip()
            n = len(text)
            if n < 120:
                print(f"{p.name:26} p.{i + 1:<3} {n:>5}  — no text —")
                continue

            mojibake = (
                sum(1 for c in text if c in MOJIBAKE_TR)
                + sum(len(m) for m in MOJIBAKE_UTF8.findall(text))
            ) / n
            unprintable = sum(
                1 for c in text
                if not c.isprintable() and not c.isspace()
                and c not in BENIGN_INVISIBLE
            ) / n
            benign = sum(1 for c in text if c in BENIGN_INVISIBLE) / n

            print(
                f"{p.name:26} p.{i + 1:<3} {n:>5} {mojibake:9.4f} "
                f"{unprintable:8.4f} {benign:7.4f}"
            )

    print()
    print("unprint: put UNPRINTABLE_MAX between the sound pages and the corrupt ones.")
    print("benign: invisible formatting chars — excluded from unprint on purpose.")
    print()


def detection():

    print("=" * 78)
    print("DETECTION TABLE")
    print("=" * 78)

    counter = {}
    for p in sorted(SAMPLES.glob("*.pdf")):
        doc = fitz.open(p)
        for i, page in enumerate(doc):
            decision, reason = needs_ocr(page)
            label = "OCR" if decision else "text"
            print(f"{p.name:26} p.{i + 1:<3} {label:6} {reason}")
            counter[reason] = counter.get(reason, 0) + 1
        doc.close()

    print("-" * 78)
    for reason, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{reason:24} {count:4} pages")


if __name__ == "__main__":
    ratios()
    detection()
