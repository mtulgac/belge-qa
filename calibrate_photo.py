import time

import cv2
import fitz
import numpy as np

from app.config import SAMPLES
from app.ocr import (
    PHOTO_SPREAD_THRESHOLD,
    illumination_spread,
    is_photo,
    method_ocr_preprocessed,
    method_ocr_raw,
    ocr_confidence,
    page_image,
)
from evaluate import cer

# Reference: the digital text layer of the regulation (page 1) — the same ground
# truth design as the bake-off, no hand transcription. Inputs that are not that
# page get no CER column; scoring them against it would print a number that
# looks like a verdict and is not one.
REFERENCE = ("tobb_yonetmelik.pdf", 0)

# Below this gap the two methods are tied — the corpus has pairs that differ by
# 0.04 points, which is reference noise, not a win for either side.
TIE_POINTS = 1.0


def synthetic_photo(img: np.ndarray) -> np.ndarray:
    """Add a camera-like radial shadow and slight focus loss to a clean page."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w * 0.35) / w) ** 2 + ((yy - h * 0.3) / h) ** 2)
    shade = np.clip(1.0 - 0.85 * r, 0.35, 1.0).astype(np.float32)
    out = np.clip(img.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (3, 3), 0)


def pdf_page(name: str, index: int) -> np.ndarray:
    doc = fitz.open(SAMPLES / name)
    img = page_image(doc[index])
    doc.close()
    return img


def reference_text() -> str:
    name, index = REFERENCE
    doc = fitz.open(SAMPLES / name)
    text = doc[index].get_text("text")
    doc.close()
    return text


def inputs() -> list[tuple[str, np.ndarray, bool]]:
    """(label, image, is the page-1 reference valid for it)."""
    clean = pdf_page("tobb_taranmis.pdf", 0)
    return [
        ("scanned p.1", clean, True),
        ("synthetic shadow", synthetic_photo(clean), True),
        ("real photo", cv2.imread(str(SAMPLES / "yonetmelik_foto.jpeg")), True),
        ("screenshot", cv2.imread(str(SAMPLES / "yonetmelik_ss.png")), True),
        ("digital render p.1", pdf_page("tobb_yonetmelik.pdf", 0), True),
        # A scanned page of a different document: no reference, so it only
        # contributes a negative sample for the spread signal.
        ("karma p.2 (scanned)", pdf_page("karma.pdf", 1), False),
    ]


def main() -> None:
    ref = reference_text()

    print("=" * 98)
    print(f"PREPROCESSING GATE — threshold: spread >= {PHOTO_SPREAD_THRESHOLD}")
    print("=" * 98)
    print(
        f"{'input':22} {'spread':>8} {'gate':>6} "
        f"{'conf_raw':>9} {'conf_pre':>9} {'conf pick':>10} "
        f"{'CER raw':>9} {'CER pre':>8} {'truth':>7} {'2-pass':>8}"
    )
    print("-" * 98)

    scored = {"gate": 0, "conf": 0, "total": 0}

    for label, img, has_ref in inputs():
        spread = illumination_spread(img)
        gate = "pre" if is_photo(img) else "raw"

        t0 = time.perf_counter()
        conf_raw = ocr_confidence(img)
        conf_pre = ocr_confidence(img, do_preprocess=True)
        two_pass = time.perf_counter() - t0
        conf_pick = "pre" if conf_pre > conf_raw else "raw"

        if has_ref:
            cer_raw = cer(method_ocr_raw(img)[0], ref) * 100
            cer_pre = cer(method_ocr_preprocessed(img)[0], ref) * 100
            if abs(cer_raw - cer_pre) < TIE_POINTS:
                truth = "tie"
            else:
                truth = "pre" if cer_pre < cer_raw else "raw"
            cer_cols = f"{cer_raw:8.2f}% {cer_pre:7.2f}%"
            if truth != "tie":
                scored["total"] += 1
                scored["gate"] += gate == truth
                scored["conf"] += conf_pick == truth
        else:
            truth, cer_cols = "—", f"{'—':>9} {'—':>8}"

        print(
            f"{label:22} {spread:8.1f} {gate:>6} "
            f"{conf_raw:9.1f} {conf_pre:9.1f} {conf_pick:>10} "
            f"{cer_cols} {truth:>7} {two_pass:7.1f}s"
        )

    print()
    print("'truth' is the actual winner by CER; 'tie' means the two methods land")
    print(f"within {TIE_POINTS} points, which this corpus cannot separate. Only the")
    print("decided rows are scored:")
    print(f"  illumination_spread gate : {scored['gate']}/{scored['total']} correct, ~0s")
    print(f"  two-pass ocr_confidence  : {scored['conf']}/{scored['total']} correct, ~3-5s per page")
    print()
    print("NOTE: CER is measured against the page-1 reference, so inputs that are")
    print("not that page carry no CER column — a number there would look like a")
    print("verdict without being one.")


if __name__ == "__main__":
    main()
