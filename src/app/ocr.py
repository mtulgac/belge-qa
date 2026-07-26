import time

import cv2
import fitz
import numpy as np
import pytesseract

# Frozen by measurement (sweep_tesseract.py)
OCR_LANG = "tur+eng"
OCR_CONFIG = "--psm 3"
RENDER_DPI = 300

# Preprocessing gate, calibrated by calibrate_photo.py. Measured spread: photo
# 72.0 and a synthetically shadowed page 135.0, against 0.0 for scanned pages,
# the screenshot and digital renders. The threshold sits high in that gap, not
# in the middle: a false positive preprocesses clean input and takes CER from
# 3.27% to 33.75%, while a false negative merely leaves a photo on raw OCR
# (21.50%). The costs are asymmetric, so the gate is conservative.
PHOTO_SPREAD_THRESHOLD = 25.0

# EasyOCR languages: Turkish and English are both Latin-script, so the combo is
# valid. The Reader is heavy to build, so it is created once, lazily.
EASYOCR_LANGS = ["tr", "en"]
_easyocr_reader = None

# RapidOCR (PaddleOCR PP-OCRv6 models on ONNX Runtime). "tr" selects the
# Turkish recognition dictionary; its Latin coverage includes English, so one
# engine serves both corpus languages (same reasoning as EASYOCR_LANGS).
RAPIDOCR_LANG = "tr"
_rapidocr_engine = None


def page_image(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:

    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _background(gray: np.ndarray) -> np.ndarray:
    """Background estimate that carries the illumination: a MORPH_CLOSE wide
    enough to swallow the glyphs."""
    h, w = gray.shape
    k = max(15, int(min(h, w) * 0.04)) | 1   # must exceed the largest glyph
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def illumination_spread(img: np.ndarray) -> float:
    """Spread of background brightness (0-255). High on a photo because of
    shadow and angled light, ~0 on a scan or screenshot — those are flat white."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bg = _background(cv2.medianBlur(gray, 3))
    p5, p95 = np.percentile(bg, [5, 95])
    return float(p95 - p5)


def is_photo(img: np.ndarray) -> bool:
    """The preprocessing gate: was this input taken with a camera?"""
    return illumination_spread(img) >= PHOTO_SPREAD_THRESHOLD


def preprocess(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.medianBlur(gray, 3)

    h, w = gray.shape
    bg = _background(gray)
    norm = cv2.divide(gray, bg, scale=255)
    # Skew: find the angle from the minimum rotated rectangle of the text pixels
    inverted = cv2.bitwise_not(norm)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) > 100:
        # minAreaRect expects (x, y) points; np.where yields (row, col) = (y, x).
        # Feeding (y, x) fits the rect to a mirrored page — the angle comes out
        # wrong and the "correction" used to rotate pages 90°.
        pts = np.column_stack([xs, ys]).astype(np.float32)
        angle = cv2.minAreaRect(pts)[-1]
        # Fold both OpenCV angle conventions ((0, 90] and [-90, 0)) into
        # (-45, 45], so an upright page maps to ~0, not ±90.
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if abs(angle) > 0.5:                    # ignore corrections below 0.5 degrees
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            norm = cv2.warpAffine(
                norm, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255,
            )

    # Illumination is flattened above, so a global Otsu is sufficient here.
    _, out = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return out


# --- Methods -----------------------------------------------------------------

def method_text_layer(page: fitz.Page) -> tuple[str, float]:

    t0 = time.perf_counter()
    text = page.get_text("text").strip()
    return text, time.perf_counter() - t0


def method_ocr_raw(page_or_img) -> tuple[str, float]:

    t0 = time.perf_counter()
    img = (
        page_image(page_or_img)
        if isinstance(page_or_img, fitz.Page)
        else page_or_img
    )
    text = pytesseract.image_to_string(img, lang=OCR_LANG, config=OCR_CONFIG)
    return text.strip(), time.perf_counter() - t0


def method_ocr_preprocessed(page_or_img) -> tuple[str, float]:

    t0 = time.perf_counter()
    img = (
        page_image(page_or_img)
        if isinstance(page_or_img, fitz.Page)
        else page_or_img
    )
    text = pytesseract.image_to_string(
        preprocess(img), lang=OCR_LANG, config=OCR_CONFIG
    )
    return text.strip(), time.perf_counter() - t0


def _get_easyocr_reader():

    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(EASYOCR_LANGS)
    return _easyocr_reader


def method_easyocr(page_or_img) -> tuple[str, float]:

    t0 = time.perf_counter()
    img = (
        page_image(page_or_img)
        if isinstance(page_or_img, fitz.Page)
        else page_or_img
    )
    lines = _get_easyocr_reader().readtext(img, detail=0, paragraph=True)
    text = "\n".join(lines)
    return text.strip(), time.perf_counter() - t0


def _get_rapidocr_engine():

    global _rapidocr_engine
    if _rapidocr_engine is None:
        import logging
        from rapidocr import RapidOCR
        # rapidocr re-arms its INFO logger inside lazily-imported submodules
        # during construction; suppress globally for the build only, so the
        # model-path chatter doesn't flood parse_all's table output.
        logging.disable(logging.INFO)
        try:
            _rapidocr_engine = RapidOCR(params={"Rec.lang_type": RAPIDOCR_LANG})
        finally:
            logging.disable(logging.NOTSET)
    return _rapidocr_engine


def method_rapidocr(page_or_img) -> tuple[str, float]:

    t0 = time.perf_counter()
    img = (
        page_image(page_or_img)
        if isinstance(page_or_img, fitz.Page)
        else page_or_img
    )
    result = _get_rapidocr_engine()(img)
    text = "\n".join(result.txts) if result.txts else ""
    return text.strip(), time.perf_counter() - t0


def ocr_confidence(page_or_img, do_preprocess: bool = False) -> float:

    img = (
        page_image(page_or_img)
        if isinstance(page_or_img, fitz.Page)
        else page_or_img
    )
    if do_preprocess:
        img = preprocess(img)
    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )
    scores = [int(c) for c in data["conf"] if int(c) >= 0]
    return sum(scores) / len(scores) if scores else 0.0
