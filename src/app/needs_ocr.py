import re
import fitz

# --- Thresholds --------------------------------------------------------------
MIN_CHARS = 120          # fewer characters than this and the page counts as empty
MOJIBAKE_MAX = 0.001     # if the ð/þ/ý ratio exceeds this, encoding is broken
UNPRINTABLE_MAX = 0.005  # ratio of unprintable characters (excluding BENIGN_INVISIBLE).
IMAGE_COVER_MIN = 0.75   # if an image covers this much of the page
IMAGE_TEXT_MAX = 600     # and there is less text than this, the body was scanned

# --- Patterns ----------------------------------------------------------------
CID = re.compile(r"\(cid:\d+\)")

# Turkish ISO-8859-9 text decoded as latin1: ğ/ş/ı come out as ð/þ/ý
MOJIBAKE_TR = set("ðþýÐÞÝ")
# UTF-8 bytes decoded as cp1252/latin1: ’ -> â€™, é -> Ã©, etc. These byte
# artifacts never occur in real TR/EN text, so one occurrence is enough.
MOJIBAKE_UTF8 = re.compile(r"â€.|Ã[\x80-\xff]")

# Invisible formatting characters that isprintable() rejects but that appear in
# perfectly clean digital text layers. Counting them as garbage sends clean
# pages to OCR, the expensive error.
BENIGN_INVISIBLE = {
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM / zero-width no-break space
}


def _ratio(text: str, chars: set) -> float:
    return sum(1 for c in text if c in chars) / len(text)

def needs_ocr(page: fitz.Page) -> tuple[bool, str]:
    text = page.get_text("text").strip()

    # 1. No text: a fully scanned page
    if len(text) < MIN_CHARS:
        return True, "no_text"

    # 2. Font not embedded: characters come out like (cid:34)
    if CID.search(text):
        return True, "broken_font"

    # 3. Broken encoding: TR ISO-8859-9 mix-up by character ratio, or UTF-8
    #    read as cp1252 by byte-artifact pattern (language-independent)
    if _ratio(text, MOJIBAKE_TR) > MOJIBAKE_MAX or MOJIBAKE_UTF8.search(text):
        return True, "mojibake"

    # 4. Garbage characters: unprintable and not a benign formatting character
    broken = sum(
        1 for c in text
        if not c.isprintable() and not c.isspace() and c not in BENIGN_INVISIBLE
    )
    if broken / len(text) > UNPRINTABLE_MAX:
        return True, "garbage_chars"

    # 5. An image covers the page and there is little text, the body was
    # scanned, and the captured text is probably just header/footer
    page_area = abs(page.rect)
    image_area = sum(abs(fitz.Rect(i["bbox"])) for i in page.get_image_info())
    if image_area / page_area > IMAGE_COVER_MIN and len(text) < IMAGE_TEXT_MAX:
        return True, "image_heavy"

    return False, "text_layer"
