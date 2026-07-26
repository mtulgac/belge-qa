from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples"
INDEX_DIR = ROOT / "data" / "index"
GOLDEN_SET = ROOT / "data" / "golden_qa_sablon.yaml"

# --- Corpus ------------------------------------------------------------------
# The indexed documents. yonetmelik_ss.png and yonetmelik_foto.jpeg are NOT here:
# both are page 1 of the regulation and exist as OCR-test assets (the difficulty
# ladder), not as answer sources.
CORPUS = [
    "anadolu_yonetmelik.pdf",
    "arxiv_2104.13437.pdf",
    "resmi_gazete_1995.pdf",
    "tuik_cpi_tr.pdf",
    "tuik_cpi_en.pdf",
]

# The same content in three formats. Exactly ONE of them is in the index at a
# time: with all three, every answer comes back in triplicate and precision
# cannot be measured. The main retrieval test runs on the default variant; the
# other two are a separate test. In that test, format is the only variable, 
# so the metric delta is the cost OCR imposes on retrieval.
VARIANT_GROUP = [
    "tobb_yonetmelik.pdf",   # digital text layer
    "tobb_taranmis.pdf",     # fully scanned
    "karma.pdf",             # pages 2 and 4 scanned, the rest text layer
]
DEFAULT_VARIANT = "tobb_yonetmelik.pdf"

# Document language. Metrics are reported TR and EN separately, averaging lets
# a good English score mask Turkish. Includes the out-of-corpus test assets.
DOC_LANG = {
    "anadolu_yonetmelik.pdf": "TR",
    "arxiv_2104.13437.pdf": "EN",
    "karma.pdf": "TR",
    "resmi_gazete_1995.pdf": "TR",
    "tobb_taranmis.pdf": "TR",
    "tobb_yonetmelik.pdf": "TR",
    "tuik_cpi_en.pdf": "EN",
    "tuik_cpi_tr.pdf": "TR",
    "yonetmelik_foto.jpeg": "TR",
    "yonetmelik_ss.png": "TR",
}


def corpus(variant: str = DEFAULT_VARIANT) -> list[str]:
    """Active corpus: the fixed documents plus one document from the variant group."""
    if variant not in VARIANT_GROUP:
        raise ValueError(f"unknown variant: {variant} (options: {VARIANT_GROUP})")
    return CORPUS + [variant]


# --- Chunking ----------------------------------------------------------------
# To be swept on Day 4; reasonable starting values for now.
CHUNK_TARGET = 800       # characters
CHUNK_OVERLAP = 150      # characters
CHUNK_MIN = 200          # a piece shorter than this is merged into the previous one
# Hard ceiling. Sentence granularity alone is not enough: a definitions article
# ("MADDE 3- (1) ... a) ... b) ...") holds no sentence end and ran to 3302
# characters, past e5-small's 512-token window — the tail of such a chunk never
# reached the vector, silently. That window stopped being the binding constraint
# when bge-m3 (8192 tokens) was frozen as the model, so the ceiling now rests on
# chunk size affecting retrieval precision, which at this value is unmeasured.
# It is swept together with target/overlap rather than kept as a settled number.
CHUNK_MAX = 1600

# --- Embedding ---------------------------------------------------------------
# One multilingual model: separate TR and EN models would mean two vector spaces
# and "Turkish question -> English document" would stop working.

# Frozen by measurement (sweep_models.py, 4 candidates, k=5): bge-m3 leads on the
# worst-case language, TR 93.8% / EN 75.0% against 62.5% / 75.0% for the best of
# the rest, and it is the only candidate that does not lose ground cross-lingually
# — the e5 family collapses there, a family trait rather than a size one. The cost
# is ~35 ms per query and ~0.46 s per chunk at index time. Table in TESTING.md.
EMBEDDING_MODEL = "BAAI/bge-m3"

# The e5 family expects these prefixes and degrades silently without them; other
# families must NOT get them, so the prefix travels with the model, not with the
# code that calls it.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def prefixes(model_name: str) -> tuple[str, str]:
    """(query prefix, passage prefix) for a model."""
    if "e5" in model_name.lower():
        return E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
    return "", ""


def model_slug(model_name: str) -> str:
    """Filesystem-safe model id, so each model keeps its own index."""
    return model_name.replace("/", "__")
