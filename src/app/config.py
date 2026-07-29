from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples"
INDEX_DIR = ROOT / "data" / "index"
GOLDEN_SET = ROOT / "data" / "golden_qa_sablon.yaml"

# Web UI runtime state. Uploaded files land in UPLOADS_DIR; their index lives under
# data/index/runtime/ via the same index_dir() layout as the measured variants, so
# retrieval needs no special case. RUNTIME_VARIANT is passed wherever a variant goes.
# Starts EMPTY (user decision): the UI searches only what the user uploaded; the baked
# corpus indexes stay measurement-only. Both paths are Docker named volumes.
UPLOADS_DIR = ROOT / "data" / "uploads"
RUNTIME_VARIANT = "runtime"

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
# Swept Day 6 (sweep_chunk.py): 12 configs, target 600-1200 x overlap 100-200,
# dense on the golden set. No monotonic chunk-size effect, so the difference is
# redundant end-to-end. 800/150 held, now a measured pick, not a default. Table in TESTING.md.
CHUNK_TARGET = 800       # characters
CHUNK_OVERLAP = 150      # characters
CHUNK_MIN = 200          # a piece shorter than this is merged into the previous one

# Swept Day 6 as 2x target (see CHUNK_TARGET): no separate gain, so 1600 stays.
CHUNK_MAX = 1600

# --- Embedding ---------------------------------------------------------------
# One multilingual model: separate TR and EN models would mean two vector spaces
# and "Turkish question -> English document" would stop working.

# Frozen by measurement (sweep_models.py, 4 candidates, k=5):
# bge-m3 leads on the worst-case language.
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


# --- Retrieval ---------------------------------------------------------------
# Frozen to "rerank" (dense -> cross-encoder) as the runtime path.
# See the Reranking block in TESTING.md.
RETRIEVER = "rerank"     # dense | bm25 | hibrit | rerank

# Reciprocal Rank Fusion. Rank-based rather than a weighted sum of scores, and
# the reason is structural rather than measured.
# In RRF a retriever that does not rank a chunk simply contributes nothing to it.
RRF_K = 10               # the constant in w/(RRF_K + rank); the paper's 60 flattens
                         # the top ranks and measured no better here
RRF_DEPTH = 20           # candidates per retriever before fusing.
# (dense, bm25). Equal weight is an assumption, not a neutral default.
RRF_WEIGHTS = (2.0, 1.0)

# Okapi defaults. NOT swept, recorded as unswept rather than presented as chosen.
BM25_K1 = 1.5
BM25_B = 0.75

# Which token pattern app.lexical uses. "keep" holds numbers and identifiers
# together ("2,50" stays one token instead of becoming "2" and "50"); "split" is
# the plain \w+ baseline.
BM25_TOKEN = "keep"

# --- Reranking ---------------------------------------------------------------

# Frozen in Day 4.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

RERANK_CANDIDATES = [
    "BAAI/bge-reranker-v2-m3",                     # 568M, quality ceiling
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 118M, fastest, TR coverage unknown
]

# How many base candidates the cross-encoder re-scores, the direct driver of
# rerank latency, one forward pass each. Frozen at 10 (swept 5/10/20).
RERANK_DEPTH = 10

# Which retriever produces the candidates the reranker re-scores. Frozen to "dense".
RERANK_BASE = "dense"    # hibrit | dense

# Device for the reranker forward pass. Pinned to CPU on purpose.
RERANK_DEVICE = "cpu"

# --- Generation (LLM) --------------------------------------------------------
# Two production tiers, user-selectable (the web UI exposes turbo/large). Both frozen
# by eval_generation.py on the golden set, chosen on the project's priority.
LLM_TIERS = {
    "turbo": "qwen3.5:4b",   # fast, interactive
    "large": "gemma4:e4b",   # best faithfulness / lowest hallucination
}
# turbo is the default: interactive latency wins for the demo UI
LLM_DEFAULT_TIER = "turbo"
LLM_MODEL = LLM_TIERS[LLM_DEFAULT_TIER]
# The eval set going forward is the two production models; the wider sweep that picked
# them is recorded in DEVLOG/TESTING, not re-run by default.
LLM_CANDIDATES = list(LLM_TIERS.values())


def resolve_model(tier_or_name: str) -> str:
    """A tier alias (turbo/large) -> its model; any other string is taken as a model
    name as-is, so callers can pass either."""
    return LLM_TIERS.get(tier_or_name, tier_or_name)


# The LLM-as-judge model for judge_generation.py (optional offline correctness layer).
JUDGE_MODEL = "gemma4:12b"

# Grounded extraction, not open generation: temperature 0 so the answer is
# reproducible and stays on the passages.
LLM_TEMPERATURE = 0.0
# Ollama context window. bge-m3 chunks cap ~512 tokens; GEN_K of them plus the
# prompt fits comfortably, and Ollama truncates silently past num_ctx otherwise.
LLM_NUM_CTX = 8192
LLM_NUM_PREDICT = 512  # default / turbo
LLM_NUM_PREDICT_BY_MODEL = {
    "gemma4:e4b": 1536,  # thinking (~700-900 tok) + answer, ~40% margin over measured 1106
}


def resolve_num_predict(model: str) -> int:
    """The output cap for a concrete model name (not a tier alias)."""
    return LLM_NUM_PREDICT_BY_MODEL.get(model, LLM_NUM_PREDICT)

# How many reranked chunks are handed to the LLM. Matches the retrieval k the
# reranker returns; the abstention gate reads the top chunk's score.
GEN_K = 5

# --- Abstention gate ---------------------------------------------------------
REJECT_THRESHOLD = 0.003
