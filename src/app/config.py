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
# Reasonable starting values, NOT yet swept — Day 4 went to retrieval/reranking
# instead, so target/overlap stay defaults rather than measured picks. To be swept
# together with CHUNK_MAX (see below).
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


# --- Retrieval ---------------------------------------------------------------
# Dense and BM25 answer different questions: the embedding matches meaning across
# wording and across languages, BM25 matches the literal string. The second is
# what a document corpus needs for article numbers, grade thresholds and proper
# nouns, where being close in meaning is worthless.
# Frozen by sweep_retrievers.py (18 fusion settings, k=5). Chosen on the worst-case
# language, the rule the project already uses: hybrid 90.6% against dense's 75.0%.
# On Turkish alone dense is still ahead (93.8% vs 90.6%), so this is a trade, not a
# clean win — and the trade is literally two questions, q009 (EN) gained against
# q011 (TR) lost, which at n=20 is inside the noise. The sturdier signals are that
# English recall goes 75.0% -> 100% and MRR 0.68 -> 0.72; MRR moves continuously per
# question rather than in 5-point steps. Table in TESTING.md.
RETRIEVER = "hibrit"     # dense | bm25 | hibrit

# Reciprocal Rank Fusion. Rank-based rather than a weighted sum of scores, and
# the reason is structural rather than measured: on the capraz_dil questions
# (Turkish query, English document) BM25 has zero term overlap by construction,
# and a weighted sum would let those zeros drag down rows where dense is at 100%.
# In RRF a retriever that does not rank a chunk simply contributes nothing to it.
RRF_K = 10               # the constant in w/(RRF_K + rank); the paper's 60 flattens
                         # the top ranks and measured no better here
RRF_DEPTH = 20           # candidates per retriever before fusing. At 50 the hybrid
                         # loses to dense outright (82.5%) and drops capraz_dil to
                         # 66.7%: past its confident hits BM25 contributes noise
                         # that still carries a full vote.
# (dense, bm25). Equal weight is an assumption, not a neutral default: BM25 alone
# reaches 62.5% recall@5 here against dense's 90.0%, so at 1:1 the weaker list
# outvotes the stronger one wherever it is merely noisy — measured, 1:1 costs 10
# points of Turkish recall at depth 20. 3:1 measured identical to 2:1.
RRF_WEIGHTS = (2.0, 1.0)

# Okapi defaults. NOT swept — recorded as unswept rather than presented as chosen.
BM25_K1 = 1.5
BM25_B = 0.75

# Which token pattern app.lexical uses. "keep" holds numbers and identifiers
# together ("2,50" stays one token instead of becoming "2" and "50"); "split" is
# the plain \w+ baseline.
#
# NOT settled by the golden set: the two tie on every metric, alone and inside the
# hybrid. "keep" is here on an isolated test instead — querying "2,50" under
# "split" ranks a document containing "2 yıl ... 50 kredi" ABOVE the one that
# actually says 2,50, because the fragments are common enough to carry no IDF.
# The golden set holds no question that reaches that failure, so on the real
# corpus this axis is unmeasured, not decided.
BM25_TOKEN = "keep"

# --- Reranking ---------------------------------------------------------------
# A cross-encoder over the base retriever's candidates: it scores each
# (question, chunk) pair jointly in one forward pass, instead of comparing two
# independently built embeddings. That joint view is why it is the candidate
# abstention signal — a single cosine could not separate answerable from
# unanswerable (Day 3: AUC 0.790, 8/10 reddet rows inside the yanitla range) and
# the RRF score is bounded flat.
#
# Frozen Day 4. The 30-question golden set cannot separate bge from the 118M mmarco
# (both hit the 97.5% recall ceiling) — an inability to measure, not evidence of
# equality — so the choice falls to prior + abstention: bge is genuinely
# multilingual / TR-strong and its abstention signal is cleaner (AUC 0.840 vs
# 0.800). mmarco stays the equivalent, lighter alternative (wins latency only on
# short chunks). Latency was measured FIRST (bench_rerank.py) because a cross-encoder
# is structurally expensive — one pass per candidate — then recall/abstention
# (sweep_rerank.py).
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Spread across the size axis so the latency x recall curve is visible. The pick
# is frozen only after measurement — this list is the search space, not a ranking.
# gte-multilingual and jina-reranker-v2 were dropped: both ship custom remote code
# that is incompatible with the pinned transformers (jina fails to import, gte
# crashes mid-scoring), so the standard-architecture pair is what we measure.
RERANK_CANDIDATES = [
    "BAAI/bge-reranker-v2-m3",                     # 568M — quality ceiling
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 118M — fastest, TR coverage unknown
]

# How many base candidates the cross-encoder re-scores — the direct driver of
# rerank latency, one forward pass each. Frozen at 10 (swept 5/10/20): recall@5
# 95.0% at ~3.3 s/query on this CPU against depth 20's 97.5% at ~6.9 s. The +2.5 is
# a single çelişki anchor (q041), not worth 2x latency; depth 10 still beats hybrid
# (92.5%) and dense (90.0%), and abstention is depth-independent. Whatever depth is
# fed, the reranker still returns the top-k.
RERANK_DEPTH = 10

# Which retriever produces the candidates the reranker re-scores. Frozen to "dense"
# by the base ablation: reranking dense's top-`depth` matches or beats reranking the
# hybrid's (97.5% vs 95.0% at depth 20), because the reranker recovers q009 — the
# proper-noun question BM25 was added for — from the dense pool's top-20. Fusion is
# therefore redundant in front of the reranker; hibrit stays in the repo as measured
# evidence, not the path.
RERANK_BASE = "dense"    # hibrit | dense

# Device for the reranker forward pass. Pinned to CPU on purpose: the on-prem
# target has no GPU (the project's on-prem assumption), and sentence-transformers
# otherwise auto-selects MPS on Apple Silicon — which both mislabels the latency
# (measured: MPS hides a 7x model-size gap behind a per-call dispatch floor) and
# ties the scores to the dev machine. CPU keeps latency honest for the deployment
# target and keeps recall reproducible across machines. Measured per-pair cost is
# model- and chunk-length-dependent (bench_rerank.py).
RERANK_DEVICE = "cpu"
