import re
from dataclasses import dataclass, field

import ollama

from app.config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, resolve_num_predict

# The sentinel the model must emit when the passages do not contain the answer.
# Chosen over free-form refusal so the gate is a deterministic string test rather
# than fuzzy phrase matching across two languages.
REFUSAL = "[YANITLANAMADI]"

# Reasoning models (were excluded as candidates, but a future one might slip in) wrap
# their scratchpad in <think>...</think>. Strip it before parsing so the sentinel and
# citations are read off the final answer only, never the reasoning trace.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Citations are passage numbers: [1], [3], or a run like [1][2]. The model only has to
# echo the number of the passage it used; generate() maps that back to the real
# (dosya, sayfa) from the hit list. This is far more robust than asking the model to
# copy file names — a small model writes "[dosya s.4]" literally ("dosya" is also a
# Turkish word) and the source is lost. The number can't be faked into a wrong file.
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Name patterns that mark a reasoning model, so `think=False` is passed to suppress the
# trace (a latency win). Independent of _THINK_RE, which is the safety net if it leaks.
# gemma4:e4b (the large tier) is a thinking model too but is DELIBERATELY not listed:
# measured, its hidden reasoning drops hallucinations on the reddet rows 4/10 -> 1/10, and
# faithfulness is the graded priority, so large keeps thinking on (it gets the wider
# num_predict in config to fit reasoning + answer). turbo (qwen3.5:4b) matches here because
# its reasoning is 4600-6000+ tokens/query — disqualifying on latency for the fast tier.
_REASONING_RE = re.compile(r"(qwen3|deepseek-?r1|[-:/]r1\b|reason)", re.IGNORECASE)

SYSTEM_PROMPT = """You are a document question-answering assistant. Answer using ONLY \
the passages provided under BELGELER. Follow these rules exactly:

1. Use only information stated in the passages. Do not add outside knowledge and do \
not infer beyond what is written.
2. If the passages do not contain the answer, reply beginning with exactly the token \
[YANITLANAMADI] and then one short sentence saying what is missing. This includes a \
question that assumes something the documents do not support (a wrong premise): reply \
[YANITLANAMADI] and state the correct premise.
3. When you do answer, cite every claim with the number of the passage it came from, \
in square brackets, e.g. [1] or [3]. Cite the passage number only, never the file name.
4. If the passages give conflicting answers, present each value with its own source. \
Do not collapse them into a single answer.
5. Answer in the same language as the question. Be concise."""


@dataclass
class GenResult:
    raw: str                                   # the model's reply, verbatim
    grounded: bool                             # False => the model emitted the sentinel
    cevap: str                                 # answer text (or the refusal reason)
    kaynaklar: list[dict] = field(default_factory=list)  # [{"dosya":..., "sayfa":int}]


def _page_label(hit) -> str:
    if hit.sayfa_baslangic == hit.sayfa_bitis:
        return f"s.{hit.sayfa_baslangic}"
    return f"s.{hit.sayfa_baslangic}-{hit.sayfa_bitis}"


def format_context(hits) -> str:
    """The passages block: each hit numbered, with a `dosya s.N` header the model
    copies into its citations."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] {h.dosya} {_page_label(h)}\n{h.metin}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits) -> list[dict]:
    context = format_context(hits)
    user = f"BELGELER:\n{context}\n\nSORU: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_citations(text: str, hits) -> list[dict]:
    """Map the passage numbers the answer cites ([1], [3], ...) back to real sources.

    Deduped, order preserved. Out-of-range numbers (the model invented a passage) are
    dropped — a citation can only point at a passage that was actually shown, which is
    the guarantee the number scheme buys over free-form file names."""
    seen, out = set(), []
    for m in _CITATION_RE.finditer(text):
        i = int(m.group(1))
        if not (1 <= i <= len(hits)):
            continue
        h = hits[i - 1]
        key = (h.dosya, h.sayfa_baslangic)  # dedup on the source, not the passage index:
        if key not in seen:                 # two chunks can share a (dosya, sayfa)
            seen.add(key)
            out.append({"dosya": key[0], "sayfa": key[1]})
    return out


def generate(question: str, hits, model: str = LLM_MODEL) -> GenResult:
    """Run one grounded generation. `hits` are the reranked chunks (retrieval.Hit)."""
    if not hits:
        # Nothing to ground on. The pipeline's stage-1 gate normally catches this
        # first; guard here too so generate() is safe to call directly.
        return GenResult(raw="", grounded=False, cevap="Belge bulunamadı.")

    kwargs = dict(
        model=model,
        messages=build_messages(question, hits),
        options={"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX,
                 "num_predict": resolve_num_predict(model)},
    )
    if _REASONING_RE.search(model):
        kwargs["think"] = False
    try:
        resp = ollama.chat(**kwargs)
    except TypeError:
        # Older ollama clients do not accept `think`; the <think> stripper covers it.
        kwargs.pop("think", None)
        resp = ollama.chat(**kwargs)

    raw = resp["message"]["content"]
    text = _strip_thinking(raw)

    if not text:
        # No answer text came back. A thinking model can spend its whole num_predict
        # budget on hidden reasoning (message.thinking) before the answer starts, or a
        # length cap can cut it off first — either way message.content is empty. Report
        # it as ungrounded so the pipeline abstains with a reason instead of surfacing a
        # blank answer. (Per-model num_predict is sized to make this rare; this is the
        # backstop for the tail.)
        return GenResult(raw=raw, grounded=False, cevap="cevap üretilemedi")

    if REFUSAL in text:
        reason = text.replace(REFUSAL, "").strip()
        return GenResult(raw=raw, grounded=False, cevap=reason)

    return GenResult(raw=raw, grounded=True, cevap=text, kaynaklar=_extract_citations(text, hits))
