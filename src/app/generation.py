import re
from dataclasses import dataclass, field

import ollama

from app.config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, resolve_num_predict

# The sentinel the model must emit when the passages do not contain the answer.
# Chosen over free-form refusal so the gate is a deterministic string test rather
# than fuzzy phrase matching across two languages.
REFUSAL = "[YANITLANAMADI]"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_CITATION_RE = re.compile(r"\[(\d+)\]")

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


def _consume_stream(chunks, on_stage, on_token) -> str:
    parts: list[str] = []
    thinking_seen = content_seen = False
    for chunk in chunks:
        msg = chunk["message"]
        if not thinking_seen and (msg.get("thinking") or "") and on_stage:
            on_stage("thinking")
            thinking_seen = True
        delta = msg.get("content") or ""
        if delta:
            if not content_seen and on_stage:
                on_stage("writing")
                content_seen = True
            parts.append(delta)
            if on_token:
                on_token(delta)
    return "".join(parts)


def generate(
    question: str,
    hits,
    model: str = LLM_MODEL,
    on_stage=None,
    on_token=None,
) -> GenResult:
    if not hits:
        # Nothing to ground on. The pipeline's stage-1 gate normally catches this
        # first; guard here too so generate() is safe to call directly.
        return GenResult(raw="", grounded=False, cevap="Belge bulunamadı.")

    stream = on_stage is not None or on_token is not None
    kwargs = dict(
        model=model,
        messages=build_messages(question, hits),
        options={"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX,
                 "num_predict": resolve_num_predict(model)},
    )
    if stream:
        kwargs["stream"] = True
    if _REASONING_RE.search(model):
        kwargs["think"] = False
    try:
        resp = ollama.chat(**kwargs)
    except TypeError:
        # Older ollama clients do not accept `think`; the <think> stripper covers it.
        kwargs.pop("think", None)
        resp = ollama.chat(**kwargs)

    raw = _consume_stream(resp, on_stage, on_token) if stream else resp["message"]["content"]
    text = _strip_thinking(raw)

    if not text:
        return GenResult(raw=raw, grounded=False, cevap="cevap üretilemedi")

    if REFUSAL in text:
        reason = text.replace(REFUSAL, "").strip()
        return GenResult(raw=raw, grounded=False, cevap=reason)

    return GenResult(raw=raw, grounded=True, cevap=text, kaynaklar=_extract_citations(text, hits))
