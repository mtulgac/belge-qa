from dataclasses import dataclass, field

from app.config import GEN_K, LLM_MODEL, REJECT_THRESHOLD, RETRIEVER
from app.generation import generate
from app.retrieval import Hit, search


@dataclass
class Answer:
    cevap: str                                     # the answer, or the refusal reason
    reddedildi: bool                               # True => the system abstained
    gerekce: str = ""                              # which stage rejected, and why
    kaynaklar: list[dict] = field(default_factory=list)  # [{"dosya":..., "sayfa":int}]
    hits: list[Hit] = field(default_factory=list)  # the retrieved chunks (diagnostics)


def answer(
    question: str,
    session_id: str | None = None,
    history: list | None = None,
    *,
    model: str = LLM_MODEL,
    k: int = GEN_K,
    retriever: str = RETRIEVER,
) -> Answer:
    """Answer one question against the index, or abstain.

    `session_id` / `history` are accepted so the signature is stable for the web UI,
    but multi-turn is not wired: the golden set is single-turn, so follow-up rewriting
    and conversation state are deferred to the web-UI day. Both are ignored here.
    """
    hits = search(question, k=k, retriever=retriever)

    # Stage 1: cheap reranker-score pre-filter. Empty hits or a top score below the
    # zero-false-reject threshold means an off-topic question — reject without paying
    # for the LLM call.
    if not hits or hits[0].skor < REJECT_THRESHOLD:
        return Answer(
            cevap="",
            reddedildi=True,
            gerekce="ilgili belge bulunamadı",
            hits=hits,
        )

    # Stage 2: LLM grounding/citation backstop — the authoritative gate.
    result = generate(question, hits, model=model)
    if not result.grounded:
        return Answer(
            cevap="",
            reddedildi=True,
            gerekce=result.cevap or "cevap belgelerde yok",
            hits=hits,
        )

    return Answer(
        cevap=result.cevap,
        reddedildi=False,
        kaynaklar=result.kaynaklar,
        hits=hits,
    )
