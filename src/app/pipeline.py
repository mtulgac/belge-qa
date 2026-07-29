from dataclasses import dataclass, field

from app.config import DEFAULT_VARIANT, GEN_K, LLM_MODEL, REJECT_THRESHOLD, RETRIEVER
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
    variant: str = DEFAULT_VARIANT,
    on_stage=None,
    on_token=None,
) -> Answer:
    if on_stage:
        on_stage("retrieval")
    hits = search(question, k=k, retriever=retriever, variant=variant, on_stage=on_stage)

    # Stage 1: cheap reranker-score pre-filter. Empty hits or a top score below the
    # zero-false-reject threshold means an off-topic question: reject without paying
    # for the LLM call.
    if not hits or hits[0].skor < REJECT_THRESHOLD:
        return Answer(
            cevap="",
            reddedildi=True,
            gerekce="ilgili belge bulunamadı",
            hits=hits,
        )

    # Stage 2: LLM grounding/citation backstop, the authoritative gate.
    if on_stage:
        on_stage("generate")
    result = generate(question, hits, model=model, on_stage=on_stage, on_token=on_token)
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
