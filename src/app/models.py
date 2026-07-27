from functools import lru_cache

import numpy as np

from app.config import RERANK_DEVICE, RERANK_MODEL


# maxsize=1: a reranker is 0.5-2 GB resident and the candidate sweep loads them in
# turn, so holding more than the one in use costs memory it never pays back —
# the same reasoning as retrieval._encoder.
@lru_cache(maxsize=1)
def _cross_encoder(model_name: str = RERANK_MODEL, device: str = RERANK_DEVICE):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, trust_remote_code=True, max_length=512, device=device)
    ce.model.eval()
    return ce


def rerank_scores(
    question: str, texts: list[str], model_name: str = RERANK_MODEL) -> np.ndarray:
    if not texts:
        return np.zeros(0, dtype=np.float64)
    import torch

    ce = _cross_encoder(model_name)
    features = ce.tokenizer(
        [question] * len(texts), list(texts),
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    features = {k: v.to(ce.model.device) for k, v in features.items()}
    with torch.no_grad():
        # reshape(-1) not squeeze: these rerankers emit [N, 1], and squeeze would
        # also collapse a single-candidate batch's leading dim.
        logits = ce.model(**features).logits.reshape(-1)
    return torch.sigmoid(logits).double().cpu().numpy()
