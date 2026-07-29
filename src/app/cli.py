import argparse
import sys

from app.config import LLM_DEFAULT_TIER, LLM_TIERS, resolve_model
from app.pipeline import Answer, answer


def _format(a: Answer) -> str:
    if a.reddedildi:
        return f"REDDEDİLDİ: {a.gerekce}"
    lines = [a.cevap]
    if a.kaynaklar:
        kaynak = ", ".join(f"{k['dosya']} s.{k['sayfa']}" for k in a.kaynaklar)
        lines.append(f"\nKaynaklar: {kaynak}")
    return "\n".join(lines)


def _ask(question: str, model: str) -> None:
    print(_format(answer(question, model=model)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Doküman Bilgi Erişim: soru sor veya reddet.")
    parser.add_argument("soru", nargs="*", help="soru (boşsa etkileşimli mod)")
    parser.add_argument("--tier", choices=list(LLM_TIERS), default=LLM_DEFAULT_TIER,
                        help=f"turbo (hızlı) | large (kaliteli); varsayılan {LLM_DEFAULT_TIER}")
    parser.add_argument("--model", default=None, help="açık Ollama modeli (--tier'ı geçersiz kılar)")
    args = parser.parse_args()
    model = args.model or resolve_model(args.tier)

    # First query loads the embedding model (~11 s), the reranker and the LLM; warn so
    # a cold start does not look like a hang.
    print(f"[{args.tier} → {model}. İlk soru modelleri yüklerken bekleyebilir.]", file=sys.stderr)

    if args.soru:
        _ask(" ".join(args.soru), model)
        return

    while True:
        try:
            question = input("\nsoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        _ask(question, model)


if __name__ == "__main__":
    main()
