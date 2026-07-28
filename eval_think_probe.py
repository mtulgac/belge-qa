import argparse
import json
from datetime import datetime, timezone

import ollama
import yaml

from app.config import (
    DEFAULT_VARIANT,
    GEN_K,
    GOLDEN_SET,
    REJECT_THRESHOLD,
    RERANK_DEPTH,
    RETRIEVER,
    ROOT,
    model_slug,
    resolve_model,
)
from eval_generation import RESULTS_DIR, measure, summarize

_real_chat = ollama.chat


def install_seam(think: bool, num_predict: int) -> None:
    """Force `think` (and optionally `num_predict`) on every generation.generate call.
    generate() calls ollama.chat by module attribute, so replacing it here reaches it."""
    def _patched(**kw):
        kw["think"] = think
        if num_predict:
            opts = dict(kw.get("options") or {})
            opts["num_predict"] = num_predict
            kw["options"] = opts
        return _real_chat(**kw)
    ollama.chat = _patched


def _load_baseline(model: str) -> dict | None:
    path = RESULTS_DIR / f"{model_slug(model)}_d{RERANK_DEPTH}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary")


def _record(m: dict, s: dict, think: bool, num_predict: int) -> "Path":
    suffix = "think" if think else "nothink"
    path = RESULTS_DIR / f"{model_slug(m['model'])}_d{RERANK_DEPTH}_{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": m["model"], "variant": m["variant"], "k": m["k"],
        "retriever": RETRIEVER, "rerank_depth": RERANK_DEPTH,
        "reject_threshold": REJECT_THRESHOLD,
        "think": think, "num_predict_override": num_predict or None,
        "summary": s, "results": m["results"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _row(label: str, s: dict) -> str:
    return (f"  {label:18} uydurma {s['uydurma']}/{s['n_reddet']}   "
            f"ger.red {s['gereksiz_red']}/{s['n_yanitla']}   "
            f"doğru {s['dogru_cevap']}/{s['n_yanitla']}   "
            f"med {s['latency_med']:.1f}s  max {s['latency_max']:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Golden-set generation eval, thinking forced on/off.")
    p.add_argument("--think", choices=["on", "off"], required=True,
                   help="force the model's thinking on or off for this run")
    p.add_argument("--models", default=None,
                   help="comma-separated models (default: large for --think off, turbo for on)")
    p.add_argument("--num-predict", type=int, default=0,
                   help="override the output cap (needed when forcing think on for turbo)")
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--k", type=int, default=GEN_K)
    p.add_argument("--limit", type=int, default=0, help="first N golden rows (smoke)")
    args = p.parse_args()

    think = args.think == "on"
    install_seam(think, args.num_predict)

    default_models = resolve_model("turbo") if think else resolve_model("large")
    models = [m.strip() for m in (args.models or default_models).split(",") if m.strip()]

    rows = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    width = 78
    for model in models:
        try:
            m = measure(model, rows, args.variant, args.k)
        except Exception as e:
            print(f"{model}: ATLANDI — {type(e).__name__}: {e}\n")
            continue
        s = summarize(m)
        path = _record(m, s, think, args.num_predict)

        print("=" * width)
        cap = f", num_predict={args.num_predict}" if args.num_predict else ""
        print(f"THINK={'ON' if think else 'False'} — {model}  "
              f"(retriever {RETRIEVER}, k={m['k']}, depth {RERANK_DEPTH}{cap})")
        print("-" * width)
        print(_row(f"think={'ON' if think else 'False'}", s))
        base = _load_baseline(model)
        if base:
            print(_row("production (base)", base))
        if s["uydurma_ids"]:
            print(f"  uydurma ids       : {s['uydurma_ids']}")
        if s["gereksiz_ids"]:
            print(f"  gereksiz red ids  : {s['gereksiz_ids']}")
        print(f"kaydedildi: {path.relative_to(ROOT)}")
        print()


if __name__ == "__main__":
    main()
