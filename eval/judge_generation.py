import argparse
import glob
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import ollama
import yaml

from app.config import (
    GOLDEN_SET,
    JUDGE_MODEL,
    ROOT,
    model_slug,
    resolve_model,
)
from app.generation import _strip_thinking
from eval_generation import answer_covers, expected_anchors

RESULTS_DIR = ROOT / "out" / "generation"
_GOLDEN = {r["id"]: r for r in yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))}

JUDGE_SYSTEM = """You grade a document-QA system's answer against a reference answer.

Reply with exactly one word on the first line — DOGRU or YANLIS — then one short line \
saying why:
- DOGRU: the answer contains the reference answer's key fact(s). Extra detail beyond the \
reference is FINE and does NOT make it wrong, as long as that extra detail is not \
fabricated and does not contradict the answer.
- YANLIS: the key fact is missing or wrong, OR the answer adds a claim that is fabricated \
or contradicts itself/the reference (a hallucination), OR it answers a question the \
reference says should be refused.

Judge only the facts. Ignore wording, language (Turkish vs English), number formatting \
(45,14 = 45.14), and spelled-out vs digit numbers. If the reference lists several values \
(a conflict), the answer must contain ALL of them to be DOGRU."""


def _verdict(raw: str) -> tuple[bool, bool]:

    u = unicodedata.normalize("NFKC", _strip_thinking(raw)).upper()
    has_dogru = "DOĞRU" in u or "DOGRU" in u
    has_yanlis = "YANLIŞ" in u or "YANLIS" in u
    if has_dogru and not has_yanlis:
        return True, True
    if has_yanlis and not has_dogru:
        return False, True
    if not has_dogru and not has_yanlis:
        return False, False  # unparsed: caller warns
    i_d = min((u.find(x) for x in ("DOĞRU", "DOGRU") if x in u), default=1 << 30)
    i_y = min((u.find(x) for x in ("YANLIŞ", "YANLIS") if x in u), default=1 << 30)
    return i_d < i_y, True


def judge_one(judge_model: str, soru: str, referans: str, cevap: str) -> tuple[bool, bool, str]:

    user = f"SORU:\n{soru}\n\nREFERANS CEVAP:\n{referans}\n\nSİSTEM CEVABI:\n{cevap}"
    kwargs = dict(
        model=judge_model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
        options={"temperature": 0.0},
    )
    try:
        resp = ollama.chat(think=False, **kwargs)
    except TypeError:
        resp = ollama.chat(**kwargs)
    raw = resp["message"]["content"]
    ok, parsed = _verdict(raw)
    return ok, parsed, _strip_thinking(raw).replace("\n", " ")[:200]


def _deterministic_ok(row_id: str, variant: str, cevap: str) -> bool:

    anchors = expected_anchors(_GOLDEN[row_id], variant)
    return bool(anchors) and all(answer_covers(cevap, a["alinti"]) for a in anchors)


def pick_judge(explicit: str) -> str:
    """Explicit judge (tier or model name) wins; otherwise the fixed independent judge."""
    return resolve_model(explicit) if explicit else JUDGE_MODEL


def judge_file(path: Path, explicit_judge: str) -> dict | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in d:
        return None
    model, variant = d["model"], d.get("variant")
    judge_model = pick_judge(explicit_judge)
    if judge_model == model:
        print(f"⚠ {model}: yargıç kendisi — self-evaluation biaslı. "
              f"--judge-model ile başka bir model verin.", file=sys.stderr)

    answered = [r for r in d["results"] if not r["reddedildi"]]
    print(f"[{model}] judge={judge_model}  {len(answered)} cevap…", file=sys.stderr, flush=True)

    records, i = [], 0
    for r in d["results"]:
        rec = {k: r[k] for k in ("id", "dil", "kategori", "beklenen", "reddedildi")}
        if r["reddedildi"]:
            rec |= {"det_ok": False, "judge_ok": False, "judge_reason": ""}
        else:
            i += 1
            ok, parsed, reason = judge_one(judge_model, r["soru"], r.get("referans_cevap", ""), r["cevap"])
            det = _deterministic_ok(r["id"], variant, r["cevap"]) if r["beklenen"] == "yanitla" else False
            rec |= {"det_ok": det, "judge_ok": ok, "judge_reason": reason, "judge_parsed": parsed}
            flag = "  ⚠ VERDICT PARSE EDİLEMEDİ" if not parsed else ("" if det == ok else "  ⚠ uyuşmuyor")
            print(f"[{model}] {i:>2}/{len(answered)} {r['id']} "
                  f"det={'✓' if det else '✗'} judge={'✓' if ok else '✗'}{flag}",
                  file=sys.stderr, flush=True)
        records.append(rec)

    # Mirror the input name (which carries the depth, e.g. qwen3.5:4b_d10) so each run's
    # judge output sits next to the run it graded.
    out = path.with_name(path.stem + ".judge.json")
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "judge_model": judge_model, "variant": variant,
        "records": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"model": model, "judge_model": judge_model, "records": records, "path": out}


def report(m: dict) -> None:
    recs = m["records"]
    yanitla = [r for r in recs if r["beklenen"] == "yanitla"]
    width = 78
    print("=" * width)
    print(f"JUDGE — {m['model']}   (yargıç: {m['judge_model']})")
    print("=" * width)

    def line(label, subset):
        det = sum(1 for r in subset if r["det_ok"])
        jud = sum(1 for r in subset if r["judge_ok"])
        print(f"  {label:6} deterministik {det:>2}/{len(subset)}   judge {jud:>2}/{len(subset)}")

    line("hepsi", yanitla)
    line("TR", [r for r in yanitla if r["dil"] == "tr"])
    line("EN", [r for r in yanitla if r["dil"] == "en"])

    disagree = [r for r in yanitla if not r["reddedildi"] and r["det_ok"] != r["judge_ok"]]
    print(f"\n  uyuşmazlık (yanitla): {len(disagree)}")
    for r in disagree:
        yon = "judge kurtardı" if r["judge_ok"] else "judge düşürdü"
        print(f"    {r['id']} [{r['dil']}/{r['kategori']}] {yon}: {r['judge_reason'][:78]}")

    uydurma = [r for r in recs if r["beklenen"] == "reddet" and not r["reddedildi"]]
    if uydurma:
        print("\n  cevap verilen reddet (uydurma) satırları:")
        for r in uydurma:
            print(f"    {r['id']} judge={'DOGRU?!' if r['judge_ok'] else 'YANLIS'}")
    print(f"\nkaydedildi: {m['path'].relative_to(ROOT)}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-as-judge over saved generation output.")
    p.add_argument("--models", default="", help="comma-separated model names; boşsa hepsi")
    p.add_argument("--judge-model", default="",
                   help="sabit yargıç (tier ya da model adı); boşsa cross-judge (turbo↔large)")
    args = p.parse_args()

    if args.models:
        # Prefix match so every depth variant of a model (e.g. <slug>_d10, _d20) is found.
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        patterns = [str(RESULTS_DIR / f"{model_slug(m)}*.json") for m in wanted]
    else:
        patterns = [str(RESULTS_DIR / "*.json")]
    files = sorted({Path(f) for pat in patterns for f in glob.glob(pat)
                    if not f.endswith(".judge.json")})

    if not files:
        print("değerlendirilecek dosya yok.", file=sys.stderr)
    for path in files:
        m = judge_file(path, args.judge_model)
        if m:
            report(m)


if __name__ == "__main__":
    main()
