#!/usr/bin/env bash
# App container entrypoint: wait for the Ollama server, optionally provision the LLMs,
# then hand off to the container command (idle by default).
set -euo pipefail

echo "Ollama bekleniyor ($OLLAMA_HOST) ..."
until curl -sf "$OLLAMA_HOST/api/version" >/dev/null 2>&1; do
    sleep 1
done
echo "Ollama hazır."

# One-time (idempotent) provisioning: pull the two production tiers into the Ollama
# volume. Skipped unless PROVISION=1 so a plain restart is instant. `ollama.pull`
# talks to the server over OLLAMA_HOST, so the models land in the ollama container.
if [ "${PROVISION:-0}" = "1" ]; then
    echo "Üretim modelleri çekiliyor (ilk sefer uzun sürer) ..."
    python - <<'PY'
import ollama
from app.config import LLM_TIERS
for model in sorted(set(LLM_TIERS.values())):
    print(f"  pull {model}", flush=True)
    ollama.pull(model)
print("  provizyon tamam.", flush=True)
PY
fi

exec "$@"
