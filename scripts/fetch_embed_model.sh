#!/usr/bin/env bash
# Download the embedding model for semantic recall.
#
# Run once. About 69 MB from HuggingFace — the same place scripts/set_voice.sh
# gets Piper voices, so if that has worked here, this will.
#
# Safe to re-run: an already-downloaded model is detected and nothing is
# fetched again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MODEL="${1:-BAAI/bge-small-en-v1.5}"
CACHE="$ROOT/models/embed"

[[ -x "$PY" ]] || { echo "No venv at $PY"; exit 1; }

echo "Model:  $MODEL"
echo "Cache:  $CACHE"
mkdir -p "$CACHE"

"$PY" - "$MODEL" "$CACHE" <<'PYEOF'
import sys, time
model_name, cache = sys.argv[1], sys.argv[2]
try:
    from fastembed import TextEmbedding
except ImportError:
    print("\nfastembed is not installed. Run:")
    print("  ~/assistant/.venv/bin/pip install fastembed")
    raise SystemExit(1)

t0 = time.time()
m = TextEmbedding(model_name, cache_dir=cache, threads=2)
print(f"ready in {time.time()-t0:.1f}s, {m.embedding_size} dimensions")

t0 = time.time()
list(m.embed(["warm up the graph"]))
first = time.time() - t0
t0 = time.time()
list(m.embed(["how fast is one sentence once it is warm"]))
print(f"first embed {first*1000:.0f}ms, then {(time.time()-t0)*1000:.0f}ms each")
PYEOF

echo
echo "Done. Restart the core service to pick it up:"
echo "  systemctl --user restart assistant-core"
echo
echo "Then watch it index your history:"
echo "  curl -s localhost:8760/memory/index | python3 -m json.tool"
