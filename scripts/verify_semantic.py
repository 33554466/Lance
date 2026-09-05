#!/usr/bin/env python3
"""Prove semantic recall actually works, against a scratch database.

This is the test I could not run while building it — HuggingFace is not
reachable from where the code was written, so the model itself was never
exercised. Everything around the model was tested; the model was not.

So run this once. It uses a TEMPORARY database and never touches your real
one, and it asks the questions that keyword search demonstrably fails:

    "what did we decide about the roof"  should find  "shingles replaced"
    "INC-4412"                           should still be found exactly
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Re-exec under the assistant's own interpreter if we were started with
# another one. numpy, fastembed and onnxruntime live in the venv, so running
# this as ./verify_semantic.py — which the shebang sends to /usr/bin/python3 —
# fails with ModuleNotFoundError even though everything is correctly
# installed. Fixing it here rather than in the instructions, because the
# instructions are not what people type.
_VENV = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if _VENV.exists() and Path(sys.executable).resolve() != _VENV.resolve():
    os.execv(str(_VENV), [str(_VENV), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from core.db import Store          # noqa: E402
from core.embed import Embedder, SemanticIndex  # noqa: E402

CORPUS = [
    "we are getting the shingles replaced next month, the quote was fourteen thousand",
    "the insurance letter came about the flood rider on the policy",
    "case INC-4412 phishing, SPF softfail and the domain was registered four days ago",
    "add milk eggs and dog food to the shopping list",
    "the gutters need clearing before the monsoon season starts",
    "drill weekend is the second saturday and the paperwork is due friday",
    "the pebble speakers connect over the aux cable from the respeaker",
    "I booked the dentist for a cleaning in the middle of october",
]

# (question, substring that must appear in the top hit, why it is interesting)
CHECKS = [
    ("what did we decide about the roof", "shingles",
     "no shared words at all — this is the case keyword search cannot do"),
    ("did anything come from the insurance company", "insurance letter",
     "paraphrase, some overlap"),
    ("INC-4412", "INC-4412",
     "exact token — keyword must still win here"),
    ("when am I at the dentist", "dentist",
     "direct"),
    ("what do I need from the shop", "milk eggs",
     "'shop' never appears in the message"),
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    sem = (cfg.get("memory", {}) or {}).get("semantic", {}) or {}
    if not sem.get("enabled"):
        print("memory.semantic.enabled is false in config.yaml")
        return 1

    emb = Embedder(cfg)
    print(f"loading {emb.model_name} ...")
    t0 = time.time()
    if emb.dim is None:
        print("\nModel would not load. Run scripts/fetch_embed_model.sh first.")
        return 1
    print(f"  ready in {time.time()-t0:.1f}s, {emb.dim} dimensions\n")

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "verify.db")
        for line in CORPUS:
            store.add_message("user", line)
        index = SemanticIndex(store, emb)

        t0 = time.time()
        n = index.backfill("message", batch=64)
        print(f"indexed {n} messages in {time.time()-t0:.1f}s\n")

        passed = failed = 0
        for question, expect, why in CHECKS:
            t0 = time.time()
            kw = store.search_ids(question, limit=20)
            ids = index.hybrid(question, kw, limit=3)
            rows = store.messages_by_ids(ids)
            ms = (time.time() - t0) * 1000
            top = rows[0]["content"] if rows else ""
            ok = expect.lower() in top.lower()
            passed, failed = (passed + ok, failed + (not ok))
            print(f"[{'PASS' if ok else 'FAIL'}] {question!r}   ({ms:.0f}ms)")
            print(f"        why: {why}")
            print(f"        keyword hits: {len(kw)}")
            print(f"        top: {top[:70]}")
            if not ok:
                print(f"        EXPECTED to contain: {expect!r}")
            print()

        print(f"{passed} passed, {failed} failed")
        if failed:
            print("\nA failure here means the model loaded but retrieval is "
                  "not doing what it should. Send me this output.")
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
