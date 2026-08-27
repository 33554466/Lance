"""Assemble what actually goes into each request.

This is where your token spend is determined, so it is worth reading rather
than skimming. Two decisions matter:

  1. The system prompt is byte-identical on every request. That is what makes
     it cacheable, and the cache is where most of your savings come from.
     Anything that varies (the time, say) must go in the user turn instead,
     or you invalidate the cache on every call.

  2. History is capped. Appending every turn forever grows input cost
     silently through the day and eventually blows the context window.
"""
from __future__ import annotations

import datetime as _dt


def _describe_place(loc: dict) -> str:
    """Turn the location config into something a sentence can contain."""
    parts = [loc.get("city"), loc.get("region"), loc.get("country")]
    named = ", ".join(p for p in parts if p)
    return named or "an unspecified location"


class ContextBuilder:
    def __init__(self, cfg: dict, store, toolbox=None):
        self.cfg = cfg
        self.store = store
        self.toolbox = toolbox
        self.history_turns = cfg["context"]["history_turns"]
        # A gap longer than this starts a fresh conversation. Without it,
        # coming back after three days makes the assistant reply to Tuesday.
        self.session_gap = float(cfg["context"].get("session_gap_minutes", 90))
        ident = cfg["identity"]
        # Rendered once at startup. Never interpolate anything time-varying
        # here — it would break prompt caching on every request.
        #
        # Location is safe to put here precisely because it does NOT vary.
        # The date is not, which is why it rides along with the user turn
        # further down.
        self._system = cfg["context"]["system_prompt"].format(
            name=ident["name"], user=ident["user"],
            place=_describe_place(cfg.get("location") or {}),
        ).strip()

    @property
    def system(self) -> str:
        """Static prompt plus whatever is currently remembered.

        Rebuilt per request because facts change. That costs one small
        SQLite read and, when a fact HAS changed, one prompt-cache miss —
        which is the correct trade: the alternative is an assistant that
        does not know something until you restart it.
        """
        block = self.toolbox.memory_block() if self.toolbox else ""
        return f"{self._system}\n\n{block}" if block else self._system

    def build(self, user_text: str) -> list[dict]:
        """Return provider-shaped messages for this exchange."""
        import time as _time
        messages = self.store.turns_since(
            _time.time() - self.session_gap * 60, self.history_turns)

        # Volatile context rides along with the user turn, keeping the
        # cached system block stable.
        now = _dt.datetime.now()
        stamp = now.strftime("%A %-d %B %Y, %-I:%M %p")
        messages.append({
            "role": "user",
            "content": f"[{stamp}]\n{user_text}",
        })
        return messages
