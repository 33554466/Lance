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


class ContextBuilder:
    def __init__(self, cfg: dict, store):
        self.cfg = cfg
        self.store = store
        self.history_turns = cfg["context"]["history_turns"]
        ident = cfg["identity"]
        # Rendered once at startup. Never interpolate anything time-varying
        # here — it would break prompt caching on every request.
        self._system = cfg["context"]["system_prompt"].format(
            name=ident["name"], user=ident["user"]
        ).strip()

    @property
    def system(self) -> str:
        return self._system

    def build(self, user_text: str) -> list[dict]:
        """Return provider-shaped messages for this exchange."""
        messages = self.store.recent_turns(self.history_turns)

        # Volatile context rides along with the user turn, keeping the
        # cached system block stable.
        now = _dt.datetime.now()
        stamp = now.strftime("%A %-d %B %Y, %-I:%M %p")
        messages.append({
            "role": "user",
            "content": f"[{stamp}]\n{user_text}",
        })
        return messages
