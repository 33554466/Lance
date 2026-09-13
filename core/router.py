"""Decide which model tier a request deserves.

Most of what a household assistant is asked is trivial. Paying top-tier rates
for "what's the weather" is how a six-dollar month becomes a forty-dollar one.

This is a keyword-and-length heuristic on purpose. It is transparent, costs
nothing, and you can read the reason it chose a tier in the log. A learned
classifier is a refinement, not a prerequisite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def _phrase(marker: str) -> re.Pattern:
    """A marker that matches whole words only.

    Plain substring matching was the original design and it was wrong in a way
    that cost real money. Against the live config:

        "start my training"      -> 'rain'   -> escalated, web search attached
        "can I borrow the drill" -> 'row'    -> escalated
        "I am impressed"         -> 'press'  -> escalated

    live_info_markers are checked before everything else, so every workout or
    army sentence containing "training" was routed as a weather question and
    carried the billable search tool. Word boundaries cost one compiled
    pattern per marker at startup and nothing per request.

    \b on both ends, and the marker escaped so a phrase with punctuation in
    it ("cost of", "who won") still means itself.
    """
    return re.compile(r"\b" + r"\s+".join(
        re.escape(w) for w in marker.split()) + r"\b", re.I)


@dataclass
class Route:
    tier: str
    model: str
    reason: str


class Router:
    def __init__(self, cfg: dict):
        r = cfg["router"]
        self.models = cfg["provider"]["models"]
        self.small_max_chars = r.get("small_max_chars", 80)
        self.escalate = [p.lower() for p in r.get("escalate_phrases", [])]
        self.markers = [m.lower() for m in r.get("complexity_markers", [])]
        # Compiled once. Keep the plain lists too — they are what gets logged
        # as the reason, and a regex in a log line explains nothing.
        self._re_markers = [(m, _phrase(m)) for m in self.markers]
        # Requests that need to look something up. These are usually SHORT —
        # "what's the weather", "who won" — so the length heuristic alone
        # would send them to the small tier, which may not carry the web
        # search tool. Anything matching here skips that shortcut.
        self.live = [m.lower() for m in r.get("live_info_markers", [])]
        self._re_live = [(m, _phrase(m)) for m in self.live]
        # Words that name something this box can DO. These requests are
        # usually short — "print my shopping list", "show the board" — so the
        # length heuristic alone sends them to the small tier, which then has
        # to pick the right one out of forty-odd tools. It often does not.
        #
        # This was found the hard way: "start today's workout" worked and
        # "display full body A workout" did not, purely because the first
        # happened to contain "today" and got escalated by the live-info list.
        self.tools = [m.lower() for m in r.get("tool_markers", [])]
        self._re_tools = [(m, _phrase(m)) for m in self.tools]

    @staticmethod
    def _first_hit(pairs, text: str) -> str | None:
        """The first marker that matches as whole words, or None."""
        for marker, pattern in pairs:
            if pattern.search(text):
                return marker
        return None

    def needs_tool(self, text: str) -> bool:
        return self._first_hit(self._re_tools, text) is not None

    def needs_live_info(self, text: str) -> bool:
        return self._first_hit(self._re_live, text) is not None

    def route(self, text: str, in_session: bool = False) -> Route:
        """`in_session` is true while a case or workout is open.

        Word lists cannot cover this. Mid-workout, "185 for 5" and "that's
        the dips" are tool calls with no keyword in them at all, and enumerating
        every lift and every phishing step is a list that is wrong the day
        someone says something new. Knowing a session is OPEN is the fact that
        actually predicts it — while one is running, almost everything said is
        aimed at the checklist.
        """
        t = text.lower().strip()

        # Explicit escalation always wins. Being able to say "think hard
        # about this" and have it mean something is worth more than any
        # amount of automatic classification.
        for phrase in self.escalate:
            if phrase in t:
                return Route("top", self.models["top"],
                             f"escalated by phrase: '{phrase}'")

        hit = self._first_hit(self._re_live, t)
        if hit:
            return Route("mid", self.models["mid"],
                         f"needs current information: '{hit}'")

        # Before the length shortcut, not after. The whole point is to catch
        # the short requests that the shortcut would otherwise hand to a model
        # choosing blind among every tool on the appliance.
        hit = self._first_hit(self._re_tools, t)
        if hit:
            return Route("mid", self.models["mid"],
                         f"names a local tool: '{hit}'")

        if in_session:
            return Route("mid", self.models["mid"],
                         "a session is open")

        hit = self._first_hit(self._re_markers, t)

        if len(t) <= self.small_max_chars and not hit:
            return Route("small", self.models["small"],
                         f"short ({len(t)} chars), no complexity markers")

        if hit:
            return Route("mid", self.models["mid"],
                         f"complexity marker: '{hit}'")

        return Route("mid", self.models["mid"], "default tier")
