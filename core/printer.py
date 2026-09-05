"""Paper. The one output that survives the machine being off.

Everything else this appliance produces is ephemeral — spoken and gone, or on a
screen in another room. A strip of paper goes in a pocket and comes to the shop
with you, and it does not need a battery, a signal, or the thing to still be
running.

Three rules shaped this file, all learned the hard way from other USB devices
on this box:

  * Open, print, close. Never hold the handle. A long-lived USB handle means
    the printer cannot be unplugged without leaving a dead file descriptor
    behind, and after a replug that handle fails forever while `lsusb` insists
    everything is fine. That is the ReSpeaker wedge, and once per appliance is
    enough.

  * Never let it block the voice loop. The tools layer already runs this on a
    worker thread, but a USB write to a printer that is off, jammed, or out of
    paper can hang for as long as the driver feels like. Every call carries a
    timeout, and a failure returns a sentence rather than raising.

  * ASCII only, transliterated deliberately. The playbook is full of em dashes
    and the transcriber emits curly quotes; both come out of an ESC/POS printer
    as mojibake because the default code page is not UTF-8. Mapping them down
    is not lossy in any way a person reading a receipt will mind.
"""
from __future__ import annotations

import logging
import textwrap
import time
import unicodedata

log = logging.getLogger("assistant.printer")

# Characters that reach here from speech, from YAML, and from the model, and
# that a code-page-437 printer renders as garbage. Mapped explicitly rather
# than stripped, because "SHA256 every attachment  hash it" reads like a typo
# while "- hash it" reads like a dash.
_TRANSLIT = {
    "—": "-", "–": "-", "‒": "-", "−": "-",   # dashes
    "‘": "'", "’": "'", "‚": ",", "‛": "'",   # quotes
    "“": '"', "”": '"', "„": '"',
    "…": "...", "•": "*", "·": "-", "′": "'",
    " ": " ", " ": " ", "​": "",
    "→": "->", "←": "<-", "×": "x", "÷": "/",
    "✓": "x", "✔": "x", "○": " ", "●": "*",
    "°": " deg", "€": "EUR", "£": "GBP",
}


def to_ascii(text: str) -> str:
    """Fold to something a receipt printer can actually render."""
    out = "".join(_TRANSLIT.get(ch, ch) for ch in str(text))
    # Decompose accents and drop the combining marks: "Ramón" -> "Ramon".
    out = unicodedata.normalize("NFKD", out)
    out = "".join(c for c in out if not unicodedata.combining(c))
    # Anything still outside ASCII would print as a random glyph. A question
    # mark is honest about the fact that something was lost.
    return out.encode("ascii", "replace").decode("ascii")


class Receipt:
    """A page being composed, as plain lines. Rendered to ESC/POS at the end.

    Building the text first and emitting it in one pass means the layout can be
    unit-tested without a printer attached, and means a formatting mistake
    cannot leave the printer half-written with the cutter never called.
    """

    def __init__(self, width: int = 42):
        self.width = max(20, int(width))
        self.lines: list[tuple[str, str]] = []   # (style, text)

    def raw(self, text: str = "", style: str = "normal") -> "Receipt":
        self.lines.append((style, to_ascii(text)))
        return self

    def rule(self, ch: str = "-") -> "Receipt":
        return self.raw(ch * self.width)

    def blank(self, n: int = 1) -> "Receipt":
        for _ in range(n):
            self.raw("")
        return self

    def centre(self, text: str, style: str = "normal") -> "Receipt":
        return self.raw(to_ascii(text).center(self.width).rstrip(), style)

    def wrap(self, text: str, indent: str = "", hang: str = "") -> "Receipt":
        """Wrap to the paper width with a hanging indent.

        Truncation is not an option here. "Pick up the prescription from the
        pharmacy on" tells you nothing about which pharmacy, and a shopping
        list that silently loses the end of an item is worse than no list.
        """
        body = to_ascii(text).strip()
        if not body:
            return self
        for line in textwrap.wrap(
                body, width=self.width,
                initial_indent=indent, subsequent_indent=hang or indent,
                break_long_words=True, break_on_hyphens=False) or [indent]:
            self.raw(line)
        return self

    def columns(self, left: str, right: str) -> "Receipt":
        """Left text, right text, padded apart. Used for headers and counts."""
        left, right = to_ascii(left), to_ascii(right)
        gap = self.width - len(left) - len(right)
        if gap < 1:
            left = left[: max(0, self.width - len(right) - 1)]
            gap = max(1, self.width - len(left) - len(right))
        return self.raw(left + " " * gap + right)

    def render(self, printer, cut: bool = True, feed: int = 3) -> None:
        """Emit to any python-escpos printer, including Dummy."""
        style = None
        for want, text in self.lines:
            if want != style:
                printer.set(**_STYLES.get(want, _STYLES["normal"]))
                style = want
            printer.text(text + "\n")
        printer.set(**_STYLES["normal"])
        if feed:
            printer.text("\n" * feed)
        if cut:
            printer.cut()

    def preview(self) -> str:
        """What it will look like, for logs and tests."""
        return "\n".join(t for _, t in self.lines)


_STYLES = {
    "normal": {"align": "left", "bold": False,
               "double_height": False, "double_width": False, "underline": 0},
    "title":  {"align": "center", "bold": True,
               "double_height": True, "double_width": True, "underline": 0},
    "head":   {"align": "center", "bold": True,
               "double_height": False, "double_width": False, "underline": 0},
    "bold":   {"align": "left", "bold": True,
               "double_height": False, "double_width": False, "underline": 0},
    "dim":    {"align": "left", "bold": False,
               "double_height": False, "double_width": False, "underline": 0},
}


class PrinterTools:
    """print_list / print_case / print_note / print_text."""

    def __init__(self, cfg: dict, store, timers=None, doctools=None):
        p = cfg.get("printer", {}) or {}
        self.enabled = bool(p.get("enabled", False))
        self.vendor = int(str(p.get("vendor_id", "0x04b8")), 16)
        self.product = int(str(p.get("product_id", "0x0e20")), 16)
        self.out_ep = int(str(p.get("out_ep", "0x01")), 16)
        raw_in = p.get("in_ep", "0x81")
        self.in_ep = int(str(raw_in), 16) if raw_in not in (None, "", "null") else None
        self.width = int(p.get("width", 42))
        self.cut = bool(p.get("cut", True))
        self.feed = int(p.get("feed_lines", 3))
        self.header = str(p.get("header", "") or "")
        self.timeout = int(p.get("timeout_ms", 5000))
        self.store = store
        # Borrowed from TimerTools so "print the honey-do list" resolves to the
        # same board the voice commands use. Two independent name resolvers
        # would drift, and the failure would be a printed list that is real but
        # not the one asked for.
        self.timers = timers
        self.docs = doctools
        if self.enabled:
            log.info("printer: %04x:%04x out=%s in=%s width=%d",
                     self.vendor, self.product, hex(self.out_ep),
                     hex(self.in_ep) if self.in_ep else "none", self.width)

    # -- the connection ----------------------------------------------

    def _send(self, receipt: Receipt) -> str:
        """Open, print, close. Returns '' on success or a spoken error."""
        try:
            from escpos.printer import Usb
        except ImportError:
            return ("The printer library is not installed. "
                    "Run pip install python-escpos.")

        kwargs = {"in_ep": self.in_ep} if self.in_ep else {}
        printer = None
        try:
            printer = Usb(self.vendor, self.product, timeout=self.timeout,
                          out_ep=self.out_ep, **kwargs)
            receipt.render(printer, cut=self.cut, feed=self.feed)
            return ""
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            log.error("print failed: %s: %s", name, exc)
            text = str(exc).lower()
            # Translate the three failures that actually happen into something
            # worth hearing out loud. "USBError errno 13" is not.
            if "no device" in text or "not found" in text:
                return "I cannot find the printer. Is it switched on?"
            if "permission" in text or "access" in text or "errno 13" in text:
                return "The printer refused access. The udev rule may be missing."
            if "timeout" in text or "timed out" in text:
                return "The printer did not respond. Check paper and power."
            return f"The printer failed: {name}."
        finally:
            if printer is not None:
                try:
                    printer.close()
                except Exception:  # noqa: BLE001
                    # Closing a handle that never opened properly raises, and
                    # that error would mask the real one from the try block.
                    pass

    def _open(self, r: Receipt, title: str, subtitle: str = "") -> Receipt:
        """Masthead, then title, then when. In that order, deliberately.

        The name goes above the title the way it does on a till receipt — you
        want to know whose paper this is before you read what it says, and a
        strip found on a worktop three days later needs both.
        """
        if self.header:
            r.centre(self.header, "head")
        r.centre(title.upper(), "title")
        if subtitle:
            r.centre(subtitle.upper(), "head")
        r.centre(time.strftime("%a %d %b   %-I:%M %p"))
        return r.rule()

    # -- layouts -----------------------------------------------------

    def build_list(self, list_name: str, spoken: str,
                   include_done: bool = False) -> Receipt:
        rows = self.store.list_read(list_name, include_done=include_done)
        r = Receipt(self.width)
        self._open(r, spoken)
        if not rows:
            r.blank().centre("nothing on it").blank()
        for row in rows:
            box = "[x] " if row["done"] else "[ ] "
            r.wrap(row["text"], indent=box, hang="    ")
        r.rule()
        open_n = sum(1 for row in rows if not row["done"])
        r.centre(f"{open_n} item{'s' if open_n != 1 else ''}")
        return r

    def build_case(self, case, only: str = "all") -> Receipt:
        """`only`: all | open | findings.

        A finished case is about ninety lines, which is a foot of paper. That
        is right for a write-up and wrong for "what have I still got left",
        which is the thing actually asked for mid-investigation.
        """
        r = Receipt(self.width)
        tag = " / ".join(x for x in (case["ref"], case["severity"]) if x)
        self._open(r, case["title"], tag)

        if case["sla_due"]:
            left = case["sla_due"] - time.time()
            from .casework import human_left
            r.columns("SLA", human_left(abs(left))
                      + (" OVER" if left < 0 else " left"))
            r.rule()

        for side, label in (("investigation", "INVESTIGATION"),
                            ("admin", "MY TASKS")):
            all_rows = self.store.case_steps(case["id"], side)
            if not all_rows:
                continue
            done = sum(1 for x in all_rows if x["done"])
            if only == "open":
                rows = [x for x in all_rows if not x["done"]]
            elif only == "findings":
                rows = [x for x in all_rows if x["finding"]]
            else:
                rows = all_rows
            r.columns(label, f"{done}/{len(all_rows)}")
            r.rule(".")
            if not rows:
                r.raw("  nothing" if only == "findings" else "  all done")
                r.rule()
                continue
            phase = None
            for row in rows:
                # Phase headers are only worth the line when more than one
                # phase survives the filter. On "print what's left" they
                # otherwise outnumber the steps.
                if row["phase"] and row["phase"] != phase:
                    phase = row["phase"]
                    r.raw(phase.upper(), "bold")
                box = "[x] " if row["done"] else "[ ] "
                r.wrap(row["text"], indent=box, hang="    ")
                if row["finding"]:
                    r.wrap(row["finding"], indent="    > ", hang="      ")
            r.rule()

        if (case["notes"] or "").strip():
            r.raw("NOTES", "bold")
            for line in case["notes"].splitlines():
                r.wrap(line, hang="    ")
            r.rule()
        return r

    # -- tool surface ------------------------------------------------

    _NAMES = {"print_list", "print_case", "print_note", "print_text"}

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "print_list",
                "description": (
                    "Print a list on paper. Use when the user asks to print "
                    "one, or wants it to take with them to the shop. Say one "
                    "short sentence out loud confirming it — do not read the "
                    "items back, the paper is the point."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "list": {"type": "string",
                                 "description": "Which list, in their words."},
                        "include_done": {
                            "type": "boolean",
                            "description": (
                                "Include completed items, ticked. Default "
                                "false — a shopping list wants what is left."
                            ),
                        },
                    },
                },
            },
            {
                "name": "print_case",
                "description": (
                    "Print the active investigation on paper. Use when they "
                    "ask to print the case, or want it for a write-up or a "
                    "handover."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "only": {
                            "type": "string",
                            "enum": ["all", "open", "findings"],
                            "description": (
                                "'all' is the full record — use it for a "
                                "write-up or handover. 'open' is only what is "
                                "left, which is what 'print what I still have "
                                "to do' means. 'findings' is only the steps "
                                "with findings recorded, for writing a report."
                            ),
                        },
                    },
                },
            },
            {
                "name": "print_note",
                "description": "Print a saved note by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "print_text",
                "description": (
                    "Print arbitrary text — something they dictated, or "
                    "something you produced that is worth having on paper. "
                    "Keep it short; this is a receipt, not a document."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string",
                                  "description": "Optional heading."},
                    },
                    "required": ["text"],
                },
            },
        ]

    def run_sync(self, name: str, args: dict) -> str | None:
        if name not in self._NAMES:
            return None
        if not self.enabled:
            return "The printer is turned off in the configuration."

        if name == "print_list":
            which = args.get("list")
            if self.timers is not None:
                key = self.timers.resolve_list(which)
                spoken = self.timers.spoken_name(key)
            else:
                key = spoken = (which or "shopping")
            rows = self.store.list_read(key, bool(args.get("include_done")))
            if not rows:
                return f"The {spoken} is empty, so there is nothing to print."
            err = self._send(self.build_list(
                key, spoken, bool(args.get("include_done"))))
            n = sum(1 for r in rows if not r["done"])
            return err or (f"Printed. {n} item{'s' if n != 1 else ''} on the "
                           f"{spoken}.")

        if name == "print_case":
            case = self.store.case_active()
            if not case:
                return "No case is open."
            only = args.get("only") or "all"
            if only not in ("all", "open", "findings"):
                only = "all"
            err = self._send(self.build_case(case, only))
            prog = self.store.case_progress(case["id"])
            done = sum(d for d, _ in prog.values())
            total = sum(t for _, t in prog.values())
            if err:
                return err
            if only == "open":
                return f"Printed what is left. {total - done} of {total}."
            if only == "findings":
                return "Printed the findings."
            return f"Printed. {done} of {total} ticked off."

        if name == "print_note":
            if self.docs is None or not self.docs.enabled:
                return "Notes are turned off in the configuration."
            path = self.docs._existing(args.get("name", ""))
            if path is None:
                return f"I cannot find a note called {args.get('name')!r}."
            r = Receipt(self.width)
            self._open(r, path.stem.replace("-", " "))
            for line in path.read_text()[:4000].splitlines():
                r.wrap(line, hang="  ") if line.strip() else r.blank()
            err = self._send(r)
            return err or f"Printed {path.stem.replace('-', ' ')}."

        text = (args.get("text") or "").strip()
        if not text:
            return "Nothing to print."
        r = Receipt(self.width)
        self._open(r, str(args.get("title") or "note"))
        for line in text.splitlines():
            r.wrap(line, hang="  ") if line.strip() else r.blank()
        err = self._send(r)
        return err or "Printed."
