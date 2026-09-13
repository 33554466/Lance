"""Device resolution across a microphone that comes and goes.

This is the bug that cost two days. resolve_devices() runs again on every
recovery attempt, and the old version wrote the resolved INDEX back over the
configured NAME — so every later call reused a number instead of enumerating.
When the array dropped off the USB bus, that number had become the mini PC's
onboard analog codec: the service bound to the wrong sound card, failed
forever on sample rate, and could not recover even after the microphone came
back.

    python -m tests.test_devices
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# The two states his machine was actually in, taken from the journal.
WITH_ARRAY = [
    {"name": "HDA Intel PCH: ALC897 Analog (hw:0,0)",
     "max_input_channels": 2, "max_output_channels": 2},
    {"name": "reSpeaker XVF3800 4-Mic Array: USB Audio (hw:1,0)",
     "max_input_channels": 6, "max_output_channels": 2},
]
# The array gone, and something else now sitting where it used to be.
WITHOUT_ARRAY = [
    {"name": "HD-Audio Generic: ALC897 Analog (hw:2,0)",
     "max_input_channels": 2, "max_output_channels": 2},
]


class FakeSd(types.ModuleType):
    """Just enough sounddevice to drive resolve_devices()."""

    def __init__(self, devices):
        super().__init__("sounddevice")
        self._devices = devices

    def query_devices(self):
        return self._devices

    def _initialize(self):
        pass

    def _terminate(self):
        pass


def _install(devices):
    sys.modules["sounddevice"] = FakeSd(devices)


def _cfg():
    return {"audio": {"input_device": "Array", "output_device": "Array",
                      "sample_rate": 16000, "usb_reset": "2886:001a"}}


def test_happy_path() -> None:
    print("\nWith the array present")
    _install(WITH_ARRAY)
    from audio import service
    cfg = _cfg()
    service.resolve_devices(cfg)
    check("resolves to the array, not the onboard codec",
          cfg["audio"]["input_device"] == 1,
          f"got index {cfg['audio']['input_device']}")
    check("the configured name is kept for next time",
          cfg["audio"]["input_name"] == "Array")

    # Resolving twice must give the same answer, by looking it up again.
    service.resolve_devices(cfg)
    check("a second resolve still lands on the array",
          cfg["audio"]["input_device"] == 1)


def test_reenumeration() -> None:
    """The index moves but the name does not. This must follow the name."""
    print("\nAfter a re-enumeration moved the index")
    _install(WITH_ARRAY)
    from audio import service
    cfg = _cfg()
    service.resolve_devices(cfg)

    # Array comes back at a different position — exactly what a USB reset does.
    _install([WITH_ARRAY[0], WITH_ARRAY[0], WITH_ARRAY[1]])
    service.resolve_devices(cfg)
    check("follows the name to its new index",
          cfg["audio"]["input_device"] == 2,
          f"got index {cfg['audio']['input_device']}")


def test_array_disappears() -> None:
    print("\nAfter the array left the USB bus")
    _install(WITH_ARRAY)
    from audio import service
    cfg = _cfg()
    service.resolve_devices(cfg)
    first = cfg["audio"]["input_device"]
    check("started out on the array", first == 1)

    _install(WITHOUT_ARRAY)
    raised = None
    try:
        service.resolve_devices(cfg)
    except SystemExit as exc:
        raised = str(exc)
    check("refuses to resolve at all", raised is not None,
          f"silently chose index {cfg['audio']['input_device']}")
    if raised:
        # Either explanation is correct and which one appears depends on
        # whether ALSA can still see the card — on a real appliance it often
        # can, because ALSA lists cards regardless of who holds them. What
        # matters is that it refused and said something actionable.
        check("explains itself with something to try",
              any(hint in raised for hint in
                  ("genuinely absent", "lsusb", "already has it open")),
              raised[:120])

    # And the wrapper turns that into "wait", not "crash".
    check("_try_resolve reports failure instead of exiting",
          service._try_resolve(cfg) is False)


def test_wrong_hardware_is_refused() -> None:
    """A name that substring-matches the wrong card must not be accepted."""
    print("\nWhen the closest match is the wrong hardware")
    _install([{"name": "Array Audio Bridge (hw:3,0)",
               "max_input_channels": 2, "max_output_channels": 2}])
    from audio import service
    cfg = {"audio": {"input_device": "reSpeaker", "output_device": "reSpeaker",
                     "sample_rate": 16000}}
    raised = None
    try:
        service.resolve_devices(cfg)
    except SystemExit as exc:
        raised = str(exc)
    check("refused", raised is not None)


def test_reset_ladder_gives_up() -> None:
    """Repeated resets knocked the array off the bus. Stop after a few."""
    print("\nThe reset ladder")
    _install(WITH_ARRAY)
    from audio import service

    service._recovery_attempts = 0
    calls = {"power": 0, "reset": 0}
    service.probe_capture = lambda cfg, timeout=4.0: False
    service.usb_power_cycle = lambda *a, **k: calls.__setitem__(
        "power", calls["power"] + 1) or False
    service.usb_reset = lambda *a, **k: calls.__setitem__(
        "reset", calls["reset"] + 1) or False

    cfg = _cfg()
    for _ in range(6):
        service.ensure_capture_works(cfg)
    check("stops resetting after the cap",
          calls["reset"] == service.MAX_RESET_ATTEMPTS,
          f"reset {calls['reset']} times")
    check("stops power cycling too",
          calls["power"] == service.MAX_RESET_ATTEMPTS,
          f"cycled {calls['power']} times")
    check("but keeps returning False so the caller keeps waiting",
          service.ensure_capture_works(cfg) is False)


def main() -> int:
    test_happy_path()
    test_reenumeration()
    test_array_disappears()
    test_wrong_hardware_is_refused()
    test_reset_ladder_gives_up()
    print()
    if FAILURES:
        print(f"\033[1;31m{len(FAILURES)} failed\033[0m")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\033[1;32mall passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
