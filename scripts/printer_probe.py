#!/usr/bin/env python3
"""Find the MUNBYN/Epson printer, describe it, and print a test receipt.

Diagnostic first, print second. If something is wrong — permissions, a kernel
driver holding the device, endpoints in an unexpected place — this says which,
in a sentence, instead of raising a libusb error code at you.

    python3 printer_probe.py            # probe and test print
    python3 printer_probe.py --dry-run  # probe only, no paper
"""
from __future__ import annotations

import sys

VENDOR, PRODUCT = 0x04B8, 0x0E20  # Seiko Epson TM-m30-II (MUNBYN ITPP047)


def main() -> int:
    dry = "--dry-run" in sys.argv

    try:
        import usb.core
        import usb.util
    except ImportError:
        print("pyusb is missing. Install it into the assistant venv:")
        print("  ~/assistant/.venv/bin/pip install pyusb python-escpos")
        return 1

    dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT)
    if dev is None:
        print(f"No device {VENDOR:04x}:{PRODUCT:04x} found.")
        print("Is it powered on and plugged in? Check `lsusb`.")
        return 1

    print(f"Found {VENDOR:04x}:{PRODUCT:04x} "
          f"on bus {dev.bus} address {dev.address}")
    for name, idx in (("manufacturer", dev.iManufacturer),
                      ("product", dev.iProduct)):
        if not idx:
            continue
        try:
            print(f"  {name}: {usb.util.get_string(dev, idx)}")
        except Exception:  # noqa: BLE001
            # Reading strings needs the same permissions as everything else,
            # and failing here is a clearer signal than failing mid-print.
            print(f"  {name}: <could not read — permissions?>")

    # Endpoints. python-escpos can guess, but guessing wrong fails as a
    # timeout thirty seconds later, which reads like a broken printer rather
    # than a wrong number.
    cfg = dev.get_active_configuration()
    out_ep = in_ep = None
    iface_num = 0
    for iface in cfg:
        for ep in iface:
            direction = usb.util.endpoint_direction(ep.bEndpointAddress)
            kind = usb.util.endpoint_type(ep.bmAttributes)
            if kind != usb.util.ENDPOINT_TYPE_BULK:
                continue
            if direction == usb.util.ENDPOINT_OUT and out_ep is None:
                out_ep, iface_num = ep.bEndpointAddress, iface.bInterfaceNumber
            elif direction == usb.util.ENDPOINT_IN and in_ep is None:
                in_ep = ep.bEndpointAddress

    print(f"  interface: {iface_num}")
    print(f"  bulk OUT : {hex(out_ep) if out_ep else 'NOT FOUND'}")
    print(f"  bulk IN  : {hex(in_ep) if in_ep else 'none (fine — write-only)'}")
    if out_ep is None:
        print("\nNo bulk OUT endpoint. This is not a printer interface — the "
              "device may be in IPP-over-USB mode.")
        return 1

    # Something else holding the device is the most common cause of a printer
    # that works once and then never again: usblp, ipp-usb, or CUPS grabbed it.
    try:
        if dev.is_kernel_driver_active(iface_num):
            print(f"  kernel driver: ATTACHED — detaching")
            dev.detach_kernel_driver(iface_num)
        else:
            print("  kernel driver: none (good)")
    except NotImplementedError:
        print("  kernel driver: cannot check on this platform")
    except usb.core.USBError as exc:
        print(f"  kernel driver: could not detach — {exc}")
        print("\nThis is almost always permissions. Install the udev rule.")
        return 1

    # Let go of the device before python-escpos opens its own handle. Holding
    # both means two claims on one interface, and set_configuration comes back
    # "Resource busy" — which looks like a printer fault and is not one.
    usb.util.dispose_resources(dev)
    del dev

    if dry:
        print("\nDry run — nothing printed.")
        print(f"For config:  usb_out_ep: {hex(out_ep)}   "
              f"usb_in_ep: {hex(in_ep) if in_ep else 'null'}")
        return 0

    try:
        from escpos.printer import Usb
    except ImportError:
        print("\npython-escpos is missing:")
        print("  ~/assistant/.venv/bin/pip install python-escpos")
        return 1

    kwargs = {"in_ep": in_ep} if in_ep else {}
    try:
        p = Usb(VENDOR, PRODUCT, timeout=5000, out_ep=out_ep, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not open the printer: {type(exc).__name__}: {exc}")
        print("If this says access denied, the udev rule is missing or the "
              "device has not been replugged since you added it.")
        return 1

    print("\nPrinting a test receipt...")
    p.set(align="center", bold=True, double_height=True, double_width=True)
    p.text("LANCE\n")
    p.set(align="center", bold=False, double_height=False, double_width=False)
    p.text("printer test\n")
    p.text("-" * 32 + "\n")
    p.set(align="left")
    p.text("If you can read this, the whole\n")
    p.text("path works: libusb, permissions,\n")
    p.text("endpoints, and ESC/POS.\n\n")
    p.text("Width check, 32 characters:\n")
    p.text("12345678901234567890123456789012\n")
    p.text("Width check, 42 characters:\n")
    p.text("123456789012345678901234567890123456789012\n\n")
    p.set(bold=True)
    p.text("Bold. ")
    p.set(bold=False, underline=1)
    p.text("Underlined. ")
    p.set(underline=0, double_height=True)
    p.text("Big.\n")
    p.set(double_height=False)
    p.text("\n")
    try:
        p.cut()
    except Exception as exc:  # noqa: BLE001
        print(f"  (cut failed: {exc} — tear it off by hand)")
    p.close()

    print("Done. Two things to tell me:")
    print("  1. Which width line ran off the edge, 32 or 42?")
    print("  2. Did it cut the paper by itself?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
