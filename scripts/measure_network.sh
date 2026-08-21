#!/usr/bin/env bash
# ============================================================
# Settle the wired-vs-wireless question on YOUR network instead of
# taking anyone's word for it — mine included.
#
#   ./scripts/measure_network.sh
#
# Run it once on Ethernet and once on Wi-Fi, then compare. What matters
# is not the averages, which will look fine either way, but the p95 and
# the maximum. A voice assistant is judged on its worst responses.
# ============================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
good() { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[1;31m✗\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

# ------------------------------------------------------------
bold "Link"
IFACE=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K\S+' || true)
if [[ -z "${IFACE:-}" ]]; then
    warn "could not identify the default interface — skipping link checks"
    KIND="unknown"
elif [[ "$IFACE" == e* ]]; then
    KIND="wired"
    good "wired ($IFACE)"
    ethtool "$IFACE" 2>/dev/null | grep -E 'Speed|Duplex' | sed 's/^/    /'
else
    KIND="wireless"
    warn "wireless ($IFACE)"
    if command -v iw >/dev/null; then
        iw dev "$IFACE" link 2>/dev/null | \
            grep -E 'SSID|freq|signal|tx bitrate|rx bitrate' | sed 's/^/    /'
        FREQ=$(iw dev "$IFACE" link 2>/dev/null | grep -oP 'freq: \K[0-9]+' || echo 0)
        SIG=$(iw dev "$IFACE" link 2>/dev/null | grep -oP 'signal: \K-[0-9]+' || echo -99)
        if (( FREQ < 3000 )) && (( FREQ > 0 )); then
            bad "on 2.4 GHz — move to 5 or 6 GHz, this band is where jitter lives"
        elif (( FREQ > 0 )); then
            good "on ${FREQ} MHz"
        fi
        if (( SIG < -70 )); then
            bad "signal ${SIG} dBm — weak. Retransmits will cause jitter spikes."
        elif (( SIG < -60 )); then
            warn "signal ${SIG} dBm — usable but not comfortable"
        else
            good "signal ${SIG} dBm"
        fi
    fi

    # THE big one. Linux Wi-Fi dynamic power management parks the radio
    # between packets. Measured impact is not subtle: hundreds of
    # milliseconds on the first packet after an idle gap. This appliance is
    # idle almost all the time, so it hits that path on essentially every
    # single request.
    PS=$(iw dev "$IFACE" get power_save 2>/dev/null | grep -oP 'Power save: \K\w+' || echo unknown)
    if [[ "$PS" == "on" ]]; then
        bad "power_save is ON — fix this before measuring anything else:"
        info "sudo iw dev $IFACE set power_save off       # now"
        info ""
        info "# and permanently:"
        info "printf '[connection]\\nwifi.powersave = 2\\n' | \\"
        info "  sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf"
        info "sudo systemctl restart NetworkManager"
    elif [[ "$PS" == "off" ]]; then
        good "power_save off"
    fi
fi

# ------------------------------------------------------------
GW=$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')
if [[ -n "${GW:-}" ]]; then
bold "Round trip to the router ($GW)"
ping -c 30 -i 0.2 -q "$GW" 2>/dev/null | tail -2 | sed 's/^/  /'
fi

bold "Round trip to the internet (1.1.1.1)"
ping -c 30 -i 0.2 -q 1.1.1.1 2>/dev/null | tail -2 | sed 's/^/  /'

# ------------------------------------------------------------
bold "TLS handshake + first byte from the API (20 samples)"
info "This is the number that actually matters. Everything above is proxy."

python3 - "$KIND" <<'PY'
import socket, ssl, statistics, sys, time

HOST, PORT, N = "api.anthropic.com", 443, 20
samples = []
ctx = ssl.create_default_context()

for _ in range(N):
    t0 = time.perf_counter()
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as raw:
            with ctx.wrap_socket(raw, server_hostname=HOST):
                samples.append((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        print(f"    connect failed: {exc}")
    time.sleep(0.15)

if not samples:
    print("    no samples — is the network up?")
    sys.exit(0)

s = sorted(samples)
p = lambda q: s[min(len(s) - 1, int(len(s) * q))]
med, p95, mx = statistics.median(s), p(0.95), s[-1]
jitter = statistics.pstdev(s)

print(f"    median {med:6.1f} ms")
print(f"    p95    {p95:6.1f} ms")
print(f"    max    {mx:6.1f} ms")
print(f"    jitter {jitter:6.1f} ms (std dev)")
print()

# The full chain to first audio is roughly 1,500 ms. Judge the network
# against that budget, not against an absolute standard.
BUDGET = 1500.0
print(f"    p95 is {p95 / BUDGET * 100:.1f}% of the ~1,500 ms budget to first sound")

if p95 < 120:
    verdict = "\033[1;32mFine.\033[0m Nobody will perceive this."
elif p95 < 250:
    verdict = "\033[1;32mFine.\033[0m Slightly slower than wired; still imperceptible in speech."
elif p95 < 500:
    verdict = "\033[1;33mNoticeable.\033[0m Replies will feel a beat slow. Worth improving."
else:
    verdict = "\033[1;31mA problem.\033[0m You will hear gaps mid-sentence when TTS runs dry."
print(f"    {verdict}")

if jitter > 80:
    print("    \033[1;33m!\033[0m High jitter matters more than high latency here: the reply")
    print("      is streamed and spoken sentence by sentence, so a spike mid-stream")
    print("      is an audible gap rather than a uniformly slower answer.")
PY

bold "How to use this"
cat <<'EOF'
  Run it on Ethernet, then on Wi-Fi, and compare the p95 — not the median.
  A voice assistant is judged on its worst responses, not its average ones.

  Rough guide, against the ~1,500 ms budget to first sound:
    p95 under 250 ms   no practical difference, use whichever is convenient
    p95 250 - 500 ms   a beat slow; try 5/6 GHz, power_save off, or MoCA
    p95 over 500 ms    fix the network before blaming the assistant
EOF
