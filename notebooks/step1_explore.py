"""
Step 1 — Dataset Exploration (READ-ONLY)
=========================================
For each device pcap (actually pcapng) in data/raw/:
  - Parse all packets using dpkt.pcapng, filter to the chosen date window (1–3 Dec 2016)
  - Report: total packet count, time span, mean/variance of packet rate
  - Confirm which fields are present (timestamp, size, direction, protocol)
  - Show how device labels are derived (MAC address from filename)

No filtering, cleaning, or transformation is performed.
"""

import os
import sys
import glob
import struct
import datetime
import statistics
import dpkt
import dpkt.pcapng

# ── Configuration ──────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
DATE_START = datetime.datetime(2016, 12, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
DATE_END   = datetime.datetime(2016, 12, 4, 0, 0, 0, tzinfo=datetime.timezone.utc)  # exclusive

DEVICE_MACS = {
    "SamsungCamera":          "00:16:6c:ab:6b:88",
    "BelkinWemoMotionSensor": "ec:1a:59:83:28:11",
    "AmazonEcho":             "44:65:0d:56:cc:d3",
    "PhilipsHue":             "00:17:88:2b:9a:25",
    "TPLinkSmartPlug":        "50:c7:bf:00:56:39",
    "NestProtectSmokeAlarm":  "18:b4:30:25:be:e4",
}

DATE_START_TS = DATE_START.timestamp()
DATE_END_TS   = DATE_END.timestamp()


def mac_bytes_to_str(mac_bytes):
    """Convert 6 raw MAC bytes to colon-separated hex string."""
    return ":".join(f"{b:02x}" for b in mac_bytes)


def mac_from_filename(fname):
    """Extract MAC from filename like 'SamsungCamera_00166cab6b88.pcap'."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        raw_mac = parts[-1]
        if len(raw_mac) == 12:
            return ":".join(raw_mac[i:i+2] for i in range(0, 12, 2))
    return None


def device_name_from_filename(fname):
    """Extract device name from filename like 'SamsungCamera_00166cab6b88.pcap'."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return base


def explore_device(filepath):
    """
    Explore a single device pcapng file.
    Returns a dict with exploration results.
    """
    device_name = device_name_from_filename(filepath)
    file_mac = mac_from_filename(filepath)
    device_mac_lower = file_mac.lower() if file_mac else None

    print(f"\n{'='*70}")
    print(f"  Device: {device_name}")
    print(f"  File:   {os.path.basename(filepath)}")
    print(f"  MAC (from filename): {file_mac}")
    print(f"  File size: {os.path.getsize(filepath) / (1024**2):.1f} MB")
    print(f"{'='*70}")

    # Counters
    total_packets = 0
    window_packets = 0
    timestamps_in_window = []
    first_ts = None
    last_ts = None
    sizes_in_window = []

    # Field presence tracking
    has_timestamp = False
    has_size = False
    has_direction = False
    has_protocol = False
    ethertypes_seen = set()
    ip_protos_seen = set()

    # Direction tracking
    outgoing_count = 0
    incoming_count = 0

    print(f"  Scanning packets (this may take a while for large files)...", flush=True)

    with open(filepath, "rb") as f:
        try:
            reader = dpkt.pcapng.Reader(f)
        except Exception:
            # Fallback to classic pcap
            f.seek(0)
            reader = dpkt.pcap.Reader(f)

        for ts, buf in reader:
            total_packets += 1

            if first_ts is None:
                first_ts = ts
            last_ts = ts

            # Quick timestamp check before doing any parsing
            if ts < DATE_START_TS or ts >= DATE_END_TS:
                # Print progress every 5M packets
                if total_packets % 5_000_000 == 0:
                    print(f"    ... processed {total_packets:,} packets ...", flush=True)
                continue

            window_packets += 1
            timestamps_in_window.append(ts)
            sizes_in_window.append(len(buf))
            has_timestamp = True
            has_size = True

            # Parse Ethernet frame
            if len(buf) >= 14:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    src_mac = mac_bytes_to_str(eth.src)
                    dst_mac = mac_bytes_to_str(eth.dst)
                    has_direction = True

                    if device_mac_lower:
                        if src_mac == device_mac_lower:
                            outgoing_count += 1
                        elif dst_mac == device_mac_lower:
                            incoming_count += 1

                    ethertypes_seen.add(eth.type)
                    has_protocol = True

                    # Get IP protocol
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip_protos_seen.add(eth.data.p)
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        ip_protos_seen.add(eth.data.nxt)
                except Exception:
                    pass

            # Print progress every 5M packets
            if total_packets % 5_000_000 == 0:
                print(f"    ... processed {total_packets:,} packets ({window_packets:,} in window) ...", flush=True)

    # ── Compute statistics ─────────────────────────────────────────────
    result = {
        "device": device_name,
        "mac": file_mac,
        "total_packets_in_file": total_packets,
        "file_first_ts": None,
        "file_last_ts": None,
        "window_packets": window_packets,
        "window_first_ts": None,
        "window_last_ts": None,
        "window_span_hours": 0,
        "mean_pkt_rate_per_sec": 0,
        "var_pkt_rate_per_sec": 0,
        "mean_pkt_size": 0,
        "has_timestamp": has_timestamp,
        "has_size": has_size,
        "has_direction": has_direction,
        "has_protocol": has_protocol,
        "outgoing": outgoing_count,
        "incoming": incoming_count,
        "ethertypes": ethertypes_seen,
        "ip_protos": ip_protos_seen,
    }

    if first_ts:
        result["file_first_ts"] = datetime.datetime.fromtimestamp(first_ts, tz=datetime.timezone.utc)
    if last_ts:
        result["file_last_ts"] = datetime.datetime.fromtimestamp(last_ts, tz=datetime.timezone.utc)

    if window_packets > 0:
        ts_sorted = sorted(timestamps_in_window)
        result["window_first_ts"] = datetime.datetime.fromtimestamp(ts_sorted[0], tz=datetime.timezone.utc)
        result["window_last_ts"] = datetime.datetime.fromtimestamp(ts_sorted[-1], tz=datetime.timezone.utc)
        span_sec = ts_sorted[-1] - ts_sorted[0]
        result["window_span_hours"] = span_sec / 3600.0

        # Packet rate: compute per-second bins
        if span_sec > 0:
            bin_size = 1.0
            num_bins = int(span_sec / bin_size) + 1
            bins = [0] * num_bins
            base_ts = ts_sorted[0]
            for t in ts_sorted:
                idx = int((t - base_ts) / bin_size)
                if idx < num_bins:
                    bins[idx] += 1
            result["mean_pkt_rate_per_sec"] = statistics.mean(bins)
            result["var_pkt_rate_per_sec"] = statistics.variance(bins) if len(bins) > 1 else 0

        result["mean_pkt_size"] = statistics.mean(sizes_in_window)

    # ── Print per-device report ────────────────────────────────────────
    print(f"\n  --- Full File Overview ---")
    print(f"  Total packets in file:  {total_packets:,}")
    if result["file_first_ts"] and result["file_last_ts"]:
        print(f"  File time range:        {result['file_first_ts']} -> {result['file_last_ts']}")

    print(f"\n  --- Date Window: {DATE_START.date()} to {(DATE_END - datetime.timedelta(days=1)).date()} ---")
    print(f"  Packets in window:      {window_packets:,}")
    if result["window_first_ts"] and result["window_last_ts"]:
        print(f"  Window time range:      {result['window_first_ts']} -> {result['window_last_ts']}")
    print(f"  Window span:            {result['window_span_hours']:.2f} hours")
    print(f"  Mean pkt rate:          {result['mean_pkt_rate_per_sec']:.2f} pkt/sec")
    print(f"  Var pkt rate:           {result['var_pkt_rate_per_sec']:.2f}")
    print(f"  Mean pkt size:          {result['mean_pkt_size']:.1f} bytes")

    print(f"\n  --- Available Fields ---")
    print(f"  Timestamp:   {'YES' if has_timestamp else 'NO (no packets in window)'}")
    print(f"  Packet size: {'YES' if has_size else 'NO'}")
    print(f"  Direction:   {'YES' if has_direction else 'NO'} (src/dst MAC)")
    if has_direction:
        print(f"    Outgoing (src=device): {outgoing_count:,}")
        print(f"    Incoming (dst=device): {incoming_count:,}")
    print(f"  Protocol:    {'YES' if has_protocol else 'NO'}")
    if ethertypes_seen:
        etype_names = {0x0800: "IPv4", 0x86DD: "IPv6", 0x0806: "ARP", 0x8100: "VLAN"}
        print(f"    Ethertypes: {', '.join(etype_names.get(e, hex(e)) for e in sorted(ethertypes_seen))}")
    if ip_protos_seen:
        proto_names = {6: "TCP", 17: "UDP", 1: "ICMP", 2: "IGMP", 58: "ICMPv6"}
        print(f"    IP protos:  {', '.join(proto_names.get(p, str(p)) for p in sorted(ip_protos_seen))}")

    print(f"\n  --- Label Derivation ---")
    print(f"  Device label '{device_name}' derived from filename prefix")
    print(f"  MAC '{file_mac}' derived from filename suffix")

    return result


def main():
    print("=" * 70)
    print("  STEP 1: DATASET EXPLORATION (READ-ONLY)")
    print(f"  Date window: {DATE_START.date()} to {(DATE_END - datetime.timedelta(days=1)).date()}")
    print("=" * 70)

    pcap_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.pcap")))
    if not pcap_files:
        print(f"\n  ERROR: No .pcap files found in {os.path.abspath(RAW_DIR)}")
        return

    print(f"\n  Found {len(pcap_files)} pcap files:")
    for f in pcap_files:
        print(f"    - {os.path.basename(f)} ({os.path.getsize(f) / (1024**2):.1f} MB)")

    results = []
    for filepath in pcap_files:
        r = explore_device(filepath)
        results.append(r)

    # ── Summary Table ──────────────────────────────────────────────────
    print("\n\n")
    print("=" * 130)
    print("  SUMMARY TABLE - Packets in window Dec 1-3, 2016")
    print("=" * 130)
    header = (
        f"{'Device':<28} {'MAC':<20} {'Pkts(total)':>14} {'Pkts(window)':>14} "
        f"{'Span(hrs)':>10} {'Rate(pkt/s)':>12} {'Var(rate)':>12} {'AvgSize':>8} "
        f"{'Dir?':>5} {'Proto?':>6}"
    )
    print(header)
    print("-" * 130)
    for r in sorted(results, key=lambda x: x["window_packets"], reverse=True):
        print(
            f"{r['device']:<28} {r['mac']:<20} {r['total_packets_in_file']:>14,} "
            f"{r['window_packets']:>14,} "
            f"{r['window_span_hours']:>10.2f} {r['mean_pkt_rate_per_sec']:>12.2f} "
            f"{r['var_pkt_rate_per_sec']:>12.2f} {r['mean_pkt_size']:>8.1f} "
            f"{'YES' if r['has_direction'] else 'NO':>5} "
            f"{'YES' if r['has_protocol'] else 'NO':>6}"
        )
    print("=" * 130)

    # ── Usability Assessment ───────────────────────────────────────────
    print("\n  USABILITY ASSESSMENT FOR DEC 1-3 WINDOW:")
    for r in sorted(results, key=lambda x: x["window_packets"], reverse=True):
        if r["window_packets"] >= 1000:
            status = "OK"
            flag = "[OK]"
        elif r["window_packets"] > 0:
            status = "LOW"
            flag = "[!!]"
        else:
            status = "EMPTY"
            flag = "[XX]"
        print(f"  {flag} {r['device']:<28} - {r['window_packets']:>10,} packets ({status})")

    print("\n  Step 1 complete. Awaiting review before proceeding to Step 2.")


if __name__ == "__main__":
    main()
