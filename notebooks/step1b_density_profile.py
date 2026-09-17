"""
Step 1b — Full-Range Packet Density Profile (READ-ONLY)
========================================================
For each device, count packets per day across the entire capture range.
Output a daily density table and identify candidate windows where all
(or most) devices have substantial traffic simultaneously.
"""

import os
import glob
import datetime
import dpkt
import dpkt.pcapng
from collections import defaultdict

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

DEVICES = [
    "AmazonEcho_44650d56ccd3",
    "BelkinWemoMotionSensor_ec1a59832811",
    "NestProtectSmokeAlarm_18b43025bee4",
    "PhilipsHue_0017882b9a25",
    "SamsungCamera_00166cab6b88",
    "TPLinkSmartPlug_50c7bf005639",
]

DEVICE_LABELS = {
    "AmazonEcho_44650d56ccd3": "AmazonEcho",
    "BelkinWemoMotionSensor_ec1a59832811": "BelkinWemo",
    "NestProtectSmokeAlarm_18b43025bee4": "NestProtect",
    "PhilipsHue_0017882b9a25": "PhilipsHue",
    "SamsungCamera_00166cab6b88": "SamsungCam",
    "TPLinkSmartPlug_50c7bf005639": "TPLinkPlug",
}

# Threshold: minimum packets in a day to be considered "non-trivial"
MIN_DAILY_PACKETS = 500


def count_packets_per_day(filepath):
    """Count packets per calendar day (UTC) across the full capture."""
    daily_counts = defaultdict(int)
    total = 0

    print(f"  Scanning {os.path.basename(filepath)} ({os.path.getsize(filepath) / (1024**2):.0f} MB)...",
          flush=True)

    with open(filepath, "rb") as f:
        try:
            reader = dpkt.pcapng.Reader(f)
        except Exception:
            f.seek(0)
            reader = dpkt.pcap.Reader(f)

        for ts, buf in reader:
            total += 1
            # Convert timestamp to UTC date
            day = datetime.date.fromtimestamp(ts)  # local, but these captures are UTC-based
            daily_counts[day] += 1

            if total % 5_000_000 == 0:
                print(f"    ... {total:,} packets ...", flush=True)

    print(f"    Done: {total:,} packets across {len(daily_counts)} days", flush=True)
    return daily_counts, total


def main():
    print("=" * 80)
    print("  STEP 1b: FULL-RANGE PACKET DENSITY PROFILE (READ-ONLY)")
    print("=" * 80)

    # Collect daily counts per device
    all_daily = {}
    all_dates = set()

    for dev_key in DEVICES:
        filepath = os.path.join(RAW_DIR, dev_key + ".pcap")
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping")
            continue
        daily, total = count_packets_per_day(filepath)
        all_daily[dev_key] = daily
        all_dates.update(daily.keys())

    if not all_dates:
        print("  ERROR: No data found.")
        return

    # Build sorted date range
    date_min = min(all_dates)
    date_max = max(all_dates)
    num_days = (date_max - date_min).days + 1
    all_dates_sorted = [date_min + datetime.timedelta(days=i) for i in range(num_days)]

    print(f"\n  Capture range: {date_min} to {date_max} ({num_days} days)")

    # ── Daily Density Table ────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  DAILY PACKET COUNTS (showing days where at least one device has > 0 packets)")
    print(f"{'='*120}")

    labels = [DEVICE_LABELS[d] for d in DEVICES]
    header = f"{'Date':<12} " + " ".join(f"{l:>12}" for l in labels) + "  All>0?"
    print(header)
    print("-" * 120)

    # Track which days all devices have traffic
    good_days = []  # days where all 6 have >= MIN_DAILY_PACKETS

    for d in all_dates_sorted:
        counts = []
        for dev_key in DEVICES:
            c = all_daily.get(dev_key, {}).get(d, 0)
            counts.append(c)

        # Skip days with zero traffic across all devices
        if sum(counts) == 0:
            continue

        all_above_min = all(c >= MIN_DAILY_PACKETS for c in counts)
        marker = " <<<" if all_above_min else ""

        row = f"{str(d):<12} " + " ".join(
            f"{c:>12,}" if c > 0 else f"{'—':>12}" for c in counts
        ) + marker
        print(row)

        if all_above_min:
            good_days.append((d, counts))

    print(f"{'='*120}")

    # ── Weekly Summary ─────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  WEEKLY SUMMARY (packets per ISO week)")
    print(f"{'='*120}")

    # Aggregate by ISO week
    weekly = defaultdict(lambda: defaultdict(int))
    for dev_key in DEVICES:
        for d, c in all_daily.get(dev_key, {}).items():
            iso_year, iso_week, _ = d.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            weekly[week_key][dev_key] += c

    week_keys_sorted = sorted(weekly.keys())

    header = f"{'Week':<12} " + " ".join(f"{DEVICE_LABELS[d]:>12}" for d in DEVICES) + "  All>500?"
    print(header)
    print("-" * 120)

    for wk in week_keys_sorted:
        counts = [weekly[wk].get(d, 0) for d in DEVICES]
        all_above = all(c >= MIN_DAILY_PACKETS for c in counts)
        marker = " <<<" if all_above else ""
        row = f"{wk:<12} " + " ".join(
            f"{c:>12,}" if c > 0 else f"{'—':>12}" for c in counts
        ) + marker
        print(row)

    print(f"{'='*120}")

    # ── Find Best 3-Day Windows ────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  CANDIDATE 3-DAY WINDOWS (all 6 devices >= 500 pkts/day each day)")
    print(f"{'='*100}")

    candidates_6 = []
    candidates_5 = []
    candidates_4 = []

    for i in range(len(all_dates_sorted) - 2):
        d1 = all_dates_sorted[i]
        d2 = all_dates_sorted[i + 1]
        d3 = all_dates_sorted[i + 2]

        window_counts = {}
        for dev_key in DEVICES:
            daily = all_daily.get(dev_key, {})
            total = daily.get(d1, 0) + daily.get(d2, 0) + daily.get(d3, 0)
            min_day = min(daily.get(d1, 0), daily.get(d2, 0), daily.get(d3, 0))
            window_counts[dev_key] = (total, min_day)

        # Count how many devices have >= 500 per day (min_day >= 500)
        # and a reasonable total (>= 2000 for enough 128-windows)
        devices_ok = sum(1 for d in DEVICES
                        if window_counts[d][0] >= 2000 and window_counts[d][1] >= 100)

        entry = {
            "start": d1,
            "end": d3,
            "counts": {DEVICE_LABELS[d]: window_counts[d] for d in DEVICES},
            "devices_ok": devices_ok,
        }

        if devices_ok == 6:
            candidates_6.append(entry)
        elif devices_ok >= 5:
            candidates_5.append(entry)
        elif devices_ok >= 4:
            candidates_4.append(entry)

    def print_candidates(candidates, label):
        if not candidates:
            print(f"\n  {label}: NONE FOUND")
            return
        print(f"\n  {label}: {len(candidates)} found, showing top 5 by total packet volume:")
        # Sort by total packets descending
        candidates.sort(key=lambda x: sum(v[0] for v in x["counts"].values()), reverse=True)
        for entry in candidates[:5]:
            print(f"\n    Window: {entry['start']} to {entry['end']}")
            for dev, (total, min_day) in sorted(entry["counts"].items()):
                flag = "OK" if total >= 2000 and min_day >= 100 else "LOW" if total > 0 else "EMPTY"
                print(f"      {dev:<14} total={total:>8,}  min_day={min_day:>6,}  [{flag}]")

    print_candidates(candidates_6, "Windows where ALL 6 devices are usable")
    print_candidates(candidates_5, "Windows where 5 of 6 devices are usable")
    print_candidates(candidates_4, "Windows where 4 of 6 devices are usable")

    # ── Per-Device Summary: Where is each device's traffic concentrated? ──
    print(f"\n\n{'='*100}")
    print("  PER-DEVICE TRAFFIC CONCENTRATION")
    print(f"{'='*100}")

    for dev_key in DEVICES:
        label = DEVICE_LABELS[dev_key]
        daily = all_daily.get(dev_key, {})
        if not daily:
            print(f"\n  {label}: NO DATA")
            continue

        total = sum(daily.values())
        sorted_days = sorted(daily.items(), key=lambda x: x[1], reverse=True)
        top10 = sorted_days[:10]

        print(f"\n  {label} — {total:,} total packets across {len(daily)} active days")
        print(f"    Top 10 days:")
        for d, c in top10:
            pct = c / total * 100
            print(f"      {d}  {c:>10,} pkts  ({pct:5.1f}%)")

    print(f"\n\n  Step 1b complete. Awaiting decision on date window and device set.")


if __name__ == "__main__":
    main()
