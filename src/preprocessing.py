"""
Step 2 — Preprocessing and Identifier Removal
===============================================
For each device in the chosen subset and date window:
  1. Read raw pcapng packets
  2. Extract ONLY: timestamp, packet_size, direction (outgoing/incoming)
  3. Strip ALL local identifiers (MAC, IP addresses) — none appear in output
  4. Save cleaned per-device CSVs to data/processed/
  5. Report before/after row counts and print a data sample

Non-negotiable rules enforced:
  - No MAC address in output columns
  - No IP address in output columns
  - Direction encoded as 0 (incoming) / 1 (outgoing) — derived from MAC match
    but MAC itself is NOT stored
  - Device label derived from filename, not from any packet field
"""

import os
import sys
import datetime
import yaml
import dpkt
import dpkt.pcapng
import csv

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
DATE_START = datetime.datetime.strptime(config["dataset"]["date_start"], "%Y-%m-%d").replace(
    tzinfo=datetime.timezone.utc
)
DATE_END = datetime.datetime.strptime(config["dataset"]["date_end"], "%Y-%m-%d").replace(
    tzinfo=datetime.timezone.utc
)
DATE_START_TS = DATE_START.timestamp()
DATE_END_TS = DATE_END.timestamp()

RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJ_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Map device names to their pcap filenames and MACs
DEVICE_FILE_MAP = {
    "AmazonEcho": ("AmazonEcho_44650d56ccd3.pcap", "44:65:0d:56:cc:d3"),
    "BelkinWemoMotionSensor": ("BelkinWemoMotionSensor_ec1a59832811.pcap", "ec:1a:59:83:28:11"),
    "PhilipsHue": ("PhilipsHue_0017882b9a25.pcap", "00:17:88:2b:9a:25"),
    "SamsungCamera": ("SamsungCamera_00166cab6b88.pcap", "00:16:6c:ab:6b:88"),
    "TPLinkSmartPlug": ("TPLinkSmartPlug_50c7bf005639.pcap", "50:c7:bf:00:56:39"),
}


def mac_bytes_to_str(mac_bytes):
    """Convert 6 raw MAC bytes to colon-separated lowercase hex string."""
    return ":".join(f"{b:02x}" for b in mac_bytes)


def preprocess_device(device_name):
    """
    Preprocess a single device's pcapng file.
    Returns (raw_count, processed_count, sample_rows).
    """
    if device_name not in DEVICE_FILE_MAP:
        print(f"  ERROR: Unknown device '{device_name}'")
        return 0, 0, []

    pcap_filename, device_mac = DEVICE_FILE_MAP[device_name]
    filepath = os.path.join(RAW_DIR, pcap_filename)
    device_mac_lower = device_mac.lower()

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return 0, 0, []

    print(f"\n  Processing {device_name} ({pcap_filename})...", flush=True)

    output_path = os.path.join(PROCESSED_DIR, f"{device_name}.csv")

    raw_count = 0
    processed_count = 0
    skipped_outside_window = 0
    skipped_non_device = 0
    skipped_parse_error = 0
    sample_rows = []

    with open(filepath, "rb") as fin, open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        # Output columns: timestamp, packet_size, direction
        # NO MAC, NO IP — only behavioral features
        writer.writerow(["timestamp", "packet_size", "direction"])

        try:
            reader = dpkt.pcapng.Reader(fin)
        except Exception:
            fin.seek(0)
            reader = dpkt.pcap.Reader(fin)

        for ts, buf in reader:
            raw_count += 1

            # Filter to date window
            if ts < DATE_START_TS or ts >= DATE_END_TS:
                skipped_outside_window += 1
                if raw_count % 5_000_000 == 0:
                    print(f"    ... scanned {raw_count:,} raw packets ...", flush=True)
                continue

            # Parse Ethernet frame to determine direction
            if len(buf) < 14:
                skipped_parse_error += 1
                continue

            try:
                eth = dpkt.ethernet.Ethernet(buf)
                src_mac = mac_bytes_to_str(eth.src)
                dst_mac = mac_bytes_to_str(eth.dst)
            except Exception:
                skipped_parse_error += 1
                continue

            # Determine direction: 1 = outgoing (device is source), 0 = incoming (device is dest)
            if src_mac == device_mac_lower:
                direction = 1  # outgoing
            elif dst_mac == device_mac_lower:
                direction = 0  # incoming
            else:
                # Packet doesn't involve this device — skip
                skipped_non_device += 1
                continue

            # Packet size: use the buffer length (captured size)
            packet_size = len(buf)

            # Write cleaned row — NO identifiers, only behavioral fields
            writer.writerow([f"{ts:.6f}", packet_size, direction])
            processed_count += 1

            # Collect sample rows (first 5)
            if len(sample_rows) < 5:
                sample_rows.append((f"{ts:.6f}", packet_size, direction))

            if raw_count % 5_000_000 == 0:
                print(f"    ... scanned {raw_count:,} raw, {processed_count:,} kept ...", flush=True)

    print(f"    Raw packets scanned:       {raw_count:,}")
    print(f"    Skipped (outside window):  {skipped_outside_window:,}")
    print(f"    Skipped (non-device MAC):  {skipped_non_device:,}")
    print(f"    Skipped (parse error):     {skipped_parse_error:,}")
    print(f"    Processed (output):        {processed_count:,}")
    print(f"    Output file: {output_path}")

    return raw_count, processed_count, sample_rows


def verify_no_identifiers(device_name):
    """
    Post-processing verification: confirm the output CSV contains
    NO MAC addresses and NO IP addresses.
    """
    output_path = os.path.join(PROCESSED_DIR, f"{device_name}.csv")
    if not os.path.exists(output_path):
        return False, "File not found"

    with open(output_path, "r") as f:
        header = f.readline().strip()
        columns = header.split(",")

        # Check column names
        forbidden_cols = {"mac", "ip", "src_mac", "dst_mac", "src_ip", "dst_ip",
                          "source_mac", "dest_mac", "source_ip", "dest_ip",
                          "mac_address", "ip_address"}
        found_forbidden = [c for c in columns if c.lower() in forbidden_cols]
        if found_forbidden:
            return False, f"Forbidden columns found: {found_forbidden}"

        # Verify expected columns
        expected = {"timestamp", "packet_size", "direction"}
        actual = set(columns)
        if actual != expected:
            return False, f"Unexpected columns: {actual} (expected {expected})"

        # Spot-check first 100 data rows for MAC/IP patterns
        import re
        mac_pattern = re.compile(r"([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}")
        ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

        for i, line in enumerate(f):
            if i >= 100:
                break
            if mac_pattern.search(line):
                return False, f"MAC address found in data row {i+1}"
            if ip_pattern.search(line):
                return False, f"IP address pattern found in data row {i+1}"

    return True, "PASS — no identifiers found"


def main():
    print("=" * 70)
    print("  STEP 2: PREPROCESSING AND IDENTIFIER REMOVAL")
    print(f"  Date window: {DATE_START.date()} to {(DATE_END - datetime.timedelta(days=1)).date()}")
    print(f"  Devices: {', '.join(DEVICE_SUBSET)}")
    print("=" * 70)

    all_results = []

    for device_name in DEVICE_SUBSET:
        raw_count, proc_count, samples = preprocess_device(device_name)
        all_results.append({
            "device": device_name,
            "raw": raw_count,
            "processed": proc_count,
            "samples": samples,
        })

    # ── Summary Table ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  BEFORE/AFTER ROW COUNTS")
    print(f"{'='*80}")
    print(f"  {'Device':<28} {'Raw (scanned)':>15} {'Processed':>15} {'Kept %':>8}")
    print(f"  {'-'*68}")
    total_raw = 0
    total_proc = 0
    for r in all_results:
        pct = (r["processed"] / r["raw"] * 100) if r["raw"] > 0 else 0
        print(f"  {r['device']:<28} {r['raw']:>15,} {r['processed']:>15,} {pct:>7.1f}%")
        total_raw += r["raw"]
        total_proc += r["processed"]
    total_pct = (total_proc / total_raw * 100) if total_raw > 0 else 0
    print(f"  {'-'*68}")
    print(f"  {'TOTAL':<28} {total_raw:>15,} {total_proc:>15,} {total_pct:>7.1f}%")
    print(f"{'='*80}")

    # ── Data Samples ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DATA SAMPLES (first 5 rows per device)")
    print(f"  Columns: timestamp, packet_size, direction (1=outgoing, 0=incoming)")
    print(f"{'='*80}")
    for r in all_results:
        print(f"\n  {r['device']}:")
        print(f"  {'timestamp':<22} {'packet_size':>12} {'direction':>10}")
        print(f"  {'-'*46}")
        for ts, size, dirn in r["samples"]:
            dir_label = "outgoing" if dirn == 1 else "incoming"
            print(f"  {ts:<22} {size:>12} {dirn:>5} ({dir_label})")

    # ── Identifier Verification ────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  IDENTIFIER REMOVAL VERIFICATION")
    print(f"{'='*80}")
    all_pass = True
    for device_name in DEVICE_SUBSET:
        passed, msg = verify_no_identifiers(device_name)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {device_name}: {msg}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All output files verified: NO local identifiers present.")
    else:
        print(f"\n  WARNING: Some files contain identifiers! Review and fix before proceeding.")

    print(f"\n  Output directory: {os.path.abspath(PROCESSED_DIR)}")
    print(f"\n  Step 2 complete. Awaiting review before proceeding to Step 3.")


if __name__ == "__main__":
    main()
