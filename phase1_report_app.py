"""
Phase 1 Report — Encrypted IoT Device Fingerprinting
Run locally with:  streamlit run phase1_report_app.py

Place this file at the project root (NOT inside src/dashboard/ — that folder
is reserved for the Phase 2 live admin dashboard, this is a static Phase 1
results report).
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Encrypted IoT Fingerprinting — Phase 1 Report",
    page_icon="🔐",
    layout="wide",
)

# ---------- Data (from Phase 1 results) ----------

devices = ["AmazonEcho", "BelkinWemoMotion", "PhilipsHue", "SamsungCamera", "TPLinkPlug"]

packets_per_device = [175410, 376601, 229885, 183539, 8544]
windows_per_device = [1370, 2942, 1795, 1433, 66]
median_iat_ms = [11.649, 1.014, 2.842, 1.940, 243]

overall_perf = pd.DataFrame({
    "Metric": ["Accuracy", "Macro F1"],
    "DWT": [93.44, 93.98],
    "Raw Stats": [99.34, 99.38],
})

per_device_f1 = pd.DataFrame({
    "Device": devices,
    "DWT F1": [91.27, 96.46, 91.86, 90.30, 100.0],
    "Raw F1": [98.76, 99.76, 99.18, 99.21, 100.0],
})

ablation = pd.DataFrame({
    "latent_dim": [2, 4, 8, 16],
    "Held-out TP rate (%)": [0.14, 0.34, 0.71, 3.23],
    "Known FP rate (%)": [1.24, 0.62, 0.78, 0.93],
})

heldout_compare = pd.DataFrame({
    "Config": ["DWT · BelkinWemo", "DWT · TPLinkPlug", "Raw · BelkinWemo", "Raw · TPLinkPlug"],
    "Detection rate (%)": [3.23, 100, 14.28, 100],
})

routing = pd.DataFrame({
    "Device": ["AmazonEcho", "BelkinWemoMotionSensor", "PhilipsHue", "SamsungCamera", "TPLinkSmartPlug (unseen)"],
    "Windows": [204, 423, 242, 189, 9],
    "VERIFY": [204, 423, 239, 182, 0],
    "BLOCK": [0, 0, 3, 7, 9],
    "Classifier Acc.": ["89.7%", "99.5%", "91.6%", "90.1%", "—"],
    "Outcome": ["Known — Verified", "Known — Verified", "Known — Verified", "Known — Verified", "Unknown — Blocked"],
})

ACCENT = "#2f6690"
ACCENT2 = "#8a5cf6"
ACCENT3 = "#e0703e"
GOOD = "#1e8e5a"
BAD = "#c2453d"

# ---------- Header ----------

st.caption("COMPUTER NETWORKS · TEAM PROJECT")
st.title("Encrypted IoT Device Fingerprinting Using Adaptive Inter-Arrival Time Analysis")
st.markdown(
    "Phase 1 results — passive device identification and open-set detection from "
    "encrypted traffic timing, without inspecting payloads. Prepared for Review 1."
)

sections = st.tabs([
    "1 · Summary", "2 · Pipeline", "3 · Dataset", "4 · IAT Rhythm",
    "5 · Classification", "6 · Open-Set (VAE)", "7 · Final Routing", "8 · What's Next",
])

# ---------- 1. Summary ----------
with sections[0]:
    st.subheader("Executive Summary")
    st.markdown(
        "The end-to-end pipeline was built and validated on 5 real IoT devices from the "
        "UNSW-IoTraffic dataset, using only packet timing metadata — no payloads, no local identifiers."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Known-device accuracy", "94.05%", "4 classes")
    c2.metric("Unknown detection rate", "100%", "TPLinkSmartPlug")
    c3.metric("False positive rate", "0.9%")
    c4.metric("Packets analyzed", "973,979", "3-day window")
    c5.metric("Total windows", "7,606")

# ---------- 2. Pipeline ----------
with sections[1]:
    st.subheader("System Pipeline")
    st.markdown(
        "Phase 1 scope — Fingerprint Manager here only performs binary **VERIFY / BLOCK** routing. "
        "Versioning, QUARANTINE, and adversarial testing are Phase 2."
    )
    st.markdown(
        "`Dataset Replay (Nov 22-24, 2016)` → `Metadata Extraction` → `IAT Analysis (log1p)` → "
        "**`DWT Features (20-d)`** / **`Raw Stats (13-d)`** → **`Random Forest`** / **`VAE`** → "
        "`Fingerprint Manager → VERIFY / BLOCK`"
    )
    st.info(
        "Boxes above run left to right. DWT/Raw features feed both the classifier "
        "(device ID) and the VAE (open-set score) in parallel."
    )

# ---------- 3. Dataset ----------
with sections[2]:
    st.subheader("Dataset & Devices")
    st.markdown(
        "5 devices selected from the UNSW-IoTraffic dataset after a density-profiling check "
        "ruled out a 6th (NestProtect — 95% of its traffic landed on a single day, unusable as a stable class)."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=devices, y=packets_per_device, marker_color=ACCENT))
        fig.update_layout(title="Packets captured per device (Nov 22–24)", yaxis_type="log", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(x=devices, y=windows_per_device, marker_color=ACCENT2))
        fig.update_layout(title="Usable windows per device (128 pkts/window)", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("TPLinkSmartPlug's low count (66) is expected — later used as the open-set test case.")

# ---------- 4. IAT Rhythm ----------
with sections[3]:
    st.subheader("The Core Signal: Inter-Arrival Time")
    st.markdown(
        "Different devices have measurably different timing rhythms — this is the entire premise "
        "the project relies on, and it holds up before any model is trained."
    )
    colors = [ACCENT, ACCENT, ACCENT, ACCENT, ACCENT3]
    fig = go.Figure(go.Bar(x=devices, y=median_iat_ms, marker_color=colors))
    fig.update_layout(title="Median inter-arrival time per device (log scale, ms)", yaxis_type="log", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nearly a 3-order-of-magnitude spread — from ~1 ms (BelkinWemoMotionSensor) to ~243 ms (TPLinkSmartPlug).")

# ---------- 5. Classification ----------
with sections[4]:
    st.subheader("Classification: Raw Stats vs. DWT")
    st.markdown(
        "Both feature sets were trained and evaluated identically on a leakage-safe, "
        "time-based split (70/15/15). Test set touched exactly once."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_bar(name="DWT", x=overall_perf["Metric"], y=overall_perf["DWT"], marker_color=ACCENT)
        fig.add_bar(name="Raw Stats", x=overall_perf["Metric"], y=overall_perf["Raw Stats"], marker_color=ACCENT3)
        fig.update_layout(title="Overall test performance", yaxis_range=[85, 100], height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_bar(name="DWT F1", x=per_device_f1["Device"], y=per_device_f1["DWT F1"], marker_color=ACCENT)
        fig.add_bar(name="Raw F1", x=per_device_f1["Device"], y=per_device_f1["Raw F1"], marker_color=ACCENT3)
        fig.update_layout(title="Per-device F1 score", yaxis_range=[85, 100], height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Finding worth explaining, not hiding.** Raw statistical features (percentiles, kurtosis, "
        "skew) outperform DWT by ~6 points accuracy on this clean data. This does not contradict the "
        "project's premise — DWT's claimed advantage in the abstract is *robustness to network jitter*, "
        "not peak accuracy on unperturbed traffic. This comparison is the exact baseline Phase 2's "
        "adversarial jitter test (Step 13) is designed to revisit."
    )

# ---------- 6. Open-Set (VAE) ----------
with sections[5]:
    st.subheader("Open-Set Detection (VAE)")
    st.markdown(
        "Can the system flag a device it has never seen, instead of forcing it into a known class? "
        "Tested by holding devices out of training entirely."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_scatter(x=ablation["latent_dim"], y=ablation["Held-out TP rate (%)"],
                         name="Held-out TP rate %", line=dict(color=ACCENT3))
        fig.add_scatter(x=ablation["latent_dim"], y=ablation["Known FP rate (%)"],
                         name="Known FP rate %", line=dict(color=ACCENT))
        fig.update_layout(title="Latent dimension ablation (held-out: BelkinWemoMotionSensor)",
                           xaxis_title="latent_dim", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Detection rate scales with capacity — shrinking the bottleneck did not help; latent_dim=16 was kept.")
    with col2:
        colors2 = [ACCENT, GOOD, ACCENT, GOOD]
        fig = go.Figure(go.Bar(x=heldout_compare["Config"], y=heldout_compare["Detection rate (%)"], marker_color=colors2))
        fig.update_layout(title="Detection rate by held-out device", yaxis_range=[0, 100], height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("An outlier device (TPLinkSmartPlug) is caught reliably; a subtle one (BelkinWemo) is not — with either feature set.")

    st.warning(
        "**Known limitation, documented rather than hidden.** Reconstruction-error VAE detection works "
        "cleanly on devices with a distinct timing profile (100% on TPLinkSmartPlug), but struggles on "
        "subtle behavioral differences (3–14% on BelkinWemoMotionSensor). This is reported as a diagnostic "
        "finding about current feature discriminability, not folded into the production decision boundary."
    )

# ---------- 7. Final Routing ----------
with sections[6]:
    st.subheader("Final Decision Engine (Test Set)")
    st.markdown(
        "Classifier and VAE trained on the same 4 known devices; TPLinkSmartPlug used as the "
        "genuine unseen device for this demonstration."
    )
    fig = go.Figure()
    fig.add_bar(name="VERIFY", x=routing["Device"], y=routing["VERIFY"], marker_color=GOOD)
    fig.add_bar(name="BLOCK", x=routing["Device"], y=routing["BLOCK"], marker_color=BAD)
    fig.update_layout(barmode="stack", title="VERIFY vs BLOCK outcomes per device", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(routing, use_container_width=True, hide_index=True)

# ---------- 8. What's Next ----------
with sections[7]:
    st.subheader("What's Next — Phase 2")
    st.markdown(
        "Review 1 covers detection. Review 2 adds adaptation and adversarial robustness — "
        "deliberately scoped as a separate milestone."
    )
    st.markdown("""
- Expand to 8–10 devices
- Full Fingerprint Manager: versioning + verified-only incremental updates
- QUARANTINE decision path (known device, anomalous behavior)
- Jitter & mimicry attack simulation — the real test of the DWT-vs-raw-stats hypothesis
- Admin dashboard (Streamlit)
""")

st.divider()
st.caption("Encrypted IoT Device Fingerprinting — Computer Networks Project · Phase 1 Report")
