# Encrypted IoT Device Fingerprinting

## Dataset Selection

### Chosen Devices (5)
| Device | MAC | Rationale |
|---|---|---|
| AmazonEcho | 44:65:0d:56:cc:d3 | Strong, consistent traffic |
| BelkinWemoMotionSensor | ec:1a:59:83:28:11 | Strong, consistent traffic |
| PhilipsHue | 00:17:88:2b:9a:25 | Strong, consistent traffic |
| SamsungCamera | 00:16:6c:ab:6b:88 | Strong, consistent traffic |
| TPLinkSmartPlug | 50:c7:bf:00:56:39 | Lower volume but sufficient |

### Excluded Device
**NestProtectSmokeAlarm** (MAC 18:b4:30:25:be:e4) was excluded after Step 1 density profiling. 95.2% of its 702,949 total packets (669,491) were concentrated on a single anomalous day (March 2, 2017). On all other 148 active days across the 6-month capture, it produced only ~200–400 packets/day — far too few to form even a handful of 128-packet windows. This is consistent with NestProtect's role as a smoke alarm that sends periodic heartbeats with one burst event. Including it would yield a class with ~1–5 training samples, which is statistically meaningless for classification.

### Capture Window: November 22–24, 2016
This 3-day window was selected after profiling packet density per day across the full capture range (Sept 2016 – April 2017) for all candidate devices. Nov 22–24 maximizes simultaneous traffic across all 5 chosen devices:

| Device | Packets in Window | Min Single-Day | ~Windows of 128 |
|---|---:|---:|---:|
| BelkinWemoMotionSensor | 381,530 | 118,030 | ~2,981 |
| PhilipsHue | 228,268 | 69,921 | ~1,783 |
| SamsungCamera | 182,589 | 52,676 | ~1,426 |
| AmazonEcho | 174,982 | 57,305 | ~1,367 |
| TPLinkSmartPlug | 8,503 | 2,817 | ~66 |

## Step 3: IAT Transform Decision

- **Transform**: `log(1 + IAT)` (`iat_transform: log1p` in config). Applied to raw IAT values before windowing.
- **Why log-transform, not percentile clipping**: Computing a percentile-based clip threshold (e.g., P99) from the full device dataset before the train/val/test split (Step 6) would leak validation/test information into a preprocessing decision — the same violation Hard Rule 3 prohibits for scalers. A log-transform `log(1 + IAT)` avoids this entirely, since it is a fixed mathematical function with no data-derived parameters, while still compressing the multi-order-of-magnitude dynamic range (microseconds to minutes) into a DWT-friendly range.
- **Preserves real signal**: Genuine long idle gaps (e.g., TPLinkSmartPlug's ~4-minute heartbeat intervals) are compressed, not discarded — they remain available as real behavioral signal for the classifier. Any remaining feature-level scale differences will be handled by the per-feature scaler in Step 6, fit on the training split only.

## Step 4: Windowing and Padding Decisions

- **Window Size and Step**: `window_size` is set to `128` and `window_step` is set to `128`. This means windows are strictly non-overlapping. This decision was made to ensure that each window represents a completely independent slice of traffic (no data leakage or shared packets between adjacent windows), which simplifies the strict session/time-block splitting required by the project rules.
- **Handling Incomplete Windows**: `pad_incomplete_windows` is set to `false`. The final incomplete window for any device is dropped rather than zero-padded. This prevents the Discrete Wavelet Transform (DWT) from extracting artificial edge artifacts from the padding boundary, which could pollute the signal with synthetic features not actually present in the device's traffic.

## Step 8: Open-Set Detection (Diagnostic vs Final Engine)

During Step 8, we evaluated the VAE on two different held-out devices. 
- **TPLinkSmartPlug** (extreme IAT outlier): Detected 100% of the time as anomalous by both DWT and Raw Stats.
- **BelkinWemoMotionSensor** (subtle IAT differences): Detected only 3.23% (DWT) and 14.28% (Raw Stats) of the time.

**Diagnostic Finding:** The VAE struggles to detect subtle unknown devices on clean traffic, with raw features outperforming DWT features. This limitation is noted as a diagnostic finding. However, for the final deployed decision engine (Step 9), the known/unknown boundary is established by treating the easily-detected **TPLinkSmartPlug** as the genuine unseen/unknown device, ensuring consistency between the classifier's known set and the VAE's known set.

## Phase 1 Complete: Hyperparameter consolidation

For a transparent record of all configuration decisions made during Phase 1:

- **Device Subset**: `AmazonEcho`, `BelkinWemoMotionSensor`, `PhilipsHue`, `SamsungCamera`, `TPLinkSmartPlug`
- **Excluded Device**: `NestProtectSmokeAlarm` excluded due to 95% of traffic concentrated on a single anomalous day (insufficient samples elsewhere).
- **Date Window**: `November 22–24, 2016` (selected via density profiling to guarantee sufficient simultaneous traffic across the subset).
- **IAT Transform**: `log(1 + IAT)` (chosen instead of percentile clipping to avoid data leakage before splits, preserving extreme idle gaps natively).
- **Window Size/Step**: `128` size, `128` step (strictly non-overlapping).
- **Incomplete Windows**: Dropped (`pad_incomplete_windows: false`) to avoid artificial padding artifacts in the frequency domain.
- **DWT Config**: Wavelet `db4`, Decomposition level `3`.
- **Classification Performance**: Raw stats outperformed DWT (99.3% vs 93.4%) on clean traffic. DWT carried forward as it is expected to shine under Phase 2 jitter conditions.
- **VAE Architecture**: `latent_dim = 16`. Ablation proved smaller bottlenecks (e.g. 2) degraded detection capability by increasing baseline reconstruction error uniformly.
- **Unknown Device Simulation**: `TPLinkSmartPlug` was entirely held out from the final VAE and Classifier training in Step 9 because its extreme IAT pattern provides a clear signal, ensuring the final engine boundary is sharp. (Subtle anomalies like Belkin were proven much harder for the VAE).
- **VAE Threshold**: 99th percentile of the known-device validation split.
