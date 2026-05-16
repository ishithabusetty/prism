# 🎛️ Audio Compliance & Safety Scanner 📡

> **Production-Grade Acoustic Intelligence Pipeline for Copyright Compliance & Regulatory Frequency Surveillance**

The Audio Compliance & Safety Scanner is an advanced AI-assisted digital signal processing (DSP) framework designed to automatically inspect incoming multimedia assets for:

- 🎵 Unauthorized copyrighted audio usage
- 📡 Emergency Alert System (EAS) tone violations
- 🚨 Regulatory acoustic anomalies
- 🔍 Structural audio fingerprint matches

The pipeline acts as an automated acoustic guardian for media ingest platforms by performing high-speed audio extraction, spectral decomposition, fingerprint generation, and compliance auditing entirely through lightweight mathematical signal-processing workflows.

---

# 🚀 Overview

Traditional moderation pipelines often rely heavily on metadata or visual analysis while ignoring the underlying acoustic structure of multimedia content.

This framework instead performs:

- 📊 Deep spectral analysis
- 🎵 Structural audio fingerprinting
- 📡 Frequency-domain anomaly tracking
- ⏱️ Sustained-duration signal validation
- 🧠 Intelligent compliance verdict generation

The system is optimized for:

- 🎬 Video platforms
- 📻 Broadcast compliance systems
- 🏢 Media ingest pipelines
- 🛡️ Regulatory monitoring frameworks
- ☁️ Cloud or local runtime execution

---

# 🧭 System Workflow

## 🔄 End-to-End Pipeline Architecture

```text
📥 Input Media File (.mp4, .mkv, .wav, .mp3)
                    │
                    ▼
        [ 🛠️ Extraction Pipeline ]
         FFmpeg Audio Demuxing
                    │
    +---------------+---------------+
    │                               │
    ▼                               ▼

📊 Acoustic Fingerprint Engine    🚨 Spectral Frequency Evaluator

1. STFT Spectrogram Generation    1. Sliding Hann Window Buffer
2. Peak Coordinate Filtering      2. Multi-Tone Goertzel Analysis
3. Hash Sequence Linking             (1050Hz / 853Hz / 960Hz)

    │                               │
    ▼                               ▼

🔍 In-Memory Match Engine         ⏱️ Continuous Duration Gate

    │                               │
    +---------------+---------------+
                    │
                    ▼

        📝 Structured JSON Verdicts
```

---

# ⚙️ Core Detection Modules

## 🎵 1. Acoustic Fingerprinting Engine

### What It Detects

- Copyright infringement
- Background music theft
- Audio sample reuse
- Re-encoded copyrighted tracks
- Pitch/speed modified audio bypass attempts

---

## 🧠 Internal Logic

### 📊 STFT Spectrogram Transformation

The system converts raw waveform data into a:

> **Time vs Frequency vs Intensity**

spectral map using:

```python
scipy.signal.stft
```

This creates a dense frequency-domain representation of the audio structure.

---

### 🌌 Constellation Map Generation

A 2D Local Maximum Filter removes weak spectral components and isolates dominant peaks.

This creates sparse structural anchor points called:

> 🎯 Constellation Maps

which remain resilient to:
- background noise
- voice overlays
- compression artifacts
- bitrate degradation

---

### 🔐 Hash Linking System

Peak coordinates are paired and hashed using:

```text
SHA-1 Fingerprint Hashing
```

These hashes form:
- structural identifiers
- temporal anchors
- audio signatures

---

### ⏱️ Time Coherence Matching

The engine validates infringement by checking whether hundreds of hashes maintain identical time offset deltas:

```math
Δt = t_ref - t_query
```

Large-scale temporal consistency mathematically proves audio duplication.

---

# 📡 2. Regulatory Multi-Tone Tracker

## What It Detects

- Unauthorized EAS signals
- Weather broadcast tones
- Emergency override frequencies
- Sustained compliance violations

---

# 🧠 Frequency Analysis Logic

Instead of expensive full-spectrum FFT processing, the system leverages:

## ⚡ Goertzel Algorithm

This enables extremely efficient targeted frequency analysis.

---

## 🎯 Target Regulatory Frequencies

| Frequency | Purpose |
|---|---|
| `1050 Hz` | National Weather Service Warning Tone |
| `853 Hz + 960 Hz` | Emergency Alert System Dual-Tone Pair |

---

## ⏱️ Sustained Duration Gate

To avoid false positives from:
- music chords
- synthetic sounds
- accidental harmonics

the detector only triggers if frequencies persist continuously for:

```text
≥ 2.0 seconds
```

---

# ✨ Core Features

## 🎵 Acoustic Intelligence

- STFT-based spectral decomposition
- Constellation-map fingerprinting
- SHA-1 structural audio hashing
- Time-coherent match verification
- In-memory reference indexing

---

## 📡 Compliance & Safety

- Multi-tone EAS detection
- Goertzel frequency evaluation
- Sliding-window DSP analysis
- Continuous-duration validation
- Automated compliance verdict generation

---

## ⚡ Performance & Deployment

- Lightweight DSP architecture
- No GPU dependency required
- Real-time compatible execution
- Cloud and local runtime support
- Google Colab compatible
- Virtual environment friendly

---

# 🛠️ Unified Tech Stack

| Category | Technology |
|---|---|
| 🐍 Core Language | Python |
| 🎬 Multimedia Processing | FFmpeg (`ffmpeg-python`) |
| 🔢 Numerical Processing | NumPy |
| 🧬 DSP & Spectral Analysis | SciPy |
| 🔐 Fingerprint Hashing | Hashlib |
| 📦 Data Structures | Collections |
| ☁️ Runtime Environment | Google Colab / Local venv |
| 📊 Signal Transformation | STFT |
| 📡 Frequency Detection | Goertzel Algorithm |

---

# 🧬 Mathematical Foundations

## 📊 STFT Transformation

```python
scipy.signal.stft
```

Transforms waveform data into frequency-domain representation.

---

## 🌌 Peak Filtering

```python
scipy.ndimage.maximum_filter
```

Performs neighborhood-based peak extraction.

---

## 🔐 SHA-1 Fingerprinting

Structural audio peaks are converted into deterministic hash signatures.

---

## 📡 Goertzel Spectral Targeting

Efficiently measures power density at:

```text
1050 Hz
853 Hz
960 Hz
```

without performing expensive global FFT analysis.

---

# 📥 Input Specifications

## Supported Media Formats

### 🎬 Video
- `.mp4`
- `.mkv`
- `.mov`
- `.webm`

### 🎵 Audio
- `.wav`
- `.mp3`
- `.m4a`
- `.ogg`

---

# 📤 Output Specifications

## JSON Compliance Verdict

```json
{
  "status": "anomaly_flagged",
  "scanned_file": "user_upload_7721.mp4",
  "audio_match": {
    "match_detected": true,
    "reference_asset_id": "copyrighted_studio_track_alpha",
    "match_timestamp_start": 12.4,
    "confidence_score": 1.0,
    "matching_hash_votes": 803
  },
  "eas_alert": {
    "eas_tone_present": false,
    "confidence_score": 0.04,
    "sustained_seconds": 0.0
  },
  "metrics": {
    "processing_time_ms": 792.1
  }
}
```

---

# 🖥️ Console Diagnostic Output

```plaintext
============================================================
                    COMPLIANCE REPORT
============================================================
Scanned Asset File:          user_upload_7721.mp4
EAS Alert Detected:          False
Unauthorized Music Matched:  True
------------------------------------------------------------
Pipeline Runtime Latency:    792.1 ms
============================================================
```

---

# 📂 Repository Structure

```text
.
├── audio_compliance_scanner.ipynb
├── reference_audio/
├── outputs/
├── README.md
```

---

# ⚙️ Installation

## Install Dependencies

```bash
pip install ffmpeg-python
pip install numpy scipy
```

---

# ☁️ Runtime Recommendations

## Recommended Environment

> ✅ Google Colab  
> ✅ Python Virtual Environment  
> ✅ Local Linux/Windows Runtime  

No GPU acceleration is required.

---

# 🎯 Processing Pipeline Summary

| Stage | Purpose |
|---|---|
| FFmpeg Demuxing | Extract raw audio |
| STFT Processing | Frequency-domain transformation |
| Peak Filtering | Structural point extraction |
| SHA-1 Hashing | Fingerprint generation |
| Reverse Match Indexing | Copyright detection |
| Goertzel Analysis | Regulatory tone detection |
| Duration Gating | False-positive suppression |
| JSON Aggregation | Compliance reporting |

---

# 🚀 Future Enhancements

- 🌐 Real-time livestream monitoring
- ☁️ Distributed cloud inference
- 🎵 Neural acoustic embeddings
- 🧠 Transformer-based audio understanding
- 📊 Dashboard integration
- 🔔 Automated moderation alerts

---

# 📜 License

This project is intended for educational, research, compliance automation, and prototype deployment purposes.

---

# 👨‍💻 Author

Developed as part of an AI-powered multimedia compliance and acoustic intelligence research initiative focused on scalable DSP-driven moderation systems.