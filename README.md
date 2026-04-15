# FauxPix 🎬 — Partial Video Deepfake Detector

**Segment-level video deepfake detection using signal-processing-first, per-clip z-scoring.**

> Detects *where* manipulation was injected — not just whether a clip is fake.

---

## Demo Results (Real Output)

```
Video: test_partial_deepfake.mp4 (8.0s, simulated splice at t=3.8s–5.2s)

Verdict:     MANIPULATED
Confidence:  95%
Segments:    6 anomaly segments detected

[HIGH] t=3.24s–4.72s | lip_sync_anomaly, texture_smoothing, gan_frequency_artifact | z=1.84
[HIGH] t=5.40s–6.88s | texture_smoothing, gan_frequency_artifact, temporal_gradient  | z=2.21
```

→ Full output: [`sample_output/sample_output.json`](sample_output/sample_output.json)

---

## Architecture

FauxPix mirrors the partial audio deepfake detection philosophy — **per-clip self-referential z-scoring** — extended to 5 video-domain signals.

```
Video Input
    │
    ├── [Signal 1] Lip Geometry         ← phoneme-viseme proxy (audio: F0 jitter)
    ├── [Signal 2] Laplacian Variance   ← GAN over-smoothing (audio: hf_ratio_z)
    ├── [Signal 3] FFT Peak Score       ← GAN frequency fingerprint (audio: MFCC shift)
    ├── [Signal 4] Landmark Velocity    ← facial feature drift (audio: F0 jitter)
    └── [Signal 5] Temporal Gradient   ← face/bg ratio anomaly (audio: spectral flatness z)
         │
         ▼
    Per-clip Z-scoring (baseline = first 30% of clip)
         │
         ▼
    Composite Score → Differential → Peak Detection
         │
         ▼
    Anomaly Segments with timestamps + confidence + triggered signals
         │
         ▼
    Forensic Report JSON
```

### Key Innovation: Per-Clip Self-Referential Z-Scoring

Most detectors compare features against a universal trained reference. This fails on:
- Compressed body cam footage
- Low-resolution surveillance video
- Phone call recordings
- Variable lighting / noisy environments

FauxPix z-scores every signal against **the same clip's own baseline** (first 30% of frames). The question becomes: *"Is this window anomalous for this clip?"* — not *"Is this anomalous in absolute terms?"*

This makes FauxPix robust to field conditions — critical for law enforcement deployments.

---

## Signal Reference

| Signal | What It Detects | Audio Analogue | Threshold |
|--------|----------------|----------------|-----------|
| Lip Geometry | Phoneme-viseme mismatch (Wav2Lip, VideoRetalking) | F0 jitter discontinuity | z > 2.0 |
| Laplacian Variance | GAN over-smoothing of face texture | hf_ratio_z (neural vocoder HF suppression) | z > 2.0 |
| FFT Peak Score | GAN upsampling frequency fingerprint | MFCC vocoder fingerprint shift | z > 2.0 |
| Landmark Velocity | Facial Feature Drift between frames | F0 jitter at splice boundary | z > 2.2 |
| Temporal Gradient | Face vs background motion inconsistency | Spectral flatness z-score | z > 2.0 |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Detection engine | Python · MediaPipe · OpenCV · NumPy · SciPy |
| API backend | FastAPI · Uvicorn |
| Frontend | React + Vite |
| Face tracking | MediaPipe Face Mesh (468 landmarks) |
| Face detection | MediaPipe Face Detection |
| Splice localization | `scipy.signal.find_peaks` on composite z-score differential |

---

## Quickstart

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI running at http://localhost:5173
```

### CLI / Script
```python
from backend.detector import FauxPixDetector

det = FauxPixDetector()
result = det.detect("your_video.mp4")

print(result.verdict)           # "MANIPULATED" | "AUTHENTIC" | "INCONCLUSIVE"
print(result.overall_confidence)
for seg in result.segments:
    print(f"t={seg.start_time}s–{seg.end_time}s: {seg.triggered_signals}")
```

---

## API Reference

```
POST /detect          Upload video, returns full detection result
GET  /signals         Signal documentation with audio analogues
GET  /health          Health check
GET  /docs            Auto-generated API docs (Swagger)
```

### Response Schema
```json
{
  "verdict": "MANIPULATED",
  "overall_confidence": 0.95,
  "summary": "MANIPULATED — 6 anomaly segment(s) detected...",
  "anomaly_segments": [
    {
      "start_time": 3.24,
      "end_time": 4.72,
      "peak_z_score": 1.839,
      "confidence": "high",
      "triggered_signals": ["lip_sync_anomaly", "texture_smoothing", "gan_frequency_artifact"],
      "description": "Anomaly at t=3.24s-4.72s: lip geometry deviation (z=1.84); GAN texture smoothing (z=2.21)"
    }
  ],
  "per_signal_zscores": { "lip_aspect_ratio": [...], "composite": [...] }
}
```

---

## Datasets for Evaluation

| Dataset | What to Test |
|---------|-------------|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | Face swap + reenactment (c23/c40 compression) |
| [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) | Audio-visual multimodal (RVFA, FVFA) |
| [LAV-DF](https://github.com/ControlNet/LAV-DF) | **Localized A-V partial deepfakes** — directly analogous to this architecture |
| [DF40](https://github.com/YZY-stack/DF40) | 40 deepfake methods — generalization test |

---

## Connection to Audio Deepfake Detection

This project is the video extension of a partial audio deepfake detector
([Deepfake-Detector](https://github.com/Rabba-Meghana/Deepfake-Detector)).

| Audio Domain | Video Domain |
|-------------|-------------|
| MFCC vocoder fingerprint shift | FFT GAN frequency fingerprint |
| HF energy ratio (hf_ratio_z) | Laplacian variance (texture smoothing) |
| F0 jitter at splice boundary | Landmark velocity spike (FFD) |
| Spectral flatness z-score | Temporal gradient face/bg ratio |
| Splice boundary peak detection | Composite z-score differential peaks |
| Per-clip baseline z-scoring | Per-clip baseline z-scoring (same) |

When combined with audio detection: **dual-modality forensic evidence** — both systems
flag the same timestamp independently, producing court-admissible chain-of-evidence documentation.

---

## Author

**Meghana Rabba**  
MS Computer Science, Illinois Institute of Technology, Chicago  
mrabba@hawk.illinoistech.edu
