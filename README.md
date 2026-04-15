# FauxPix 🎬 — Partial Video Deepfake Detector

**6-signal segment-level video deepfake detection.**  
Detects *where* manipulation was injected — not just whether a clip is fake.

Signal 6 uses **Groq Whisper** (same API as the companion audio detector) for phoneme-viseme synchrony analysis — the direct bridge between audio and video deepfake detection.

---

## Real Output

```
Video:      test_partial_deepfake.mp4 (8s, splice injected at t=3.8s–5.2s)
Verdict:    MANIPULATED
Confidence: 95%
Segments:   6 anomaly segments detected

[HIGH] t=3.24s–4.72s | lip_sync_anomaly, texture_smoothing, gan_frequency_artifact | z=1.84
[HIGH] t=5.40s–6.88s | texture_smoothing, gan_frequency_artifact, temporal_gradient  | z=2.21
```

Full output → [`sample_output/sample_output.json`](sample_output/sample_output.json)

---

## Architecture

```
Video Input
    │
    ├── [S1] Lip Geometry (MediaPipe)        ← phoneme-viseme proxy    (audio: F0 jitter)
    ├── [S2] Laplacian Variance              ← GAN over-smoothing       (audio: hf_ratio_z)
    ├── [S3] FFT Peak Score                  ← GAN frequency fingerprint(audio: MFCC shift)
    ├── [S4] Landmark Velocity               ← Facial Feature Drift     (audio: F0 jitter)
    ├── [S5] Temporal Gradient               ← face vs bg ratio         (audio: spectral flatness)
    └── [S6] Phoneme-Viseme Sync (Groq)  ★  ← Whisper timestamps × lip shape
         │                                       KEY SIGNAL — MAIA ↔ FauxPix bridge
         ▼
    Per-clip self-referential z-scoring  ← core innovation
         ▼
    Composite score → differential → find_peaks()
         ▼
    Anomaly segments: timestamp + confidence + triggered signals
         ▼
    Forensic JSON report
```

### Core Innovation: Per-Clip Self-Referential Z-Scoring

Every signal is z-scored against **the same clip's own baseline** (first 30% of frames).  
Works on: compressed body cam footage · surveillance video · phone recordings · variable lighting.  
*Same philosophy as the companion partial audio deepfake detector.*

### Signal 6: Groq Whisper Phoneme-Viseme

1. Extract audio from video via ffmpeg
2. Send to `whisper-large-v3-turbo` on Groq → **word-level timestamps**
3. Map each word's first phoneme to expected viseme (lip shape)
4. Check MediaPipe lip aspect ratio at that frame
5. If actual LAR outside expected range → **phoneme-viseme mismatch**

High-value phonemes: bilabials (`b`,`p`,`m`) require closed lips (LAR < 0.06).  
If Groq says `"been"` is being spoken but lips are open → lip-sync deepfake confirmed.

**Bridge to MAIA:** When MAIA flags audio synthesis at t=4.0s AND FauxPix flags phoneme-viseme mismatch at t=4.0s → dual-modality forensic evidence from independent signal domains.

---

## Signal Reference

| # | Signal | Targets | Audio Analogue | Threshold |
|---|--------|---------|----------------|-----------|
| 1 | Lip Geometry | Wav2Lip, VideoRetalking | F0 jitter | z > 2.0 |
| 2 | Laplacian Variance | GAN face synthesis | hf_ratio_z | z > 2.0 |
| 3 | FFT Peak Score | StyleGAN, diffusion models | MFCC vocoder shift | z > 2.0 |
| 4 | Landmark Velocity (FFD) | Face reenactment | F0 jitter | z > 2.2 |
| 5 | Temporal Gradient | All compositing deepfakes | Spectral flatness z | z > 2.0 |
| 6 | **Phoneme-Viseme (Groq)** | **Lip-sync deepfakes** | **MAIA audio detection** | **z > 1.8** |

---

## Quickstart

### 1. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) → Create API key → copy it.  
*(Same key works for audio detector + FauxPix Signal 6)*

### 2. Backend
```bash
cd backend
pip install -r requirements.txt

# With Groq (all 6 signals):
GROQ_API_KEY=gsk_your_key uvicorn main:app --reload

# Without Groq (5 signals, no key needed):
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
# Paste Groq key in the UI field, drop a video, run detection
```

### 4. Python (direct)
```python
import os
os.environ["GROQ_API_KEY"] = "gsk_your_key"   # optional

from backend.detector import FauxPixDetector
det    = FauxPixDetector()
result = det.detect("your_video.mp4")

print(result.verdict)            # MANIPULATED / AUTHENTIC / INCONCLUSIVE
print(result.overall_confidence)
for seg in result.segments:
    print(f"t={seg.start_time}s–{seg.end_time}s | {seg.triggered_signals}")

# Groq phoneme-viseme report
if result.phoneme_viseme_report:
    pv = result.phoneme_viseme_report
    print(f"Transcript: {pv.transcript}")
    print(f"Mismatches: {pv.mismatch_count}/{pv.word_count} words ({pv.mismatch_rate:.1%})")
```

---

## API

```
POST /detect          Upload video (pass X-Groq-Api-Key header for Signal 6)
GET  /signals         All 6 signal descriptions with audio analogues
GET  /health          Health check + groq_configured status
GET  /docs            Swagger UI
```

### Response (key fields)
```json
{
  "verdict": "MANIPULATED",
  "overall_confidence": 0.95,
  "groq_transcript": "I have never and will never sell user data",
  "anomaly_segments": [{
    "start_time": 4.0, "end_time": 5.5,
    "confidence": "high",
    "triggered_signals": ["phoneme_viseme_mismatch", "lip_sync_anomaly", "landmark_jitter"],
    "description": "Anomaly t=4.00s–5.50s: phoneme-viseme mismatch (z=2.31) — Groq Whisper; lip geometry deviation (z=1.94)"
  }],
  "phoneme_viseme_report": {
    "transcript": "...",
    "word_count": 42,
    "mismatch_count": 7,
    "mismatch_rate": 0.167,
    "mismatches": [{"word":"been","time":4.02,"viseme_group":"bilabial_stop","actual_lar":0.21,"expected_range":[0,0.06]}]
  }
}
```

---

## Evaluation Datasets

| Dataset | Relevant For |
|---------|-------------|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | Standard benchmark (c23/c40 compression) |
| [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) | Multimodal — RVFA, FVFA categories |
| [LAV-DF](https://github.com/ControlNet/LAV-DF) | **Localized A-V partial deepfakes** — directly matches this architecture |
| [DF40](https://github.com/YZY-stack/DF40) | 40 methods — generalization test |

---

## Connection to Audio Detector

| Audio ([Deepfake-Detector](https://github.com/Rabba-Meghana/Deepfake-Detector)) | Video (FauxPix) |
|---|---|
| MFCC vocoder fingerprint | FFT GAN frequency fingerprint |
| hf_ratio_z (HF energy) | Laplacian variance (texture smoothing) |
| F0 jitter at splice boundary | Landmark velocity spike (FFD) |
| Spectral flatness z-score | Temporal gradient face/bg ratio |
| Groq Whisper transcription | Groq Whisper phoneme timestamps |
| Per-clip baseline z-scoring | Per-clip baseline z-scoring |

**Dual-modality forensic report:** MAIA catches audio synthesis + FauxPix catches visual manipulation at the same timestamp → independent confirmation from two signal domains.

---

**Meghana Rabba** · MS CS, Illinois Institute of Technology · mrabba@hawk.illinoistech.edu
