"""
FauxPix FastAPI Backend
Mirrors the audio deepfake detector's FastAPI architecture.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, json, logging
from detector import FauxPixDetector, DetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FauxPix — Partial Video Deepfake Detector",
    description=(
        "Segment-level video deepfake detection using signal-processing-first "
        "multi-signal analysis. Detects WHERE manipulation was injected, not just "
        "whether a clip is fake. Per-clip self-referential z-scoring ensures robustness "
        "on compressed, low-quality, and field-recorded footage."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FauxPixDetector()


def result_to_dict(r: DetectionResult) -> dict:
    return {
        "verdict": r.verdict,
        "overall_confidence": r.overall_confidence,
        "summary": r.summary,
        "processing_time_sec": r.processing_time,
        "video_info": {
            "total_frames": r.total_frames,
            "fps": r.fps,
            "duration_sec": round(r.total_frames / r.fps, 2) if r.fps else 0,
        },
        "anomaly_segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "peak_z_score": s.peak_z_score,
                "confidence": s.confidence,
                "triggered_signals": s.triggered_signals,
                "description": s.description,
            }
            for s in r.segments
        ],
        "per_signal_zscores": {
            k: v[:200]  # truncate for response size
            for k, v in r.per_signal_zscores.items()
        },
        "signal_descriptions": {
            "lip_aspect_ratio":   "Lip geometry deviation — phoneme-viseme proxy. Spikes indicate lip-sync manipulation.",
            "laplacian_variance": "Face texture energy — GAN over-smoothing analogue of audio HF suppression.",
            "fft_peak_score":     "GAN frequency fingerprint — periodic peaks from generator upsampling artifacts.",
            "landmark_velocity":  "Facial landmark jitter — visual analogue of audio F0 jitter at splice boundaries.",
            "temporal_gradient":  "Face vs background temporal gradient ratio — composite region mismatch.",
            "composite":          "Weighted composite of all signals. Peak detection run on differential of this series.",
        },
    }


@app.get("/")
def root():
    return {
        "name": "FauxPix",
        "version": "1.0.0",
        "description": "Partial video deepfake detector — segment-level forensic analysis",
        "endpoints": {
            "POST /detect": "Upload video for analysis",
            "GET /health": "Health check",
            "GET /signals": "Signal documentation",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "detector": "FauxPixDetector v1.0"}


@app.get("/signals")
def signals():
    return {
        "signals": [
            {
                "id": "lip_aspect_ratio",
                "name": "Lip Geometry (Phoneme-Viseme Proxy)",
                "audio_analogue": "F0 jitter discontinuity at splice boundary",
                "targets": "Wav2Lip, VideoRetalking, SadTalker lip-sync deepfakes",
                "threshold_z": 2.0,
            },
            {
                "id": "laplacian_variance",
                "name": "Face Texture Energy",
                "audio_analogue": "hf_ratio_z — neural vocoder HF energy suppression",
                "targets": "GAN face synthesis, face swap models",
                "threshold_z": 2.0,
            },
            {
                "id": "fft_peak_score",
                "name": "GAN Frequency Fingerprint",
                "audio_analogue": "MFCC vocoder fingerprint shift",
                "targets": "StyleGAN, DALL-E video, diffusion-based synthesis",
                "threshold_z": 2.0,
            },
            {
                "id": "landmark_velocity",
                "name": "Facial Landmark Jitter (FFD)",
                "audio_analogue": "F0 jitter z-score against clip baseline",
                "targets": "Face reenactment, partial face manipulation",
                "threshold_z": 2.2,
            },
            {
                "id": "temporal_gradient",
                "name": "Temporal Gradient Anomaly",
                "audio_analogue": "Spectral flatness z-score vs clip baseline",
                "targets": "All frame-level compositing deepfakes",
                "threshold_z": 2.0,
            },
        ],
        "key_innovation": (
            "All signals are z-scored against the clip's OWN baseline (first 30% of frames). "
            "This makes detection robust to compression artifacts, noisy recording conditions, "
            "and variable quality footage — critical for law enforcement / body cam use cases."
        ),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Supported: {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Running FauxPix detection on {file.filename} ({len(content)//1024}KB)")
        result = detector.detect(tmp_path)
        return JSONResponse(result_to_dict(result))
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
