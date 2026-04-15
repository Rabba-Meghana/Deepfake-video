"""
FauxPix FastAPI Backend — 6-signal video deepfake detector
Accepts GROQ_API_KEY via env var or X-Groq-Api-Key header.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, logging, json
import numpy as np
from typing import Optional
from detector import FauxPixDetector, DetectionResult

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FauxPix — Partial Video Deepfake Detector",
    description=(
        "6-signal segment-level video deepfake detection. "
        "Signal 6 (Groq Whisper phoneme-viseme) requires GROQ_API_KEY. "
        "All signals z-scored against the clip's own baseline — "
        "robust on compressed, noisy, body cam footage."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)


def make_response(r: DetectionResult) -> dict:
    pv = None
    if r.phoneme_viseme_report:
        pv = {
            "groq_active":    r.phoneme_viseme_report.groq_used,
            "transcript":     r.phoneme_viseme_report.transcript,
            "word_count":     r.phoneme_viseme_report.word_count,
            "mismatch_count": r.phoneme_viseme_report.mismatch_count,
            "mismatch_rate":  r.phoneme_viseme_report.mismatch_rate,
            "mismatches":     r.phoneme_viseme_report.mismatches,
            "error":          r.phoneme_viseme_report.error,
        }

    return {
        "verdict":            r.verdict,
        "overall_confidence": r.overall_confidence,
        "summary":            r.summary,
        "groq_transcript":    r.groq_transcript,
        "processing_time_sec": r.processing_time,
        "video_info": {
            "total_frames": r.total_frames,
            "fps":          r.fps,
            "duration_sec": round(r.total_frames / r.fps, 2) if r.fps else 0,
        },
        "anomaly_segments": [
            {
                "start_time":       s.start_time,
                "end_time":         s.end_time,
                "start_frame":      s.start_frame,
                "end_frame":        s.end_frame,
                "peak_z_score":     s.peak_z_score,
                "confidence":       s.confidence,
                "triggered_signals": s.triggered_signals,
                "description":      s.description,
            }
            for s in r.segments
        ],
        "phoneme_viseme_report": pv,
        "per_signal_zscores": {k: v[:300] for k, v in r.per_signal_zscores.items()},
        "signal_descriptions": {
            "lip_aspect_ratio":        "Lip geometry — phoneme-viseme proxy. Spikes = lip-sync manipulation.",
            "laplacian_variance":      "Face texture energy — GAN over-smoothing (audio: hf_ratio_z).",
            "fft_peak_score":          "GAN upsampling fingerprint — checkerboard artifacts in FFT.",
            "landmark_velocity":       "Facial Feature Drift — visual analogue of audio F0 jitter.",
            "temporal_gradient":       "Face vs background temporal gradient ratio.",
            "phoneme_viseme_mismatch": "Groq Whisper word timestamps x MediaPipe lip geometry. "
                                       "KEY SIGNAL — direct bridge to MAIA audio detection.",
            "composite":               "Weighted composite. Peak detection on differential localizes splices.",
        },
    }


@app.get("/")
def root():
    return {
        "name": "FauxPix", "version": "2.0.0",
        "signals": 6,
        "signal_6": "Groq Whisper phoneme-viseme (set GROQ_API_KEY or pass X-Groq-Api-Key header)",
        "endpoints": {"POST /detect": "Upload video", "GET /signals": "Signal docs", "GET /health": "Health"},
    }

@app.get("/health")
def health():
    groq_key = os.environ.get("GROQ_API_KEY")
    return {
        "status": "ok",
        "groq_configured": bool(groq_key),
        "signals_active": 6 if groq_key else 5,
    }

@app.get("/signals")
def signals():
    return {
        "signals": [
            {"id": "lip_aspect_ratio",        "name": "Lip Geometry",             "audio_analogue": "F0 jitter",         "threshold_z": 2.0},
            {"id": "laplacian_variance",       "name": "Texture Energy",           "audio_analogue": "hf_ratio_z",        "threshold_z": 2.0},
            {"id": "fft_peak_score",           "name": "GAN Frequency Fingerprint","audio_analogue": "MFCC vocoder shift", "threshold_z": 2.0},
            {"id": "landmark_velocity",        "name": "Landmark Jitter (FFD)",    "audio_analogue": "F0 jitter",         "threshold_z": 2.2},
            {"id": "temporal_gradient",        "name": "Temporal Gradient",        "audio_analogue": "Spectral flatness z","threshold_z": 2.0},
            {"id": "phoneme_viseme_mismatch",  "name": "Phoneme-Viseme Sync (Groq)","audio_analogue": "MAIA audio detection","threshold_z": 1.8,
             "requires": "GROQ_API_KEY", "weight": "0.24 (highest — most specific to lip-sync deepfakes)"},
        ],
        "key_innovation": "Per-clip self-referential z-scoring. Robust to compression and noisy field footage.",
        "groq_model": "whisper-large-v3-turbo",
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    x_groq_api_key: Optional[str] = Header(default=None),
):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format {ext}. Allowed: {allowed}")

    # Groq key: header > env var
    groq_key = x_groq_api_key or os.environ.get("GROQ_API_KEY")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Detecting: {file.filename} ({len(content)//1024}KB) groq={'yes' if groq_key else 'no'}")
        det    = FauxPixDetector(groq_api_key=groq_key)
        result = det.detect(tmp_path)
        response = json.loads(json.dumps(make_response(result), cls=NumpyEncoder))
        # Add Groq LLM forensic explanation if key available and segments found
        if groq_key and response["anomaly_segments"]:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                segs_text = "\n".join([
                    f"- t={s['start_time']}s-{s['end_time']}s [{s['confidence']}]: {', '.join(s['triggered_signals'])} (peak z={s['peak_z_score']})"
                    for s in response["anomaly_segments"]
                ])
                pv_text = ""
                if response.get("phoneme_viseme_report"):
                    pv = response["phoneme_viseme_report"]
                    pv_text = f"Phoneme-viseme analysis: {pv['mismatch_count']}/{pv['word_count']} word mismatches. Transcript: '{pv['transcript'][:200]}'"
                prompt = f"""You are a forensic video analyst. A deepfake detection system analyzed a video and found:

Verdict: {response['verdict']} (confidence: {response['overall_confidence']*100:.0f}%)
Duration: {response['video_info']['duration_sec']}s at {response['video_info']['fps']}fps

Anomaly segments detected:
{segs_text}

{pv_text}

Signal explanations:
- lip_sync_anomaly: lip geometry deviated from clip baseline (phoneme-viseme mismatch proxy)
- texture_smoothing: face texture over-smoothed (GAN/neural synthesis signature)  
- gan_frequency_artifact: periodic peaks in FFT spectrum (GAN upsampling fingerprint)
- landmark_jitter: facial landmark micro-jitter between frames (Facial Feature Drift)
- temporal_gradient_anomaly: face region motion inconsistent with background
- phoneme_viseme_mismatch: Groq Whisper confirmed audio phoneme does not match lip shape

Write a 3-4 sentence forensic explanation of these findings for a law enforcement audience. Be specific about timestamps and what each signal means. If phoneme-viseme shows 0 mismatches, note that audio-visual sync appears authentic. Be honest if results are ambiguous."""
                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=300,
                )
                response["forensic_explanation"] = chat.choices[0].message.content
                logger.info("Groq forensic explanation generated")
            except Exception as e:
                logger.warning(f"Groq explanation failed: {e}")
                response["forensic_explanation"] = None
        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
