"""
FauxPix FastAPI Backend — Combined Audio + Video Deepfake Detector
7 video signals + 6 audio signals (MAIA-parallel).
Signal 6 (Groq Whisper phoneme-viseme) requires GROQ_API_KEY.
Signal 7 (cross-frame periodicity + splice) and all audio signals always active.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, logging, json
import numpy as np
from typing import Optional
from detector import FauxPixDetector, DetectionResult, AudioDetectionResult

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FauxPix — Audio + Video Deepfake Detector",
    description=(
        "Combined audio + video deepfake detection. "
        "VIDEO: 7 signals — Signal 6 (Groq Whisper phoneme-viseme) requires GROQ_API_KEY, "
        "Signal 7 (cross-frame periodicity + splice) always active. "
        "AUDIO: 6 MAIA-parallel signals (F0 jitter, HF ratio, spectral flatness, "
        "MFCC delta, ZCR, audio splice) — all always active, no API key required. "
        "All signals z-scored against the clip's own baseline."
    ),
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)


def make_audio_response(ar: AudioDetectionResult) -> dict:
    if ar is None:
        return None
    return {
        "has_audio":          ar.has_audio,
        "verdict":            ar.verdict,
        "overall_confidence": ar.overall_confidence,
        "summary":            ar.summary,
        "sample_rate":        ar.sample_rate,
        "duration_sec":       ar.duration_sec,
        "anomaly_segments": [
            {
                "start_time":        s.start_time,
                "end_time":          s.end_time,
                "peak_z_score":      s.peak_z_score,
                "confidence":        s.confidence,
                "triggered_signals": s.triggered_signals,
                "description":       s.description,
            }
            for s in ar.segments
        ],
        "per_signal_zscores": {k: v[:500] for k, v in ar.per_signal_zscores.items()},
        "signal_descriptions": {
            "f0_jitter":         "F0 pitch jitter — TTS/vocoders are unnaturally smooth. Real voices have micro-variations.",
            "hf_ratio":          "High-frequency energy ratio — GAN over-smoothing removes HF content (MAIA: hf_ratio_z).",
            "spectral_flatness": "Wiener entropy — neural codec artifacts create anomalous spectral flatness.",
            "mfcc_delta":        "Mel-cepstral energy drift — temporal inconsistency in vocal tract modelling.",
            "zcr":               "Zero-crossing rate — unnatural voicing transitions from TTS boundary stitching.",
            "audio_splice":      "Audio cross-frame splice boundary — left vs right window distributional shift (same as video Signal 7B).",
            "audio_composite":   "Weighted composite of all 6 audio signals.",
        },
    }


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
                "start_time":        s.start_time,
                "end_time":          s.end_time,
                "start_frame":       s.start_frame,
                "end_frame":         s.end_frame,
                "peak_z_score":      s.peak_z_score,
                "confidence":        s.confidence,
                "triggered_signals": s.triggered_signals,
                "description":       s.description,
                "modality":          s.modality,
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
            "phoneme_viseme_mismatch": "Groq Whisper word timestamps × MediaPipe lip geometry — direct MAIA audio bridge.",
            "periodicity":             "Signal 7A — GAN frame periodicity. Autocorrelation of 4-signal feature matrix across lags 4-24 frames.",
            "splice":                  "Signal 7B — Video splice boundary. Sliding window distributional shift (15 frames each side).",
            "composite":               "Weighted composite of all video signals.",
        },
        "audio": make_audio_response(r.audio_result),
    }


@app.get("/")
def root():
    return {
        "name": "FauxPix", "version": "4.0.0",
        "modalities": ["video", "audio"],
        "video_signals": 7,
        "audio_signals": 6,
        "video_signal_6": "Groq Whisper phoneme-viseme (set GROQ_API_KEY or pass X-Groq-Api-Key header)",
        "video_signal_7": "Cross-frame sliding window: GAN periodicity (autocorr lags 4-24) + splice boundary",
        "audio_signals_desc": "F0 jitter, HF ratio, spectral flatness, MFCC delta, ZCR, audio splice — always active",
        "endpoints": {"POST /detect": "Upload video (audio+video analyzed)", "GET /signals": "Signal docs", "GET /health": "Health"},
    }

@app.get("/health")
def health():
    groq_key = os.environ.get("GROQ_API_KEY")
    return {
        "status": "ok",
        "groq_configured": bool(groq_key),
        "video_signals_active": 7 if groq_key else 6,
        "audio_signals_active": 6,
        "video_signal_7": "always active — cross-frame comparison, no API key required",
        "audio_detection": "always active — MAIA-parallel 6-signal pipeline",
    }

@app.get("/signals")
def signals():
    return {
        "video_signals": [
            {"id": "lip_aspect_ratio",        "name": "Lip Geometry",              "audio_analogue": "F0 jitter",           "threshold_z": 3.2},
            {"id": "laplacian_variance",       "name": "Texture Energy",            "audio_analogue": "hf_ratio_z",          "threshold_z": 3.5},
            {"id": "fft_peak_score",           "name": "GAN Frequency Fingerprint", "audio_analogue": "MFCC vocoder shift",   "threshold_z": 3.5},
            {"id": "landmark_velocity",        "name": "Landmark Jitter (FFD)",     "audio_analogue": "F0 jitter",           "threshold_z": 3.8},
            {"id": "temporal_gradient",        "name": "Temporal Gradient",         "audio_analogue": "Spectral flatness z",  "threshold_z": 3.5},
            {"id": "phoneme_viseme_mismatch",  "name": "Phoneme-Viseme Sync (Groq)","audio_analogue": "MAIA audio detection", "threshold_z": 1.8,
             "requires": "GROQ_API_KEY", "weight": "0.20 — high-specificity, can trigger MANIPULATED alone"},
            {"id": "periodicity",              "name": "GAN Frame Periodicity",     "audio_analogue": "Periodic artifact detection",
             "method": "Autocorrelation of 4-signal feature matrix at lags 4-24 frames.", "threshold": 0.25},
            {"id": "splice",                   "name": "Video Splice Boundary",     "audio_analogue": "Audio splice detection",
             "method": "Sliding window (15 frames each side). Welch-like statistic per signal column.", "threshold": 2.5},
        ],
        "audio_signals": [
            {"id": "f0_jitter",         "name": "F0 Pitch Jitter",        "video_analogue": "Landmark velocity",    "threshold_z": 3.0},
            {"id": "hf_ratio",          "name": "HF Energy Ratio",         "video_analogue": "Laplacian variance",   "threshold_z": 3.5},
            {"id": "spectral_flatness", "name": "Spectral Flatness",        "video_analogue": "FFT peak score",       "threshold_z": 3.5},
            {"id": "mfcc_delta",        "name": "MFCC Delta",               "video_analogue": "Temporal gradient",    "threshold_z": 3.0},
            {"id": "zcr",               "name": "Zero-Crossing Rate",       "video_analogue": "Lip geometry",         "threshold_z": 3.2},
            {"id": "audio_splice",      "name": "Audio Splice Boundary",    "video_analogue": "Video splice (Sig 7B)","threshold_z": 2.5,
             "weight": "high-specificity, can trigger MANIPULATED alone"},
        ],
        "key_innovations": [
            "Per-clip self-referential z-scoring — robust to compression and noisy field footage.",
            "Video Signal 7 compares frames against each other (not just vs clip mean) — catches GAN rhythm and splice cuts.",
            "Audio signals compare windows against each other — same temporal coherence approach as MAIA.",
            "Phoneme-Viseme (Signal 6) is the direct audio-video bridge: Groq Whisper timestamps vs MediaPipe lip geometry.",
            "Audio+Video combined verdict: if both modalities flag MANIPULATED, confidence is boosted.",
            "2+ signals required for MANIPULATED verdict (phoneme-viseme, audio splice, video splice are high-specificity exceptions).",
        ],
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

        # Groq LLM forensic explanation
        if groq_key and (response["anomaly_segments"] or
                         (response.get("audio") and response["audio"].get("anomaly_segments"))):
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)

                vid_segs = "\n".join([
                    f"  VIDEO t={s['start_time']}s-{s['end_time']}s [{s['confidence']}]: "
                    f"{', '.join(s['triggered_signals'])} (peak z={s['peak_z_score']}, modality={s.get('modality','video')})"
                    for s in response["anomaly_segments"]
                ]) or "  No video anomalies detected."

                aud_segs = ""
                if response.get("audio") and response["audio"].get("anomaly_segments"):
                    aud_segs = "\n".join([
                        f"  AUDIO t={s['start_time']}s-{s['end_time']}s [{s['confidence']}]: "
                        f"{', '.join(s['triggered_signals'])} (peak z={s['peak_z_score']})"
                        for s in response["audio"]["anomaly_segments"]
                    ])
                else:
                    aud_segs = f"  Audio verdict: {response.get('audio', {}).get('verdict', 'NO_AUDIO')}"

                pv_text = ""
                if response.get("phoneme_viseme_report"):
                    pv = response["phoneme_viseme_report"]
                    pv_text = (f"\nPhoneme-Viseme bridge: {pv['mismatch_count']}/{pv['word_count']} "
                               f"word mismatches ({pv['mismatch_rate']*100:.1f}%). "
                               f"Transcript: '{pv['transcript'][:200]}'")

                prompt = f"""You are a forensic multimedia analyst. A deepfake detection system analyzed a video file for both audio and video manipulation.

Overall video verdict: {response['verdict']} (confidence: {response['overall_confidence']*100:.0f}%)
Duration: {response['video_info']['duration_sec']}s at {response['video_info']['fps']}fps

Video anomaly segments:
{vid_segs}

Audio anomaly segments:
{aud_segs}
{pv_text}

Signal glossary:
VIDEO:
- lip_sync_anomaly: lip geometry deviated from clip baseline
- texture_smoothing: GAN over-smoothing of face texture
- gan_frequency_artifact: GAN upsampling checkerboard in FFT
- landmark_jitter: facial landmark micro-jitter (Facial Feature Drift)
- temporal_gradient_anomaly: face motion inconsistent with background
- phoneme_viseme_mismatch: Groq Whisper confirmed audio phoneme ≠ lip shape
- gan_periodicity: GAN generator artifact repeating at fixed frame interval
- splice_boundary: video splice cut (abrupt statistical shift between windows)

AUDIO:
- f0_jitter_anomaly: pitch jitter spike — TTS unnaturally smooth or stitched
- hf_smoothing: high-frequency energy removed — GAN/neural codec signature
- spectral_flatness_anomaly: Wiener entropy spike — codec artifact
- mfcc_drift: mel-cepstral temporal inconsistency
- zcr_anomaly: zero-crossing rate anomaly — unnatural voicing transition
- audio_splice_boundary: audio statistical shift — consistent with audio splice/edit

Write a 4-5 sentence forensic report for a law enforcement audience. Be specific about timestamps. Note if audio and video anomalies co-occur (stronger evidence). If phoneme-viseme shows 0 mismatches, note that audio-visual sync appears authentic at the word level. Be honest if results are ambiguous or confidence is moderate."""

                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=400,
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
