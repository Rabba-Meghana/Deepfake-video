"""
FauxPix — Partial Video Deepfake Detector
==========================================
Signal-processing-first pipeline for segment-level video deepfake detection.
Detects WHERE in a video manipulation was injected, not just whether a clip is fake.

Architecture mirrors the partial audio deepfake approach (per-clip z-scoring),
extended to 6 video-domain signals.

Author: Meghana Rabba
Affiliation: Illinois Institute of Technology / BLK-BX Research
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy import stats
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameFeatures:
    frame_idx: int
    timestamp: float
    # Signal 1 - Lip geometry (phoneme-viseme proxy)
    lip_aspect_ratio: float = 0.0
    lip_area: float = 0.0
    # Signal 2 - Texture (GAN over-smoothing analogue of HF energy)
    laplacian_var: float = 0.0
    # Signal 3 - GAN frequency fingerprint
    fft_peak_score: float = 0.0
    # Signal 4 - Landmark jitter (visual F0 jitter analogue)
    landmark_velocity: float = 0.0
    # Signal 5 - Temporal gradient (inter-frame face region change)
    temporal_gradient: float = 0.0
    # Composite
    composite_score: float = 0.0
    composite_z: float = 0.0


@dataclass
class AnomalySegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    peak_z_score: float
    triggered_signals: List[str]
    confidence: str  # "high" | "medium" | "low"
    description: str


@dataclass
class DetectionResult:
    verdict: str           # "MANIPULATED" | "AUTHENTIC" | "INCONCLUSIVE"
    overall_confidence: float  # 0-1
    segments: List[AnomalySegment]
    frame_features: List[FrameFeatures]
    per_signal_zscores: dict
    processing_time: float
    total_frames: int
    fps: float
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Signal extractors
# ─────────────────────────────────────────────────────────────────────────────

class LipGeometryExtractor:
    """
    Signal 1 — Phoneme-Viseme Proxy
    Measures lip aspect ratio and area per frame.
    Anomalies in the lip geometry time-series relative to clip baseline
    indicate lip-sync manipulation (e.g. Wav2Lip injection).
    """

    # MediaPipe mouth landmark indices
    UPPER_LIP = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270]
    LOWER_LIP = [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87]
    LIP_TOP = [13]
    LIP_BOTTOM = [14]
    LIP_LEFT = [61]
    LIP_RIGHT = [291]

    def extract(self, landmarks, frame_w: int, frame_h: int) -> Tuple[float, float]:
        if landmarks is None:
            return 0.0, 0.0
        lm = landmarks.landmark
        top = lm[13]
        bottom = lm[14]
        left = lm[61]
        right = lm[291]
        vertical = abs(top.y - bottom.y) * frame_h
        horizontal = abs(left.x - right.x) * frame_w
        lip_ar = vertical / (horizontal + 1e-6)
        lip_area = vertical * horizontal
        return float(lip_ar), float(lip_area)


class TextureAnalyzer:
    """
    Signal 2 — Laplacian Variance (GAN Over-Smoothing Detector)
    Real faces have natural micro-texture. GAN-generated faces over-smooth
    high-frequency detail — exactly analogous to neural vocoder HF energy suppression.
    Low Laplacian variance in the face crop = synthetic smoothing signature.
    """

    def extract(self, face_crop: np.ndarray) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())


class GANFrequencyAnalyzer:
    """
    Signal 3 — FFT Peak Score (GAN Frequency Fingerprint)
    GAN upsampling operations leave periodic peaks in the 2D FFT power spectrum.
    These 'checkerboard artifacts' appear at spatial frequencies corresponding
    to the generator's upsampling stride — a fingerprint of synthetic generation.
    """

    def extract(self, face_crop: np.ndarray) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        if gray.shape[0] < 32 or gray.shape[1] < 32:
            return 0.0
        resized = cv2.resize(gray, (128, 128)).astype(np.float32)
        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        # Mask DC component
        h, w = magnitude.shape
        magnitude[h//2-4:h//2+4, w//2-4:w//2+4] = 0
        # Score = ratio of max off-center peak to mean
        score = float(magnitude.max() / (magnitude.mean() + 1e-6))
        return score


class LandmarkVelocityTracker:
    """
    Signal 4 — Facial Landmark Velocity (Visual F0-Jitter Analogue)
    Real facial motion is smooth between frames — landmark positions change
    continuously following natural head motion and expression dynamics.
    Deepfakes introduce frame-to-frame micro-jitter (Facial Feature Drift / FFD)
    as the generator produces independent per-frame errors.
    Velocity z-score against clip baseline catches these discontinuities —
    exactly as F0 jitter z-score catches audio splice boundaries.
    """

    def __init__(self):
        self.prev_landmarks = None

    def extract(self, landmarks, frame_w: int, frame_h: int) -> float:
        if landmarks is None:
            self.prev_landmarks = None
            return 0.0
        # Key landmarks: eyes, nose tip, mouth corners (stable across expressions)
        KEY_POINTS = [1, 4, 33, 263, 61, 291, 199]
        curr = np.array([[landmarks.landmark[i].x * frame_w,
                          landmarks.landmark[i].y * frame_h]
                         for i in KEY_POINTS])
        if self.prev_landmarks is None:
            self.prev_landmarks = curr
            return 0.0
        velocity = float(np.mean(np.linalg.norm(curr - self.prev_landmarks, axis=1)))
        self.prev_landmarks = curr
        return velocity


class TemporalGradientAnalyzer:
    """
    Signal 5 — Temporal Gradient Anomaly (Face vs Background Ratio)
    In real video, the temporal gradient (pixel delta between adjacent frames)
    is consistent between the face region and the background.
    In deepfakes, the composite face region has a different temporal gradient
    profile — the generator produces slightly different pixel values each frame
    in ways that don't match natural scene motion.
    Elevated face-to-background temporal gradient ratio = manipulation flag.
    """

    def __init__(self):
        self.prev_frame = None

    def extract(self, frame: np.ndarray, face_bbox: Optional[Tuple]) -> float:
        if self.prev_frame is None or frame is None:
            self.prev_frame = frame.copy() if frame is not None else None
            return 0.0
        diff = cv2.absdiff(frame, self.prev_frame).astype(np.float32)
        self.prev_frame = frame.copy()
        if face_bbox is None:
            return float(diff.mean())
        x, y, w, h = face_bbox
        face_diff = diff[y:y+h, x:x+w]
        mask = np.ones(diff.shape[:2], dtype=bool)
        mask[y:y+h, x:x+w] = False
        bg_diff = diff[mask]
        face_mean = face_diff.mean() if face_diff.size > 0 else 0
        bg_mean = bg_diff.mean() if bg_diff.size > 0 else 1e-6
        return float(face_mean / (bg_mean + 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# Core detector
# ─────────────────────────────────────────────────────────────────────────────

class FauxPixDetector:
    """
    Segment-level partial video deepfake detector.

    Key design principle (same as audio partial deepfake detector):
    ALL features are z-scored against THE SAME CLIP'S OWN BASELINE.
    This makes detection robust to recording conditions, compression,
    resolution, and lighting — critical for real-world forensic use.
    """

    # Z-score thresholds for anomaly flagging
    THRESHOLDS = {
        "lip_aspect_ratio": 2.0,
        "laplacian_var":     2.0,
        "fft_peak_score":    2.0,
        "landmark_velocity": 2.2,
        "temporal_gradient": 2.0,
    }

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.lip_extractor = LipGeometryExtractor()
        self.texture_analyzer = TextureAnalyzer()
        self.fft_analyzer = GANFrequencyAnalyzer()
        self.landmark_tracker = LandmarkVelocityTracker()
        self.temporal_analyzer = TemporalGradientAnalyzer()

    def _get_face_bbox(self, detections, frame_w, frame_h):
        if not detections:
            return None
        d = detections[0]
        bb = d.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * frame_w))
        y = max(0, int(bb.ymin * frame_h))
        w = min(int(bb.width * frame_w), frame_w - x)
        h = min(int(bb.height * frame_h), frame_h - y)
        return (x, y, w, h)

    def _zscore_series(self, values: List[float], baseline_frac: float = 0.3) -> List[float]:
        """
        Per-clip self-referential z-scoring.
        Baseline = first baseline_frac of clip (assumed unmanipulated).
        This is the core innovation: normalize against THIS clip, not a universal reference.
        """
        arr = np.array(values, dtype=np.float64)
        n_baseline = max(10, int(len(arr) * baseline_frac))
        baseline = arr[:n_baseline]
        mu = baseline.mean()
        sigma = baseline.std()
        if sigma < 1e-8:
            sigma = arr.std() + 1e-8
        return ((arr - mu) / sigma).tolist()

    def _composite_score(self, ff: FrameFeatures, zscores: dict) -> float:
        signals = [
            abs(zscores["lip_ar"][ff.frame_idx]),
            abs(zscores["lap"][ff.frame_idx]),
            abs(zscores["fft"][ff.frame_idx]),
            abs(zscores["vel"][ff.frame_idx]),
            abs(zscores["tg"][ff.frame_idx]),
        ]
        # Weight: lip sync + landmark velocity highest (most discriminative)
        weights = [0.30, 0.15, 0.15, 0.25, 0.15]
        return float(np.dot(signals, weights))

    def detect(self, video_path: str) -> DetectionResult:
        t0 = time.time()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_features: List[FrameFeatures] = []

        # Reset stateful trackers
        self.landmark_tracker.prev_landmarks = None
        self.temporal_analyzer.prev_frame = None

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        ) as face_mesh, \
        self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.4
        ) as face_det:

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection for bbox
                det_results = face_det.process(rgb)
                face_bbox = None
                if det_results.detections:
                    face_bbox = self._get_face_bbox(det_results.detections, frame_w, frame_h)

                # Face mesh for landmarks
                mesh_results = face_mesh.process(rgb)
                landmarks = None
                if mesh_results.multi_face_landmarks:
                    landmarks = mesh_results.multi_face_landmarks[0]

                # Face crop
                face_crop = None
                if face_bbox:
                    x, y, w, h = face_bbox
                    if w > 0 and h > 0:
                        face_crop = frame[y:y+h, x:x+w]

                ff = FrameFeatures(frame_idx=frame_idx, timestamp=ts)

                # Extract all signals
                ff.lip_aspect_ratio, ff.lip_area = self.lip_extractor.extract(landmarks, frame_w, frame_h)
                ff.laplacian_var = self.texture_analyzer.extract(face_crop)
                ff.fft_peak_score = self.fft_analyzer.extract(face_crop)
                ff.landmark_velocity = self.landmark_tracker.extract(landmarks, frame_w, frame_h)
                ff.temporal_gradient = self.temporal_analyzer.extract(frame, face_bbox)

                raw_features.append(ff)
                frame_idx += 1

        cap.release()

        if len(raw_features) < 20:
            return DetectionResult(
                verdict="INCONCLUSIVE", overall_confidence=0.0,
                segments=[], frame_features=raw_features,
                per_signal_zscores={}, processing_time=time.time()-t0,
                total_frames=frame_idx, fps=fps,
                summary="Insufficient frames for analysis."
            )

        # Per-clip z-scoring (the core innovation)
        zscores = {
            "lip_ar": self._zscore_series([f.lip_aspect_ratio for f in raw_features]),
            "lap":    self._zscore_series([f.laplacian_var     for f in raw_features]),
            "fft":    self._zscore_series([f.fft_peak_score    for f in raw_features]),
            "vel":    self._zscore_series([f.landmark_velocity for f in raw_features]),
            "tg":     self._zscore_series([f.temporal_gradient for f in raw_features]),
        }

        # Composite score per frame
        for ff in raw_features:
            ff.composite_score = self._composite_score(ff, zscores)

        # Z-score the composite (second-order normalization)
        comp_scores = [f.composite_score for f in raw_features]
        comp_z = self._zscore_series(comp_scores)
        for i, ff in enumerate(raw_features):
            ff.composite_z = comp_z[i]

        # Peak detection on composite z-score differential
        comp_z_arr = np.array(comp_z)
        diff_signal = np.abs(np.diff(comp_z_arr, prepend=comp_z_arr[0]))
        peaks, props = find_peaks(diff_signal, height=1.8, distance=int(fps * 0.5))

        # Build anomaly segments around peaks
        segments: List[AnomalySegment] = []
        for peak_idx in peaks:
            # Window around peak
            w_start = max(0, peak_idx - int(fps * 0.5))
            w_end = min(len(raw_features) - 1, peak_idx + int(fps * 1.0))

            # Which signals fired?
            triggered = []
            window_frames = raw_features[w_start:w_end+1]
            peak_z = max(abs(comp_z[i]) for i in range(w_start, w_end+1))

            for f in window_frames:
                if abs(zscores["lip_ar"][f.frame_idx]) > self.THRESHOLDS["lip_aspect_ratio"]:
                    if "lip_sync_anomaly" not in triggered:
                        triggered.append("lip_sync_anomaly")
                if abs(zscores["lap"][f.frame_idx]) > self.THRESHOLDS["laplacian_var"]:
                    if "texture_smoothing" not in triggered:
                        triggered.append("texture_smoothing")
                if abs(zscores["fft"][f.frame_idx]) > self.THRESHOLDS["fft_peak_score"]:
                    if "gan_frequency_artifact" not in triggered:
                        triggered.append("gan_frequency_artifact")
                if abs(zscores["vel"][f.frame_idx]) > self.THRESHOLDS["landmark_velocity"]:
                    if "landmark_jitter" not in triggered:
                        triggered.append("landmark_jitter")
                if abs(zscores["tg"][f.frame_idx]) > self.THRESHOLDS["temporal_gradient"]:
                    if "temporal_gradient_anomaly" not in triggered:
                        triggered.append("temporal_gradient_anomaly")

            if not triggered:
                continue

            confidence = "high" if len(triggered) >= 3 else "medium" if len(triggered) == 2 else "low"
            t_start = raw_features[w_start].timestamp
            t_end = raw_features[w_end].timestamp

            desc_parts = []
            if "lip_sync_anomaly" in triggered:
                desc_parts.append(f"lip geometry deviation (z={abs(zscores['lip_ar'][peak_idx]):.2f})")
            if "landmark_jitter" in triggered:
                desc_parts.append(f"landmark velocity spike (z={abs(zscores['vel'][peak_idx]):.2f})")
            if "texture_smoothing" in triggered:
                desc_parts.append(f"GAN texture smoothing (z={abs(zscores['lap'][peak_idx]):.2f})")
            if "gan_frequency_artifact" in triggered:
                desc_parts.append(f"GAN frequency fingerprint (z={abs(zscores['fft'][peak_idx]):.2f})")
            if "temporal_gradient_anomaly" in triggered:
                desc_parts.append(f"temporal gradient anomaly (z={abs(zscores['tg'][peak_idx]):.2f})")

            segments.append(AnomalySegment(
                start_frame=w_start, end_frame=w_end,
                start_time=round(t_start, 3), end_time=round(t_end, 3),
                peak_z_score=round(peak_z, 3),
                triggered_signals=triggered,
                confidence=confidence,
                description=f"Anomaly at t={t_start:.2f}s-{t_end:.2f}s: " + "; ".join(desc_parts)
            ))

        # Verdict
        high_conf = [s for s in segments if s.confidence == "high"]
        med_conf  = [s for s in segments if s.confidence == "medium"]

        if high_conf:
            verdict = "MANIPULATED"
            overall_conf = min(0.95, 0.70 + 0.08 * len(high_conf))
        elif med_conf:
            verdict = "MANIPULATED"
            overall_conf = min(0.75, 0.50 + 0.08 * len(med_conf))
        elif segments:
            verdict = "INCONCLUSIVE"
            overall_conf = 0.40
        else:
            verdict = "AUTHENTIC"
            overall_conf = 0.85

        summary = (
            f"{verdict} — {len(segments)} anomaly segment(s) detected across "
            f"{frame_idx} frames ({frame_idx/fps:.1f}s). "
            f"Overall confidence: {overall_conf:.0%}."
        )

        per_signal_zscores = {
            "lip_aspect_ratio": [round(z, 3) for z in zscores["lip_ar"]],
            "laplacian_variance": [round(z, 3) for z in zscores["lap"]],
            "fft_peak_score": [round(z, 3) for z in zscores["fft"]],
            "landmark_velocity": [round(z, 3) for z in zscores["vel"]],
            "temporal_gradient": [round(z, 3) for z in zscores["tg"]],
            "composite": [round(z, 3) for z in comp_z],
        }

        return DetectionResult(
            verdict=verdict,
            overall_confidence=round(overall_conf, 3),
            segments=segments,
            frame_features=raw_features,
            per_signal_zscores=per_signal_zscores,
            processing_time=round(time.time() - t0, 2),
            total_frames=frame_idx,
            fps=fps,
            summary=summary
        )
