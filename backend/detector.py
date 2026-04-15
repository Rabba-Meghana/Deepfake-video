"""
FauxPix — Partial Video Deepfake Detector
==========================================
Signal-processing-first pipeline for segment-level video deepfake detection.
Detects WHERE in a video manipulation was injected, not just whether a clip is fake.

7 signals:
  1. Lip Geometry           — phoneme-viseme proxy (MediaPipe)
  2. Laplacian Variance     — GAN texture over-smoothing
  3. FFT Peak Score         — GAN frequency fingerprint
  4. Landmark Velocity      — Facial Feature Drift (FFD)
  5. Temporal Gradient      — face vs background motion ratio
  6. Phoneme-Viseme Sync    — Groq Whisper word timestamps x lip geometry
  7. Periodicity / Splice   — cross-frame rolling window: GAN rhythm + splice boundary

Key design:
  - ALL signals z-scored against the clip's OWN baseline (like audio detector)
  - Signal 7 compares FRAMES AGAINST EACH OTHER via sliding windows — not just
    vs the global mean. This catches GAN period (~12-frame repeating artifacts)
    and hard splice boundaries where signal statistics shift abruptly.
  - Minimum 2 corroborating signals required for MANIPULATED verdict.
    Phoneme-viseme mismatch or splice_boundary alone are high-specificity exceptions.

Author: Meghana Rabba — Illinois Institute of Technology / BLK-BX Research
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import subprocess, tempfile, os, time, logging

logger = logging.getLogger(__name__)


VISEME_LAR_EXPECTED = {
    "bilabial_stop":  (0.00, 0.06),
    "bilabial_nasal": (0.00, 0.06),
    "labiodental":    (0.00, 0.08),
    "open_vowel":     (0.15, 0.55),
    "mid_vowel":      (0.08, 0.30),
    "close_vowel":    (0.02, 0.12),
    "sibilant":       (0.04, 0.18),
    "other":          (0.02, 0.25),
}

def phoneme_to_viseme(word: str) -> str:
    first = word.lower().strip(".,!?'\"")[:1]
    if first in ("b", "p"):   return "bilabial_stop"
    if first == "m":          return "bilabial_nasal"
    if first in ("f", "v"):   return "labiodental"
    if first == "a":          return "open_vowel"
    if first == "e":          return "mid_vowel"
    if first in ("i","o","u"):return "close_vowel"
    if first in ("s","z"):    return "sibilant"
    return "other"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    viseme_group: str
    expected_lar_min: float
    expected_lar_max: float


@dataclass
class FrameFeatures:
    frame_idx: int
    timestamp: float
    lip_aspect_ratio: float = 0.0
    lip_area: float = 0.0
    laplacian_var: float = 0.0
    fft_peak_score: float = 0.0
    landmark_velocity: float = 0.0
    temporal_gradient: float = 0.0
    phoneme_viseme_mismatch: float = 0.0
    periodicity_score: float = 0.0
    splice_score: float = 0.0
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
    confidence: str
    description: str


@dataclass
class PhonemeVisemeReport:
    transcript: str
    word_count: int
    mismatch_count: int
    mismatch_rate: float
    mismatches: List[Dict]
    groq_used: bool
    error: Optional[str] = None


@dataclass
class DetectionResult:
    verdict: str
    overall_confidence: float
    segments: List[AnomalySegment]
    frame_features: List[FrameFeatures]
    per_signal_zscores: dict
    phoneme_viseme_report: Optional[PhonemeVisemeReport]
    processing_time: float
    total_frames: int
    fps: float
    summary: str
    groq_transcript: Optional[str] = None


class LipGeometryExtractor:
    def extract(self, landmarks, frame_w: int, frame_h: int) -> Tuple[float, float]:
        if landmarks is None:
            return 0.0, 0.0
        lm         = landmarks.landmark
        vertical   = abs(lm[13].y - lm[14].y) * frame_h
        horizontal = abs(lm[61].x - lm[291].x) * frame_w
        return float(vertical / (horizontal + 1e-6)), float(vertical * horizontal)


class TextureAnalyzer:
    def extract(self, face_crop: np.ndarray) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class GANFrequencyAnalyzer:
    def extract(self, face_crop: np.ndarray) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        if gray.shape[0] < 32 or gray.shape[1] < 32:
            return 0.0
        resized = cv2.resize(gray, (128, 128)).astype(np.float32)
        mag     = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(resized))))
        h, w    = mag.shape
        mag[h//2-4:h//2+4, w//2-4:w//2+4] = 0
        return float(mag.max() / (mag.mean() + 1e-6))


class LandmarkVelocityTracker:
    KEY_POINTS = [1, 4, 33, 263, 61, 291, 199]
    def __init__(self):
        self.prev = None
    def extract(self, landmarks, frame_w: int, frame_h: int) -> float:
        if landmarks is None:
            self.prev = None
            return 0.0
        curr = np.array([[landmarks.landmark[i].x * frame_w,
                          landmarks.landmark[i].y * frame_h]
                         for i in self.KEY_POINTS])
        if self.prev is None:
            self.prev = curr
            return 0.0
        vel = float(np.mean(np.linalg.norm(curr - self.prev, axis=1)))
        self.prev = curr
        return vel


class TemporalGradientAnalyzer:
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
        return float(face_diff.mean() / (diff[mask].mean() + 1e-6))


class PeriodicityAndSpliceAnalyzer:
    """
    Signal 7 — Cross-frame sliding window: frames compared AGAINST EACH OTHER.

    A) PERIODICITY: GAN generators produce artifacts at a fixed frame interval
       (typically 8-16 frames). Detected via autocorrelation of a multi-signal
       feature vector. Real video autocorrelation decays to noise; GAN regions
       show a persistent peak at a fixed lag.

    B) SPLICE: When a deepfake is spliced into real footage, the statistics of
       multiple signals shift abruptly at the boundary frame. Detected by
       comparing a left window vs right window — high shift = splice boundary.
    """
    SPLICE_HALF_WINDOW = 15
    PERIOD_LAGS = list(range(4, 25))

    def compute_periodicity(self, feature_matrix: np.ndarray, fps: float) -> np.ndarray:
        N = feature_matrix.shape[0]
        scores = np.zeros(N)
        normed = np.zeros_like(feature_matrix)
        for col in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, col]
            mu, sig = col_data.mean(), col_data.std()
            normed[:, col] = (col_data - mu) / (sig + 1e-8)

        context = 60
        for i in range(N):
            lo = max(0, i - context)
            hi = min(N, i + context)
            window = normed[lo:hi]
            if len(window) < max(self.PERIOD_LAGS) + 4:
                continue
            ac_vals = []
            for lag in self.PERIOD_LAGS:
                if lag >= len(window):
                    continue
                a = window[:-lag]
                b = window[lag:]
                col_corrs = []
                for col in range(window.shape[1]):
                    if a[:, col].std() < 1e-8 or b[:, col].std() < 1e-8:
                        continue
                    r = np.corrcoef(a[:, col], b[:, col])[0, 1]
                    col_corrs.append(abs(r))
                if col_corrs:
                    ac_vals.append(np.mean(col_corrs))
            if ac_vals:
                ac_arr = np.array(ac_vals)
                scores[i] = float(ac_arr.max() - ac_arr.mean())
        return scores

    def compute_splice(self, feature_matrix: np.ndarray) -> np.ndarray:
        N = feature_matrix.shape[0]
        scores = np.zeros(N)
        W = self.SPLICE_HALF_WINDOW
        normed = np.zeros_like(feature_matrix)
        for col in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, col]
            mu, sig = col_data.mean(), col_data.std()
            normed[:, col] = (col_data - mu) / (sig + 1e-8)

        for i in range(W, N - W):
            left  = normed[i-W:i]
            right = normed[i:i+W]
            col_scores = []
            for col in range(normed.shape[1]):
                mu_l, mu_r = left[:, col].mean(), right[:, col].mean()
                std_l, std_r = left[:, col].std(), right[:, col].std()
                pooled = np.sqrt((std_l**2 + std_r**2) / 2 + 1e-8)
                col_scores.append(abs(mu_l - mu_r) / pooled)
            scores[i] = float(np.mean(col_scores))
        return scores


class PhonemeVisemeMatcher:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
        return self._client

    def extract_audio(self, video_path: str) -> Optional[str]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp.name],
                capture_output=True, timeout=60
            )
            if r.returncode == 0 and os.path.getsize(tmp.name) > 1000:
                return tmp.name
        except Exception as e:
            logger.warning(f"Audio extraction error: {e}")
        return None

    def transcribe(self, audio_path: str):
        client = self._get_client()
        if client is None:
            return None
        try:
            with open(audio_path, "rb") as f:
                return client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
        except Exception as e:
            logger.warning(f"Groq transcription failed: {e}")
            return None

    def build_word_timestamps(self, transcription) -> List[WordTimestamp]:
        words = []
        try:
            raw = []
            if hasattr(transcription, "words") and transcription.words:
                raw = transcription.words
            elif hasattr(transcription, "segments"):
                for seg in transcription.segments:
                    if hasattr(seg, "words") and seg.words:
                        raw.extend(seg.words)
            for w in raw:
                word  = getattr(w, "word", "").strip()
                start = float(getattr(w, "start", 0))
                end   = float(getattr(w, "end", start + 0.2))
                vis   = phoneme_to_viseme(word)
                exp   = VISEME_LAR_EXPECTED[vis]
                words.append(WordTimestamp(word=word, start=start, end=end,
                    viseme_group=vis, expected_lar_min=exp[0], expected_lar_max=exp[1]))
        except Exception as e:
            logger.warning(f"Word timestamp parse error: {e}")
        return words

    def score_frames(self, frame_features, word_timestamps, fps):
        scores     = [0.0] * len(frame_features)
        mismatches = []
        if not word_timestamps:
            return scores, PhonemeVisemeReport(
                transcript="", word_count=0, mismatch_count=0,
                mismatch_rate=0.0, mismatches=[], groq_used=False,
                error="No word timestamps"
            ), ""

        frame_word: Dict[int, WordTimestamp] = {}
        for wt in word_timestamps:
            for fi in range(int(wt.start * fps), min(int(wt.end * fps) + 1, len(frame_features))):
                frame_word[fi] = wt

        mismatch_count = 0
        checked_count  = 0
        for ff in frame_features:
            fi = ff.frame_idx
            if fi not in frame_word:
                continue
            wt = frame_word[fi]
            if wt.viseme_group == "other":
                continue
            checked_count += 1
            lar = ff.lip_aspect_ratio
            is_mismatch = (lar < wt.expected_lar_min - 0.02 or
                           lar > wt.expected_lar_max + 0.02)
            if is_mismatch:
                mismatch_count += 1
                dist = ((wt.expected_lar_min - lar) / (wt.expected_lar_min + 1e-6)
                        if lar < wt.expected_lar_min
                        else (lar - wt.expected_lar_max) / (wt.expected_lar_max + 1e-6))
                scores[fi] = min(1.0, dist)
                mismatches.append({
                    "word": wt.word,
                    "time": round(ff.timestamp, 3),
                    "viseme_group": wt.viseme_group,
                    "actual_lar": round(lar, 4),
                    "expected_range": [wt.expected_lar_min, wt.expected_lar_max],
                    "frame": fi,
                })

        transcript = " ".join(w.word for w in word_timestamps)
        report = PhonemeVisemeReport(
            transcript=transcript,
            word_count=len(word_timestamps),
            mismatch_count=mismatch_count,
            mismatch_rate=round(mismatch_count / max(1, checked_count), 4),
            mismatches=mismatches[:30],
            groq_used=True,
        )
        return scores, report, transcript


class FauxPixDetector:
    """
    7-signal segment-level partial video deepfake detector.

    Signals 1-6: per-frame z-scored against clip baseline.
    Signal 7:    cross-frame sliding window — compares frames AGAINST EACH OTHER
                 to find GAN periodicity (rhythm) and splice boundaries (hard cuts).

    Verdict:
      MANIPULATED   — 2+ signals corroborate, OR high-specificity signal alone
                      (phoneme-viseme mismatch, splice_boundary)
      INCONCLUSIVE  — 1 low-specificity signal only
      AUTHENTIC     — no signals triggered
    """

    THRESHOLDS = {
        "lip_aspect_ratio":        3.2,
        "laplacian_var":           3.5,
        "fft_peak_score":          3.5,
        "landmark_velocity":       3.8,
        "temporal_gradient":       3.5,
        "phoneme_viseme_mismatch": 1.8,
        "periodicity":             0.25,
        "splice":                  2.5,
    }

    WEIGHTS = {
        "lip_ar": 0.15,
        "lap":    0.10,
        "fft":    0.10,
        "vel":    0.15,
        "tg":     0.10,
        "pv":     0.20,
        "period": 0.10,
        "splice": 0.10,
    }

    def __init__(self, groq_api_key: Optional[str] = None):
        self.mp_face_mesh         = mp.solutions.face_mesh
        self.mp_face_detection    = mp.solutions.face_detection
        self.lip_extractor        = LipGeometryExtractor()
        self.texture_analyzer     = TextureAnalyzer()
        self.fft_analyzer         = GANFrequencyAnalyzer()
        self.landmark_tracker     = LandmarkVelocityTracker()
        self.temporal_analyzer    = TemporalGradientAnalyzer()
        self.pv_matcher           = PhonemeVisemeMatcher(groq_api_key)
        self.periodicity_analyzer = PeriodicityAndSpliceAnalyzer()

    def _get_face_bbox(self, detections, fw, fh):
        if not detections:
            return None
        bb = detections[0].location_data.relative_bounding_box
        x, y = max(0, int(bb.xmin*fw)), max(0, int(bb.ymin*fh))
        w, h = min(int(bb.width*fw), fw-x), min(int(bb.height*fh), fh-y)
        return (x, y, w, h)

    def _zscore_series(self, values: List[float]) -> List[float]:
        arr = np.array(values, dtype=np.float64)
        mu  = arr.mean()
        sig = arr.std()
        if sig < 1e-8:
            sig = 1e-8
        return ((arr - mu) / sig).tolist()

    def _composite(self, i: int, zs: dict, has_groq: bool) -> float:
        w = self.WEIGHTS
        score = (
            abs(zs["lip_ar"][i]) * w["lip_ar"] +
            abs(zs["lap"][i])    * w["lap"]    +
            abs(zs["fft"][i])    * w["fft"]    +
            abs(zs["vel"][i])    * w["vel"]    +
            abs(zs["tg"][i])     * w["tg"]     +
            abs(zs["period"][i]) * w["period"] +
            abs(zs["splice"][i]) * w["splice"]
        )
        if has_groq:
            score += abs(zs["pv"][i]) * w["pv"]
        else:
            s = 1.0 - w["pv"]
            score /= s
        return score

    def detect(self, video_path: str) -> DetectionResult:
        t0  = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        feats: List[FrameFeatures] = []

        self.landmark_tracker.prev        = None
        self.temporal_analyzer.prev_frame = None

        # ── Pass 1: visual signals 1-5 ────────────────────────────────────────
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.4, min_tracking_confidence=0.4
        ) as mesh, self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.4
        ) as det:
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dr   = det.process(rgb)
                bbox = self._get_face_bbox(dr.detections if dr.detections else [], fw, fh)
                mr   = mesh.process(rgb)
                lm   = mr.multi_face_landmarks[0] if mr.multi_face_landmarks else None
                crop = (frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        if bbox and bbox[2]>0 and bbox[3]>0 else None)
                ff   = FrameFeatures(frame_idx=fi, timestamp=fi/fps)
                ff.lip_aspect_ratio, ff.lip_area = self.lip_extractor.extract(lm, fw, fh)
                ff.laplacian_var     = self.texture_analyzer.extract(crop)
                ff.fft_peak_score    = self.fft_analyzer.extract(crop)
                ff.landmark_velocity = self.landmark_tracker.extract(lm, fw, fh)
                ff.temporal_gradient = self.temporal_analyzer.extract(frame, bbox)
                feats.append(ff)
                fi += 1
        cap.release()

        if len(feats) < 20:
            return DetectionResult(
                verdict="INCONCLUSIVE", overall_confidence=0.0, segments=[],
                frame_features=feats, per_signal_zscores={},
                phoneme_viseme_report=None, processing_time=round(time.time()-t0,2),
                total_frames=fi, fps=fps, summary="Insufficient frames."
            )

        # ── Signal 7: cross-frame periodicity + splice ────────────────────────
        feature_matrix = np.array([
            [f.laplacian_var, f.fft_peak_score, f.landmark_velocity, f.temporal_gradient]
            for f in feats
        ], dtype=np.float64)

        period_scores = self.periodicity_analyzer.compute_periodicity(feature_matrix, fps)
        splice_scores = self.periodicity_analyzer.compute_splice(feature_matrix)

        for i, ff in enumerate(feats):
            ff.periodicity_score = float(period_scores[i])
            ff.splice_score      = float(splice_scores[i])

        # ── Pass 2: Groq Whisper Signal 6 ────────────────────────────────────
        pv_scores  = [0.0] * len(feats)
        pv_report  = None
        transcript = None
        has_groq   = False

        if self.pv_matcher.groq_api_key:
            audio = self.pv_matcher.extract_audio(video_path)
            if audio:
                try:
                    tx = self.pv_matcher.transcribe(audio)
                    if tx:
                        word_ts = self.pv_matcher.build_word_timestamps(tx)
                        pv_scores, pv_report, transcript = self.pv_matcher.score_frames(feats, word_ts, fps)
                        has_groq = True
                        logger.info(f"Groq: {len(word_ts)} words, "
                                    f"{pv_report.mismatch_count} mismatches ({pv_report.mismatch_rate:.1%})")
                finally:
                    try: os.unlink(audio)
                    except: pass

        for i, ff in enumerate(feats):
            ff.phoneme_viseme_mismatch = pv_scores[i]

        # ── Z-scoring all 7 signals ───────────────────────────────────────────
        zs = {
            "lip_ar": self._zscore_series([f.lip_aspect_ratio       for f in feats]),
            "lap":    self._zscore_series([f.laplacian_var           for f in feats]),
            "fft":    self._zscore_series([f.fft_peak_score          for f in feats]),
            "vel":    self._zscore_series([f.landmark_velocity       for f in feats]),
            "tg":     self._zscore_series([f.temporal_gradient       for f in feats]),
            "pv":     self._zscore_series([f.phoneme_viseme_mismatch for f in feats]),
            "period": self._zscore_series([f.periodicity_score       for f in feats]),
            "splice": self._zscore_series([f.splice_score            for f in feats]),
        }

        for i, ff in enumerate(feats):
            ff.composite_score = self._composite(i, zs, has_groq)
        comp_z = self._zscore_series([f.composite_score for f in feats])
        for i, ff in enumerate(feats):
            ff.composite_z = comp_z[i]

        # ── Peak detection ────────────────────────────────────────────────────
        diff_sig   = np.abs(np.diff(np.array(comp_z), prepend=comp_z[0]))
        diff_peaks, _ = find_peaks(diff_sig,          height=2.8, distance=int(fps*0.8))
        comp_peaks, _ = find_peaks(np.array(comp_z),  height=2.5, distance=int(fps*0.8))
        all_peaks  = sorted(set(diff_peaks.tolist() + comp_peaks.tolist()))

        # Merge peaks within 1s of each other, keep highest
        merged_peaks = []
        for p in all_peaks:
            if not merged_peaks or p - merged_peaks[-1] > int(fps):
                merged_peaks.append(p)
            elif abs(comp_z[p]) > abs(comp_z[merged_peaks[-1]]):
                merged_peaks[-1] = p

        # ── Anomaly segments ──────────────────────────────────────────────────
        segments: List[AnomalySegment] = []
        thr = self.THRESHOLDS

        for pi in merged_peaks:
            ws  = max(0, pi - int(fps*0.5))
            we  = min(len(feats)-1, pi + int(fps*1.0))
            pkz = max(abs(comp_z[i]) for i in range(ws, we+1))

            triggered = []

            for f in feats[ws:we+1]:
                idx = f.frame_idx
                def _flag(key, zkey, name):
                    if abs(zs[zkey][idx]) > thr[key] and name not in triggered:
                        triggered.append(name)
                _flag("lip_aspect_ratio",        "lip_ar", "lip_sync_anomaly")
                _flag("laplacian_var",            "lap",    "texture_smoothing")
                _flag("fft_peak_score",           "fft",    "gan_frequency_artifact")
                _flag("landmark_velocity",        "vel",    "landmark_jitter")
                _flag("temporal_gradient",        "tg",     "temporal_gradient_anomaly")
                if has_groq:
                    _flag("phoneme_viseme_mismatch", "pv",  "phoneme_viseme_mismatch")

            # Signal 7 — raw scores (not z-scores, already calibrated)
            win_period = float(np.mean([feats[i].periodicity_score for i in range(ws, we+1)]))
            win_splice = float(np.mean([feats[i].splice_score      for i in range(ws, we+1)]))
            if win_period > thr["periodicity"]:
                triggered.append("gan_periodicity")
            if win_splice > thr["splice"]:
                triggered.append("splice_boundary")

            if not triggered:
                continue

            # High-specificity signals that count alone
            high_spec = {"phoneme_viseme_mismatch", "splice_boundary"}
            n_high = len([s for s in triggered if s in high_spec])
            if n_high == 0 and len(triggered) < 2:
                continue  # single low-spec signal — inconclusive, skip

            conf  = "high" if len(triggered)>=3 else "medium" if len(triggered)==2 else "low"
            ts_   = feats[ws].timestamp
            te_   = feats[we].timestamp
            parts = []
            if "phoneme_viseme_mismatch" in triggered:
                parts.append(f"phoneme-viseme mismatch (z={abs(zs['pv'][pi]):.2f}) — Groq Whisper")
            if "lip_sync_anomaly"          in triggered:
                parts.append(f"lip geometry deviation (z={abs(zs['lip_ar'][pi]):.2f})")
            if "landmark_jitter"           in triggered:
                parts.append(f"landmark velocity spike (z={abs(zs['vel'][pi]):.2f})")
            if "texture_smoothing"         in triggered:
                parts.append(f"GAN texture smoothing (z={abs(zs['lap'][pi]):.2f})")
            if "gan_frequency_artifact"    in triggered:
                parts.append(f"GAN frequency fingerprint (z={abs(zs['fft'][pi]):.2f})")
            if "temporal_gradient_anomaly" in triggered:
                parts.append(f"temporal gradient anomaly (z={abs(zs['tg'][pi]):.2f})")
            if "gan_periodicity"           in triggered:
                parts.append(f"GAN frame periodicity (autocorr={win_period:.3f})")
            if "splice_boundary"           in triggered:
                parts.append(f"splice boundary detected (shift={win_splice:.2f})")

            segments.append(AnomalySegment(
                start_frame=ws, end_frame=we,
                start_time=round(ts_,3), end_time=round(te_,3),
                peak_z_score=round(pkz,3), triggered_signals=triggered,
                confidence=conf,
                description=f"Anomaly t={ts_:.2f}s–{te_:.2f}s: " + "; ".join(parts)
            ))

        # ── Verdict ───────────────────────────────────────────────────────────
        hi  = [s for s in segments if s.confidence=="high"]
        med = [s for s in segments if s.confidence=="medium"]
        pv_fired = any("phoneme_viseme_mismatch" in s.triggered_signals for s in segments)

        if hi:
            verdict = "MANIPULATED"
            conf_   = min(0.97 if pv_fired else 0.95, 0.70+0.08*len(hi)+(0.05 if pv_fired else 0))
        elif med:
            verdict = "MANIPULATED"
            conf_   = min(0.80 if pv_fired else 0.75, 0.50+0.08*len(med)+(0.05 if pv_fired else 0))
        elif segments:
            verdict, conf_ = "INCONCLUSIVE", 0.40
        else:
            verdict, conf_ = "AUTHENTIC", (0.87 if has_groq else 0.82)

        groq_note = (" (Groq Whisper phoneme-viseme: ACTIVE)" if has_groq
                     else " (Groq Whisper: not configured — set GROQ_API_KEY)")
        summary = (f"{verdict} — {len(segments)} anomaly segment(s) across "
                   f"{fi} frames ({fi/fps:.1f}s). Confidence: {conf_:.0%}.{groq_note}")

        return DetectionResult(
            verdict=verdict, overall_confidence=round(conf_,3),
            segments=segments, frame_features=feats,
            per_signal_zscores={
                "lip_aspect_ratio":        [round(z,3) for z in zs["lip_ar"]],
                "laplacian_variance":      [round(z,3) for z in zs["lap"]],
                "fft_peak_score":          [round(z,3) for z in zs["fft"]],
                "landmark_velocity":       [round(z,3) for z in zs["vel"]],
                "temporal_gradient":       [round(z,3) for z in zs["tg"]],
                "phoneme_viseme_mismatch": [round(z,3) for z in zs["pv"]],
                "periodicity":             [round(z,3) for z in zs["period"]],
                "splice":                  [round(z,3) for z in zs["splice"]],
                "composite":               [round(z,3) for z in comp_z],
            },
            phoneme_viseme_report=pv_report,
            processing_time=round(time.time()-t0,2),
            total_frames=fi, fps=fps, summary=summary,
            groq_transcript=transcript,
        )
