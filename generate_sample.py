"""
Generate a realistic sample_output.json by running FauxPix on a
synthetically constructed test video — one with a simulated 'splice'
at t=4.0s where we inject artificial texture/motion artifacts.
This demonstrates real detection on real signal data.
"""

import sys
sys.path.insert(0, "/home/claude/fauxpix/backend")

import cv2
import numpy as np
import json
import os
from detector import FauxPixDetector

def make_test_video(path: str, fps: int = 25, duration: float = 8.0):
    """
    Create a test video that simulates a partial deepfake splice:
    - Frames 0-3.8s: normal face region with natural texture/motion
    - Frames 3.8-5.2s: simulated synthetic insertion (over-smoothed, GAN-like patterns)
    - Frames 5.2-8.0s: return to normal
    """
    w, h = 480, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    total_frames = int(fps * duration)
    splice_start = int(fps * 3.8)
    splice_end   = int(fps * 5.2)

    rng = np.random.default_rng(42)

    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background — natural texture
        bg = rng.integers(60, 90, (h, w, 3), dtype=np.uint8)
        frame[:] = bg

        # Face oval
        face_cx, face_cy = w // 2, h // 2
        face_rx, face_ry = 90, 110

        in_splice = splice_start <= i < splice_end

        if in_splice:
            # Simulated GAN face: over-smooth, slightly uniform, periodic FFT artifacts
            face_color = np.array([200, 175, 160], dtype=np.float32)
            face_region = np.ones((h, w, 3), dtype=np.float32) * face_color
            # Add very subtle noise (GAN over-smoothing)
            face_region += rng.normal(0, 2, (h, w, 3))
            # Inject periodic pattern (GAN upsampling artifact)
            for row in range(0, h, 16):
                face_region[row, :, :] += 4
            face_region = np.clip(face_region, 0, 255).astype(np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360, 255, -1)
            frame[mask > 0] = face_region[mask > 0]
        else:
            # Natural face: organic texture, pores, subtle noise
            face_base = np.array([195, 160, 145], dtype=np.float32)
            face_region = np.ones((h, w, 3), dtype=np.float32) * face_base
            face_region += rng.normal(0, 8, (h, w, 3))  # natural micro-texture
            face_region = np.clip(face_region, 0, 255).astype(np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360, 255, -1)
            frame[mask > 0] = face_region[mask > 0]

        # Eyes
        t = i / fps
        blink = np.exp(-((t % 4 - 2) ** 2) / 0.02)
        eye_h = max(2, int(10 * (1 - blink)))
        # In splice: slightly asymmetric eye timing (deepfake artifact)
        if in_splice:
            eye_h_r = max(2, int(10 * (1 - blink * 0.8)))
        else:
            eye_h_r = eye_h
        cv2.ellipse(frame, (face_cx - 28, face_cy - 20), (18, eye_h), 0, 0, 360, (50, 40, 35), -1)
        cv2.ellipse(frame, (face_cx + 28, face_cy - 20), (18, eye_h_r), 0, 0, 360, (50, 40, 35), -1)

        # Mouth — simulate lip motion; in splice, over-smooth/wrong shape
        mouth_y = face_cy + 35
        if in_splice:
            # Unnatural lip shape during splice
            mouth_open = int(4 + 2 * np.sin(2 * np.pi * 2.5 * t + 1.5))
        else:
            mouth_open = int(4 + 6 * abs(np.sin(np.pi * 2.2 * t)))
        cv2.ellipse(frame, (face_cx, mouth_y), (22, max(2, mouth_open)), 0, 0, 360, (90, 60, 60), -1)

        # Nose
        cv2.circle(frame, (face_cx, face_cy + 5), 6, (170, 140, 130), -1)

        out.write(frame)

    out.release()
    print(f"Test video written: {path} ({total_frames} frames, {duration}s)")


if __name__ == "__main__":
    os.makedirs("/home/claude/fauxpix/sample_output", exist_ok=True)
    video_path = "/home/claude/fauxpix/sample_output/test_partial_deepfake.mp4"

    print("Creating synthetic test video with splice at t=3.8s-5.2s ...")
    make_test_video(video_path, fps=25, duration=8.0)

    print("Running FauxPix detector ...")
    det = FauxPixDetector()
    result = det.detect(video_path)

    out = {
        "test_video": "test_partial_deepfake.mp4",
        "injected_splice": {"start": 3.8, "end": 5.2, "description": "Simulated GAN texture + unnatural lip motion"},
        "verdict": result.verdict,
        "overall_confidence": result.overall_confidence,
        "summary": result.summary,
        "processing_time_sec": result.processing_time,
        "video_info": {
            "total_frames": result.total_frames,
            "fps": result.fps,
            "duration_sec": round(result.total_frames / result.fps, 2) if result.fps else 0,
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
            for s in result.segments
        ],
        "per_signal_zscores_sample": {
            k: v[:50] for k, v in result.per_signal_zscores.items()
        },
    }

    out_path = "/home/claude/fauxpix/sample_output/sample_output.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSample output written: {out_path}")
    print(f"\nVerdict: {result.verdict}")
    print(f"Confidence: {result.overall_confidence:.0%}")
    print(f"Summary: {result.summary}")
    print(f"Anomaly segments detected: {len(result.segments)}")
    for s in result.segments:
        print(f"  [{s.confidence.upper()}] t={s.start_time}s-{s.end_time}s | signals: {s.triggered_signals} | z={s.peak_z_score}")
