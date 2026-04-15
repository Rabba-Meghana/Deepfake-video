import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const SIGNAL_META = {
  lip_aspect_ratio:    { label: "Lip Geometry",        color: "#7c3aed", audio: "F0 jitter z-score" },
  laplacian_variance:  { label: "Texture Energy",      color: "#2563eb", audio: "hf_ratio_z" },
  fft_peak_score:      { label: "GAN Fingerprint",     color: "#059669", audio: "MFCC shift" },
  landmark_velocity:   { label: "Landmark Jitter",     color: "#d97706", audio: "F0 jitter" },
  temporal_gradient:   { label: "Temporal Gradient",   color: "#dc2626", audio: "Spectral flatness z" },
  composite:           { label: "Composite Score",     color: "#0d0d1a", audio: "Combined z-score" },
};

function Sparkline({ values, color, threshold = 2.0 }) {
  if (!values || values.length === 0) return null;
  const W = 300, H = 48;
  const max = Math.max(...values.map(Math.abs), threshold + 0.5);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - ((v + max) / (2 * max)) * H;
    return `${x},${y}`;
  }).join(" ");
  const threshY = H - ((threshold + max) / (2 * max)) * H;
  const negThreshY = H - ((-threshold + max) / (2 * max)) * H;

  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      <line x1={0} y1={threshY} x2={W} y2={threshY} stroke="#ef4444" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.5} />
      <line x1={0} y1={negThreshY} x2={W} y2={negThreshY} stroke="#ef4444" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.5} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} opacity={0.85} />
      {values.map((v, i) =>
        Math.abs(v) > threshold ? (
          <circle
            key={i}
            cx={(i / (values.length - 1)) * W}
            cy={H - ((v + max) / (2 * max)) * H}
            r={3}
            fill="#ef4444"
            opacity={0.9}
          />
        ) : null
      )}
    </svg>
  );
}

function SegmentBadge({ seg }) {
  const bg = seg.confidence === "high" ? "#fef2f2" : seg.confidence === "medium" ? "#fffbeb" : "#f0fdf4";
  const border = seg.confidence === "high" ? "#fca5a5" : seg.confidence === "medium" ? "#fcd34d" : "#86efac";
  const textColor = seg.confidence === "high" ? "#991b1b" : seg.confidence === "medium" ? "#92400e" : "#166534";
  return (
    <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: "10px 14px", marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: textColor }}>
          t={seg.start_time}s – {seg.end_time}s
        </span>
        <span style={{ fontSize: 11, fontWeight: 700, color: textColor, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {seg.confidence} • z={seg.peak_z_score}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 4 }}>
        {seg.triggered_signals.map(s => (
          <span key={s} style={{ fontSize: 10, background: "#1e1b4b", color: "#c4b5fd", borderRadius: 4, padding: "2px 6px" }}>
            {s.replace(/_/g, " ")}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 11, color: "#6b7280" }}>{seg.description}</div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/detect`, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Detection failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const verdictColor = result?.verdict === "MANIPULATED" ? "#991b1b"
    : result?.verdict === "AUTHENTIC" ? "#166534" : "#92400e";
  const verdictBg = result?.verdict === "MANIPULATED" ? "#fef2f2"
    : result?.verdict === "AUTHENTIC" ? "#f0fdf4" : "#fffbeb";

  return (
    <div style={{ minHeight: "100vh", background: "#f8f7ff", fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ background: "#0d0d1a", color: "white", padding: "16px 32px", display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#7c3aed", boxShadow: "0 0 8px #7c3aed" }} />
        <span style={{ fontWeight: 800, fontSize: 18, letterSpacing: "0.04em" }}>FauxPix</span>
        <span style={{ color: "#9ca3af", fontSize: 12, marginLeft: 4 }}>Partial Video Deepfake Detector</span>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "#6b7280" }}>
          5-signal · segment-level · per-clip z-scoring
        </span>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px" }}>

        {/* Upload */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current.click()}
          style={{
            border: `2px dashed ${dragging ? "#7c3aed" : file ? "#7c3aed" : "#d1d5db"}`,
            borderRadius: 12, padding: "32px 24px", textAlign: "center",
            background: dragging ? "#ede9fe" : "#fff", cursor: "pointer",
            transition: "all 0.15s", marginBottom: 20,
          }}
        >
          <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }}
            onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
          <div style={{ fontSize: 32, marginBottom: 8 }}>🎬</div>
          {file ? (
            <div>
              <div style={{ fontWeight: 700, color: "#7c3aed" }}>{file.name}</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>{(file.size / 1024 / 1024).toFixed(1)} MB</div>
            </div>
          ) : (
            <div>
              <div style={{ fontWeight: 600, color: "#374151" }}>Drop video here or click to upload</div>
              <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>MP4, MOV, AVI, MKV, WebM</div>
            </div>
          )}
        </div>

        <button
          onClick={analyze}
          disabled={!file || loading}
          style={{
            width: "100%", padding: "14px", borderRadius: 10, border: "none",
            background: !file || loading ? "#e5e7eb" : "#0d0d1a",
            color: !file || loading ? "#9ca3af" : "white",
            fontWeight: 700, fontSize: 15, cursor: !file || loading ? "not-allowed" : "pointer",
            marginBottom: 24, transition: "all 0.15s",
          }}
        >
          {loading ? "⏳ Analyzing…" : "🔍 Run FauxPix Detection"}
        </button>

        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 8, padding: 14, color: "#991b1b", marginBottom: 20 }}>
            {error}
          </div>
        )}

        {result && (
          <>
            {/* Verdict */}
            <div style={{ background: verdictBg, border: `2px solid ${verdictColor}`, borderRadius: 12, padding: "20px 24px", marginBottom: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                <span style={{ fontSize: 28 }}>
                  {result.verdict === "MANIPULATED" ? "🚨" : result.verdict === "AUTHENTIC" ? "✅" : "⚠️"}
                </span>
                <div>
                  <div style={{ fontWeight: 800, fontSize: 22, color: verdictColor }}>{result.verdict}</div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    Confidence: {(result.overall_confidence * 100).toFixed(0)}% · {result.processing_time_sec}s · {result.video_info.total_frames} frames @ {result.video_info.fps.toFixed(1)}fps
                  </div>
                </div>
              </div>
              <div style={{ fontSize: 13, color: "#374151" }}>{result.summary}</div>
            </div>

            {/* Segments */}
            {result.anomaly_segments.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: "#0d0d1a", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Anomaly Segments ({result.anomaly_segments.length})
                </div>
                {result.anomaly_segments.map((s, i) => <SegmentBadge key={i} seg={s} />)}
              </div>
            )}

            {/* Signal traces */}
            <div style={{ marginBottom: 16, fontWeight: 700, fontSize: 14, color: "#0d0d1a", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Signal Traces (per-clip z-scores)
            </div>
            <div style={{ display: "grid", gap: 12 }}>
              {Object.entries(SIGNAL_META).map(([key, meta]) => {
                const vals = result.per_signal_zscores[key];
                if (!vals) return null;
                return (
                  <div key={key} style={{ background: "#fff", borderRadius: 10, padding: "12px 16px", border: "1px solid #e5e7eb" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 13, color: meta.color }}>{meta.label}</span>
                        <span style={{ fontSize: 11, color: "#9ca3af", marginLeft: 8 }}>audio analogue: {meta.audio}</span>
                      </div>
                      <span style={{ fontSize: 11, color: "#6b7280" }}>
                        max |z| = {Math.max(...vals.map(Math.abs)).toFixed(2)}
                      </span>
                    </div>
                    <Sparkline values={vals} color={meta.color} threshold={key === "landmark_velocity" ? 2.2 : 2.0} />
                  </div>
                );
              })}
            </div>

            {/* Architecture note */}
            <div style={{ marginTop: 24, background: "#ede9fe", borderRadius: 10, padding: "12px 16px", fontSize: 12, color: "#4c1d95" }}>
              <strong>Architecture note:</strong> All signals z-scored against this clip's own baseline (first 30% of frames). 
              Robust to compression, noise, and variable recording conditions. 
              Peak detection run on composite z-score differential to localize splice boundaries — 
              same approach as partial audio deepfake detection.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
