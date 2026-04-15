import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const SIGNAL_META = {
  lip_aspect_ratio:        { label: "Lip Geometry",          color: "#7c3aed", audio: "F0 jitter z-score" },
  laplacian_variance:      { label: "Texture Energy",        color: "#2563eb", audio: "hf_ratio_z" },
  fft_peak_score:          { label: "GAN Fingerprint",       color: "#059669", audio: "MFCC shift" },
  landmark_velocity:       { label: "Landmark Jitter (FFD)", color: "#d97706", audio: "F0 jitter" },
  temporal_gradient:       { label: "Temporal Gradient",     color: "#dc2626", audio: "Spectral flatness z" },
  phoneme_viseme_mismatch: { label: "Phoneme-Viseme (Groq)", color: "#0891b2", audio: "MAIA audio detection" },
  composite:               { label: "Composite Score",       color: "#0d0d1a", audio: "Combined z-score" },
};

function Sparkline({ values, color, threshold = 2.0 }) {
  if (!values || values.length === 0) return null;
  const W = 320, H = 50;
  const max = Math.max(...values.map(Math.abs), threshold + 0.5);
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * W},${H - ((v + max) / (2 * max)) * H}`
  ).join(" ");
  const tY  = H - ((threshold  + max) / (2 * max)) * H;
  const ntY = H - ((-threshold + max) / (2 * max)) * H;
  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      <line x1={0} y1={tY}  x2={W} y2={tY}  stroke="#ef4444" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.5} />
      <line x1={0} y1={ntY} x2={W} y2={ntY} stroke="#ef4444" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.5} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} opacity={0.85} />
      {values.map((v, i) =>
        Math.abs(v) > threshold ? (
          <circle key={i} cx={(i/(values.length-1))*W} cy={H-((v+max)/(2*max))*H}
            r={3} fill="#ef4444" opacity={0.9} />
        ) : null
      )}
    </svg>
  );
}

function SegmentBadge({ seg }) {
  const conf  = seg.confidence;
  const bg    = conf==="high" ? "#fef2f2" : conf==="medium" ? "#fffbeb" : "#f0fdf4";
  const bdr   = conf==="high" ? "#fca5a5" : conf==="medium" ? "#fcd34d" : "#86efac";
  const tc    = conf==="high" ? "#991b1b" : conf==="medium" ? "#92400e" : "#166534";
  const hasPV = seg.triggered_signals.includes("phoneme_viseme_mismatch");
  return (
    <div style={{ background: bg, border: `1px solid ${bdr}`, borderRadius: 8,
                  padding: "10px 14px", marginBottom: 8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:4 }}>
        <span style={{ fontWeight:700, fontSize:13, color:tc }}>
          t={seg.start_time}s – {seg.end_time}s
          {hasPV && <span style={{ marginLeft:8, fontSize:11, background:"#0891b2",
            color:"white", borderRadius:4, padding:"1px 6px" }}>🎤 Groq</span>}
        </span>
        <span style={{ fontSize:11, fontWeight:700, color:tc, textTransform:"uppercase", letterSpacing:"0.05em" }}>
          {conf} · z={seg.peak_z_score}
        </span>
      </div>
      <div style={{ display:"flex", flexWrap:"wrap", gap:4, marginBottom:4 }}>
        {seg.triggered_signals.map(s => (
          <span key={s} style={{ fontSize:10, background:"#1e1b4b", color:"#c4b5fd",
            borderRadius:4, padding:"2px 6px" }}>{s.replace(/_/g," ")}</span>
        ))}
      </div>
      <div style={{ fontSize:11, color:"#6b7280" }}>{seg.description}</div>
    </div>
  );
}

function PhonemeVisemePanel({ report, transcript }) {
  const [expanded, setExpanded] = useState(false);
  if (!report) return null;
  return (
    <div style={{ background:"#ecfeff", border:"1px solid #67e8f9", borderRadius:10,
                  padding:"14px 16px", marginBottom:20 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
        <span style={{ fontWeight:700, fontSize:13, color:"#0e7490" }}>
          🎤 Signal 6 — Groq Whisper Phoneme-Viseme Analysis
        </span>
        <span style={{ fontSize:11, color:"#0e7490" }}>
          {report.mismatch_count}/{report.word_count} words mismatched ({(report.mismatch_rate*100).toFixed(1)}%)
        </span>
      </div>
      {transcript && (
        <div style={{ fontSize:11, color:"#155e75", background:"white", borderRadius:6,
                      padding:"8px 10px", marginBottom:8, fontStyle:"italic" }}>
          "{transcript}"
        </div>
      )}
      {report.mismatches.length > 0 && (
        <>
          <button onClick={() => setExpanded(e=>!e)}
            style={{ fontSize:11, color:"#0e7490", background:"none", border:"1px solid #67e8f9",
                     borderRadius:4, padding:"3px 8px", cursor:"pointer", marginBottom:6 }}>
            {expanded ? "Hide" : "Show"} {report.mismatches.length} mismatch detail(s)
          </button>
          {expanded && (
            <div style={{ maxHeight:200, overflowY:"auto" }}>
              {report.mismatches.map((m,i) => (
                <div key={i} style={{ fontSize:10, padding:"3px 6px", borderBottom:"1px solid #cffafe",
                                      display:"flex", gap:12 }}>
                  <span style={{ fontWeight:700, color:"#0e7490", minWidth:60 }}>"{m.word}"</span>
                  <span style={{ color:"#374151" }}>t={m.time}s</span>
                  <span style={{ color:"#6b7280" }}>viseme: {m.viseme_group}</span>
                  <span style={{ color:"#dc2626" }}>LAR={m.actual_lar}</span>
                  <span style={{ color:"#059669" }}>expected [{m.expected_range[0]}–{m.expected_range[1]}]</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function App() {
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [groqKey, setGroqKey]     = useState("");
  const [showKey, setShowKey]     = useState(false);
  const inputRef = useRef();

  const handleFile = useCallback(f => { setFile(f); setResult(null); setError(null); }, []);
  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) handleFile(f);
  }, [handleFile]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const headers = {};
      if (groqKey.trim()) headers["X-Groq-Api-Key"] = groqKey.trim();
      const res = await fetch(`${API}/detect`, { method:"POST", body:fd, headers });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Detection failed"); }
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const verdictColor = result?.verdict==="MANIPULATED" ? "#991b1b"
    : result?.verdict==="AUTHENTIC" ? "#166534" : "#92400e";
  const verdictBg = result?.verdict==="MANIPULATED" ? "#fef2f2"
    : result?.verdict==="AUTHENTIC" ? "#f0fdf4" : "#fffbeb";

  return (
    <div style={{ minHeight:"100vh", background:"#f8f7ff", fontFamily:"'Inter',system-ui,sans-serif" }}>
      {/* Header */}
      <div style={{ background:"#0d0d1a", color:"white", padding:"16px 32px",
                    display:"flex", alignItems:"center", gap:12 }}>
        <div style={{ width:10, height:10, borderRadius:"50%", background:"#7c3aed",
                      boxShadow:"0 0 8px #7c3aed" }} />
        <span style={{ fontWeight:800, fontSize:18, letterSpacing:"0.04em" }}>FauxPix</span>
        <span style={{ color:"#9ca3af", fontSize:12, marginLeft:4 }}>Partial Video Deepfake Detector</span>
        <span style={{ marginLeft:"auto", fontSize:11, color:"#6b7280" }}>
          6-signal · segment-level · Groq Whisper · per-clip z-scoring
        </span>
      </div>

      <div style={{ maxWidth:940, margin:"0 auto", padding:"32px 24px" }}>

        {/* Groq key input */}
        <div style={{ background:"#ecfeff", border:"1px solid #67e8f9", borderRadius:10,
                      padding:"12px 16px", marginBottom:16, display:"flex", alignItems:"center", gap:10 }}>
          <span style={{ fontSize:12, color:"#0e7490", fontWeight:700, whiteSpace:"nowrap" }}>
            🎤 Groq API Key (Signal 6):
          </span>
          <input
            type={showKey ? "text" : "password"}
            value={groqKey}
            onChange={e => setGroqKey(e.target.value)}
            placeholder="gsk_... (get free key at console.groq.com)"
            style={{ flex:1, padding:"6px 10px", borderRadius:6, border:"1px solid #67e8f9",
                     fontSize:12, fontFamily:"monospace", outline:"none" }}
          />
          <button onClick={() => setShowKey(s=>!s)}
            style={{ fontSize:11, background:"none", border:"1px solid #67e8f9",
                     borderRadius:4, padding:"4px 8px", cursor:"pointer", color:"#0e7490" }}>
            {showKey ? "Hide" : "Show"}
          </button>
          <span style={{ fontSize:10, color:"#6b7280", whiteSpace:"nowrap" }}>
            Optional — enables phoneme-viseme detection
          </span>
        </div>

        {/* Upload */}
        <div
          onDragOver={e=>{e.preventDefault();setDragging(true);}}
          onDragLeave={()=>setDragging(false)}
          onDrop={onDrop}
          onClick={()=>inputRef.current.click()}
          style={{ border:`2px dashed ${dragging||file?"#7c3aed":"#d1d5db"}`,
                   borderRadius:12, padding:"28px 24px", textAlign:"center",
                   background:dragging?"#ede9fe":"#fff", cursor:"pointer",
                   transition:"all 0.15s", marginBottom:16 }}
        >
          <input ref={inputRef} type="file" accept="video/*" style={{ display:"none" }}
            onChange={e=>e.target.files[0]&&handleFile(e.target.files[0])} />
          <div style={{ fontSize:28, marginBottom:6 }}>🎬</div>
          {file ? (
            <div>
              <div style={{ fontWeight:700, color:"#7c3aed" }}>{file.name}</div>
              <div style={{ fontSize:12, color:"#6b7280" }}>{(file.size/1024/1024).toFixed(1)} MB</div>
            </div>
          ) : (
            <div>
              <div style={{ fontWeight:600, color:"#374151" }}>Drop video or click to upload</div>
              <div style={{ fontSize:12, color:"#9ca3af", marginTop:4 }}>MP4 · MOV · AVI · MKV · WebM</div>
            </div>
          )}
        </div>

        <button onClick={analyze} disabled={!file||loading}
          style={{ width:"100%", padding:"13px", borderRadius:10, border:"none",
                   background:!file||loading?"#e5e7eb":"#0d0d1a",
                   color:!file||loading?"#9ca3af":"white",
                   fontWeight:700, fontSize:15, cursor:!file||loading?"not-allowed":"pointer",
                   marginBottom:24, transition:"all 0.15s" }}>
          {loading ? "⏳ Analyzing…" : `🔍 Run FauxPix${groqKey?" (6 signals)": " (5 signals)"}`}
        </button>

        {error && (
          <div style={{ background:"#fef2f2", border:"1px solid #fca5a5", borderRadius:8,
                        padding:14, color:"#991b1b", marginBottom:20 }}>{error}</div>
        )}

        {result && (
          <>
            {/* Verdict */}
            <div style={{ background:verdictBg, border:`2px solid ${verdictColor}`,
                          borderRadius:12, padding:"20px 24px", marginBottom:20 }}>
              <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:8 }}>
                <span style={{ fontSize:26 }}>
                  {result.verdict==="MANIPULATED"?"🚨":result.verdict==="AUTHENTIC"?"✅":"⚠️"}
                </span>
                <div>
                  <div style={{ fontWeight:800, fontSize:22, color:verdictColor }}>{result.verdict}</div>
                  <div style={{ fontSize:12, color:"#6b7280" }}>
                    Confidence: {(result.overall_confidence*100).toFixed(0)}% ·
                    {result.processing_time_sec}s ·
                    {result.video_info.total_frames} frames @ {result.video_info.fps?.toFixed(1)}fps
                  </div>
                </div>
              </div>
              <div style={{ fontSize:13, color:"#374151" }}>{result.summary}</div>
            </div>

            {/* Phoneme-Viseme Signal 6 */}
            <PhonemeVisemePanel
              report={result.phoneme_viseme_report}
              transcript={result.groq_transcript}
            />

            {/* Segments */}
            {result.anomaly_segments?.length > 0 && (
              <div style={{ marginBottom:24 }}>
                <div style={{ fontWeight:700, fontSize:13, color:"#0d0d1a", marginBottom:8,
                              textTransform:"uppercase", letterSpacing:"0.06em" }}>
                  Anomaly Segments ({result.anomaly_segments.length})
                </div>
                {result.anomaly_segments.map((s,i)=><SegmentBadge key={i} seg={s}/>)}
              </div>
            )}

            {/* Signal traces */}
            <div style={{ fontWeight:700, fontSize:13, color:"#0d0d1a", marginBottom:10,
                          textTransform:"uppercase", letterSpacing:"0.06em" }}>
              Signal Traces — per-clip z-scores
            </div>
            <div style={{ display:"grid", gap:10 }}>
              {Object.entries(SIGNAL_META).map(([key, meta]) => {
                const vals = result.per_signal_zscores?.[key];
                if (!vals) return null;
                const isGroq = key === "phoneme_viseme_mismatch";
                return (
                  <div key={key} style={{ background:"#fff", borderRadius:10,
                    padding:"12px 16px", border:`1px solid ${isGroq?"#67e8f9":"#e5e7eb"}` }}>
                    <div style={{ display:"flex", justifyContent:"space-between",
                                  alignItems:"center", marginBottom:6 }}>
                      <div>
                        <span style={{ fontWeight:700, fontSize:13, color:meta.color }}>
                          {meta.label}
                          {isGroq && <span style={{ fontSize:10, marginLeft:6, color:"#0891b2" }}>
                            (Groq Whisper)</span>}
                        </span>
                        <span style={{ fontSize:11, color:"#9ca3af", marginLeft:8 }}>
                          audio analogue: {meta.audio}
                        </span>
                      </div>
                      <span style={{ fontSize:11, color:"#6b7280" }}>
                        max |z| = {Math.max(...vals.map(Math.abs)).toFixed(2)}
                      </span>
                    </div>
                    <Sparkline values={vals} color={meta.color}
                      threshold={key==="landmark_velocity"?2.2:key==="phoneme_viseme_mismatch"?1.8:2.0}/>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop:20, background:"#ede9fe", borderRadius:10,
                          padding:"12px 16px", fontSize:12, color:"#4c1d95" }}>
              <strong>Architecture:</strong> All 6 signals z-scored against this clip's own baseline
              (first 30% of frames). Signal 6 uses Groq whisper-large-v3-turbo for word-level
              timestamps, then checks actual lip geometry (MediaPipe) against expected viseme
              shapes for each phoneme — the MAIA audio ↔ FauxPix video bridge.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
