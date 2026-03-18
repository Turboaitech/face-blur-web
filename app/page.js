"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";
const BLUR_MODES = [
  { id: "gaussian", label: "高斯模糊", icon: "◎" },
  { id: "mosaic", label: "像素马赛克", icon: "▦" },
  { id: "black", label: "纯黑遮挡", icon: "■" },
];

// ─── Main Component ──────────────────────────────────────────────────────────
export default function FaceBlurApp() {
  const [status, setStatus] = useState("idle"); // idle | loading-model | ready | detecting | done | error
  const [image, setImage] = useState(null);
  const [faces, setFaces] = useState([]);
  const [blurMode, setBlurMode] = useState("gaussian");
  const [blurStrength, setBlurStrength] = useState(40);
  const [expandRatio, setExpandRatio] = useState(0.3);
  const [errorMsg, setErrorMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [faceApi, setFaceApi] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);

  const canvasRef = useRef(null);
  const origCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const imgRef = useRef(null);

  // ─── Load face-api.js dynamically ──────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (modelLoaded) return;
    setStatus("loading-model");
    try {
      const faceapi = await import("face-api.js");
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      setFaceApi(faceapi);
      setModelLoaded(true);
      setStatus("ready");
    } catch (e) {
      console.error(e);
      setErrorMsg("模型加载失败，请检查网络连接后刷新页面");
      setStatus("error");
    }
  }, [modelLoaded]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // ─── Handle image upload ───────────────────────────────────────────────────
  const handleImage = useCallback(
    async (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      setFaces([]);
      setStatus("detecting");
      setErrorMsg("");

      const url = URL.createObjectURL(file);
      setImage(url);

      const img = new Image();
      img.onload = async () => {
        imgRef.current = img;

        // Draw original to hidden canvas for reference
        const origCanvas = origCanvasRef.current;
        origCanvas.width = img.width;
        origCanvas.height = img.height;
        const origCtx = origCanvas.getContext("2d");
        origCtx.drawImage(img, 0, 0);

        // Draw to visible canvas
        const canvas = canvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        // Detect faces
        if (!faceApi) {
          setErrorMsg("模型尚未加载完成，请稍等...");
          setStatus("error");
          return;
        }

        try {
          const detections = await faceApi.detectAllFaces(
            img,
            new faceApi.SsdMobilenetv1Options({ minConfidence: 0.3 })
          );

          const faceBoxes = detections.map((d) => ({
            x: d.box.x,
            y: d.box.y,
            w: d.box.width,
            h: d.box.height,
            enabled: true,
          }));

          setFaces(faceBoxes);

          if (faceBoxes.length === 0) {
            setErrorMsg("未检测到人脸，可尝试调整参数或换一张更清晰的照片");
          }

          setStatus("done");
          applyBlur(img, canvas, faceBoxes, blurMode, blurStrength, expandRatio);
        } catch (e) {
          console.error(e);
          setErrorMsg("检测过程出错: " + e.message);
          setStatus("error");
        }
      };
      img.src = url;
    },
    [faceApi, blurMode, blurStrength, expandRatio]
  );

  // ─── Apply blur to canvas ─────────────────────────────────────────────────
  const applyBlur = useCallback(
    (img, canvas, faceBoxes, mode, strength, expand) => {
      if (!img || !canvas) return;
      const ctx = canvas.getContext("2d");

      // Redraw original
      ctx.drawImage(img, 0, 0);

      const activeFaces = faceBoxes.filter((f) => f.enabled);
      if (activeFaces.length === 0) return;

      for (const face of activeFaces) {
        const ex = face.w * expand;
        const ey = face.h * expand;
        const fx = Math.max(0, face.x - ex);
        const fy = Math.max(0, face.y - ey);
        const fw = Math.min(img.width - fx, face.w + ex * 2);
        const fh = Math.min(img.height - fy, face.h + ey * 2);

        if (mode === "gaussian") {
          // Multi-pass box blur approximation of gaussian
          ctx.save();
          ctx.beginPath();
          ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
          ctx.clip();

          const passes = Math.ceil(strength / 5);
          for (let i = 0; i < passes; i++) {
            ctx.filter = `blur(${Math.min(strength, 50)}px)`;
            ctx.drawImage(
              canvas,
              fx - 50, fy - 50, fw + 100, fh + 100,
              fx - 50, fy - 50, fw + 100, fh + 100
            );
          }
          ctx.filter = "none";
          ctx.restore();
        } else if (mode === "mosaic") {
          // Pixelate
          const blockSize = Math.max(5, Math.ceil(strength / 3));
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = fw;
          tempCanvas.height = fh;
          const tempCtx = tempCanvas.getContext("2d");
          tempCtx.drawImage(canvas, fx, fy, fw, fh, 0, 0, fw, fh);

          const smallW = Math.max(1, Math.floor(fw / blockSize));
          const smallH = Math.max(1, Math.floor(fh / blockSize));

          const smallCanvas = document.createElement("canvas");
          smallCanvas.width = smallW;
          smallCanvas.height = smallH;
          const smallCtx = smallCanvas.getContext("2d");
          smallCtx.imageSmoothingEnabled = false;
          smallCtx.drawImage(tempCanvas, 0, 0, smallW, smallH);

          ctx.save();
          ctx.beginPath();
          ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
          ctx.clip();
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(smallCanvas, 0, 0, smallW, smallH, fx, fy, fw, fh);
          ctx.imageSmoothingEnabled = true;
          ctx.restore();
        } else if (mode === "black") {
          ctx.save();
          ctx.fillStyle = "#000";
          ctx.beginPath();
          ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }
    },
    []
  );

  // Re-apply when settings change
  useEffect(() => {
    if (status === "done" && imgRef.current && canvasRef.current) {
      applyBlur(imgRef.current, canvasRef.current, faces, blurMode, blurStrength, expandRatio);
    }
  }, [blurMode, blurStrength, expandRatio, faces, status, applyBlur]);

  // ─── Toggle individual face ────────────────────────────────────────────────
  const toggleFace = (index) => {
    setFaces((prev) =>
      prev.map((f, i) => (i === index ? { ...f, enabled: !f.enabled } : f))
    );
  };

  // ─── Download result ───────────────────────────────────────────────────────
  const download = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = "face_blurred.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  // ─── Reset ─────────────────────────────────────────────────────────────────
  const reset = () => {
    setImage(null);
    setFaces([]);
    setStatus(modelLoaded ? "ready" : "idle");
    setErrorMsg("");
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // ─── Drag & drop ──────────────────────────────────────────────────────────
  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleImage(e.dataTransfer.files[0]); };

  // ─── Paste from clipboard ──────────────────────────────────────────────────
  useEffect(() => {
    const onPaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          handleImage(item.getAsFile());
          break;
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleImage]);

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
          --bg: #0a0a0c;
          --surface: #141418;
          --surface-2: #1c1c22;
          --border: #2a2a32;
          --border-focus: #5b5bf0;
          --text: #e8e8ed;
          --text-2: #8888a0;
          --accent: #6c6cf0;
          --accent-glow: rgba(108, 108, 240, 0.15);
          --danger: #f06c6c;
          --success: #6cf0a0;
          --warning: #f0d06c;
          --radius: 12px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
          font-family: 'Noto Sans SC', -apple-system, sans-serif;
          background: var(--bg);
          color: var(--text);
          min-height: 100vh;
          overflow-x: hidden;
        }

        .app-container {
          max-width: 1100px;
          margin: 0 auto;
          padding: 40px 24px 80px;
        }

        /* ── Header ─────────────────────────────── */
        .header {
          text-align: center;
          margin-bottom: 48px;
        }

        .header h1 {
          font-size: 2.2rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          background: linear-gradient(135deg, #e8e8ed 0%, #6c6cf0 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 8px;
        }

        .header p {
          color: var(--text-2);
          font-size: 0.95rem;
          font-weight: 300;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          margin-top: 12px;
          padding: 4px 14px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 100px;
          font-size: 0.75rem;
          color: var(--text-2);
          font-family: 'JetBrains Mono', monospace;
        }

        .badge .dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: var(--success);
          animation: pulse 2s infinite;
        }
        .badge .dot.loading { background: var(--warning); }
        .badge .dot.error { background: var(--danger); animation: none; }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }

        /* ── Upload Zone ────────────────────────── */
        .upload-zone {
          position: relative;
          border: 2px dashed var(--border);
          border-radius: var(--radius);
          padding: 64px 24px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s;
          background: var(--surface);
        }

        .upload-zone:hover, .upload-zone.drag-over {
          border-color: var(--accent);
          background: var(--accent-glow);
        }

        .upload-zone .icon {
          font-size: 3rem;
          margin-bottom: 16px;
          opacity: 0.6;
        }

        .upload-zone h3 {
          font-size: 1.1rem;
          font-weight: 500;
          margin-bottom: 8px;
        }

        .upload-zone .hint {
          font-size: 0.8rem;
          color: var(--text-2);
        }

        .upload-zone .hint kbd {
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 1px 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.72rem;
        }

        /* ── Workspace ──────────────────────────── */
        .workspace {
          display: grid;
          grid-template-columns: 1fr 280px;
          gap: 24px;
          align-items: start;
        }

        @media (max-width: 768px) {
          .workspace {
            grid-template-columns: 1fr;
          }
        }

        .canvas-wrapper {
          position: relative;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          overflow: hidden;
        }

        .canvas-wrapper canvas {
          display: block;
          width: 100%;
          height: auto;
        }

        .canvas-overlay {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(10, 10, 12, 0.8);
          backdrop-filter: blur(4px);
        }

        .spinner {
          width: 40px; height: 40px;
          border: 3px solid var(--border);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        /* ── Panel ──────────────────────────────── */
        .panel {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 20px;
        }

        .panel h4 {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--text-2);
          margin-bottom: 16px;
          font-weight: 500;
        }

        .panel-section + .panel-section {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid var(--border);
        }

        /* ── Mode Selector ──────────────────────── */
        .mode-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }

        .mode-btn {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          padding: 10px 4px;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          color: var(--text-2);
          font-size: 0.72rem;
        }

        .mode-btn:hover {
          border-color: var(--text-2);
        }

        .mode-btn.active {
          border-color: var(--accent);
          background: var(--accent-glow);
          color: var(--text);
        }

        .mode-btn .mode-icon {
          font-size: 1.3rem;
        }

        /* ── Sliders ────────────────────────────── */
        .slider-group {
          margin-bottom: 16px;
        }

        .slider-label {
          display: flex;
          justify-content: space-between;
          font-size: 0.8rem;
          color: var(--text-2);
          margin-bottom: 8px;
        }

        .slider-label span:last-child {
          font-family: 'JetBrains Mono', monospace;
          color: var(--text);
        }

        input[type="range"] {
          -webkit-appearance: none;
          width: 100%;
          height: 4px;
          border-radius: 2px;
          background: var(--border);
          outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px; height: 16px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
          border: 2px solid var(--bg);
          box-shadow: 0 0 8px var(--accent-glow);
        }

        /* ── Face list ──────────────────────────── */
        .face-list { display: flex; flex-direction: column; gap: 6px; }

        .face-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          background: var(--surface-2);
          border-radius: 8px;
          font-size: 0.82rem;
          cursor: pointer;
          transition: background 0.15s;
        }

        .face-item:hover { background: var(--border); }

        .face-check {
          width: 18px; height: 18px;
          border-radius: 4px;
          border: 1.5px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          transition: all 0.15s;
          font-size: 0.7rem;
        }

        .face-check.checked {
          background: var(--accent);
          border-color: var(--accent);
        }

        .no-face {
          font-size: 0.82rem;
          color: var(--text-2);
          text-align: center;
          padding: 12px;
        }

        /* ── Buttons ────────────────────────────── */
        .btn-row {
          display: flex;
          gap: 8px;
          margin-top: 20px;
        }

        .btn {
          flex: 1;
          padding: 10px 16px;
          border: none;
          border-radius: 8px;
          font-size: 0.85rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          font-family: 'Noto Sans SC', sans-serif;
        }

        .btn-primary {
          background: var(--accent);
          color: #fff;
        }
        .btn-primary:hover { filter: brightness(1.15); }
        .btn-primary:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .btn-ghost {
          background: var(--surface-2);
          color: var(--text-2);
          border: 1px solid var(--border);
        }
        .btn-ghost:hover { border-color: var(--text-2); color: var(--text); }

        /* ── Error ──────────────────────────────── */
        .error-msg {
          margin-top: 12px;
          padding: 10px 14px;
          background: rgba(240, 108, 108, 0.08);
          border: 1px solid rgba(240, 108, 108, 0.2);
          border-radius: 8px;
          font-size: 0.8rem;
          color: var(--danger);
        }

        /* ── Footer ─────────────────────────────── */
        .footer {
          text-align: center;
          margin-top: 48px;
          font-size: 0.75rem;
          color: var(--text-2);
          opacity: 0.6;
        }
      `}</style>

      <div className="app-container">
        <header className="header">
          <h1>人脸模糊工具</h1>
          <p>上传图片 → 自动检测人脸 → 一键模糊，全程浏览器端处理，图片不会上传至服务器</p>
          <div className="badge">
            <span
              className={`dot ${
                status === "loading-model" ? "loading" : status === "error" ? "error" : ""
              }`}
            />
            {status === "loading-model"
              ? "模型加载中..."
              : modelLoaded
              ? "face-api.js · 就绪"
              : "等待初始化"}
          </div>
        </header>

        {!image ? (
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <div className="icon">📷</div>
            <h3>点击选择图片，或拖拽到此处</h3>
            <p className="hint">
              支持 JPG / PNG / WebP &nbsp;|&nbsp; 也可以 <kbd>Ctrl+V</kbd> 粘贴截图
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleImage(e.target.files[0])}
            />
          </div>
        ) : (
          <div className="workspace">
            <div className="canvas-wrapper">
              <canvas ref={canvasRef} />
              {status === "detecting" && (
                <div className="canvas-overlay">
                  <div className="spinner" />
                </div>
              )}
            </div>

            <div>
              <div className="panel">
                <div className="panel-section">
                  <h4>模糊模式</h4>
                  <div className="mode-grid">
                    {BLUR_MODES.map((m) => (
                      <button
                        key={m.id}
                        className={`mode-btn ${blurMode === m.id ? "active" : ""}`}
                        onClick={() => setBlurMode(m.id)}
                      >
                        <span className="mode-icon">{m.icon}</span>
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="panel-section">
                  <h4>参数调整</h4>
                  <div className="slider-group">
                    <div className="slider-label">
                      <span>模糊强度</span>
                      <span>{blurStrength}</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="80"
                      value={blurStrength}
                      onChange={(e) => setBlurStrength(Number(e.target.value))}
                    />
                  </div>
                  <div className="slider-group">
                    <div className="slider-label">
                      <span>扩展范围</span>
                      <span>{Math.round(expandRatio * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="80"
                      value={expandRatio * 100}
                      onChange={(e) => setExpandRatio(Number(e.target.value) / 100)}
                    />
                  </div>
                </div>

                <div className="panel-section">
                  <h4>检测到的人脸 ({faces.filter((f) => f.enabled).length}/{faces.length})</h4>
                  {faces.length > 0 ? (
                    <div className="face-list">
                      {faces.map((f, i) => (
                        <div key={i} className="face-item" onClick={() => toggleFace(i)}>
                          <div className={`face-check ${f.enabled ? "checked" : ""}`}>
                            {f.enabled ? "✓" : ""}
                          </div>
                          <span>人脸 #{i + 1}</span>
                          <span style={{ marginLeft: "auto", fontSize: "0.7rem", color: "var(--text-2)", fontFamily: "'JetBrains Mono', monospace" }}>
                            {Math.round(f.w)}×{Math.round(f.h)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="no-face">
                      {status === "detecting" ? "检测中..." : "未检测到人脸"}
                    </div>
                  )}
                </div>

                {errorMsg && <div className="error-msg">{errorMsg}</div>}

                <div className="btn-row">
                  <button className="btn btn-ghost" onClick={reset}>
                    重选图片
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={download}
                    disabled={status !== "done" || faces.filter((f) => f.enabled).length === 0}
                  >
                    下载结果
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        <canvas ref={origCanvasRef} style={{ display: "none" }} />

        <div className="footer">
          纯前端处理 · 图片不会离开你的浏览器 · Powered by face-api.js + Next.js
        </div>
      </div>
    </>
  );
}
