"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";
const BLUR_MODES = [
  { id: "gaussian", label: "高斯模糊", icon: "◎" },
  { id: "mosaic", label: "像素马赛克", icon: "▦" },
  { id: "black", label: "纯黑遮挡", icon: "■" },
];

export default function FaceBlurApp() {
  const [status, setStatus] = useState("idle");
  const [image, setImage] = useState(null);
  const [faces, setFaces] = useState([]);
  const [blurMode, setBlurMode] = useState("gaussian");
  const [blurStrength, setBlurStrength] = useState(40);
  const [expandRatio, setExpandRatio] = useState(0.3);
  const [cutoutExpand, setCutoutExpand] = useState(0.5);
  const [cutoutBg, setCutoutBg] = useState("white");
  const [errorMsg, setErrorMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [faceApi, setFaceApi] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState("blur");
  const [cutoutUrls, setCutoutUrls] = useState([]);

  const canvasRef = useRef(null);
  const origCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const imgRef = useRef(null);

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

  useEffect(() => { loadModels(); }, [loadModels]);

  const handleImage = useCallback(
    async (file) => {
      if (!file || !file.type.startsWith("image/")) return;
      setFaces([]); setCutoutUrls([]); setStatus("detecting"); setErrorMsg("");
      const url = URL.createObjectURL(file);
      setImage(url);
      const img = new Image();
      img.onload = async () => {
        imgRef.current = img;
        const origCanvas = origCanvasRef.current;
        origCanvas.width = img.width; origCanvas.height = img.height;
        origCanvas.getContext("2d").drawImage(img, 0, 0);
        const canvas = canvasRef.current;
        canvas.width = img.width; canvas.height = img.height;
        canvas.getContext("2d").drawImage(img, 0, 0);
        if (!faceApi) { setErrorMsg("模型尚未加载完成，请稍等..."); setStatus("error"); return; }
        try {
          const detections = await faceApi.detectAllFaces(img, new faceApi.SsdMobilenetv1Options({ minConfidence: 0.3 }));
          const faceBoxes = detections.map((d) => ({ x: d.box.x, y: d.box.y, w: d.box.width, h: d.box.height, enabled: true }));
          setFaces(faceBoxes);
          if (faceBoxes.length === 0) setErrorMsg("未检测到人脸，可尝试调整参数或换一张更清晰的照片");
          setStatus("done");
          applyBlur(img, canvas, faceBoxes, blurMode, blurStrength, expandRatio);
          generateCutouts(img, faceBoxes, cutoutExpand, cutoutBg);
        } catch (e) { console.error(e); setErrorMsg("检测过程出错: " + e.message); setStatus("error"); }
      };
      img.src = url;
    },
    [faceApi, blurMode, blurStrength, expandRatio, cutoutExpand, cutoutBg]
  );

  const applyBlur = useCallback((img, canvas, faceBoxes, mode, strength, expand) => {
    if (!img || !canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const activeFaces = faceBoxes.filter((f) => f.enabled);
    if (activeFaces.length === 0) return;
    for (const face of activeFaces) {
      const ex = face.w * expand, ey = face.h * expand;
      const fx = Math.max(0, face.x - ex), fy = Math.max(0, face.y - ey);
      const fw = Math.min(img.width - fx, face.w + ex * 2), fh = Math.min(img.height - fy, face.h + ey * 2);
      if (mode === "gaussian") {
        ctx.save(); ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
        ctx.clip();
        const passes = Math.ceil(strength / 5);
        for (let i = 0; i < passes; i++) {
          ctx.filter = `blur(${Math.min(strength, 50)}px)`;
          ctx.drawImage(canvas, fx - 50, fy - 50, fw + 100, fh + 100, fx - 50, fy - 50, fw + 100, fh + 100);
        }
        ctx.filter = "none"; ctx.restore();
      } else if (mode === "mosaic") {
        const blockSize = Math.max(5, Math.ceil(strength / 3));
        const tc = document.createElement("canvas"); tc.width = fw; tc.height = fh;
        tc.getContext("2d").drawImage(canvas, fx, fy, fw, fh, 0, 0, fw, fh);
        const sw = Math.max(1, Math.floor(fw / blockSize)), sh = Math.max(1, Math.floor(fh / blockSize));
        const sc = document.createElement("canvas"); sc.width = sw; sc.height = sh;
        const sctx = sc.getContext("2d"); sctx.imageSmoothingEnabled = false;
        sctx.drawImage(tc, 0, 0, sw, sh);
        ctx.save(); ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
        ctx.clip(); ctx.imageSmoothingEnabled = false;
        ctx.drawImage(sc, 0, 0, sw, sh, fx, fy, fw, fh);
        ctx.imageSmoothingEnabled = true; ctx.restore();
      } else if (mode === "black") {
        ctx.save(); ctx.fillStyle = "#000"; ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2);
        ctx.fill(); ctx.restore();
      }
    }
  }, []);

  const generateCutouts = useCallback((img, faceBoxes, expand, bg) => {
    if (!img || faceBoxes.length === 0) { setCutoutUrls([]); return; }
    const urls = faceBoxes.map((face) => {
      const ex = face.w * expand, ey = face.h * expand;
      const fx = Math.max(0, face.x - ex);
      const fy = Math.max(0, face.y - ey * 1.2);
      const fw = Math.min(img.width - fx, face.w + ex * 2);
      const fh = Math.min(img.height - fy, face.h + ey * 2.8);

      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = fw; cropCanvas.height = fh;
      cropCanvas.getContext("2d").drawImage(img, fx, fy, fw, fh, 0, 0, fw, fh);

      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = fw; maskCanvas.height = fh;
      const maskCtx = maskCanvas.getContext("2d");
      const cx = fw / 2, cy = fh / 2;
      const rx = fw * 0.46, ry = fh * 0.46;
      const maxR = Math.max(rx, ry);
      const gradient = maskCtx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
      gradient.addColorStop(0, "rgba(255,255,255,1)");
      gradient.addColorStop(0.78, "rgba(255,255,255,1)");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      maskCtx.fillStyle = gradient;
      maskCtx.beginPath();
      maskCtx.save();
      maskCtx.translate(cx, cy);
      maskCtx.scale(rx / maxR, ry / maxR);
      maskCtx.arc(0, 0, maxR, 0, Math.PI * 2);
      maskCtx.restore();
      maskCtx.fill();

      const c = document.createElement("canvas");
      c.width = fw; c.height = fh;
      const ctx = c.getContext("2d");
      ctx.drawImage(maskCanvas, 0, 0);
      ctx.globalCompositeOperation = "source-in";
      ctx.drawImage(cropCanvas, 0, 0);
      ctx.globalCompositeOperation = "source-over";

      if (bg === "white") {
        const fc = document.createElement("canvas"); fc.width = fw; fc.height = fh;
        const fctx = fc.getContext("2d");
        fctx.fillStyle = "#ffffff"; fctx.fillRect(0, 0, fw, fh);
        fctx.drawImage(c, 0, 0);
        return fc.toDataURL("image/png");
      }
      return c.toDataURL("image/png");
    });
    setCutoutUrls(urls);
  }, []);

  useEffect(() => {
    if (status === "done" && imgRef.current && canvasRef.current) {
      applyBlur(imgRef.current, canvasRef.current, faces, blurMode, blurStrength, expandRatio);
      generateCutouts(imgRef.current, faces, cutoutExpand, cutoutBg);
    }
  }, [blurMode, blurStrength, expandRatio, cutoutExpand, cutoutBg, faces, status, applyBlur, generateCutouts]);

  const toggleFace = (i) => setFaces((prev) => prev.map((f, j) => (j === i ? { ...f, enabled: !f.enabled } : f)));

  const downloadBlurred = () => {
    const c = canvasRef.current; if (!c) return;
    const a = document.createElement("a"); a.download = "face_blurred.png"; a.href = c.toDataURL("image/png"); a.click();
  };
  const downloadCutout = (url, i) => {
    const a = document.createElement("a"); a.download = `face_cutout_${i + 1}.png`; a.href = url; a.click();
  };
  const downloadAllCutouts = () => cutoutUrls.forEach((url, i) => setTimeout(() => downloadCutout(url, i), i * 200));

  const reset = () => { setImage(null); setFaces([]); setCutoutUrls([]); setStatus(modelLoaded ? "ready" : "idle"); setErrorMsg(""); setActiveTab("blur"); };

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleImage(e.dataTransfer.files[0]); };

  useEffect(() => {
    const onPaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) { if (item.type.startsWith("image/")) { handleImage(item.getAsFile()); break; } }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleImage]);

  const hasFaces = faces.length > 0;
  const hasEnabled = faces.filter((f) => f.enabled).length > 0;

  return (
    <>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
        :root {
          --bg:#0a0a0c;--surface:#141418;--surface-2:#1c1c22;--border:#2a2a32;
          --text:#e8e8ed;--text-2:#8888a0;--accent:#6c6cf0;--accent-glow:rgba(108,108,240,0.15);
          --accent-2:#f06cb0;--accent-2-glow:rgba(240,108,176,0.15);
          --danger:#f06c6c;--success:#6cf0a0;--warning:#f0d06c;--radius:12px;
        }
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Noto Sans SC',-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
        .app{max-width:1100px;margin:0 auto;padding:40px 24px 80px}
        .hd{text-align:center;margin-bottom:48px}
        .hd h1{font-size:2.2rem;font-weight:700;letter-spacing:-0.03em;background:linear-gradient(135deg,#e8e8ed 0%,#6c6cf0 50%,#f06cb0 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
        .hd p{color:var(--text-2);font-size:0.95rem;font-weight:300}
        .badge{display:inline-flex;align-items:center;gap:6px;margin-top:12px;padding:4px 14px;background:var(--surface);border:1px solid var(--border);border-radius:100px;font-size:0.75rem;color:var(--text-2);font-family:'JetBrains Mono',monospace}
        .dot{width:6px;height:6px;border-radius:50%;background:var(--success);animation:pulse 2s infinite}
        .dot.ld{background:var(--warning)}.dot.er{background:var(--danger);animation:none}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

        .uz{border:2px dashed var(--border);border-radius:var(--radius);padding:64px 24px;text-align:center;cursor:pointer;transition:all .3s;background:var(--surface)}
        .uz:hover,.uz.dg{border-color:var(--accent);background:var(--accent-glow)}
        .uz .ic{font-size:3rem;margin-bottom:16px;opacity:.6}
        .uz h3{font-size:1.1rem;font-weight:500;margin-bottom:8px}
        .uz .ht{font-size:.8rem;color:var(--text-2)}
        .uz kbd{background:var(--surface-2);border:1px solid var(--border);border-radius:4px;padding:1px 6px;font-family:'JetBrains Mono',monospace;font-size:.72rem}

        .ws{display:grid;grid-template-columns:1fr 300px;gap:24px;align-items:start}
        @media(max-width:768px){.ws{grid-template-columns:1fr}}

        .tabs{display:flex;gap:2px;background:var(--surface-2);border-radius:10px;padding:3px;margin-bottom:16px}
        .tab{flex:1;padding:8px 12px;background:none;border:none;border-radius:8px;color:var(--text-2);font-size:.82rem;font-weight:500;cursor:pointer;transition:all .2s;font-family:'Noto Sans SC',sans-serif}
        .tab:hover{color:var(--text)}
        .tab.on{background:var(--surface);color:var(--text);box-shadow:0 1px 4px rgba(0,0,0,.3)}
        .tab.on.t1{border-bottom:2px solid var(--accent)}.tab.on.t2{border-bottom:2px solid var(--accent-2)}

        .cw{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
        .cw canvas{display:block;width:100%;height:auto}
        .ov{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(10,10,12,.8);backdrop-filter:blur(4px)}
        .sp{width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
        @keyframes spin{to{transform:rotate(360deg)}}

        .gal{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;min-height:200px}
        .gg{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px}
        .gc{position:relative;background:var(--surface-2);border:1px solid var(--border);border-radius:10px;overflow:hidden;transition:all .2s;cursor:pointer}
        .gc:hover{border-color:var(--accent-2);box-shadow:0 0 20px var(--accent-2-glow);transform:translateY(-2px)}
        .gc img{display:block;width:100%;height:auto;background:repeating-conic-gradient(#1c1c22 0% 25%,#141418 0% 50%) 50%/16px 16px}
        .gc.wb img{background:#fff}
        .gc-l{position:absolute;bottom:8px;left:8px;background:rgba(10,10,12,.8);backdrop-filter:blur(6px);border-radius:6px;padding:3px 10px;font-size:.7rem;color:var(--text-2);font-family:'JetBrains Mono',monospace}
        .gc-d{position:absolute;top:8px;right:8px;background:var(--accent-2);border:none;border-radius:6px;padding:4px 10px;font-size:.7rem;color:#fff;cursor:pointer;opacity:0;transition:opacity .2s;font-family:'Noto Sans SC',sans-serif}
        .gc:hover .gc-d{opacity:1}
        .ge{display:flex;align-items:center;justify-content:center;height:200px;color:var(--text-2);font-size:.9rem}

        .pn{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
        .pn h4{font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-2);margin-bottom:16px;font-weight:500}
        .ps+.ps{margin-top:20px;padding-top:20px;border-top:1px solid var(--border)}

        .mg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
        .mb{display:flex;flex-direction:column;align-items:center;gap:4px;padding:10px 4px;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;cursor:pointer;transition:all .2s;color:var(--text-2);font-size:.72rem}
        .mb:hover{border-color:var(--text-2)}.mb.on{border-color:var(--accent);background:var(--accent-glow);color:var(--text)}
        .mi{font-size:1.3rem}

        .sg{margin-bottom:16px}
        .sl{display:flex;justify-content:space-between;font-size:.8rem;color:var(--text-2);margin-bottom:8px}
        .sl span:last-child{font-family:'JetBrains Mono',monospace;color:var(--text)}
        input[type=range]{-webkit-appearance:none;width:100%;height:4px;border-radius:2px;background:var(--border);outline:none}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);cursor:pointer;border:2px solid var(--bg)}

        .bt{display:flex;gap:8px}.bts{display:flex;flex-direction:column;gap:8px;margin-top:20px}
        .bn{flex:1;padding:10px 16px;border:none;border-radius:8px;font-size:.85rem;font-weight:500;cursor:pointer;transition:all .2s;font-family:'Noto Sans SC',sans-serif}
        .b1{background:var(--accent);color:#fff}.b1:hover{filter:brightness(1.15)}.b1:disabled{opacity:.4;cursor:not-allowed}
        .b2{background:var(--accent-2);color:#fff}.b2:hover{filter:brightness(1.15)}.b2:disabled{opacity:.4;cursor:not-allowed}
        .bg{background:var(--surface-2);color:var(--text-2);border:1px solid var(--border)}.bg:hover{border-color:var(--text-2);color:var(--text)}

        .bgt{display:flex;gap:8px}
        .bo{flex:1;padding:8px;border:1px solid var(--border);border-radius:8px;background:var(--surface-2);color:var(--text-2);font-size:.78rem;cursor:pointer;text-align:center;transition:all .2s;font-family:'Noto Sans SC',sans-serif}
        .bo:hover{border-color:var(--text-2)}.bo.on{border-color:var(--accent-2);background:var(--accent-2-glow);color:var(--text)}

        .fl{display:flex;flex-direction:column;gap:6px}
        .fi{display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--surface-2);border-radius:8px;font-size:.82rem;cursor:pointer;transition:background .15s}
        .fi:hover{background:var(--border)}
        .fc{width:18px;height:18px;border-radius:4px;border:1.5px solid var(--border);display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;font-size:.7rem}
        .fc.ck{background:var(--accent);border-color:var(--accent)}
        .nf{font-size:.82rem;color:var(--text-2);text-align:center;padding:12px}
        .em{margin-top:12px;padding:10px 14px;background:rgba(240,108,108,.08);border:1px solid rgba(240,108,108,.2);border-radius:8px;font-size:.8rem;color:var(--danger)}
        .ft{text-align:center;margin-top:48px;font-size:.75rem;color:var(--text-2);opacity:.6}
      `}</style>

      <div className="app">
        <header className="hd">
          <h1>人脸模糊 & 抠脸工具</h1>
          <p>上传图片 → 自动检测人脸 → 同时生成模糊版本和抠脸头像，全程浏览器端处理</p>
          <div className="badge">
            <span className={`dot ${status === "loading-model" ? "ld" : status === "error" ? "er" : ""}`} />
            {status === "loading-model" ? "模型加载中..." : modelLoaded ? "face-api.js · 就绪" : "等待初始化"}
          </div>
        </header>

        {!image ? (
          <div className={`uz ${dragOver ? "dg" : ""}`} onClick={() => fileInputRef.current?.click()} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
            <div className="ic">📷</div>
            <h3>点击选择图片，或拖拽到此处</h3>
            <p className="ht">支持 JPG / PNG / WebP &nbsp;|&nbsp; 也可以 <kbd>Ctrl+V</kbd> 粘贴截图</p>
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleImage(e.target.files[0])} />
          </div>
        ) : (
          <div className="ws">
            <div>
              <div className="tabs">
                <button className={`tab ${activeTab === "blur" ? "on t1" : ""}`} onClick={() => setActiveTab("blur")}>🌫 模糊结果</button>
                <button className={`tab ${activeTab === "cutout" ? "on t2" : ""}`} onClick={() => setActiveTab("cutout")}>✂️ 抠脸结果 {cutoutUrls.length > 0 && `(${cutoutUrls.length})`}</button>
              </div>
              {activeTab === "blur" ? (
                <div className="cw">
                  <canvas ref={canvasRef} />
                  {status === "detecting" && <div className="ov"><div className="sp" /></div>}
                </div>
              ) : (
                <div className="gal">
                  {cutoutUrls.length > 0 ? (
                    <div className="gg">
                      {cutoutUrls.map((url, i) => (
                        <div key={i} className={`gc ${cutoutBg === "white" ? "wb" : ""}`}>
                          <img src={url} alt={`Face ${i + 1}`} />
                          <span className="gc-l">#{i + 1}</span>
                          <button className="gc-d" onClick={(e) => { e.stopPropagation(); downloadCutout(url, i); }}>下载</button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="ge">{status === "detecting" ? "检测中..." : "未检测到人脸"}</div>
                  )}
                </div>
              )}
            </div>

            <div>
              <div className="pn">
                <div className="ps">
                  <h4>🌫 模糊设置</h4>
                  <div className="mg">
                    {BLUR_MODES.map((m) => (
                      <button key={m.id} className={`mb ${blurMode === m.id ? "on" : ""}`} onClick={() => setBlurMode(m.id)}>
                        <span className="mi">{m.icon}</span>{m.label}
                      </button>
                    ))}
                  </div>
                  <div style={{ marginTop: 16 }}>
                    <div className="sg"><div className="sl"><span>模糊强度</span><span>{blurStrength}</span></div>
                      <input type="range" min="5" max="80" value={blurStrength} onChange={(e) => setBlurStrength(Number(e.target.value))} /></div>
                    <div className="sg"><div className="sl"><span>模糊范围</span><span>{Math.round(expandRatio * 100)}%</span></div>
                      <input type="range" min="0" max="80" value={expandRatio * 100} onChange={(e) => setExpandRatio(Number(e.target.value) / 100)} /></div>
                  </div>
                </div>

                <div className="ps">
                  <h4>✂️ 抠脸设置</h4>
                  <div className="sg"><div className="sl"><span>裁剪范围</span><span>{Math.round(cutoutExpand * 100)}%</span></div>
                    <input type="range" min="20" max="100" value={cutoutExpand * 100} onChange={(e) => setCutoutExpand(Number(e.target.value) / 100)} /></div>
                  <div className="sl" style={{ marginBottom: 8 }}><span>背景</span></div>
                  <div className="bgt">
                    <button className={`bo ${cutoutBg === "white" ? "on" : ""}`} onClick={() => setCutoutBg("white")}>白色背景</button>
                    <button className={`bo ${cutoutBg === "transparent" ? "on" : ""}`} onClick={() => setCutoutBg("transparent")}>透明背景</button>
                  </div>
                </div>

                <div className="ps">
                  <h4>检测到的人脸 ({faces.filter((f) => f.enabled).length}/{faces.length})</h4>
                  {hasFaces ? (
                    <div className="fl">
                      {faces.map((f, i) => (
                        <div key={i} className="fi" onClick={() => toggleFace(i)}>
                          <div className={`fc ${f.enabled ? "ck" : ""}`}>{f.enabled ? "✓" : ""}</div>
                          <span>人脸 #{i + 1}</span>
                          <span style={{ marginLeft: "auto", fontSize: "0.7rem", color: "var(--text-2)", fontFamily: "'JetBrains Mono',monospace" }}>{Math.round(f.w)}×{Math.round(f.h)}</span>
                        </div>
                      ))}
                    </div>
                  ) : <div className="nf">{status === "detecting" ? "检测中..." : "未检测到人脸"}</div>}
                </div>

                {errorMsg && <div className="em">{errorMsg}</div>}

                <div className="bts">
                  <div className="bt">
                    <button className="bn b1" onClick={downloadBlurred} disabled={status !== "done" || !hasEnabled}>下载模糊图</button>
                    <button className="bn b2" onClick={downloadAllCutouts} disabled={cutoutUrls.length === 0}>下载全部抠脸</button>
                  </div>
                  <button className="bn bg" onClick={reset} style={{ flex: "none" }}>重选图片</button>
                </div>
              </div>
            </div>
          </div>
        )}

        <canvas ref={origCanvasRef} style={{ display: "none" }} />
        <div className="ft">纯前端处理 · 图片不会离开你的浏览器 · Powered by face-api.js + Next.js</div>
      </div>
    </>
  );
}
