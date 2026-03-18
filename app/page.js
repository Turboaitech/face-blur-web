"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";
const SEG_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const SEG_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const L = {
  en: {
    title: "Face Blur & Extract",
    sub: "Upload an image — faces are auto-detected, blurred, and extracted (face only, white background). 100% browser-side, nothing uploaded.",
    loading: "Loading models…", ready: "Models ready", wait: "Initializing…",
    upload: "Click to select image, or drag & drop here",
    hint: "JPG / PNG / WebP", paste: "paste screenshot",
    blurred: "Blurred", isolated: "Face Only (White BG)",
    blurSettings: "Blur Settings", gaussian: "Gaussian", mosaic: "Mosaic", blackout: "Blackout",
    strength: "Strength", range: "Range",
    isoSettings: "Isolation Settings",
    threshold: "Edge Threshold",
    thresholdHelp: "Higher = tighter cut, Lower = more included",
    isoExpand: "Region Expand",
    isoExpandHelp: "How far beyond face box to include (hair, chin)",
    morphClose: "Edge Smoothing",
    faces: "Detected Faces", face: "Face", detecting: "Processing…",
    noFace: "No faces detected", noFaceHint: "Try a clearer photo",
    dlBlur: "Download Blurred", dlIso: "Download Face", dlBoth: "Download Both",
    reselect: "Choose Another Image", settings: "Settings",
    footer: "100% client-side · Images never leave your browser",
  },
  zh: {
    title: "人脸模糊 & 人脸提取",
    sub: "上传图片 → 自动检测 → 模糊处理 + 仅提取人脸（白色背景）。全程浏览器处理，图片不上传。",
    loading: "模型加载中…", ready: "模型就绪", wait: "初始化中…",
    upload: "点击选择图片，或拖拽到此处",
    hint: "JPG / PNG / WebP", paste: "粘贴截图",
    blurred: "模糊版", isolated: "仅人脸（白底）",
    blurSettings: "模糊设置", gaussian: "高斯模糊", mosaic: "像素马赛克", blackout: "纯黑遮挡",
    strength: "模糊强度", range: "模糊范围",
    isoSettings: "抠图设置",
    threshold: "边缘阈值",
    thresholdHelp: "越高越紧贴轮廓，越低包含越多",
    isoExpand: "区域扩展",
    isoExpandHelp: "超出人脸框多远（包含头发、下巴）",
    morphClose: "边缘平滑",
    faces: "检测到的人脸", face: "人脸", detecting: "处理中…",
    noFace: "未检测到人脸", noFaceHint: "可尝试换一张更清晰的照片",
    dlBlur: "下载模糊版", dlIso: "下载人脸", dlBoth: "全部下载",
    reselect: "重选图片", settings: "设置",
    footer: "纯前端处理 · 图片不会离开你的浏览器",
  },
};

export default function App() {
  const [lang, setLang] = useState("en");
  const t = L[lang];

  const [status, setStatus] = useState("idle");
  const [image, setImage] = useState(null);
  const [faces, setFaces] = useState([]);
  const [segMask, setSegMask] = useState(null); // Float32Array of confidence per pixel
  const [blurMode, setBlurMode] = useState("gaussian");
  const [blurStrength, setBlurStrength] = useState(40);
  const [blurExpand, setBlurExpand] = useState(0.3);
  const [isoThreshold, setIsoThreshold] = useState(0.5);
  const [isoExpand, setIsoExpand] = useState(0.35);
  const [morphIter, setMorphIter] = useState(1);
  const [errorMsg, setErrorMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [faceApi, setFaceApi] = useState(null);
  const [segmenter, setSegmenter] = useState(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  const blurCanvasRef = useRef(null);
  const isoCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const imgRef = useRef(null);
  const imgDimsRef = useRef({ w: 0, h: 0 });

  const MODES = [
    { id: "gaussian", label: t.gaussian, icon: "◎" },
    { id: "mosaic", label: t.mosaic, icon: "▦" },
    { id: "black", label: t.blackout, icon: "■" },
  ];

  // ── Load both models ───────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (modelsLoaded) return;
    setStatus("loading-model");
    try {
      // face-api for face detection (blur rectangles)
      const fa = await import("face-api.js");
      await fa.nets.ssdMobilenetv1.loadFromUri(FACE_MODEL_URL);
      setFaceApi(fa);

      // MediaPipe Selfie Segmenter for pixel-accurate person segmentation
      const vision = await import("@mediapipe/tasks-vision");
      const { ImageSegmenter, FilesetResolver } = vision;
      const wasmFileset = await FilesetResolver.forVisionTasks(SEG_WASM_URL);
      const seg = await ImageSegmenter.createFromOptions(wasmFileset, {
        baseOptions: { modelAssetPath: SEG_MODEL_URL, delegate: "GPU" },
        runningMode: "IMAGE",
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      });
      setSegmenter(seg);

      setModelsLoaded(true);
      setStatus("ready");
    } catch (e) {
      console.error(e);
      setErrorMsg("Model load failed: " + e.message);
      setStatus("error");
    }
  }, [modelsLoaded]);

  useEffect(() => { loadModels(); }, [loadModels]);

  // ── Handle image ───────────────────────────────────────────────────────────
  const handleImage = useCallback(async (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setFaces([]); setSegMask(null); setStatus("detecting"); setErrorMsg("");
    const url = URL.createObjectURL(file); setImage(url);

    const img = new Image();
    img.onload = async () => {
      imgRef.current = img;
      imgDimsRef.current = { w: img.width, h: img.height };
      [blurCanvasRef, isoCanvasRef].forEach(r => { r.current.width = img.width; r.current.height = img.height; });
      blurCanvasRef.current.getContext("2d").drawImage(img, 0, 0);

      if (!faceApi || !segmenter) { setErrorMsg("Models not ready yet"); setStatus("error"); return; }

      try {
        // 1. Face detection for blur
        const det = await faceApi.detectAllFaces(img, new faceApi.SsdMobilenetv1Options({ minConfidence: 0.3 }));
        const boxes = det.map(d => ({ x: d.box.x, y: d.box.y, w: d.box.width, h: d.box.height, enabled: true }));
        setFaces(boxes);
        if (!boxes.length) setErrorMsg(t.noFaceHint);

        // 2. Person segmentation for isolation
        const result = segmenter.segment(img);
        if (result.confidenceMasks && result.confidenceMasks.length > 0) {
          const maskData = result.confidenceMasks[0].getAsFloat32Array();
          // Store a copy since MediaPipe may reclaim the buffer
          setSegMask(new Float32Array(maskData));
        }

        setStatus("done");
      } catch (e) { console.error(e); setErrorMsg(e.message); setStatus("error"); }
    };
    img.src = url;
  }, [faceApi, segmenter, t.noFaceHint]);

  // ── Render blur canvas ─────────────────────────────────────────────────────
  const renderBlur = useCallback((img, canvas, faceBoxes, mode, strength, expand) => {
    if (!img || !canvas) return;
    const ctx = canvas.getContext("2d"); ctx.drawImage(img, 0, 0);
    for (const f of faceBoxes.filter(f => f.enabled)) {
      const ex = f.w * expand, ey = f.h * expand;
      const fx = Math.max(0, f.x - ex), fy = Math.max(0, f.y - ey);
      const fw = Math.min(img.width - fx, f.w + ex * 2), fh = Math.min(img.height - fy, f.h + ey * 2);
      if (mode === "gaussian") {
        ctx.save(); ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2); ctx.clip();
        for (let i = 0; i < Math.ceil(strength / 5); i++) {
          ctx.filter = `blur(${Math.min(strength, 50)}px)`;
          ctx.drawImage(canvas, fx - 50, fy - 50, fw + 100, fh + 100, fx - 50, fy - 50, fw + 100, fh + 100);
        }
        ctx.filter = "none"; ctx.restore();
      } else if (mode === "mosaic") {
        const bs = Math.max(5, Math.ceil(strength / 3));
        const tc = document.createElement("canvas"); tc.width = fw; tc.height = fh;
        tc.getContext("2d").drawImage(canvas, fx, fy, fw, fh, 0, 0, fw, fh);
        const sw = Math.max(1, Math.floor(fw / bs)), sh = Math.max(1, Math.floor(fh / bs));
        const sc = document.createElement("canvas"); sc.width = sw; sc.height = sh;
        const sctx = sc.getContext("2d"); sctx.imageSmoothingEnabled = false; sctx.drawImage(tc, 0, 0, sw, sh);
        ctx.save(); ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2); ctx.clip();
        ctx.imageSmoothingEnabled = false; ctx.drawImage(sc, 0, 0, sw, sh, fx, fy, fw, fh);
        ctx.imageSmoothingEnabled = true; ctx.restore();
      } else {
        ctx.save(); ctx.fillStyle = "#000"; ctx.beginPath();
        ctx.ellipse(fx + fw / 2, fy + fh / 2, fw / 2, fh / 2, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore();
      }
    }
  }, []);

  // ── Render isolate canvas (pixel-accurate segmentation) ────────────────────
  const renderIsolate = useCallback((img, canvas, mask, faceBoxes, threshold, expand, morphIterations) => {
    if (!img || !canvas || !mask) return;
    const W = img.width, H = img.height;
    const ctx = canvas.getContext("2d");

    // 1. Get original pixel data
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = W; srcCanvas.height = H;
    const srcCtx = srcCanvas.getContext("2d");
    srcCtx.drawImage(img, 0, 0);
    const srcData = srcCtx.getImageData(0, 0, W, H);

    // 2. Build a region mask from enabled face bounding boxes (expanded)
    // Only pixels within these regions will be considered for isolation
    const enabledFaces = faceBoxes.filter(f => f.enabled);
    const regionMask = new Uint8Array(W * H);
    for (const f of enabledFaces) {
      // Expand the face box to include hair and chin — face only, no body
      const ex = f.w * expand;
      const ey = f.h * expand;
      const x0 = Math.max(0, Math.floor(f.x - ex * 0.8));
      const y0 = Math.max(0, Math.floor(f.y - ey * 1.0)); // above for hair/forehead
      const x1 = Math.min(W, Math.ceil(f.x + f.w + ex * 0.8));
      const y1 = Math.min(H, Math.ceil(f.y + f.h + ey * 0.3)); // just chin, no neck/body
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          regionMask[y * W + x] = 1;
        }
      }
    }

    // 3. Create binary mask: segmentation confidence AND within face region
    const binaryMask = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++) {
      binaryMask[i] = (mask[i] >= threshold && regionMask[i]) ? 1 : 0;
    }

    // 4. Morphological close (dilate then erode) to fill small holes
    const morphClose = (input, w, h, iterations) => {
      let current = new Uint8Array(input);
      for (let iter = 0; iter < iterations; iter++) {
        const dilated = new Uint8Array(w * h);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            let val = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                  if (current[ny * w + nx]) { val = 1; break; }
                }
              }
              if (val) break;
            }
            dilated[y * w + x] = val;
          }
        }
        const eroded = new Uint8Array(w * h);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            let val = 1;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                  if (!dilated[ny * w + nx]) { val = 0; break; }
                } else { val = 0; break; }
              }
              if (!val) break;
            }
            eroded[y * w + x] = val;
          }
        }
        current = eroded;
      }
      return current;
    };

    const cleanMask = morphIterations > 0
      ? morphClose(binaryMask, W, H, morphIterations)
      : binaryMask;

    // 5. Apply mask: keep original pixels where mask=1, pure #FFFFFF everywhere else
    // Hard binary cutoff — no feathering, no blending
    const outData = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const pi = i * 4;
      if (cleanMask[i]) {
        outData.data[pi]     = srcData.data[pi];
        outData.data[pi + 1] = srcData.data[pi + 1];
        outData.data[pi + 2] = srcData.data[pi + 2];
        outData.data[pi + 3] = 255;
      } else {
        outData.data[pi]     = 255;
        outData.data[pi + 1] = 255;
        outData.data[pi + 2] = 255;
        outData.data[pi + 3] = 255;
      }
    }
    ctx.putImageData(outData, 0, 0);
  }, []);

  // ── Re-render on changes ───────────────────────────────────────────────────
  useEffect(() => {
    if (status === "done" && imgRef.current) {
      renderBlur(imgRef.current, blurCanvasRef.current, faces, blurMode, blurStrength, blurExpand);
      renderIsolate(imgRef.current, isoCanvasRef.current, segMask, faces, isoThreshold, isoExpand, morphIter);
    }
  }, [status, faces, segMask, blurMode, blurStrength, blurExpand, isoThreshold, isoExpand, morphIter, renderBlur, renderIsolate]);

  // ── Actions ────────────────────────────────────────────────────────────────
  const toggleFace = (i) => setFaces(p => p.map((f, j) => j === i ? { ...f, enabled: !f.enabled } : f));
  const dl = (c, name) => { if (!c) return; const a = document.createElement("a"); a.download = name; a.href = c.toDataURL("image/png"); a.click(); };
  const dlBoth = () => { dl(blurCanvasRef.current, "blurred.png"); setTimeout(() => dl(isoCanvasRef.current, "isolated.png"), 300); };
  const reset = () => { setImage(null); setFaces([]); setSegMask(null); setStatus(modelsLoaded ? "ready" : "idle"); setErrorMsg(""); setPanelOpen(false); };

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleImage(e.dataTransfer.files[0]); };
  useEffect(() => {
    const fn = (e) => { const items = e.clipboardData?.items; if (!items) return; for (const it of items) { if (it.type.startsWith("image/")) { handleImage(it.getAsFile()); break; } } };
    window.addEventListener("paste", fn); return () => window.removeEventListener("paste", fn);
  }, [handleImage]);

  const hasEnabled = faces.some(f => f.enabled);

  const panel = (
    <div className="pn">
      <div className="ps">
        <h4>🌫 {t.blurSettings}</h4>
        <div className="mg">
          {MODES.map(m => (
            <button key={m.id} className={`mb ${blurMode === m.id ? "on" : ""}`} onClick={() => setBlurMode(m.id)}>
              <span className="mi">{m.icon}</span>{m.label}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 14 }}>
          <div className="sg"><div className="sl"><span>{t.strength}</span><span>{blurStrength}</span></div>
            <input type="range" min="5" max="80" value={blurStrength} onChange={e => setBlurStrength(+e.target.value)} /></div>
          <div className="sg"><div className="sl"><span>{t.range}</span><span>{Math.round(blurExpand * 100)}%</span></div>
            <input type="range" min="0" max="80" value={blurExpand * 100} onChange={e => setBlurExpand(+e.target.value / 100)} /></div>
        </div>
      </div>
      <div className="ps">
        <h4>✂️ {t.isoSettings}</h4>
        <div className="sg"><div className="sl"><span>{t.threshold}</span><span>{isoThreshold.toFixed(2)}</span></div>
          <input type="range" min="10" max="90" value={isoThreshold * 100} onChange={e => setIsoThreshold(+e.target.value / 100)} />
          <div className="sh">{t.thresholdHelp}</div></div>
        <div className="sg"><div className="sl"><span>{t.isoExpand}</span><span>{Math.round(isoExpand * 100)}%</span></div>
          <input type="range" min="20" max="150" value={isoExpand * 100} onChange={e => setIsoExpand(+e.target.value / 100)} />
          <div className="sh">{t.isoExpandHelp}</div></div>
        <div className="sg"><div className="sl"><span>{t.morphClose}</span><span>{morphIter}</span></div>
          <input type="range" min="0" max="5" value={morphIter} onChange={e => setMorphIter(+e.target.value)} /></div>
      </div>
      <div className="ps">
        <h4>{t.faces} ({faces.filter(f => f.enabled).length}/{faces.length})</h4>
        {faces.length > 0 ? (
          <div className="fl">
            {faces.map((f, i) => (
              <div key={i} className="fi" onClick={() => toggleFace(i)}>
                <div className={`fc ${f.enabled ? "ck" : ""}`}>{f.enabled ? "✓" : ""}</div>
                <span>{t.face} #{i + 1}</span>
                <span className="fi-sz">{Math.round(f.w)}×{Math.round(f.h)}</span>
              </div>
            ))}
          </div>
        ) : <div className="nf">{status === "detecting" ? t.detecting : t.noFace}</div>}
      </div>
      {errorMsg && <div className="em">{errorMsg}</div>}
      <div className="bts">
        <div className="bt">
          <button className="bn b1" onClick={() => dl(blurCanvasRef.current, "blurred.png")} disabled={status !== "done" || !hasEnabled}>{t.dlBlur}</button>
          <button className="bn b2" onClick={() => dl(isoCanvasRef.current, "isolated.png")} disabled={status !== "done" || !segMask}>{t.dlIso}</button>
        </div>
        <button className="bn b3" onClick={dlBoth} disabled={status !== "done"}>{t.dlBoth}</button>
        <button className="bn bg0" onClick={reset}>{t.reselect}</button>
      </div>
    </div>
  );

  return (
    <>
      <style jsx global>{`
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#08080a;--s1:#111114;--s2:#1a1a1f;--s3:#222228;
  --bd:#2a2a32;--bdh:#3a3a44;
  --t1:#ededf0;--t2:#82829a;--t3:#5a5a70;
  --ac:#6e6ef0;--acg:rgba(110,110,240,.12);
  --pk:#e86cb0;--pkg:rgba(232,108,176,.12);
  --rd:#f06c6c;--gn:#6cf0a0;--yl:#f0d06c;
  --r:12px;--rs:8px;
  --f:'DM Sans','Noto Sans SC',-apple-system,sans-serif;
  --m:'JetBrains Mono',monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--f);background:var(--bg);color:var(--t1);min-height:100vh;-webkit-font-smoothing:antialiased}

.app{max-width:1200px;margin:0 auto;padding:28px 16px 72px}
@media(min-width:768px){.app{padding:44px 28px 80px}}

.lang{position:fixed;top:12px;right:12px;z-index:100;display:flex;
  background:var(--s2);border:1px solid var(--bd);border-radius:20px;padding:2px;font-size:.73rem}
.lang button{padding:4px 12px;border:none;border-radius:18px;background:none;
  color:var(--t2);cursor:pointer;transition:all .2s;font-family:var(--f);font-weight:500}
.lang button.on{background:var(--s3);color:var(--t1)}

.hd{text-align:center;margin-bottom:32px}
@media(min-width:768px){.hd{margin-bottom:48px}}
.hd h1{font-size:1.5rem;font-weight:700;letter-spacing:-.03em;line-height:1.3;
  background:linear-gradient(135deg,var(--t1),var(--ac) 50%,var(--pk));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px}
@media(min-width:768px){.hd h1{font-size:2.3rem}}
.hd p{color:var(--t2);font-size:.82rem;font-weight:300;max-width:560px;margin:0 auto;line-height:1.6}
@media(min-width:768px){.hd p{font-size:.92rem}}
.badge{display:inline-flex;align-items:center;gap:6px;margin-top:10px;padding:3px 12px;
  background:var(--s1);border:1px solid var(--bd);border-radius:100px;font-size:.7rem;color:var(--t2);font-family:var(--m)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--gn);animation:pulse 2s infinite}
.dot.ld{background:var(--yl)}.dot.er{background:var(--rd);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.uz{border:2px dashed var(--bd);border-radius:var(--r);padding:48px 16px;text-align:center;
  cursor:pointer;transition:all .3s;background:var(--s1)}
@media(min-width:768px){.uz{padding:72px 32px}}
.uz:hover,.uz.dg{border-color:var(--ac);background:var(--acg)}
.uz-ic{font-size:2.4rem;margin-bottom:10px;opacity:.5}
@media(min-width:768px){.uz-ic{font-size:3rem}}
.uz h3{font-size:.95rem;font-weight:500;margin-bottom:6px}
@media(min-width:768px){.uz h3{font-size:1.12rem}}
.uz-h{font-size:.75rem;color:var(--t2)}
.uz-h kbd{background:var(--s2);border:1px solid var(--bd);border-radius:4px;padding:1px 5px;font-family:var(--m);font-size:.68rem}

.ws{display:flex;flex-direction:column;gap:16px}
@media(min-width:900px){.ws{display:grid;grid-template-columns:1fr 280px;gap:20px;align-items:start}}

.duo{display:flex;flex-direction:column;gap:12px}
@media(min-width:600px){.duo{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.out{position:relative;background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);overflow:hidden}
.out canvas{display:block;width:100%;height:auto}
.out-label{position:absolute;top:8px;left:8px;background:rgba(8,8,10,.75);backdrop-filter:blur(6px);
  border-radius:6px;padding:3px 10px;font-size:.7rem;font-weight:600;color:var(--t2);pointer-events:none}
.out-label.lb{border-left:2px solid var(--ac)}
.out-label.li{border-left:2px solid var(--pk)}
.ov{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(8,8,10,.85);backdrop-filter:blur(4px)}
.sp{width:32px;height:32px;border:3px solid var(--bd);border-top-color:var(--ac);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.mob-tgl{display:flex;position:sticky;bottom:12px;z-index:50;padding:0 12px;margin-top:8px}
@media(min-width:900px){.mob-tgl{display:none}}
.mob-tgl button{flex:1;padding:11px;border:none;border-radius:var(--r);background:var(--ac);color:#fff;
  font-size:.85rem;font-weight:600;font-family:var(--f);cursor:pointer;box-shadow:0 4px 20px rgba(110,110,240,.3)}
.mob-ov{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.6);backdrop-filter:blur(3px)}
.mob-ov.open{display:flex;flex-direction:column;justify-content:flex-end}
@media(min-width:900px){.mob-ov{display:none!important}}
.mob-in{background:var(--bg);border-top:1px solid var(--bd);border-radius:20px 20px 0 0;
  padding:16px 16px 28px;max-height:80vh;overflow-y:auto}
.mob-bar{width:36px;height:4px;background:var(--bd);border-radius:2px;margin:0 auto 14px}
.dsk{display:none}@media(min-width:900px){.dsk{display:block}}

.pn{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);padding:18px}
.pn h4{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--t2);margin-bottom:12px;font-weight:600}
.ps+.ps{margin-top:16px;padding-top:16px;border-top:1px solid var(--bd)}

.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.mb{display:flex;flex-direction:column;align-items:center;gap:3px;padding:8px 2px;
  background:var(--s2);border:1px solid var(--bd);border-radius:var(--rs);cursor:pointer;
  transition:all .2s;color:var(--t2);font-size:.68rem;font-family:var(--f)}
.mb:hover{border-color:var(--bdh)}.mb.on{border-color:var(--ac);background:var(--acg);color:var(--t1)}
.mi{font-size:1.1rem}

.sg{margin-bottom:12px}
.sl{display:flex;justify-content:space-between;font-size:.75rem;color:var(--t2);margin-bottom:5px}
.sl span:last-child{font-family:var(--m);color:var(--t1);font-size:.72rem}
.sh{font-size:.65rem;color:var(--t3);margin-top:4px}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;border-radius:2px;background:var(--bd);outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;
  background:var(--ac);cursor:pointer;border:2px solid var(--bg)}

.fl{display:flex;flex-direction:column;gap:4px}
.fi{display:flex;align-items:center;gap:8px;padding:6px 8px;background:var(--s2);border-radius:var(--rs);
  font-size:.78rem;cursor:pointer;transition:background .15s}
.fi:hover{background:var(--s3)}
.fi-sz{margin-left:auto;font-size:.65rem;color:var(--t2);font-family:var(--m)}
.fc{width:15px;height:15px;border-radius:3px;border:1.5px solid var(--bd);display:flex;align-items:center;
  justify-content:center;flex-shrink:0;transition:all .15s;font-size:.6rem}
.fc.ck{background:var(--ac);border-color:var(--ac)}
.nf{font-size:.78rem;color:var(--t2);text-align:center;padding:10px}
.em{margin-top:8px;padding:7px 10px;background:rgba(240,108,108,.06);border:1px solid rgba(240,108,108,.15);
  border-radius:var(--rs);font-size:.75rem;color:var(--rd)}

.bts{display:flex;flex-direction:column;gap:6px;margin-top:16px}
.bt{display:flex;gap:6px}
.bn{flex:1;padding:9px 10px;border:none;border-radius:var(--rs);font-size:.8rem;font-weight:600;
  cursor:pointer;transition:all .15s;font-family:var(--f)}
.b1{background:var(--ac);color:#fff}.b1:hover{filter:brightness(1.12)}.b1:disabled{opacity:.3;cursor:not-allowed}
.b2{background:var(--pk);color:#fff}.b2:hover{filter:brightness(1.12)}.b2:disabled{opacity:.3;cursor:not-allowed}
.b3{background:linear-gradient(135deg,var(--ac),var(--pk));color:#fff}.b3:hover{filter:brightness(1.1)}.b3:disabled{opacity:.3;cursor:not-allowed}
.bg0{background:var(--s2);color:var(--t2);border:1px solid var(--bd)}.bg0:hover{border-color:var(--bdh);color:var(--t1)}
.ft{text-align:center;margin-top:44px;font-size:.7rem;color:var(--t3);line-height:1.7}
      `}</style>

      <div className="lang">
        <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button>
        <button className={lang === "zh" ? "on" : ""} onClick={() => setLang("zh")}>中文</button>
      </div>

      <div className="app">
        <header className="hd">
          <h1>{t.title}</h1>
          <p>{t.sub}</p>
          <div className="badge">
            <span className={`dot ${status === "loading-model" ? "ld" : status === "error" ? "er" : ""}`} />
            {status === "loading-model" ? t.loading : modelsLoaded ? t.ready : t.wait}
          </div>
        </header>

        {!image ? (
          <div className={`uz ${dragOver ? "dg" : ""}`} onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
            <div className="uz-ic">📷</div>
            <h3>{t.upload}</h3>
            <p className="uz-h">{t.hint} &nbsp;|&nbsp; <kbd>Ctrl+V</kbd> {t.paste}</p>
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={e => handleImage(e.target.files[0])} />
          </div>
        ) : (
          <>
            <div className="ws">
              <div>
                <div className="duo">
                  <div className="out">
                    <span className="out-label lb">🌫 {t.blurred}</span>
                    <canvas ref={blurCanvasRef} />
                    {status === "detecting" && <div className="ov"><div className="sp" /></div>}
                  </div>
                  <div className="out">
                    <span className="out-label li">✂️ {t.isolated}</span>
                    <canvas ref={isoCanvasRef} />
                    {status === "detecting" && <div className="ov"><div className="sp" /></div>}
                  </div>
                </div>
              </div>
              <div className="dsk">{panel}</div>
            </div>
            <div className="mob-tgl">
              <button onClick={() => setPanelOpen(true)}>⚙️ {t.settings}</button>
            </div>
            <div className={`mob-ov ${panelOpen ? "open" : ""}`} onClick={() => setPanelOpen(false)}>
              <div className="mob-in" onClick={e => e.stopPropagation()}>
                <div className="mob-bar" />{panel}
              </div>
            </div>
          </>
        )}

        <div className="ft">{t.footer}<br />Powered by face-api.js + MediaPipe + Next.js</div>
      </div>
    </>
  );
}
