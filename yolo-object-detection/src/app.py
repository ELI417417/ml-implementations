"""
Flask web application for YOLO object detection.

Serves a drag-and-drop frontend and a JSON API endpoint.
"""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from detector import YOLODetector

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.onnx"
detector: YOLODetector | None = None


def get_detector() -> YOLODetector:
    global detector
    if detector is None:
        detector = YOLODetector(str(MODEL_PATH))
    return detector


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO Object Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 2rem; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #38bdf8; }
  .subtitle { color: #94a3b8; margin-bottom: 2rem; }
  .upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 3rem 2rem;
                 text-align: center; cursor: pointer; transition: border-color .2s; margin-bottom: 1rem; }
  .upload-zone:hover, .upload-zone.drag { border-color: #38bdf8; background: #1e293b; }
  .upload-zone input { display: none; }
  .controls { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center; }
  .controls label { font-size: 0.85rem; color: #94a3b8; }
  .controls input[type=range] { width: 120px; }
  button { background: #38bdf8; color: #0f172a; border: none; padding: 0.6rem 1.5rem;
           border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.9rem; }
  button:hover { background: #7dd3fc; }
  #result { position: relative; display: inline-block; max-width: 100%; margin-top: 1rem; }
  #result img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; }
  #stats { margin-top: 0.75rem; font-size: 0.85rem; color: #94a3b8; }
  .detection-chip { display: inline-block; background: #1e293b; border: 1px solid #334155;
                    border-radius: 20px; padding: 0.25rem 0.75rem; margin: 0.2rem; font-size: 0.8rem; }
  .detection-chip .cls { color: #38bdf8; font-weight: 600; }
  .detection-chip .conf { color: #94a3b8; }
  .spinner { display: none; text-align: center; padding: 2rem; }
</style>
</head>
<body>
<div class="container">
  <h1>YOLO Object Detection</h1>
  <p class="subtitle">Upload an image — detect 80 COCO classes with YOLOv8</p>

  <div class="controls">
    <label>Confidence: <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.25">
      <span id="confVal">0.25</span></label>
    <label>IoU: <input type="range" id="iou" min="0.1" max="0.9" step="0.05" value="0.45">
      <span id="iouVal">0.45</span></label>
  </div>

  <div class="upload-zone" id="dropZone">
    <p>Drop an image here or click to browse</p>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div class="spinner" id="spinner">Detecting...</div>
  <div id="result"></div>
  <div id="stats"></div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const confSlider = document.getElementById('conf');
const iouSlider = document.getElementById('iou');
const confVal = document.getElementById('confVal');
const iouVal = document.getElementById('iouVal');

confSlider.oninput = () => confVal.textContent = confSlider.value;
iouSlider.oninput = () => iouVal.textContent = iouSlider.value;

dropZone.onclick = () => fileInput.click();
dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add('drag'); };
dropZone.ondragleave = () => dropZone.classList.remove('drag');
dropZone.ondrop = e => { e.preventDefault(); dropZone.classList.remove('drag');
                         handleFile(e.dataTransfer.files[0]); };
fileInput.onchange = e => handleFile(e.target.files[0]);

async function handleFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append('image', file);
  form.append('conf', confSlider.value);
  form.append('iou', iouSlider.value);

  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result').innerHTML = '';
  document.getElementById('stats').innerHTML = '';

  try {
    const resp = await fetch('/api/detect', { method: 'POST', body: form });
    const data = await resp.json();
    document.getElementById('result').innerHTML =
      `<img src="data:image/jpeg;base64,${data.image}" alt="detection result">`;
    let stats = `<p>Found <b>${data.count}</b> object(s)</p>`;
    data.detections.forEach(d => {
      stats += `<span class="detection-chip">
        <span class="cls">${d.class_name}</span>
        <span class="conf">${(d.confidence*100).toFixed(0)}%</span>
      </span>`;
    });
    document.getElementById('stats').innerHTML = stats;
  } catch(e) {
    document.getElementById('stats').innerHTML =
      `<p style="color:#f87171">Error: ${e.message}</p>`;
  } finally {
    document.getElementById('spinner').style.display = 'none';
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))

    # Read image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run detection
    det = get_detector()
    det.conf_threshold = conf
    det.iou_threshold = iou
    detections = det.detect(image)

    # Draw bounding boxes
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (56, 189, 248), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)

    # Encode result
    _, buffer = cv2.imencode(".jpg", image)
    img_b64 = base64.b64encode(buffer).decode()

    return jsonify({
        "count": len(detections),
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox),
            }
            for d in detections
        ],
        "image": img_b64,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
