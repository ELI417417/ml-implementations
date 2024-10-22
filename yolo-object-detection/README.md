# YOLO Object Detection Web App
## Real-time object detection with a Flask + ONNX backend

A web-based object detection app using a YOLOv8 model exported to ONNX. Upload an image or use your webcam, and get bounding box predictions with confidence scores in real-time.

## Features

- Upload images for detection via drag-and-drop web UI
- Real-time webcam detection with live bounding boxes
- YOLOv8 ONNX model for fast CPU/GPU inference
- Non-maximum suppression (NMS) implemented from scratch
- COCO class labels (80 classes)
- Adjustable confidence threshold and IoU threshold
- REST API endpoint for programmatic access

## Architecture

```
Browser (HTML/JS) → Flask API → ONNX Runtime → YOLOv8 → NMS → JSON(bboxes, classes, scores)
```

## Quick Start

```bash
pip install -r requirements.txt
python src/download_model.py          # downloads YOLOv8n ONNX
python src/app.py                      # starts Flask server at :5000
```

Open http://localhost:5000 and upload an image or click "Webcam".

## API

```
POST /api/detect
Content-Type: multipart/form-data
Body: image=<file>, conf=<float>, iou=<float>

Response:
{
  "detections": [
    {"class": "person", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    ...
  ],
  "count": 3
}
```

## Project Structure

```
yolo-object-detection/
├── src/
│   ├── app.py              # Flask server + frontend
│   ├── detector.py         # YOLO inference + post-processing
│   ├── download_model.py   # Download YOLOv8 ONNX from Ultralytics
│   └── utils.py            # NMS, image processing helpers
├── tests/
│   └── test_detector.py
├── notebooks/
│   └── benchmark.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- ONNX Runtime: https://onnxruntime.ai/
