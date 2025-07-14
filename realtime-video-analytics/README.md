# Real-time Video Analytics Dashboard
## Face detection, motion tracking & activity monitoring

A real-time computer vision dashboard built with OpenCV and Streamlit. Process webcam feeds or video files with face detection, motion analysis, and activity statistics — all in a live updating dashboard.

## Features

- **Face Detection**: Haar Cascade + optional DNN face detector
- **Motion Detection**: Frame differencing with adaptive thresholding
- **Object Tracking**: Centroid-based tracker with unique IDs
- **Activity Heatmap**: Visualize motion density over time
- **Real-time Charts**: Objects detected, motion level over time
- **Event Logging**: Save timestamped detection events
- **Video Recording**: Record processed streams with annotations
- **Multiple Input Sources**: Webcam, video file, RTSP stream

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Live Feed** | Processed video with bounding box overlays |
| **Analytics** | Charts: object count, motion index, FPS |
| **Heatmap** | Cumulative motion density overlay |
| **Events** | Timestamped detection log with thumbnails |
| **Settings** | Detection thresholds, input source, recording |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure

```
realtime-video-analytics/
├── src/
│   ├── app.py                # Streamlit dashboard
│   ├── detector.py           # Face + motion detection
│   ├── tracker.py            # Centroid-based object tracking
│   ├── pipeline.py           # Video processing pipeline
│   └── analytics.py          # Metrics + statistics
├── tests/
│   └── test_detector.py
├── notebooks/
│   └── benchmark_pipeline.ipynb
├── assets/
├── requirements.txt
└── README.md
```
