"""
Streamlit dashboard for real-time video analytics.
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from analytics import AnalyticsCollector
from pipeline import VideoPipeline


def main():
    st.set_page_config(
        page_title="Video Analytics Dashboard",
        page_icon="📹",
        layout="wide",
    )

    st.title("📹 Real-time Video Analytics Dashboard")
    st.markdown("Face detection · Motion tracking · Live analytics")

    # Sidebar
    with st.sidebar:
        st.subheader("Controls")

        source_type = st.selectbox(
            "Input Source",
            ["Webcam", "Video File", "RTSP Stream"],
        )
        if source_type == "Webcam":
            source = st.number_input("Camera ID", 0, 10, 0)
        elif source_type == "Video File":
            uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
            source = None
            if uploaded:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp.write(uploaded.read())
                source = temp.name
        else:
            source = st.text_input("RTSP URL", "rtsp://")

        st.divider()

        enable_face = st.checkbox("Face Detection", value=True)
        face_method = st.selectbox("Face Detector", ["haar", "dnn"]) if enable_face else "haar"

        enable_motion = st.checkbox("Motion Detection", value=True)
        motion_threshold = st.slider("Motion Threshold", 5, 100, 25) if enable_motion else 25
        motion_min_area = st.slider("Min Motion Area", 100, 5000, 500) if enable_motion else 500

        enable_tracking = st.checkbox("Object Tracking", value=False)

        st.divider()

        resize_width = st.select_slider(
            "Processing Resolution",
            options=[320, 480, 640, 960, 1280],
            value=640,
        )

    # Main area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Live Feed", "Analytics", "Heatmap", "Events"]
    )

    live_placeholder = tab1.empty()
    chart_col1, chart_col2 = tab2.columns(2)
    heatmap_placeholder = tab3.empty()
    events_placeholder = tab4.empty()

    # Initialize pipeline
    if source is not None and "pipeline" not in st.session_state:
        st.session_state.pipeline = VideoPipeline(
            source=source,
            enable_face_detection=enable_face,
            enable_motion_detection=enable_motion,
            enable_tracking=enable_tracking,
            face_method=face_method,
            motion_threshold=motion_threshold,
            motion_min_area=motion_min_area,
            resize_width=resize_width,
        )
        st.session_state.analytics = AnalyticsCollector()
        st.session_state.stream = st.session_state.pipeline.stream()

    # Control buttons
    col_start, col_stop = st.sidebar.columns(2)
    with col_start:
        if st.button("Start", type="primary", use_container_width=True):
            if source is not None:
                st.session_state.pipeline = VideoPipeline(
                    source=source,
                    enable_face_detection=enable_face,
                    enable_motion_detection=enable_motion,
                    enable_tracking=enable_tracking,
                    face_method=face_method,
                    motion_threshold=motion_threshold,
                    motion_min_area=motion_min_area,
                    resize_width=resize_width,
                )
                st.session_state.analytics = AnalyticsCollector()
                st.session_state.stream = st.session_state.pipeline.stream()
                st.rerun()
    with col_stop:
        if st.button("Stop", use_container_width=True):
            if "pipeline" in st.session_state:
                st.session_state.pipeline.stop()
                del st.session_state.pipeline
                del st.session_state.stream
                st.rerun()

    # Render loop
    if "stream" in st.session_state and st.session_state.get("pipeline"):
        try:
            result = next(st.session_state.stream)

            # Tab 1: Live Feed
            live_placeholder.image(
                result.annotated, channels="BGR",
                use_column_width=True
            )

            # Update analytics
            st.session_state.analytics.update(result.to_dict())
            summary = st.session_state.analytics.summary()

            # Tab 2: Charts
            history = list(st.session_state.pipeline.stats.history)
            if history:
                df = pd.DataFrame(history)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

                with chart_col1:
                    fig_faces = px.line(
                        df, x="timestamp", y="face_count",
                        title="Faces Detected Over Time",
                        color_discrete_sequence=["#38bdf8"],
                    )
                    fig_faces.update_layout(
                        height=300, margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_faces, use_container_width=True)

                with chart_col2:
                    fig_motion = px.line(
                        df, x="timestamp", y="motion_level",
                        title="Motion Level Over Time",
                        color_discrete_sequence=["#4ade80"],
                    )
                    fig_motion.update_layout(
                        height=300, margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_motion, use_container_width=True)

                # FPS gauge
                fig_fps = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=summary["fps"]["current"],
                    title={"text": "FPS"},
                    gauge={"axis": {"range": [0, 30]},
                           "bar": {"color": "#38bdf8"}},
                ))
                fig_fps.update_layout(
                    height=200, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_fps, use_container_width=True)

            # Tab 3: Heatmap
            heatmap = st.session_state.pipeline.get_heatmap_image()
            if heatmap is not None:
                heatmap_placeholder.image(
                    heatmap, channels="BGR",
                    use_column_width=True,
                    caption="Motion Heatmap (cumulative)",
                )

            # Tab 4: Events
            events = list(st.session_state.analytics.events)
            if events:
                events_df = pd.DataFrame([
                    {"time": e.timestamp, "type": e.event_type,
                     "details": str(e.details)}
                    for e in reversed(events[-50:])
                ])
                events_placeholder.dataframe(events_df, use_container_width=True)
            else:
                events_placeholder.info("No events logged yet")

            # Auto-rerun for live feed
            time.sleep(0.05)
            st.rerun()

        except StopIteration:
            st.info("Video stream ended.")
            if "pipeline" in st.session_state:
                st.session_state.pipeline.close()
                del st.session_state.pipeline
                del st.session_state.stream
        except Exception as e:
            st.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    main()
