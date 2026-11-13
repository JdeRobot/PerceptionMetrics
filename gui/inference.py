import streamlit as st
from gui.tasks.image_detection.inference import render_image_detection_inference
from gui.tasks.image_segmentation.inference import render_image_segmentation_inference
from gui.tasks.lidar_segmentation.inference import render_lidar_segmentation_inference


def inference_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        render_image_detection_inference()
        return

    if task == "Image Segmentation":
        render_image_segmentation_inference()
        return

    if task == "Lidar Segmentation":
        render_lidar_segmentation_inference()
        return

    st.error(f"Unsupported task for inference: {task}")
