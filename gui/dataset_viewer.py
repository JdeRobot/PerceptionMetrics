import streamlit as st
from gui.tasks.image_detection.dataset_viewer import render_image_detection_viewer
from gui.tasks.image_segmentation.dataset_viewer import (
    render_image_segmentation_viewer,
)
from gui.tasks.lidar_segmentation.dataset_viewer import (
    render_lidar_segmentation_viewer,
)


def dataset_viewer_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        render_image_detection_viewer()
        return

    if task == "Image Segmentation":
        render_image_segmentation_viewer()
        return
    if task == "Lidar Segmentation":
        render_lidar_segmentation_viewer()
        return

    st.error(f"Unsupported task for dataset viewer: {task}")
