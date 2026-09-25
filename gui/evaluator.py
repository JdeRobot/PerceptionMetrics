import streamlit as st
from gui.tasks.image_detection.evaluator import render_image_detection_evaluator
from gui.tasks.image_segmentation.evaluator import render_image_segmentation_evaluator
from gui.tasks.lidar_segmentation.evaluator import render_lidar_segmentation_evaluator


def evaluator_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        render_image_detection_evaluator()
        return

    if task == "Image Segmentation":
        render_image_segmentation_evaluator()
        return

    if task == "Lidar Segmentation":
        render_lidar_segmentation_evaluator()
        return

    st.error(f"Unsupported task for evaluator: {task}")
