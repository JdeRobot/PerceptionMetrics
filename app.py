import streamlit as st

from perceptionmetrics.utils.torch import get_device_info
from gui.dataset_viewer import dataset_viewer_tab
from gui.evaluator import evaluator_tab
from gui.inference import inference_tab
from gui.tasks.image_detection.sidebar import render_image_detection_sidebar
from gui.tasks.image_segmentation.sidebar import render_image_segmentation_sidebar
from gui.tasks.lidar_segmentation.sidebar import render_lidar_segmentation_sidebar
st.set_page_config(page_title="PerceptionMetrics", layout="wide")

PAGES = {
    "Dataset Viewer": dataset_viewer_tab,
    "Inference": inference_tab,
    "Evaluator": evaluator_tab,
}

best_device, available_devices = get_device_info()

# Shared state
st.session_state.setdefault("task", "Image Detection")
st.session_state.setdefault("dataset_path", "")
st.session_state.setdefault("split", "test")
st.session_state.setdefault("device", best_device)

# Image detection state
st.session_state.setdefault("dataset_type", "YOLO")
st.session_state.setdefault("config_option", "Manual Configuration")
st.session_state.setdefault("confidence_threshold", 0.5)
st.session_state.setdefault("nms_threshold", 0.5)
st.session_state.setdefault("max_detections", -1)
st.session_state.setdefault("batch_size", 1)
st.session_state.setdefault("evaluation_step", 5)
st.session_state.setdefault("detection_model", None)
st.session_state.setdefault("detection_model_loaded", False)

# Image segmentation state
st.session_state.setdefault("segmentation_model", None)
st.session_state.setdefault("segmentation_model_loaded", False)
st.session_state.setdefault("segmentation_model_type", "Torch Model File")
st.session_state.setdefault("segmentation_model_path", "")
st.session_state.setdefault("segmentation_config_path", "")
st.session_state.setdefault("segmentation_ontology_path", "")

with st.sidebar:
    task = st.selectbox(
        "Task",
        ["Image Detection", "Image Segmentation", "Lidar Segmentation"],
        key="task",
        help="",
    )

    if task == "Image Detection":
        render_image_detection_sidebar(available_devices)
    elif task == "Image Segmentation":
        render_image_segmentation_sidebar(available_devices)
    elif task == "Lidar Segmentation":
        render_lidar_segmentation_sidebar(available_devices)
    else:
        st.error(f"Unsupported task: {task}")
        
    


tab1, tab2, tab3 = st.tabs(["Dataset Viewer", "Inference", "Evaluator"])

with tab1:
    dataset_viewer_tab()
with tab2:
    inference_tab()
with tab3:
    evaluator_tab()
