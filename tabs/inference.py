import streamlit as st
import os
import json
import tempfile
from detectionmetrics.models.torch_detection import TorchImageDetectionModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import supervision as sv

def draw_detections(image, predictions, label_map=None):
    """
    Draw color-coded bounding boxes and labels on the image using supervision.
    Args:
        image: PIL Image
        predictions: dict with 'boxes', 'labels', 'scores' (torch tensors)
        label_map: dict mapping label indices to class names (optional)
    Returns:
        np.ndarray with detections drawn (for st.image)
    """
    import numpy as np
    from supervision.draw.color import ColorPalette
    from supervision.detection.annotate import BoxAnnotator
    from supervision.detection.core import Detections

    img_np = np.array(image)
    boxes = predictions.get('boxes', torch.empty(0)).cpu().numpy()
    labels = predictions.get('labels', torch.empty(0)).cpu().numpy()
    scores = predictions.get('scores', torch.empty(0)).cpu().numpy()
    if label_map:
        class_names = [label_map.get(label, str(label)) for label in labels]
    else:
        class_names = [str(label) for label in labels]
    unique_class_names = list({name for name in class_names})
    palette = ColorPalette.default()
    class_name_to_color = {name: palette.by_idx(i) for i, name in enumerate(unique_class_names)}
    box_colors = [class_name_to_color[name] for name in class_names]
    detections = Detections(
        xyxy=boxes,
        class_id=labels.astype(int)
    )
    annotator = BoxAnnotator(color=palette, text_scale=0.7, text_thickness=1, text_padding=2)
    annotated_img = annotator.annotate(
        scene=img_np,
        detections=detections,
        labels=[f"{name}: {score:.2f}" for name, score in zip(class_names, scores)]
    )
    return annotated_img

def inference_tab():
    st.header("Inference")

    # Use columns for model file and config option
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload model file", type=["pt", "onnx", "h5", "pb", "pth"], key="model_file")
    with col2:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        config_option = st.radio("Model configuration:", ["Upload config JSON", "Set config in UI"])

    config_data = None
    config_path = None
    if config_option == "Upload config JSON":
        config_file = st.file_uploader("Upload config JSON file", type=["json"], key="config_file")
        if config_file is not None:
            try:
                config_data = json.load(config_file)
                st.success("Config loaded successfully.")
                # Save uploaded config to a temp file for model init
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
                    json.dump(config_data, tmp_cfg)
                    config_path = tmp_cfg.name
            except Exception as e:
                st.error(f"Failed to load config: {e}")
    else:
        # Manual config fields in two rows, multiple columns
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2 = st.columns(2)
        with row1_col1:
            confidence_threshold = st.number_input("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Minimum confidence score for detections.")
        with row1_col2:
            nms_threshold = st.number_input("NMS Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Non-maximum suppression threshold.")
        with row1_col3:
            max_detections_per_image = st.number_input("Max Detections/Image", min_value=1, max_value=1000, value=100, step=1, help="Maximum detections per image.")
        with row2_col1:
            device = st.selectbox("Device", ["cpu", "gpu"], help="Device to run inference on.")
        with row2_col2:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=1, step=1, help="Batch size for inference.")
        config_data = {
            "confidence_threshold": confidence_threshold,
            "nms_threshold": nms_threshold,
            "max_detections_per_image": max_detections_per_image,
            "device": device,
            "batch_size": batch_size
        }
        # Save manual config to a temp file for model init
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_cfg:
            json.dump(config_data, tmp_cfg)
            config_path = tmp_cfg.name
        st.info("Manual config will be used.")

    # Ontology file uploader
    ontology_file = st.file_uploader("Upload ontology JSON file", type=["json"], key="ontology_file")
    ontology_path = None
    if ontology_file is not None:
        try:
            ontology_data = json.load(ontology_file)
            st.success("Ontology loaded successfully.")
            # Save uploaded ontology to a temp file for model init
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_ont:
                json.dump(ontology_data, tmp_ont)
                ontology_path = tmp_ont.name
        except Exception as e:
            st.error(f"Failed to load ontology: {e}")

    # --- Model Loading Section ---
    load_col, status_col = st.columns([1, 2])
    with load_col:
        load_model = st.button("Load Model")
    with status_col:
        model_loaded = st.session_state.get('detection_model', None) is not None
        if model_loaded:
            st.success("Model loaded and ready for inference.")
        else:
            st.info("Model not loaded.")

    # Handle model loading
    if load_model:
        if model_file is None:
            st.error("Please upload a model file before loading.")
        elif config_path is None:
            st.error("Please provide a valid model configuration before loading.")
        elif ontology_path is None:
            st.error("Please upload an ontology file before loading.")
        else:
            try:
                # Save uploaded model file to a temp file and use its path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt', mode='wb') as tmp_model:
                    tmp_model.write(model_file.read())
                    model_temp_path = tmp_model.name
                model = TorchImageDetectionModel(
                    model=model_temp_path,
                    model_cfg=config_path,
                    ontology_fname=ontology_path
                )
                st.session_state['detection_model'] = model
                st.success("Model loaded successfully and ready for inference.")
            except Exception as e:
                st.session_state['detection_model'] = None
                st.error(f"Failed to load model: {e}")

    # Use columns for image uploader and Run Inference button
    img_col, btn_col = st.columns([2, 1])
    with img_col:
        image_file = st.file_uploader("Upload image for inference", type=["jpg", "jpeg", "png", "bmp"], key="image_file")
    with btn_col:
        st.markdown("<div style='height: 2.6rem;'></div>", unsafe_allow_html=True)
        run_inference = st.button("Run Inference")

    # Run Inference logic (only if model is loaded)
    if run_inference:
        model = st.session_state.get('detection_model', None)
        if model is None:
            st.error("Please load the model before running inference.")
        elif image_file is None:
            st.error("Please upload an image file.")
        else:
            try:
                # Read image from uploaded file
                image = Image.open(image_file).convert("RGB")
                predictions = model.inference(image)
                label_map = getattr(model, 'idx_to_class_name', None)
                result_img = draw_detections(image.copy(), predictions, label_map)
                st.image(result_img, caption="Detection Results", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to run inference: {e}") 