import streamlit as st
import os
import json
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
    boxes = predictions.get("boxes", torch.empty(0)).cpu().numpy()
    labels = predictions.get("labels", torch.empty(0)).cpu().numpy()
    scores = predictions.get("scores", torch.empty(0)).cpu().numpy()
    if label_map:
        class_names = [label_map.get(label, str(label)) for label in labels]
    else:
        class_names = [str(label) for label in labels]
    unique_class_names = list({name for name in class_names})
    palette = ColorPalette.default()
    class_name_to_color = {
        name: palette.by_idx(i) for i, name in enumerate(unique_class_names)
    }
    box_colors = [class_name_to_color[name] for name in class_names]
    detections = Detections(xyxy=boxes, class_id=labels.astype(int))
    annotator = BoxAnnotator(
        color=palette, text_scale=0.7, text_thickness=1, text_padding=2
    )
    annotated_img = annotator.annotate(
        scene=img_np,
        detections=detections,
        labels=[f"{name}: {score:.2f}" for name, score in zip(class_names, scores)],
    )
    return annotated_img


def inference_tab():
    st.header("Model Inference")
    st.markdown("Select an image and run inference using the loaded model.")

    # Check if a model has been loaded and saved in session
    if (
        "detection_model" not in st.session_state
        or st.session_state.detection_model is None
    ):
        st.warning("⚠️ Load a model from the sidebar to start inference")
        return

    st.success("Model loaded and saved. You can now select an image.")

    # Image picker in the tab
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="inference_image_file",
        help="Upload an image to run inference",
    )

    if image_file is not None:
        with st.spinner("Running inference..."):
            try:
                image = Image.open(image_file).convert("RGB")
                predictions = st.session_state.detection_model.inference(image)
                label_map = getattr(
                    st.session_state.detection_model, "idx_to_class_name", None
                )
                result_img = draw_detections(image.copy(), predictions, label_map)

                st.markdown("#### Detection Results")
                st.image(result_img, caption="Detection Results", width="stretch")

                # Display detection statistics
                if (
                    predictions.get("scores") is not None
                    and len(predictions["scores"]) > 0
                ):
                    st.markdown("#### Detection Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", len(predictions["scores"]))
                    with col2:
                        avg_confidence = float(predictions["scores"].mean())
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    with col3:
                        max_confidence = float(predictions["scores"].max())
                        st.metric("Max Confidence", f"{max_confidence:.3f}")

                    # Display and download detection results
                    st.markdown("#### Detection Results")

                    # Convert predictions to JSON format
                    detection_results = []
                    boxes = predictions.get("boxes", torch.empty(0)).cpu().numpy()
                    labels = predictions.get("labels", torch.empty(0)).cpu().numpy()
                    scores = predictions.get("scores", torch.empty(0)).cpu().numpy()

                    for i in range(len(scores)):
                        class_name = (
                            label_map.get(int(labels[i]), f"class_{labels[i]}")
                            if label_map
                            else f"class_{labels[i]}"
                        )
                        detection_results.append(
                            {
                                "detection_id": i,
                                "class_id": int(labels[i]),
                                "class_name": class_name,
                                "confidence": float(scores[i]),
                                "bbox": {
                                    "x1": float(boxes[i][0]),
                                    "y1": float(boxes[i][1]),
                                    "x2": float(boxes[i][2]),
                                    "y2": float(boxes[i][3]),
                                },
                                "bbox_xyxy": boxes[i].tolist(),
                            }
                        )

                    with st.expander(" View Detection Results (JSON)", expanded=False):
                        st.json(detection_results)

                    json_str = json.dumps(detection_results, indent=2)
                    st.download_button(
                        label="Download Detection Results as JSON",
                        data=json_str,
                        file_name="detection_results.json",
                        mime="application/json",
                        help="Download the detection results as a JSON file",
                    )
                else:
                    st.info("No detections found in the image.")
            except Exception as e:
                st.error(f"Failed to run inference: {e}")
