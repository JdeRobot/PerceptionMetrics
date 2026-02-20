from typing import Optional

import streamlit as st
import json
from PIL import Image
import torch


def draw_detections(image: Image, predictions: dict, label_map: Optional[dict] = None):
    """Draw color-coded bounding boxes and labels on the image using supervision.

    :param image: PIL Image
    :type image: Image.Image
    :param predictions: dict with 'boxes', 'labels', 'scores' (torch tensors)
    :type predictions: dict
    :param label_map: dict mapping label indices to class names (optional)
    :type label_map: dict
    :return: np.ndarray with detections drawn (for st.image)
    :rtype: np.ndarray
    """
    from perceptionmetrics.utils import image as ui

    boxes = predictions.get("boxes", torch.empty(0)).cpu().numpy()
    class_ids = predictions.get("labels", torch.empty(0)).cpu().numpy().astype(int)

    scores_tensor = predictions.get("scores")
    if scores_tensor is not None and len(scores_tensor) > 0:
        scores = scores_tensor.cpu().numpy()
    else:
        scores = None

    if label_map:
        class_names = [label_map.get(int(label), str(label)) for label in class_ids]
    else:
        class_names = [str(label) for label in class_ids]

    return ui.draw_detections(
        image=image,
        boxes=boxes,
        class_ids=class_ids,
        class_names=class_names,
        scores=scores,
    )


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
                predictions, sample_tensor = st.session_state.detection_model.predict(
                    image, return_sample=True
                )
                from torchvision.transforms import v2 as transforms

                img_to_draw = transforms.ToPILImage()(sample_tensor[0])
                label_map = getattr(
                    st.session_state.detection_model, "idx_to_class_name", None
                )
                result_img = draw_detections(img_to_draw, predictions, label_map)

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
