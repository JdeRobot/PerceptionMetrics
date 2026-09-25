import numpy as np
import streamlit as st
from PIL import Image

from perceptionmetrics.utils import conversion as uc


def render_image_segmentation_inference():
    """Render the image segmentation inference tab in the Streamlit app."""
    st.header("Model Inference")
    st.markdown("Select an image and run inference using the loaded model.")

    model = st.session_state.get("segmentation_model")
    if model is None:
        st.warning("Load a segmentation model from the sidebar to start inference.")
        return

    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="segmentation_inference_image_file",
        help="Upload an image to run segmentation inference.",
    )

    if image_file is None:
        return

    with st.spinner("Running segmentation inference..."):
        try:
            image = Image.open(image_file).convert("RGB")
            pred = model.predict(image)
        except Exception as exc:
            st.error(f"Failed to run inference: {exc}")
            return

    pred = uc.label_to_rgb(pred, model.ontology)
    pred = pred.resize(image.size)
    prediction_overlay = _overlay_mask(image, pred, opacity=0.45)

    cols = st.columns(3)
    with cols[0]:
        st.image(image, caption="Image", use_container_width=True)
    with cols[1]:
        st.image(pred, caption="Prediction", use_container_width=True)
    with cols[2]:
        st.image(
            prediction_overlay,
            caption="Prediction Overlay",
            use_container_width=True,
        )


def _overlay_mask(image, mask_rgb, opacity):
    """Overlay a segmentation mask on an image with a given opacity.
    :param image: PIL.Image, the original image
    :param mask_rgb: PIL.Image, the segmentation mask in RGB format
    :param opacity: float, the opacity of the mask overlay (0.0 to 1.0)
    :return: PIL.Image, the image with the mask overlay
    :rtype: PIL.Image
    """
    image_np = np.array(image)
    mask_np = np.array(mask_rgb)
    overlay = ((1.0 - opacity) * image_np + opacity * mask_np).astype(np.uint8)
    return Image.fromarray(overlay)
