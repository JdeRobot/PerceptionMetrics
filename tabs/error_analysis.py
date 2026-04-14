import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.error_utils import load_mask, compute_error_map, compute_class_metrics

"""
Error Analysis Tab (Streamlit)

This tab helps visualize how well a segmentation model is performing by
comparing a ground truth mask with a predicted mask.

Instead of just showing overall metrics, it highlights exactly where the
model is correct or wrong at the pixel level.

Structure:
- Input: ground truth mask + prediction mask (+ optional image)
- Output: error map, overlay visualization, and per-class accuracy

All computation logic is handled in utils/error_utils.py.
This file is only responsible for UI rendering.
"""


def render_error_analysis():
    """
    Renders the Error Analysis tab.

    Flow:
    1. User uploads ground truth and prediction masks
    2. Validate inputs (presence + shape)
    3. Compute error map and metrics
    4. Display results (images, overlay, class stats)

    This function only handles UI.
    All calculations are done in utils/error_utils.py.
    """
    st.header("Error Analysis — Semantic Segmentation")
    st.markdown(
        "Compare Ground Truth and Prediction masks to visualize pixel-wise errors."
    )

    # INPUTS INSIDE TAB (NOT SIDEBAR)
    st.subheader("Upload Inputs")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_gt = st.file_uploader(
            "Ground Truth Mask (.png)", type=["png"], key="ea_gt"
        )

    with col2:
        uploaded_pred = st.file_uploader(
            "Prediction Mask (.png)", type=["png"], key="ea_pred"
        )

    uploaded_image = st.file_uploader(
        "Original Image (optional)", type=["png", "jpg", "jpeg"], key="ea_img"
    )

    # VALIDATION
    if uploaded_gt is None or uploaded_pred is None:
        st.info("Please upload both Ground Truth and Prediction masks.")
        return

    try:
        gt = load_mask(uploaded_gt)
        pred = load_mask(uploaded_pred)

        if gt.shape != pred.shape:
            st.error(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")
            return

        # COMPUTE
        error_map = compute_error_map(gt, pred)

        total_pixels = gt.size
        correct_pixels = int(np.sum(gt == pred))
        overall_acc = correct_pixels / total_pixels

        # METRICS
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Accuracy", f"{overall_acc * 100:.2f}%")
        m2.metric("Correct Pixels", f"{correct_pixels:,}")
        m3.metric("Incorrect Pixels", f"{total_pixels - correct_pixels:,}")

        st.markdown("---")

        # VISUALS
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Ground Truth")
            uploaded_gt.seek(0)
            st.image(uploaded_gt, width="stretch")

        with col2:
            st.subheader("Prediction")
            uploaded_pred.seek(0)
            st.image(uploaded_pred, width="stretch")

        with col3:
            st.subheader("Error Map")
            st.image(
                error_map, width="stretch", caption="Green = Correct | Red = Incorrect"
            )

        # OVERLAY
        if uploaded_image is not None:
            st.markdown("---")
            st.subheader("Overlay View")

            original = np.array(
                Image.open(uploaded_image)
                .convert("RGB")
                .resize((gt.shape[1], gt.shape[0]))
            )

            alpha = 0.5
            overlay = (alpha * original + (1 - alpha) * error_map).astype(np.uint8)

            mode = st.radio("Display Mode", ["Error Map", "Overlay"], horizontal=True)

            if mode == "Overlay":
                st.image(overlay, width="stretch")
            else:
                st.image(error_map, width="stretch")

        # CLASS METRICS
        st.markdown("---")
        st.subheader("Class-wise Accuracy")

        class_metrics = compute_class_metrics(gt, pred)

        df = pd.DataFrame(class_metrics)
        df.columns = ["Class ID", "Total Pixels", "Correct Pixels", "Accuracy"]
        df["Accuracy (%)"] = (df["Accuracy"] * 100).round(2)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df, use_container_width=True)

        with col2:
            st.bar_chart(df.set_index("Class ID")["Accuracy (%)"])

    except Exception as e:
        st.error(f"Error: {str(e)}")
