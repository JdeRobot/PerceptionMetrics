import numpy as np
import streamlit as st
from PIL import Image

from utils.error_utils import (
    load_mask,
    compute_error_map,
    build_ontology_from_masks,
    compute_metrics,
)


def render_error_analysis():
    """Render the Error Analysis tab.

    Flow
    ----
    1. User uploads a Ground Truth mask and a Prediction mask (both PNG).
    2. An optional original RGB image may be uploaded for overlay view.
    3. Masks are validated (presence + shape agreement).
    4. ``compute_metrics()`` delegates entirely to
       ``SegmentationMetricsFactory`` — no inline metric math.
    5. Results are displayed as:
       - Summary metric cards (overall accuracy, macro-IoU, macro-F1).
       - Pixel-wise error map (green = correct, red = wrong).
       - Optional overlay of error map on the original image.
       - Full per-class metrics DataFrame (same structure as ``pm_evaluate``).
       - Bar chart of per-class IoU for quick visual scanning.
    """
    st.header("Error Analysis — Semantic Segmentation")
    st.markdown(
        "Compare a Ground Truth mask with a Prediction mask to visualise "
        "pixel-wise errors and standard segmentation metrics."
    )
    st.info(
        "ℹ️ Metrics are computed using `SegmentationMetricsFactory` — the same "
        "engine used by `pm_evaluate` and the Evaluator tab. "
        "No model inference is performed here.",
        icon="ℹ️",
    )

    st.subheader("Upload Inputs")

    col_gt, col_pred = st.columns(2)
    with col_gt:
        uploaded_gt = st.file_uploader(
            "Ground Truth Mask (.png)",
            type=["png"],
            key="ea_gt",
            help="Grayscale PNG where each pixel value is a class index.",
        )
    with col_pred:
        uploaded_pred = st.file_uploader(
            "Prediction Mask (.png)",
            type=["png"],
            key="ea_pred",
            help="Grayscale PNG where each pixel value is a class index.",
        )

    uploaded_image = st.file_uploader(
        "Original Image (optional — for overlay view)",
        type=["png", "jpg", "jpeg"],
        key="ea_img",
    )

    if uploaded_gt is None or uploaded_pred is None:
        st.info("Please upload both a Ground Truth mask and a Prediction mask.")
        return

    try:
        gt = load_mask(uploaded_gt)
        pred = load_mask(uploaded_pred)
    except Exception as exc:
        st.error(f" Failed to load masks: {exc}")
        return

    if gt.shape != pred.shape:
        st.error(
            f"Shape mismatch — GT: {gt.shape}, Prediction: {pred.shape}. "
            "Both masks must have identical height × width."
        )
        return

    try:
        ontology = build_ontology_from_masks(gt, pred)
        metrics_df = compute_metrics(gt, pred, ontology)
    except Exception as exc:
        st.error(f" Metrics computation failed: {exc}")
        import traceback

        st.code(traceback.format_exc())
        return

    st.markdown("---")
    st.subheader("Overall Metrics")

    # ``get_metrics_dataframe`` returns a DataFrame indexed by class name
    # with "macro" and "micro" as aggregate rows.
    macro_row = metrics_df.get("macro") if "macro" in metrics_df.columns else None

    # Pixel accuracy = micro-accuracy (TP_all / total pixels)
    micro_row = metrics_df.get("micro") if "micro" in metrics_df.columns else None

    c1, c2, c3, c4 = st.columns(4)

    pixel_acc = float(micro_row["accuracy"]) if micro_row is not None else float("nan")
    macro_iou = float(macro_row["iou"]) if macro_row is not None else float("nan")
    macro_f1 = float(macro_row["f1_score"]) if macro_row is not None else float("nan")
    macro_pre = float(macro_row["precision"]) if macro_row is not None else float("nan")

    c1.metric("Pixel Accuracy", f"{pixel_acc * 100:.2f}%")
    c2.metric("Mean IoU (macro)", f"{macro_iou:.4f}")
    c3.metric("Mean F1 (macro)", f"{macro_f1:.4f}")
    c4.metric("Mean Precision", f"{macro_pre:.4f}")

    st.markdown("---")
    st.subheader("Visual Inspection")

    error_map = compute_error_map(gt, pred)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Ground Truth**")
        uploaded_gt.seek(0)
        st.image(uploaded_gt, use_container_width=True)
    with col_b:
        st.markdown("**Prediction**")
        uploaded_pred.seek(0)
        st.image(uploaded_pred, use_container_width=True)
    with col_c:
        st.markdown("**Error Map**")
        st.image(
            error_map,
            use_container_width=True,
            caption="🟢 Correct  |  🔴 Incorrect",
        )

    if uploaded_image is not None:
        st.markdown("---")
        st.subheader("Overlay View")

        original = np.array(
            Image.open(uploaded_image).convert("RGB").resize((gt.shape[1], gt.shape[0]))
        )

        alpha = st.slider("Error map opacity", 0.0, 1.0, 0.5, 0.05, key="ea_alpha")
        overlay = np.clip(
            alpha * error_map.astype(np.float32)
            + (1.0 - alpha) * original.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        view_mode = st.radio(
            "Display mode",
            ["Overlay", "Error Map only"],
            horizontal=True,
            key="ea_view_mode",
        )
        st.image(
            overlay if view_mode == "Overlay" else error_map,
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Per-Class Metrics")
    st.caption(
        "Rows correspond to individual classes detected in the masks; "
        "`macro` and `micro` rows are aggregate statistics."
    )

    # Round floats for display
    display_df = metrics_df.T.copy()  # transpose: classes as rows, metrics as columns
    float_cols = display_df.select_dtypes(include="float").columns
    display_df[float_cols] = display_df[float_cols].round(4)

    st.dataframe(display_df, use_container_width=True)

    # Extract per-class IoU (exclude aggregate rows)
    aggregate_rows = {"macro", "micro"}
    class_cols = [c for c in metrics_df.columns if c not in aggregate_rows]

    if class_cols:
        st.markdown("---")
        st.subheader("Per-Class IoU")

        iou_series = metrics_df.loc["iou", class_cols].astype(float)
        iou_df = iou_series.reset_index()
        iou_df.columns = ["Class", "IoU"]
        iou_df = iou_df.sort_values("IoU", ascending=False)

        st.bar_chart(iou_df.set_index("Class")["IoU"])

    st.markdown("---")
    csv_data = display_df.to_csv(index=True)
    st.download_button(
        label="📥 Download metrics as CSV",
        data=csv_data,
        file_name="error_analysis_metrics.csv",
        mime="text/csv",
    )
