"""Computational cost component for Image Segmentation in Streamlit GUI."""

from typing import Optional

import pandas as pd
import streamlit as st


def render_image_segmentation_computational_cost() -> None:
    """Render the computational cost tab for Image Segmentation models.

    Allows users to configure input image dimensions, inference runs,
    trigger latency and memory estimation, and download results as CSV.
    """
    st.header("Computational Cost Analysis")
    st.markdown(
        "Analyze the computational cost (parameters, model size, inference latency) "
        "of the loaded Image Segmentation model."
    )

    if (
        not st.session_state.get("segmentation_model_loaded", False)
        or st.session_state.get("segmentation_model") is None
    ):
        st.warning(
            "⚠️ Load an image segmentation model from the sidebar to analyze its computational cost."
        )
        return

    st.success("Model is loaded and ready for computational cost analysis.")

    col1, col2 = st.columns(2)
    with col1:
        img_height = st.number_input(
            "Input Height",
            min_value=32,
            value=640,
            step=32,
            key="is_cost_height",
            help="Height of the dummy input image in pixels.",
        )
        img_width = st.number_input(
            "Input Width",
            min_value=32,
            value=640,
            step=32,
            key="is_cost_width",
            help="Width of the dummy input image in pixels.",
        )
    with col2:
        runs = st.number_input(
            "Inference Runs",
            min_value=1,
            value=30,
            step=5,
            key="is_cost_runs",
            help="Number of inference iterations to average over.",
        )
        warm_up_runs = st.number_input(
            "Warm-up Runs",
            min_value=0,
            value=5,
            step=1,
            key="is_cost_warm_up_runs",
            help="Number of warm-up iterations prior to timing.",
        )

    if st.button(
        "Calculate Computational Cost",
        type="primary",
        key="is_cost_calc_btn",
    ):
        with st.spinner("Calculating computational cost..."):
            try:
                cost_df = st.session_state.segmentation_model.get_computational_cost(
                    image_size=(int(img_height), int(img_width)),
                    runs=int(runs),
                    warm_up_runs=int(warm_up_runs),
                )
                st.session_state.is_cost_results = cost_df
            except Exception as e:
                st.error(f"Error calculating computational cost: {e}")

    results_df = st.session_state.get("is_cost_results")
    if (
        results_df is not None
        and isinstance(results_df, pd.DataFrame)
        and not results_df.empty
    ):
        _render_cost_metrics(results_df, download_key="is_cost_download_btn")


def _render_cost_metrics(cost_df: pd.DataFrame, download_key: str) -> None:
    """Render summary metrics, tabular view, and CSV download button.

    :param cost_df: DataFrame containing computational cost metrics
    :type cost_df: pd.DataFrame
    :param download_key: Unique widget key for the download button
    :type download_key: str
    """
    st.markdown("### Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)

    input_shape = (
        str(cost_df["input_shape"].iloc[0]) if "input_shape" in cost_df else "N/A"
    )
    m1.metric("Input Shape", input_shape)

    if "n_params" in cost_df and pd.notna(cost_df["n_params"].iloc[0]):
        params_val = cost_df["n_params"].iloc[0]
        params_m = params_val / 1e6
        m2.metric("Parameters (M)", f"{params_m:.2f}")
    else:
        m2.metric("Parameters (M)", "N/A")

    if (
        "size_mb" in cost_df
        and cost_df["size_mb"].iloc[0] is not None
        and pd.notna(cost_df["size_mb"].iloc[0])
    ):
        m3.metric("Model Size", f"{float(cost_df['size_mb'].iloc[0]):.2f} MB")
    else:
        m3.metric("Model Size", "N/A")

    if "inference_time_s" in cost_df and pd.notna(cost_df["inference_time_s"].iloc[0]):
        inf_time_s = float(cost_df["inference_time_s"].iloc[0])
        latency_ms = inf_time_s * 1000.0
        m4.metric("Latency (ms)", f"{latency_ms:.2f}")
        fps = (1.0 / inf_time_s) if inf_time_s > 0 else 0.0
        m5.metric("FPS", f"{fps:.1f}" if fps > 0 else "N/A")
    else:
        m4.metric("Latency (ms)", "N/A")
        m5.metric("FPS", "N/A")

    with st.expander("Detailed Results", expanded=True):
        st.dataframe(cost_df, use_container_width=True)

    csv_data = cost_df.to_csv(index=False)
    st.download_button(
        label="📥 Download results as CSV",
        data=csv_data,
        file_name="computational_cost.csv",
        mime="text/csv",
        key=download_key,
    )
