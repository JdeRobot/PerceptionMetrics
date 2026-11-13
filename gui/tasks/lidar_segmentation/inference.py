import os
import tempfile

import numpy as np
import streamlit as st

from gui.tasks.lidar_segmentation.dataset_viewer import (
    classes_dataframe,
    get_label_colors,
    get_label_names,
    intensity_to_colors,
    render_intensity_clip_slider,
    render_point_cloud_plotly,
    subsample_points,
)


def render_lidar_segmentation_inference():
    st.header("LiDAR Model Inference")
    st.markdown("Upload a SemanticKITTI `.bin` point cloud and run inference.")

    model = st.session_state.get("lidar_model")
    if model is None:
        st.warning("Load a LiDAR segmentation model from the sidebar to start inference.")
        return

    points_file = st.file_uploader(
        "Choose a point cloud",
        type=["bin"],
        key="lidar_inference_points_file",
        help="Upload a SemanticKITTI .bin point cloud file.",
    )

    col1, col2 = st.columns(2)
    with col1:
        point_size = st.slider(
            "Point Size",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            key="semantic_kitti_inference_point_size",
        )
    with col2:
        max_points = st.number_input(
            "Max Points",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=1000,
            key="semantic_kitti_inference_max_points",
            help="Subsamples large frames before rendering to keep the GUI responsive.",
        )

    if st.button("Reset Top View", key="semantic_kitti_inference_reset_top_view"):
        st.session_state.semantic_kitti_inference_view_reset = (
            st.session_state.get("semantic_kitti_inference_view_reset", 0) + 1
        )

    if points_file is None:
        st.info("Upload a `.bin` point cloud to run inference.")
        st.session_state.pop("lidar_inference_result", None)
        st.session_state.pop("lidar_inference_file_name", None)
        return
    current_file_name = points_file.name
    previous_file_name = st.session_state.get("lidar_inference_file_name")

    if previous_file_name != current_file_name:
        st.session_state.pop("lidar_inference_result", None)
        st.session_state["lidar_inference_file_name"] = current_file_name

    try:
        uploaded_points = np.frombuffer(points_file.getbuffer(), dtype=np.float32).reshape(
            -1, 4
        )
    except Exception as exc:
        st.error(f"Failed to read uploaded point cloud: {exc}")
        return

    intensity_clip_range = render_intensity_clip_slider(
        uploaded_points[:, 3],
        key="semantic_kitti_inference_intensity_clip_range",
    )

    if st.button(
        "Run LiDAR Inference",
        type="primary",
        key="run_lidar_segmentation_inference",
    ):
        with st.spinner("Running LiDAR inference..."):
            points_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_points:
                    tmp_points.write(points_file.getbuffer())
                    points_path = tmp_points.name

                points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
                pred = model.predict(
                    points_fname=points_path,
                    has_intensity=True,
                )

                st.session_state["lidar_inference_result"] = {
                    "file_name": points_file.name,
                    "points": points,
                    "pred": pred,
                }
                st.success("Inference completed.")
            except Exception as exc:
                st.error(f"Failed to run inference for '{points_file.name}': {exc}")
                return
            finally:
                if points_path and os.path.isfile(points_path):
                    os.unlink(points_path)

        
    
    result = st.session_state.get("lidar_inference_result")
    if result is None:
        st.info("Click Run LiDAR Inference to visualize predictions.")
        return
    points = result["points"]
    pred = result["pred"]

    points, pred = subsample_points(points, pred, max_points)
    pred_colors = get_label_colors(pred, model.ontology)
    pred_names = get_label_names(pred, model.ontology)
    intensity_colors = intensity_to_colors(points[:, 3], intensity_clip_range)

    st.caption(f"{result['file_name']} ({len(points)} points shown)")

    pred_col, intensity_col = st.columns(2)
    with pred_col:
        st.markdown("#### Prediction")
        render_point_cloud_plotly(
            points=points[:, :3],
            colors=pred_colors,
            point_size=point_size,
            hover_text=pred_names,
            chart_key=(
                "semantic_kitti_inference_prediction_"
                f"{st.session_state.get('semantic_kitti_inference_view_reset', 0)}"
            ),
        )
    with intensity_col:
        st.markdown("#### Intensity")
        render_point_cloud_plotly(
            points=points[:, :3],
            colors=intensity_colors,
            point_size=point_size,
            color_values=np.clip(points[:, 3], *intensity_clip_range),
            colorbar_title="Intensity",
            color_range=intensity_clip_range,
            chart_key=(
                "semantic_kitti_inference_intensity_"
                f"{st.session_state.get('semantic_kitti_inference_view_reset', 0)}"
            ),
        )

    with st.expander("Model Classes", expanded=False):
        st.dataframe(classes_dataframe(model.ontology), use_container_width=True)
