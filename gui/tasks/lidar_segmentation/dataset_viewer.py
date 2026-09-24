import os

import numpy as np
import pandas as pd
import plotly.graph_objects as graph_objects
import streamlit as st

from perceptionmetrics.datasets.semantickitti import (
    SemanticKITTILiDARSegmentationDataset,
)


def render_lidar_segmentation_viewer():
    """
    Render the LiDAR segmentation dataset viewer in Streamlit.
    """
    st.header("LiDAR Dataset Viewer")

    dataset_type = st.session_state.get("lidar_dataset_type", "SemanticKITTI")
    if dataset_type != "SemanticKITTI":
        st.info(f"{dataset_type} LiDAR segmentation viewer is not wired yet.")
        return

    dataset_path = st.session_state.get("dataset_path", "")
    config_path = st.session_state.get("lidar_config_path", "")
    split = st.session_state.get("split", "val")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("Please select a valid SemanticKITTI dataset folder.")
        return

    if not config_path or not os.path.isfile(config_path):
        st.warning("Please select a valid SemanticKITTI config YAML file.")
        return

    try:
        dataset = load_semantic_kitti_dataset(dataset_path, config_path, split)
    except Exception as exc:
        st.error(f"Failed to load SemanticKITTI dataset: {exc}")
        return

    frames = dataset.dataset.index.astype(str).tolist()

    if not frames:
        st.warning(f"No SemanticKITTI frames found for split '{split}'.")
        return

    if st.session_state.get("semantic_kitti_selected_frame") not in frames:
        st.session_state.semantic_kitti_selected_frame = frames[0]

    selected_frame = st.selectbox(
        "Frame",
        frames,
        key="semantic_kitti_selected_frame",
    )

    color_options = ["Semantic Labels", "Intensity"]
    if split == "test":
        color_options = ["Intensity"]
    if st.session_state.get("semantic_kitti_color_by") not in color_options:
        st.session_state.semantic_kitti_color_by = color_options[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        point_size = st.slider(
            "Point Size",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            key="semantic_kitti_point_size",
        )

    with col2:
        color_by = st.selectbox(
            "Color By",
            color_options,
            key="semantic_kitti_color_by",
        )

    with col3:
        max_points = st.number_input(
            "Max Points",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=1000,
            key="semantic_kitti_max_points",
            help="Subsamples large frames before rendering to keep the GUI responsive.",
        )

    if st.button("Reset Top View", key="semantic_kitti_reset_top_view"):
        st.session_state.semantic_kitti_view_reset = (
            st.session_state.get("semantic_kitti_view_reset", 0) + 1
        )

    try:
        row = dataset.dataset.loc[selected_frame]
        points = dataset.read_points(row["points"])
        labels = dataset.read_label(row["label"]) if row["label"] else None

        points, labels = subsample_points(points, labels, max_points)
        intensity_clip_range = None
        if color_by == "Semantic Labels" and labels is not None:
            colors = get_label_colors(labels, dataset.ontology)
            hover_text = get_label_names(labels, dataset.ontology)
            color_source = "semantic labels"
        else:
            intensity_clip_range = render_intensity_clip_slider(
                points[:, 3],
                key="semantic_kitti_intensity_clip_range",
            )
            colors = intensity_to_colors(points[:, 3], intensity_clip_range)
            hover_text = None
            color_source = "intensity"

    except Exception as exc:
        st.error(f"Failed to read frame '{selected_frame}': {exc}")
        return

    st.caption(
        f"{selected_frame} ({len(points)} points shown, colored by {color_source})"
    )

    render_point_cloud_plotly(
        points=points[:, :3],
        colors=colors,
        point_size=point_size,
        hover_text=hover_text,
        color_values=(
            np.clip(points[:, 3], *intensity_clip_range)
            if color_by == "Intensity"
            else None
        ),
        colorbar_title="Intensity" if color_by == "Intensity" else None,
        color_range=intensity_clip_range if color_by == "Intensity" else None,
        chart_key=f"semantic_kitti_viewer_{st.session_state.get('semantic_kitti_view_reset', 0)}",
    )

    with st.expander("Classes", expanded=False):
        st.dataframe(classes_dataframe(dataset.ontology), use_container_width=True)


def load_semantic_kitti_dataset(dataset_path, config_path, split):
    dataset_key = (
        "semantic_kitti_lidar_segmentation_dataset",
        os.path.abspath(dataset_path),
        os.path.abspath(config_path),
        split,
    )

    if dataset_key not in st.session_state:
        st.session_state[dataset_key] = SemanticKITTILiDARSegmentationDataset(
            dataset_dir=dataset_path,
            config_fname=config_path,
            split=split,
        )

    return st.session_state[dataset_key]


def render_point_cloud_plotly(
    points,
    colors,
    point_size=2.0,
    hover_text=None,
    color_values=None,
    colorbar_title=None,
    color_range=None,
    chart_key=None,
):
    """
    Render an interactive 3D point cloud in Streamlit using Plotly.
    :param points: Point cloud data as a numpy array of shape (N, 3) or (N, 4) where N is the number of points.
    :type points: np.ndarray
    :param colors: Colors for each point as a numpy array of shape (N, 3) with RGB values in [0, 1] or [0, 255].
    :type colors: np.ndarray
    :param point_size: Size of each point in the visualization.
    :type point_size: float
    :param hover_text: Text to display on hover for each point.
    :type hover_text: list[str] or None
    :param color_values: Numeric values used for the Plotly colorbar.
    :type color_values: np.ndarray or None
    :param colorbar_title: Title to display on the colorbar.
    :type colorbar_title: str or None
    :param color_range: Min and max values used for color scaling.
    :type color_range: tuple[float, float] or None
    :param chart_key: Key for the Streamlit chart.
    :type chart_key: str or None
    """

    if points.size == 0:
        st.warning("No points to render.")
        return

    marker_color = None
    marker_colorscale = None
    marker_showscale = False
    marker_colorbar = None

    if color_values is not None:
        marker_color = np.asarray(color_values, dtype=np.float32)
        marker_colorscale = [[0.0, "rgb(0,89,13)"], [1.0, "rgb(255,242,0)"]]
        marker_showscale = True
        marker_colorbar = dict(title=colorbar_title or "Value")
    else:
        colors = np.asarray(colors, dtype=np.float32)

        if colors.max() <= 1.0:
            colors = colors * 255.0

        colors = np.clip(colors, 0, 255).astype(np.uint8)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in colors]

    marker = dict(
        size=point_size,
        color=marker_color,
        colorscale=marker_colorscale,
        showscale=marker_showscale,
        colorbar=marker_colorbar,
        opacity=0.95,
    )
    if color_range is not None and color_range[0] != color_range[1]:
        marker["cmin"] = color_range[0]
        marker["cmax"] = color_range[1]

    fig = graph_objects.Figure(
        data=[
            graph_objects.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
                text=hover_text,
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>"
                    "class=%{text}<extra></extra>"
                    if hover_text is not None
                    else "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35),
            camera=dict(
                eye=dict(x=0.0, y=0.0, z=2.0),
                up=dict(x=0, y=1, z=0),
            ),
        ),
    )

    st.plotly_chart(fig, width="stretch", key=chart_key)


def get_label_colors(labels, ontology):
    """ Get colors for each label based on the provided ontology.

    :param labels: Array of label IDs for each point in the point cloud.
    :type labels: np.ndarray
    :param ontology: Ontology dictionary mapping class names to their properties, including color.
    :type ontology: dict
    :return: Array of RGB colors corresponding to each label.
    :rtype: np.ndarray
    """
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    idx_to_rgb = {
        int(class_data["idx"]): np.array(class_data["rgb"], dtype=np.float32) / 255.0
        for class_data in ontology.values()
    }

    for label_id in np.unique(labels):
        colors[labels == label_id] = idx_to_rgb.get(int(label_id), [1.0, 1.0, 1.0])

    return colors


def get_label_names(labels, ontology):
    """ Get class names for each label based on the provided ontology.
    
    :param labels: Array of label IDs for each point in the point cloud.
    :type labels: np.ndarray
    :param ontology: Ontology dictionary mapping class names to their properties, including name.
    :type ontology: dict
    :return: List of class names corresponding to each label.
    :rtype: list[str]
    """
    idx_to_name = {
        int(class_data["idx"]): class_name
        for class_name, class_data in ontology.items()
    }
    return [idx_to_name.get(int(label), str(int(label))) for label in labels]


def classes_dataframe(ontology):
    """Convert the ontology dictionary to a pandas DataFrame for display.
    
    :param ontology: Ontology dictionary mapping class names to their properties.
    :type ontology: dict
    :return: Pandas DataFrame containing class information.
    :rtype: pd.DataFrame
    """
    rows = []
    for class_name, class_data in ontology.items():
        rows.append(
            {
                "class": class_name,
                "id": class_data["idx"],
                "rgb": class_data.get("rgb"),
            }
        )

    return pd.DataFrame(rows)


def render_intensity_clip_slider(intensity, key):
    """Render an intensity clipping slider and return the selected range."""
    intensity = np.asarray(intensity, dtype=np.float32)
    min_intensity = float(np.min(intensity))
    max_intensity = float(np.max(intensity))

    if min_intensity == max_intensity:
        st.info(f"Intensity is constant at {min_intensity:.3f}.")
        return min_intensity, max_intensity

    clip_range = st.slider(
        "Intensity Clip Range",
        min_value=min_intensity,
        max_value=max_intensity,
        value=(min_intensity, max_intensity),
        step=(max_intensity - min_intensity) / 100.0,
        key=key,
        help="Values outside this range are clipped before color normalization.",
    )
    st.caption(f"Intensity clipped to [{clip_range[0]:.3f}, {clip_range[1]:.3f}]")
    return clip_range


def intensity_to_colors(intensity, clip_range=None):
    """Convert intensity values to RGB colors for visualization.

    :param intensity: Array of intensity values.
    :type intensity: np.ndarray
    :param clip_range: Min and max intensity values to clip before normalization.
    :type clip_range: tuple[float, float] or None
    :return: Array of RGB colors.
    :rtype: np.ndarray
    """
    intensity = intensity.astype(np.float32)

    if intensity.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if clip_range is None:
        min_intensity = float(intensity.min())
        max_intensity = float(intensity.max())
    else:
        min_intensity, max_intensity = clip_range

    intensity = np.clip(intensity, min_intensity, max_intensity)
    intensity = intensity - min_intensity

    intensity_range = max_intensity - min_intensity

    if intensity_range > 0:
        intensity = intensity / intensity_range

    green = np.array([0.0, 0.35, 0.05], dtype=np.float32)
    yellow = np.array([1.0, 0.95, 0.0], dtype=np.float32)
    return green + intensity[:, None] * (yellow - green)


def subsample_points(points, labels, max_points):
    """Subsample the point cloud and labels to a maximum number of points.
    
    :param points: Array of 3D points.
    :type points: np.ndarray
    :param labels: Array of label IDs for each point in the point cloud.
    :type labels: np.ndarray
    :param max_points: Maximum number of points to include in the subsampled point cloud.
    :type max_points: int
    :return: Subsampled array of 3D points and corresponding labels.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if len(points) <= max_points:
        return points, labels

    indices = np.linspace(0, len(points) - 1, int(max_points), dtype=int)

    if labels is None:
        return points[indices], None

    return points[indices], labels[indices]
