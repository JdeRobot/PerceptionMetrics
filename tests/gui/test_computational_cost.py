"""Regression tests for the Computational Cost GUI tab across modalities."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def get_by_label(elements, label):
    """Return the first Streamlit testing element with the given label."""
    return next(element for element in elements if element.label == label)


def test_computational_cost_tab_presence():
    """Verify that Computational Cost tab is present in default layout."""
    app = AppTest.from_file("app.py").run(timeout=10)

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert "Computational Cost" in tab_labels


def test_computational_cost_warning_when_model_not_loaded():
    """Verify warnings are displayed when no model is loaded across tasks."""
    app = AppTest.from_file("app.py").run(timeout=10)

    # 1. Image Detection
    warning_text = "\n".join(w.value for w in app.warning)
    assert "Load an image detection model from the sidebar" in warning_text

    # 2. Image Segmentation
    get_by_label(app.selectbox, "Task").select("Image Segmentation").run(timeout=10)
    warning_text = "\n".join(w.value for w in app.warning)
    assert "Load an image segmentation model from the sidebar" in warning_text

    # 3. Lidar Segmentation
    get_by_label(app.selectbox, "Task").select("Lidar Segmentation").run(timeout=10)
    warning_text = "\n".join(w.value for w in app.warning)
    assert "Load a lidar segmentation model from the sidebar" in warning_text


def test_image_detection_computational_cost_execution():
    """Verify computational cost calculation for Image Detection."""
    app = AppTest.from_file("app.py").run(timeout=10)

    mock_model = MagicMock()
    mock_model.get_computational_cost.return_value = pd.DataFrame(
        {
            "input_shape": ["1x3x640x640"],
            "n_params": [3200000],
            "size_mb": [12.8],
            "inference_time_s": [0.020],
        }
    )

    app.session_state["detection_model"] = mock_model
    app.session_state["detection_model_loaded"] = True
    app.run(timeout=10)

    assert not app.exception
    success_text = "\n".join(s.value for s in app.success)
    assert "Model is loaded and ready" in success_text

    # Check input widgets exist
    assert any(ni.label == "Input Height" for ni in app.number_input)
    assert any(ni.label == "Input Width" for ni in app.number_input)
    assert any(ni.label == "Inference Runs" for ni in app.number_input)
    assert any(ni.label == "Warm-up Runs" for ni in app.number_input)

    # Trigger calculation
    calc_button = next(
        b for b in app.button if b.label == "Calculate Computational Cost"
    )
    calc_button.click().run(timeout=10)

    assert not app.exception
    mock_model.get_computational_cost.assert_called_once_with(
        image_size=(640, 640),
        runs=30,
        warm_up_runs=5,
    )

    # Verify metrics rendered
    metric_labels = [m.label for m in app.metric]
    assert "Input Shape" in metric_labels
    assert "Parameters (M)" in metric_labels
    assert "Model Size" in metric_labels
    assert "Latency (ms)" in metric_labels
    assert "FPS" in metric_labels

    # Verify results DataFrame and download button rendered
    assert len(app.dataframe) > 0
    download_buttons = app.get("download_button")
    assert any(db.label == "📥 Download results as CSV" for db in download_buttons)


def test_image_segmentation_computational_cost_execution():
    """Verify computational cost calculation for Image Segmentation."""
    app = AppTest.from_file("app.py").run(timeout=10)
    get_by_label(app.selectbox, "Task").select("Image Segmentation").run(timeout=10)

    mock_model = MagicMock()
    mock_model.get_computational_cost.return_value = pd.DataFrame(
        {
            "input_shape": ["1x3x512x1024"],
            "n_params": [15000000],
            "size_mb": [58.2],
            "inference_time_s": [0.045],
        }
    )

    app.session_state["segmentation_model"] = mock_model
    app.session_state["segmentation_model_loaded"] = True
    app.run(timeout=10)

    assert not app.exception
    calc_button = next(
        b for b in app.button if b.label == "Calculate Computational Cost"
    )
    calc_button.click().run(timeout=10)

    assert not app.exception
    mock_model.get_computational_cost.assert_called_once_with(
        image_size=(640, 640),
        runs=30,
        warm_up_runs=5,
    )

    metric_labels = [m.label for m in app.metric]
    assert "Input Shape" in metric_labels
    assert "Parameters (M)" in metric_labels
    assert "Model Size" in metric_labels
    download_buttons = app.get("download_button")
    assert any(db.label == "📥 Download results as CSV" for db in download_buttons)


def test_lidar_segmentation_computational_cost_execution():
    """Verify computational cost calculation for LiDAR Segmentation."""
    app = AppTest.from_file("app.py").run(timeout=10)
    get_by_label(app.selectbox, "Task").select("Lidar Segmentation").run(timeout=10)

    mock_model = MagicMock()
    mock_model.get_computational_cost.return_value = pd.DataFrame(
        {
            "input_shape": ["100000x3"],
            "n_params": [8500000],
            "size_mb": [34.0],
            "inference_time_s": [0.080],
        }
    )

    app.session_state["lidar_model"] = mock_model
    app.session_state["lidar_model_loaded"] = True
    app.run(timeout=10)

    assert not app.exception
    assert any(ni.label == "Number of Points" for ni in app.number_input)
    assert any(cb.label == "Has Intensity" for cb in app.checkbox)
    assert any(ni.label == "X Min" for ni in app.number_input)

    calc_button = next(
        b for b in app.button if b.label == "Calculate Computational Cost"
    )
    calc_button.click().run(timeout=10)

    assert not app.exception
    mock_model.get_computational_cost.assert_called_once_with(
        point_cloud_range=(-50, -50, -5, 50, 50, 5),
        num_points=100000,
        has_intensity=False,
        runs=30,
        warm_up_runs=5,
    )

    metric_labels = [m.label for m in app.metric]
    assert "Input Shape" in metric_labels
    assert "Parameters (M)" in metric_labels
    download_buttons = app.get("download_button")
    assert any(db.label == "📥 Download results as CSV" for db in download_buttons)


def test_computational_cost_handles_calculation_error():
    """Verify exceptions during cost estimation are gracefully presented."""
    app = AppTest.from_file("app.py").run(timeout=10)

    mock_model = MagicMock()
    mock_model.get_computational_cost.side_effect = RuntimeError("CUDA Out of Memory")

    app.session_state["detection_model"] = mock_model
    app.session_state["detection_model_loaded"] = True
    app.run(timeout=10)

    calc_button = next(
        b for b in app.button if b.label == "Calculate Computational Cost"
    )
    calc_button.click().run(timeout=10)

    assert not app.exception
    error_text = "\n".join(e.value for e in app.error)
    assert "Error calculating computational cost: CUDA Out of Memory" in error_text
