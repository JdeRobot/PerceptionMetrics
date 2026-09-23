import yaml
import numpy as np
from streamlit.testing.v1 import AppTest


def create_tiny_semantickitti_dataset(tmp_path):
    """Create a minimal SemanticKITTI-style validation split for GUI tests."""
    sequence = "08"
    frame = "000000"

    sequence_dir = tmp_path / "dataset" / "sequences" / sequence
    velodyne_dir = sequence_dir / "velodyne"
    labels_dir = sequence_dir / "labels"
    velodyne_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    points = np.array(
        [
            [1.0, 0.0, 0.0, 0.1],
            [2.0, 0.5, 0.0, 0.5],
            [3.0, 1.0, 0.0, 0.9],
            [4.0, 1.5, 0.0, 0.3],
        ],
        dtype=np.float32,
    )
    points.tofile(velodyne_dir / f"{frame}.bin")

    labels = np.array([0, 10, 10, 40], dtype=np.uint32)
    labels.tofile(labels_dir / f"{frame}.label")

    config = {
        "labels": {
            0: "unlabeled",
            10: "car",
            40: "road",
        },
        "color_map": {
            0: [0, 0, 0],
            10: [245, 150, 100],
            40: [255, 0, 255],
        },
        "content": {
            0: 0.1,
            10: 0.2,
            40: 0.7,
        },
        "learning_map": {
            0: 0,
            10: 1,
            40: 2,
        },
        "learning_map_inv": {
            0: 0,
            1: 10,
            2: 40,
        },
        "split": {
            "train": [],
            "valid": [8],
            "test": [],
        },
    }

    config_path = tmp_path / "semantic-kitti.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return tmp_path, config_path


def get_by_label(elements, label):
    """Return the first Streamlit testing element with the given label."""
    return next(element for element in elements if element.label == label)


def file_uploader_labels(app):
    """Return labels for file uploaders rendered by AppTest."""
    return [
        element.label
        for element in app
        if getattr(element, "type", None) == "file_uploader"
    ]


def select_lidar_segmentation_task(app):
    """Switch the app to the LiDAR segmentation task."""
    get_by_label(app.selectbox, "Task").select("Lidar Segmentation").run(timeout=10)
    return app


def test_lidar_segmentation_gui_default_view():
    """Verify that the Streamlit app opens the LiDAR segmentation task."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    assert not app.exception
    assert app.session_state["task"] == "Lidar Segmentation"
    assert app.session_state["lidar_dataset_type"] == "SemanticKITTI"

    tab_labels = [tab.label for tab in app.tabs]
    assert tab_labels == ["Dataset Viewer", "Inference", "Evaluator"]

    selectbox_labels = [selectbox.label for selectbox in app.selectbox]
    assert "Task" in selectbox_labels
    assert "Type" in selectbox_labels
    assert "Split" in selectbox_labels
    assert "Model Type" in selectbox_labels
    assert "Device" in selectbox_labels

    text_input_labels = [text_input.label for text_input in app.text_input]
    assert "Dataset Folder" in text_input_labels
    assert "SemanticKITTI Config YAML" in text_input_labels
    assert "MMDetection3D Config" in text_input_labels
    assert "Checkpoint" in text_input_labels
    assert "Model Config JSON" in text_input_labels
    assert "Ontology File" in text_input_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid SemanticKITTI dataset folder." in warning_text
    assert "Load a LiDAR segmentation model from the sidebar to start inference." in warning_text


def test_lidar_segmentation_sidebar_updates_dataset_inputs():
    """Verify that LiDAR dataset sidebar inputs update Streamlit session state."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(
        "/tmp/not_a_real_semantickitti_dataset"
    ).run(timeout=10)
    get_by_label(app.text_input, "SemanticKITTI Config YAML").input(
        "/tmp/semantic-kitti.yaml"
    ).run(timeout=10)

    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == "/tmp/not_a_real_semantickitti_dataset"
    assert app.session_state["lidar_config_path"] == "/tmp/semantic-kitti.yaml"


def test_lidar_segmentation_sidebar_updates_model_inputs():
    """Verify that LiDAR model sidebar inputs update Streamlit session state."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    get_by_label(app.selectbox, "Model Type").select("MMDetection3D").run(timeout=10)
    get_by_label(app.text_input, "MMDetection3D Config").input(
        "/tmp/mmdet3d_config.py"
    ).run(timeout=10)
    get_by_label(app.text_input, "Checkpoint").input("/tmp/model.pth").run(timeout=10)
    get_by_label(app.text_input, "Model Config JSON").input(
        "/tmp/lidar_model_cfg.json"
    ).run(timeout=10)
    get_by_label(app.text_input, "Ontology File").input(
        "/tmp/lidar_ontology.json"
    ).run(timeout=10)

    assert app.session_state["lidar_model_type"] == "MMDetection3D"
    assert app.session_state["lidar_mmdet3d_config_path"] == "/tmp/mmdet3d_config.py"
    assert app.session_state["lidar_checkpoint_path"] == "/tmp/model.pth"
    assert app.session_state["lidar_model_config_path"] == "/tmp/lidar_model_cfg.json"
    assert app.session_state["lidar_ontology_path"] == "/tmp/lidar_ontology.json"


def test_lidar_segmentation_gui_loads_tiny_semantickitti_dataset(tmp_path):
    """Verify that the dataset viewer can load and display SemanticKITTI data."""
    dataset_path, config_path = create_tiny_semantickitti_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )
    get_by_label(app.text_input, "SemanticKITTI Config YAML").input(
        str(config_path)
    ).run(timeout=10)

    assert not app.exception
    assert app.session_state["lidar_dataset_type"] == "SemanticKITTI"
    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == str(dataset_path)
    assert app.session_state["lidar_config_path"] == str(config_path)

    selectbox_labels = [selectbox.label for selectbox in app.selectbox]
    assert "Frame" in selectbox_labels
    assert "Color By" in selectbox_labels

    slider_labels = [slider.label for slider in app.slider]
    assert "Point Size" in slider_labels

    number_input_labels = [number_input.label for number_input in app.number_input]
    assert "Max Points" in number_input_labels

    button_labels = [button.label for button in app.button]
    assert "Reset Top View" in button_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid SemanticKITTI dataset folder." not in warning_text
    assert "Please select a valid SemanticKITTI config YAML file." not in warning_text


def test_lidar_segmentation_gui_switches_to_intensity_controls(tmp_path):
    """Verify that intensity color mode renders the intensity clipping control."""
    dataset_path, config_path = create_tiny_semantickitti_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )
    get_by_label(app.text_input, "SemanticKITTI Config YAML").input(
        str(config_path)
    ).run(timeout=10)
    get_by_label(app.selectbox, "Color By").select("Intensity").run(timeout=10)

    assert app.session_state["semantic_kitti_color_by"] == "Intensity"

    slider_labels = [slider.label for slider in app.slider]
    assert "Point Size" in slider_labels
    assert any("Intensity" in label for label in slider_labels)


def test_lidar_segmentation_evaluator_uses_loaded_semantickitti_dataset(tmp_path):
    """Verify that evaluator sees the selected SemanticKITTI dataset before model load."""
    dataset_path, config_path = create_tiny_semantickitti_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)
    select_lidar_segmentation_task(app)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )
    get_by_label(app.text_input, "SemanticKITTI Config YAML").input(
        str(config_path)
    ).run(timeout=10)

    success_text = "\n".join(success.value for success in app.success)
    warning_text = "\n".join(warning.value for warning in app.warning)

    assert f"Dataset loaded: {dataset_path} (val split) - 1 samples" in success_text
    assert "No LiDAR model loaded" in warning_text

    run_evaluation = next(
        button for button in app.button if button.label == "🚀 Run Evaluation"
    )
    assert run_evaluation.disabled