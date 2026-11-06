import io
import json

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from gui.tasks.image_detection.dataset_viewer import (
    get_image_detection_dataset_key,
    load_image_detection_dataset,
)


def create_tiny_coco_dataset(tmp_path):
    """Create a minimal COCO-style validation split for GUI tests."""
    image_dir = tmp_path / "images" / "val2017"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()

    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(image_dir / "sample.jpg")

    annotation = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.jpg",
                "width": 32,
                "height": 32,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 5, 12, 10],
                "area": 120,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "car",
                "supercategory": "vehicle",
            }
        ],
    }

    with open(annotation_dir / "instances_val2017.json", "w", encoding="utf-8") as f:
        json.dump(annotation, f)

    return tmp_path


def create_tiny_yolo_dataset(tmp_path):
    """Create a minimal YOLO dataset with one train and one val sample.

    :return: Dataset directory and the YAML configuration as bytes
    :rtype: tuple
    """
    for split, name in [("train", "train_sample"), ("val", "val_sample")]:
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), color=(40, 80, 120)).save(image_dir / f"{name}.jpg")
        (label_dir / f"{name}.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    config = "train: images/train\nval: images/val\nnames:\n  0: car\n"
    return tmp_path, config.encode("utf-8")


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


def test_image_detection_gui_default_view():
    """Verify that the Streamlit app opens on the image detection task."""
    app = AppTest.from_file("app.py").run(timeout=10)

    assert not app.exception
    assert app.session_state["task"] == "Image Detection"
    assert app.session_state["dataset_type"] == "YOLO"

    tab_labels = [tab.label for tab in app.tabs]
    assert tab_labels == ["Dataset Viewer", "Inference", "Evaluator"]

    selectbox_labels = [selectbox.label for selectbox in app.selectbox]
    assert "Task" in selectbox_labels
    assert "Type" in selectbox_labels
    assert "Split" in selectbox_labels

    uploader_labels = file_uploader_labels(app)
    assert "Dataset Configuration (.yaml)" in uploader_labels
    assert "Model File (.pt, .onnx, .h5, .pb, .pth, .torchscript)" in uploader_labels
    assert "Ontology File (.json)" in uploader_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid dataset folder." in warning_text
    assert "Load a model from the sidebar to start inference" in warning_text


def test_image_detection_gui_switches_from_yolo_to_coco():
    """Verify that dataset-specific sidebar inputs update when switching type."""
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.selectbox, "Type").select("COCO").run(timeout=10)

    assert app.session_state["dataset_type"] == "COCO"

    uploader_labels = file_uploader_labels(app)
    assert "Dataset Configuration (.yaml)" not in uploader_labels


def test_image_detection_gui_updates_dataset_inputs():
    """Verify that sidebar dataset inputs update Streamlit session state."""
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(
        "/tmp/not_a_real_detection_dataset"
    ).run(timeout=10)

    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == "/tmp/not_a_real_detection_dataset"


def test_image_detection_sidebar_updates_model_config_inputs():
    """Verify that manual model configuration controls update session state."""
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.slider, "Confidence Threshold").set_value(0.7).run(timeout=10)
    get_by_label(app.slider, "NMS Threshold").set_value(0.3).run(timeout=10)
    get_by_label(app.number_input, "Max Detections/Image").set_value(25).run(
        timeout=10
    )
    get_by_label(app.selectbox, "Model Format").select("YOLO").run(timeout=10)
    get_by_label(app.number_input, "Batch Size").set_value(4).run(timeout=10)
    get_by_label(app.number_input, "Evaluation Step").set_value(10).run(timeout=10)

    assert app.session_state["confidence_threshold"] == 0.7
    assert app.session_state["nms_threshold"] == 0.3
    assert app.session_state["max_detections"] == 25
    assert app.session_state["model_format"] == "YOLO"
    assert app.session_state["batch_size"] == 4
    assert app.session_state["evaluation_step"] == 10

    get_by_label(app.radio, "Resize Strategy").set_value("Fixed Dimensions").run(
        timeout=10
    )
    get_by_label(app.number_input, "Image Resize Height").set_value(480).run(
        timeout=10
    )
    get_by_label(app.number_input, "Image Resize Width").set_value(640).run(timeout=10)

    assert app.session_state["resize_strategy"] == "Fixed Dimensions"
    assert app.session_state["resize_height"] == 480
    assert app.session_state["resize_width"] == 640

    get_by_label(app.checkbox, "Enable Padding to Closest Multiple").uncheck().run(
        timeout=10
    )
    get_by_label(app.checkbox, "Enable Center Crop").check().run(timeout=10)
    get_by_label(app.number_input, "Crop Height").set_value(320).run(timeout=10)
    get_by_label(app.number_input, "Crop Width").set_value(320).run(timeout=10)

    assert app.session_state["enable_pad"] is False
    assert app.session_state["enable_crop"] is True
    assert app.session_state["crop_height"] == 320
    assert app.session_state["crop_width"] == 320


def test_image_detection_gui_loads_tiny_coco_dataset(tmp_path):
    """Verify that the dataset viewer can load and display a small COCO dataset."""
    dataset_path = create_tiny_coco_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.selectbox, "Type").select("COCO").run(timeout=10)
    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )

    assert not app.exception
    assert app.session_state["dataset_type"] == "COCO"
    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == str(dataset_path)

    dataset_key = get_image_detection_dataset_key("COCO", str(dataset_path), "val")
    assert dataset_key in app.session_state
    assert len(app.session_state[dataset_key].dataset) == 1

    button_labels = [button.label for button in app.button]
    assert "⟨" in button_labels
    assert "⟩" in button_labels
    assert "🔍" in button_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid dataset folder." not in warning_text


def test_image_detection_evaluator_uses_loaded_coco_dataset(tmp_path):
    """Verify that the evaluator sees the selected COCO dataset before model load."""
    dataset_path = create_tiny_coco_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.selectbox, "Type").select("COCO").run(timeout=10)
    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )

    success_text = "\n".join(success.value for success in app.success)
    warning_text = "\n".join(warning.value for warning in app.warning)

    assert f"Dataset loaded: {dataset_path} (val split) - 1 samples" in success_text
    assert "No model loaded" in warning_text

    run_evaluation = next(
        button for button in app.button if button.label == "🚀 Run Evaluation"
    )
    assert run_evaluation.disabled


def test_image_detection_evaluator_does_not_reject_yolo(tmp_path):
    """Verify that selecting YOLO no longer shows a COCO-only warning (#529)."""
    dataset_path, _ = create_tiny_yolo_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)

    get_by_label(app.selectbox, "Type").select("YOLO").run(timeout=10)
    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )

    warning_text = "\n".join(warning.value for warning in app.warning)

    assert not app.exception
    assert "Only COCO datasets are currently supported" not in warning_text
    assert "Please upload the YOLO dataset configuration" in warning_text


def test_load_image_detection_dataset_yolo(tmp_path):
    """Verify that the shared loader builds a YOLO dataset filtered by split."""
    dataset_path, config = create_tiny_yolo_dataset(tmp_path)

    dataset = load_image_detection_dataset(
        "YOLO", str(dataset_path), "val", io.BytesIO(config)
    )

    assert len(dataset.dataset) == 1
    assert set(dataset.dataset["split"]) == {"val"}
    assert "car" in dataset.ontology


def test_load_image_detection_dataset_yolo_requires_config(tmp_path):
    """Verify that YOLO loading fails with a clear message without a config."""
    dataset_path, _ = create_tiny_yolo_dataset(tmp_path)

    with pytest.raises(FileNotFoundError, match="YOLO dataset configuration"):
        load_image_detection_dataset("YOLO", str(dataset_path), "val")


def test_load_image_detection_dataset_coco(tmp_path):
    """Verify that the shared loader builds a COCO dataset."""
    dataset_path = create_tiny_coco_dataset(tmp_path)

    dataset = load_image_detection_dataset("COCO", str(dataset_path), "val")

    assert len(dataset.dataset) == 1


def test_image_detection_dataset_key_depends_on_type():
    """Verify that COCO and YOLO datasets on the same path are cached separately."""
    assert get_image_detection_dataset_key(
        "COCO", "/data", "val"
    ) != get_image_detection_dataset_key("YOLO", "/data", "val")
