import json

from PIL import Image
from streamlit.testing.v1 import AppTest


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

    dataset_key = f"{dataset_path}_val"
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
