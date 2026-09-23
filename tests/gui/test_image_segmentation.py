import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


def create_tiny_cityscapes_dataset(tmp_path):
    """Create a minimal Cityscapes-style validation split for GUI tests."""
    city = "frankfurt"
    sample_id = "frankfurt_000000_000294"

    image_dir = (
        tmp_path
        / "leftImg8bit_trainvaltest"
        / "leftImg8bit"
        / "val"
        / city
    )
    label_dir = tmp_path / "gtFine" / "val" / city
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_path = image_dir / f"{sample_id}_leftImg8bit.png"
    label_path = label_dir / f"{sample_id}_gtFine_labelIds.png"

    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(image_path)

    label = np.zeros((32, 32), dtype=np.uint8)
    label[8:24, 8:24] = 7
    Image.fromarray(label).save(label_path)

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


def select_image_segmentation_task(app):
    """Switch the app to the image segmentation task."""
    get_by_label(app.selectbox, "Task").select("Image Segmentation").run(timeout=10)
    return app


def test_image_segmentation_gui_default_view():
    """Verify that the Streamlit app opens the image segmentation task."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    assert not app.exception
    assert app.session_state["task"] == "Image Segmentation"
    assert app.session_state["segmentation_dataset_type"] == "Cityscapes"

    tab_labels = [tab.label for tab in app.tabs]
    assert tab_labels == ["Dataset Viewer", "Inference", "Evaluator"]

    selectbox_labels = [selectbox.label for selectbox in app.selectbox]
    assert "Task" in selectbox_labels
    assert "Type" in selectbox_labels
    assert "Split" in selectbox_labels

    text_input_labels = [text_input.label for text_input in app.text_input]
    assert "Dataset Folder" in text_input_labels
    assert "Image Directory" in text_input_labels
    assert "Label Directory" in text_input_labels
    assert "Image Suffix" in text_input_labels
    assert "Label Suffix" in text_input_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid image segmentation dataset folder." in warning_text
    assert "Load a segmentation model from the sidebar to start inference." in warning_text


def test_image_segmentation_sidebar_switches_from_cityscapes_to_nuimages():
    """Verify that dataset-specific sidebar inputs update when switching type."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    get_by_label(app.selectbox, "Type").select("NuImages").run(timeout=10)

    assert app.session_state["segmentation_dataset_type"] == "NuImages"

    text_input_labels = [text_input.label for text_input in app.text_input]
    assert "Version" in text_input_labels
    assert "Generated Labels Directory" in text_input_labels
    assert "Image Directory" not in text_input_labels
    assert "Label Directory" not in text_input_labels


def test_image_segmentation_sidebar_updates_cityscapes_inputs():
    """Verify that Cityscapes sidebar inputs update Streamlit session state."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(
        "/tmp/not_a_real_segmentation_dataset"
    ).run(timeout=10)
    get_by_label(app.text_input, "Image Directory").input("leftImg8bit").run(timeout=10)
    get_by_label(app.text_input, "Label Directory").input("gtFine").run(timeout=10)
    get_by_label(app.text_input, "Image Suffix").input("_leftImg8bit.png").run(
        timeout=10
    )
    get_by_label(app.text_input, "Label Suffix").input("_gtFine_labelIds.png").run(
        timeout=10
    )
    get_by_label(app.checkbox, "Use Train IDs").check().run(timeout=10)

    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == "/tmp/not_a_real_segmentation_dataset"
    assert app.session_state["segmentation_image_dir"] == "leftImg8bit"
    assert app.session_state["segmentation_label_dir"] == "gtFine"
    assert app.session_state["segmentation_image_suffix"] == "_leftImg8bit.png"
    assert app.session_state["segmentation_label_suffix"] == "_gtFine_labelIds.png"
    assert app.session_state["segmentation_use_train_id"] is True


def test_image_segmentation_sidebar_updates_model_inputs():
    """Verify that segmentation model sidebar inputs update session state."""
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    get_by_label(app.selectbox, "Model Type").select("Hugging Face SegFormer").run(
        timeout=10
    )
    get_by_label(app.text_input, "Model Name or Folder").input(
        "tejasstanley/segformer-cityscapes"
    ).run(timeout=10)
    get_by_label(app.text_input, "Config File").input(
        "/tmp/segformer_cityscapes_cfg.json"
    ).run(timeout=10)
    get_by_label(app.text_input, "Ontology File").input(
        "/tmp/cityscapes_trainid_ontology.json"
    ).run(timeout=10)

    assert app.session_state["segmentation_model_type"] == "Hugging Face SegFormer"
    assert (
        app.session_state["segmentation_model_path"]
        == "tejasstanley/segformer-cityscapes"
    )
    assert app.session_state["segmentation_config_path"] == (
        "/tmp/segformer_cityscapes_cfg.json"
    )
    assert app.session_state["segmentation_ontology_path"] == (
        "/tmp/cityscapes_trainid_ontology.json"
    )


def test_image_segmentation_gui_loads_tiny_cityscapes_dataset(tmp_path):
    """Verify that the dataset viewer can load and display a small Cityscapes dataset."""
    dataset_path = create_tiny_cityscapes_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    get_by_label(app.selectbox, "Type").select("Cityscapes").run(timeout=10)
    get_by_label(app.selectbox, "Split").select("val").run(timeout=10)
    get_by_label(app.text_input, "Dataset Folder").input(str(dataset_path)).run(
        timeout=10
    )

    assert not app.exception
    assert app.session_state["segmentation_dataset_type"] == "Cityscapes"
    assert app.session_state["split"] == "val"
    assert app.session_state["dataset_path"] == str(dataset_path)

    success_text = "\n".join(success.value for success in app.success)
    assert f"Dataset loaded: {dataset_path} (val split) - 1 samples" in success_text

    button_labels = [button.label for button in app.button]
    assert "⟨" in button_labels
    assert "⟩" in button_labels
    assert "🔍" in button_labels    

    slider_labels = [slider.label for slider in app.slider]
    assert "Mask Opacity" in slider_labels

    warning_text = "\n".join(warning.value for warning in app.warning)
    assert "Please select a valid image segmentation dataset folder." not in warning_text


def test_image_segmentation_evaluator_uses_loaded_cityscapes_dataset(tmp_path):
    """Verify that evaluator sees the selected Cityscapes dataset before model load."""
    dataset_path = create_tiny_cityscapes_dataset(tmp_path)
    app = AppTest.from_file("app.py").run(timeout=10)
    select_image_segmentation_task(app)

    get_by_label(app.selectbox, "Type").select("Cityscapes").run(timeout=10)
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