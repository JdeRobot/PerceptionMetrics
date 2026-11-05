from contextlib import nullcontext
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

from PIL import Image
import torch

from gui.tasks.image_detection.inference import (
    render_image_detection_inference,
    run_image_detection_inference,
)


def test_inference_draws_detections_on_original_image():
    """Verify normalized model input is not reused as the display image."""
    image = Image.new("RGB", (4, 4), color=(128, 64, 32))
    predictions = {"boxes": [], "labels": [], "scores": []}
    model = Mock()
    model.predict.return_value = predictions
    model.idx_to_class_name = {0: "object"}

    with patch(
        "gui.tasks.image_detection.inference.draw_detections",
        return_value="rendered-image",
    ) as draw_detections:
        result_predictions, result_image = run_image_detection_inference(model, image)

    model.predict.assert_called_once_with(image)
    draw_detections.assert_called_once_with(image, predictions, model.idx_to_class_name)
    assert result_predictions is predictions
    assert result_image == "rendered-image"


def test_inference_keeps_uploaded_rgb_pixels_without_detections():
    """The displayed image should retain its original color values."""
    image = Image.new("RGB", (4, 4), color=(128, 64, 32))
    model = Mock()
    model.predict.return_value = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
    }
    model.idx_to_class_name = {}

    _, result_image = run_image_detection_inference(model, image)

    assert tuple(result_image[0, 0]) == (128, 64, 32)


def test_inference_renders_details_for_detected_objects():
    """A detected object must appear in the results panel without an error."""
    image_file = BytesIO()
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(image_file, "PNG")
    image_file.seek(0)
    predictions = {
        "boxes": torch.tensor([[1, 2, 6, 7]], dtype=torch.float32),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    model = Mock()
    model.predict.return_value = predictions
    model.idx_to_class_name = {1: "object"}
    streamlit = MagicMock()
    streamlit.session_state.__contains__.return_value = True
    streamlit.session_state.detection_model = model
    streamlit.file_uploader.return_value = image_file
    streamlit.spinner.return_value = nullcontext()
    streamlit.expander.return_value = nullcontext()
    streamlit.columns.return_value = [nullcontext()] * 3

    with patch("gui.tasks.image_detection.inference.st", streamlit), patch(
        "gui.tasks.image_detection.inference.draw_detections",
        return_value="rendered-image",
    ):
        render_image_detection_inference()

    streamlit.error.assert_not_called()
    streamlit.image.assert_called_once()
    assert streamlit.json.call_args.args[0][0]["class_name"] == "object"
