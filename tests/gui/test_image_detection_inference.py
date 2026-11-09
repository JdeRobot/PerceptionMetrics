from contextlib import nullcontext
from io import BytesIO
import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image
import pytest

from gui.tasks.image_detection.inference import (
    render_image_detection_inference,
    run_image_detection_inference,
)


def _prediction_tensor(values):
    """Provide tensor operations used by inference without requiring PyTorch.

    :param values: NumPy array containing prediction data.
    :type values: numpy.ndarray
    :return: Tensor-like mock backed by the supplied array.
    :rtype: unittest.mock.MagicMock
    """
    tensor = MagicMock(spec=["cpu", "numpy", "mean", "max", "__len__"])
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = values
    tensor.__len__.return_value = len(values)
    tensor.mean.side_effect = values.mean
    tensor.max.side_effect = values.max
    return tensor


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
        "boxes": _prediction_tensor(np.empty((0, 4))),
        "labels": _prediction_tensor(np.empty((0,), dtype=np.int64)),
        "scores": _prediction_tensor(np.empty((0,))),
    }
    model.idx_to_class_name = {}

    _, result_image = run_image_detection_inference(model, image)

    np.testing.assert_array_equal(result_image, np.asarray(image))


@pytest.mark.parametrize(
    "label_map, expected_class_name",
    [({1: "object"}, "object"), (None, "class_1"), ({2: "other"}, "class_1")],
)
def test_inference_renders_details_for_detected_objects(label_map, expected_class_name):
    """A detected object must appear in the results panel without an error."""
    image_file = BytesIO()
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(image_file, "PNG")
    image_file.seek(0)
    predictions = {
        "boxes": _prediction_tensor(np.array([[1, 2, 6, 7]], dtype=np.float32)),
        "labels": _prediction_tensor(np.array([1], dtype=np.int64)),
        "scores": _prediction_tensor(np.array([0.9], dtype=np.float32)),
    }
    model = Mock()
    model.predict.return_value = predictions
    model.idx_to_class_name = label_map
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
    streamlit.json.assert_called_once()
    result = streamlit.json.call_args.args[0][0]
    assert result["class_name"] == expected_class_name
    assert result["class_id"] == 1
    assert result["confidence"] == pytest.approx(0.9)
    assert result["bbox_xyxy"] == [1, 2, 6, 7]
    streamlit.metric.assert_any_call("Total Detections", 1)
    streamlit.metric.assert_any_call("Avg Confidence", "0.900")
    streamlit.metric.assert_any_call("Max Confidence", "0.900")
    streamlit.download_button.assert_called_once()
    assert json.loads(streamlit.download_button.call_args.kwargs["data"]) == [result]
