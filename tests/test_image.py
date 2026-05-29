import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
import pytest
import supervision as sv
from perceptionmetrics.utils.image import draw_detections


def test_draw_detections_calls_supervision():
    # Setup
    img = Image.new("RGB", (100, 100))
    boxes = np.array([[0, 0, 10, 10]])
    class_ids = np.array([0])
    class_names = ["cat"]
    scores = np.array([0.9])

    mock_box_annotator = MagicMock()
    mock_box_annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("perceptionmetrics.utils.image.sv.Detections") as mock_detections, patch(
        "perceptionmetrics.utils.image.sv.BoxAnnotator", return_value=mock_box_annotator
    ), patch("perceptionmetrics.utils.image.sv.Color"), patch(
        "perceptionmetrics.utils.image.sv.ColorPalette"
    ):

        draw_detections(img, boxes, class_ids, class_names, scores)

        assert mock_detections.called
        args, kwargs = mock_detections.call_args
        assert np.array_equal(kwargs["xyxy"], boxes)
        assert np.array_equal(kwargs["class_id"], class_ids)
        assert np.array_equal(kwargs["confidence"], scores)

        assert mock_box_annotator.annotate.called


def test_draw_detections_label_construction():
    # Setup
    img = Image.new("RGB", (100, 100))
    boxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
    class_ids = np.array([0, 1])
    class_names = ["cat", "dog"]
    scores = np.array([0.9, 0.8])

    # First BoxAnnotator (older style) will fail on annotate
    mock_box_annotator_old = MagicMock()
    mock_box_annotator_old.annotate.side_effect = TypeError(
        "Simulation of mismatching arguments"
    )

    # Second BoxAnnotator (modern style) will succeed
    mock_box_annotator_new = MagicMock()
    mock_box_annotator_new.annotate.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    mock_label_annotator = MagicMock()
    mock_label_annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "perceptionmetrics.utils.image.sv.BoxAnnotator",
        side_effect=[mock_box_annotator_old, mock_box_annotator_new],
    ), patch(
        "perceptionmetrics.utils.image.sv.LabelAnnotator",
        return_value=mock_label_annotator,
    ), patch(
        "perceptionmetrics.utils.image.sv.Color"
    ), patch(
        "perceptionmetrics.utils.image.sv.ColorPalette"
    ):

        draw_detections(img, boxes, class_ids, class_names, scores)

        # Check labels passed to LabelAnnotator
        assert mock_label_annotator.annotate.called
        args, kwargs = mock_label_annotator.annotate.call_args
        labels = kwargs.get("labels")
        assert labels == ["cat: 0.90", "dog: 0.80"]


def test_draw_detections_missing_names():
    img = Image.new("RGB", (100, 100))
    boxes = np.array([[0, 0, 10, 10]])
    class_ids = np.array([5])
    class_names = []  # No name for ID 5

    mock_box_annotator_old = MagicMock()
    mock_box_annotator_old.annotate.side_effect = TypeError()
    mock_box_annotator_new = MagicMock()
    mock_box_annotator_new.annotate.return_value = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    mock_label_annotator = MagicMock()
    mock_label_annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "perceptionmetrics.utils.image.sv.BoxAnnotator",
        side_effect=[mock_box_annotator_old, mock_box_annotator_new],
    ), patch(
        "perceptionmetrics.utils.image.sv.LabelAnnotator",
        return_value=mock_label_annotator,
    ), patch(
        "perceptionmetrics.utils.image.sv.Color"
    ), patch(
        "perceptionmetrics.utils.image.sv.ColorPalette"
    ):

        draw_detections(img, boxes, class_ids, class_names)

        assert mock_label_annotator.annotate.called
        args, kwargs = mock_label_annotator.annotate.call_args
        labels = kwargs.get("labels")
        assert labels == ["5"]
