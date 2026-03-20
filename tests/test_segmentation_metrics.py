import math

import numpy as np
from numpy.testing import assert_allclose
import pytest
from perceptionmetrics.utils.detection_metrics import DetectionMetricsFactory
from perceptionmetrics.utils.segmentation_metrics import SegmentationMetricsFactory
from perceptionmetrics.utils.detection_metrics import compute_iou_matrix, compute_iou


@pytest.fixture
def metrics_factory():
    """Fixture to create a SegmentationMetricsFactory instance for testing"""
    return SegmentationMetricsFactory(n_classes=3)


def test_update_confusion_matrix(metrics_factory):
    """Test confusion matrix updates correctly"""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])

    metrics_factory.update(pred, gt)
    confusion_matrix = metrics_factory.get_confusion_matrix()

    expected = np.array(
        [
            [1, 0, 0],  # True class 0
            [0, 1, 1],  # True class 1
            [0, 1, 1],  # True class 2
        ]
    )
    assert np.array_equal(confusion_matrix, expected), "Confusion matrix mismatch"


def test_get_tp_fp_fn_tn(metrics_factory):
    pred = np.array([0, 1, 1, 2, 2])
    gt = np.array([0, 1, 1, 2, 2])
    metrics_factory.update(pred, gt)

    assert np.array_equal(metrics_factory.get_tp(), np.array([1, 2, 2]))
    assert np.array_equal(metrics_factory.get_fp(), np.array([0, 0, 0]))
    assert np.array_equal(metrics_factory.get_fn(), np.array([0, 0, 0]))
    assert np.array_equal(metrics_factory.get_tn(), np.array([4, 3, 3]))


def test_recall(metrics_factory):
    """Test recall calculation"""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])

    metrics_factory.update(pred, gt)

    expected_recall = np.array([1.0, 0.5, 0.5])
    computed_recall = metrics_factory.get_recall()

    assert np.allclose(computed_recall, expected_recall, equal_nan=True)


def test_accuracy(metrics_factory):
    """Test global accuracy calculation (non per-class)"""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])

    metrics_factory.update(pred, gt)

    TP = metrics_factory.get_tp(per_class=False)
    FP = metrics_factory.get_fp(per_class=False)
    FN = metrics_factory.get_fn(per_class=False)
    TN = metrics_factory.get_tn(per_class=False)

    total = TP + FP + FN + TN
    expected_accuracy = (TP + TN) / total if total > 0 else math.nan

    computed_accuracy = metrics_factory.get_accuracy(per_class=False)
    assert np.isclose(computed_accuracy, expected_accuracy, equal_nan=True)


def test_f1_score(metrics_factory):
    """Test F1-score calculation"""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])

    metrics_factory.update(pred, gt)

    precision = np.array([1.0, 0.5, 0.5])
    recall = np.array([1.0, 0.5, 0.5])
    expected_f1 = 2 * (precision * recall) / (precision + recall)

    computed_f1 = metrics_factory.get_f1_score()

    assert np.allclose(computed_f1, expected_f1, equal_nan=True)


def test_edge_cases(metrics_factory):
    """Test edge cases like empty arrays and division by zero"""
    pred = np.array([])
    gt = np.array([])

    with pytest.raises(AssertionError):
        metrics_factory.update(pred, gt)

    empty_metrics_factory = SegmentationMetricsFactory(n_classes=3)

    assert np.isnan(empty_metrics_factory.get_precision(per_class=False))
    assert np.isnan(empty_metrics_factory.get_recall(per_class=False))
    assert np.isnan(empty_metrics_factory.get_f1_score(per_class=False))
    assert np.isnan(empty_metrics_factory.get_iou(per_class=False))


def test_macro_micro_weighted(metrics_factory):
    """Test macro, micro, and weighted metric averaging"""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])

    metrics_factory.update(pred, gt)

    macro_f1 = metrics_factory.get_averaged_metric("f1_score", method="macro")
    micro_f1 = metrics_factory.get_averaged_metric("f1_score", method="micro")

    weights = np.array([0.2, 0.5, 0.3])
    weighted_f1 = metrics_factory.get_averaged_metric(
        "f1_score", method="weighted", weights=weights
    )

    assert 0 <= macro_f1 <= 1
    assert 0 <= micro_f1 <= 1
    assert 0 <= weighted_f1 <= 1


# Tests for SegmentationMetricsFactory.reset()
def test_segmentation_reset_clears_data_and_allows_reuse():
    """Test reset() clears state and supports repeated reuse cycles."""
    factory = SegmentationMetricsFactory(n_classes=3)
    expected_empty = np.zeros((3, 3), dtype=np.int64)
    pred = np.array([0, 0, 1, 1, 2, 2])
    gt = np.array([0, 1, 0, 1, 2, 2])
    expected_cm = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 2],
        ]
    )

    for _ in range(3):
        factory.update(pred, gt)
        assert np.array_equal(factory.get_confusion_matrix(), expected_cm)

        factory.reset()
        assert np.array_equal(factory.get_confusion_matrix(), expected_empty)


# Tests for DetectionMetricsFactory.reset()
@pytest.fixture
def detection_factory():
    """Fixture to create a DetectionMetricsFactory instance for testing."""
    return DetectionMetricsFactory(iou_threshold=0.5, num_classes=3)


def test_detection_reset_clears_data_and_allows_reuse():
    """Test reset() clears state and supports repeated reuse cycles."""
    factory = DetectionMetricsFactory(iou_threshold=0.5, num_classes=3)
    gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    gt_labels = np.array([0, 1])
    pred_boxes = np.array([[0, 0, 10, 10]])
    pred_labels = np.array([0])
    pred_scores = np.array([0.8])

    for _ in range(3):
        factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
        metrics = factory.compute_metrics()

        assert metrics[0]["TP"] == 1
        assert metrics[0]["FN"] == 0
        assert metrics[1]["TP"] == 0
        assert metrics[1]["FN"] == 1
        assert len(factory.results) > 0
        assert len(factory.raw_data) > 0
        assert sum(factory.gt_counts.values()) > 0

        factory.reset()
        assert len(factory.results) == 0
        assert len(factory.raw_data) == 0
        assert sum(factory.gt_counts.values()) == 0


def compute_iou_matrix_reference(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Compute IoU matrix between pred and gt boxes.

    :param pred_boxes: Predicted bounding boxes, shape (num_pred, 4)
    :type pred_boxes: np.ndarray
    :param gt_boxes: Ground truth bounding boxes, shape (num_gt, 4)
    :type gt_boxes: np.ndarray
    :return: IoU matrix with shape (num_pred, num_gt)
    :rtype: np.ndarray
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)
    return iou_matrix


# tests for vectorized iou calculation test
def test_compute_iou_matrix():
    rng = np.random.default_rng(42)

    # perfect overlap — diagonal must be 1.0
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    assert_allclose(np.diag(compute_iou_matrix(boxes, boxes)), 1.0)

    # no overlap — must be exactly 0.0
    assert (
        compute_iou_matrix(
            np.array([[0, 0, 10, 10.0]]), np.array([[20, 20, 30, 30.0]])
        )[0, 0]
        == 0.0
    )

    # empty inputs — shape must be correct
    dummy, empty = np.array([[0, 0, 10, 10.0]]), np.empty((0, 4))
    assert compute_iou_matrix(empty, dummy).shape == (0, 1)
    assert compute_iou_matrix(dummy, empty).shape == (1, 0)

    # random batches - must match reference implementation
    for N, M in [(1, 1), (3, 2), (200, 150)]:
        p_xy = rng.uniform(0, 500, (N, 2))
        gt_xy = rng.uniform(0, 500, (M, 2))
        pred = np.column_stack([p_xy, p_xy + rng.uniform(10, 100, (N, 2))])
        gt = np.column_stack([gt_xy, gt_xy + rng.uniform(10, 100, (M, 2))])
        assert_allclose(
            compute_iou_matrix(pred, gt),
            compute_iou_matrix_reference(pred, gt),
            atol=1e-10,
        )
