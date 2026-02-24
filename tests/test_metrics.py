import math

import numpy as np
import pytest
from perceptionmetrics.utils.detection_metrics import DetectionMetricsFactory
from perceptionmetrics.utils.segmentation_metrics import SegmentationMetricsFactory


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
def test_segmentation_reset_clears_data(metrics_factory):
    """Test that reset() clears all accumulated data."""
    pred = np.array([0, 1, 2, 2, 1])
    gt = np.array([0, 1, 1, 2, 2])
    metrics_factory.update(pred, gt)

    # Verify data was accumulated
    assert metrics_factory.get_confusion_matrix().sum() > 0

    # Reset and verify empty state
    metrics_factory.reset()
    expected_empty = np.zeros((3, 3), dtype=np.int64)
    assert np.array_equal(metrics_factory.get_confusion_matrix(), expected_empty)


def test_segmentation_reset_allows_reuse(metrics_factory):
    """Test that factory can be reused after reset()."""
    # First evaluation
    pred1 = np.array([0, 1, 2])
    gt1 = np.array([0, 1, 2])
    metrics_factory.update(pred1, gt1)
    metrics_factory.reset()

    # Second evaluation with different data
    pred2 = np.array([0, 0, 1, 1, 2, 2])
    gt2 = np.array([0, 1, 0, 1, 2, 2])
    metrics_factory.update(pred2, gt2)

    # Verify correct metrics for second evaluation only
    expected_cm = np.array(
        [
            [1, 1, 0],  # True class 0: 1 TP, 1 FN (predicted as 1)
            [1, 1, 0],  # True class 1: 1 FP (from class 0), 1 TP
            [0, 0, 2],  # True class 2: 2 TP
        ]
    )
    assert np.array_equal(metrics_factory.get_confusion_matrix(), expected_cm)


def test_segmentation_reset_multiple_cycles():
    """Test multiple reset/reuse cycles produce consistent results."""
    factory = SegmentationMetricsFactory(n_classes=2)

    for _ in range(3):
        pred = np.array([0, 1, 0, 1])
        gt = np.array([0, 1, 0, 1])
        factory.update(pred, gt)

        # Check perfect prediction
        assert np.array_equal(factory.get_tp(), np.array([2, 2]))
        factory.reset()

        # Verify empty after reset
        assert factory.get_confusion_matrix().sum() == 0


# Tests for DetectionMetricsFactory.reset()
@pytest.fixture
def detection_factory():
    """Fixture to create a DetectionMetricsFactory instance for testing."""
    return DetectionMetricsFactory(iou_threshold=0.5, num_classes=3)


def test_detection_reset_clears_data(detection_factory):
    """Test that reset() clears all accumulated data."""
    gt_boxes = np.array([[0, 0, 10, 10]])
    gt_labels = np.array([0])
    pred_boxes = np.array([[0, 0, 10, 10]])
    pred_labels = np.array([0])
    pred_scores = np.array([0.9])

    detection_factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)

    # Verify data was accumulated
    assert len(detection_factory.results) > 0
    assert len(detection_factory.raw_data) > 0
    assert sum(detection_factory.gt_counts.values()) > 0

    # Reset and verify empty state
    detection_factory.reset()
    assert len(detection_factory.results) == 0
    assert len(detection_factory.raw_data) == 0
    assert sum(detection_factory.gt_counts.values()) == 0


def test_detection_reset_allows_reuse(detection_factory):
    """Test that factory can be reused after reset()."""
    # First evaluation
    gt_boxes1 = np.array([[0, 0, 10, 10]])
    gt_labels1 = np.array([0])
    pred_boxes1 = np.array([[0, 0, 10, 10]])
    pred_labels1 = np.array([0])
    pred_scores1 = np.array([0.9])
    detection_factory.update(
        gt_boxes1, gt_labels1, pred_boxes1, pred_labels1, pred_scores1
    )
    detection_factory.reset()

    # Second evaluation with different data
    gt_boxes2 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    gt_labels2 = np.array([0, 1])
    pred_boxes2 = np.array([[0, 0, 10, 10]])
    pred_labels2 = np.array([0])
    pred_scores2 = np.array([0.8])
    detection_factory.update(
        gt_boxes2, gt_labels2, pred_boxes2, pred_labels2, pred_scores2
    )

    metrics = detection_factory.compute_metrics()

    # Class 0 should have perfect recall (1 TP, 0 FN)
    assert metrics[0]["TP"] == 1
    assert metrics[0]["FN"] == 0
    # Class 1 should have 0 TP and 1 FN (missed detection)
    assert metrics[1]["TP"] == 0
    assert metrics[1]["FN"] == 1


def test_detection_reset_multiple_cycles():
    """Test multiple reset/reuse cycles produce consistent results."""
    factory = DetectionMetricsFactory(iou_threshold=0.5)

    for _ in range(3):
        gt_boxes = np.array([[0, 0, 10, 10]])
        gt_labels = np.array([0])
        pred_boxes = np.array([[0, 0, 10, 10]])
        pred_labels = np.array([0])
        pred_scores = np.array([0.9])

        factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
        metrics = factory.compute_metrics()

        # Should have perfect AP for class 0
        assert metrics[0]["AP"] == 1.0
        factory.reset()

        # Verify empty after reset
        assert len(factory.results) == 0
        assert len(factory.raw_data) == 0
