import numpy as np
import pytest
from perceptionmetrics.utils.segmentation_metrics import SegmentationMetricsFactory
from perceptionmetrics.utils.detection_metrics import (
    compute_iou,
    compute_iou_matrix,
)


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


# ---------------------------------------------------------------------------
# Tests for compute_iou_matrix (vectorized)
# ---------------------------------------------------------------------------


def _iou_matrix_reference(pred_boxes, gt_boxes):
    """Scalar reference implementation using compute_iou in a double loop."""
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            matrix[i, j] = compute_iou(pb, gb)
    return matrix


class TestComputeIoUMatrix:
    def test_matches_scalar_reference(self):
        """Vectorized result must be identical to the scalar double-loop."""
        pred = np.array(
            [[10, 10, 50, 50], [30, 30, 70, 70], [100, 100, 200, 200]]
        )
        gt = np.array([[10, 10, 50, 50], [40, 40, 80, 80]])
        expected = _iou_matrix_reference(pred, gt)
        result = compute_iou_matrix(pred, gt)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_perfect_overlap(self):
        """Identical boxes should yield IoU = 1.0."""
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        result = compute_iou_matrix(boxes, boxes)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes should yield IoU = 0.0."""
        pred = np.array([[0, 0, 10, 10]])
        gt = np.array([[20, 20, 30, 30]])
        result = compute_iou_matrix(pred, gt)
        assert result[0, 0] == 0.0

    def test_empty_predictions(self):
        """Empty pred_boxes should return an empty matrix."""
        pred = np.empty((0, 4))
        gt = np.array([[0, 0, 10, 10]])
        result = compute_iou_matrix(pred, gt)
        assert result.shape == (0, 1)

    def test_empty_ground_truth(self):
        """Empty gt_boxes should return an empty matrix."""
        pred = np.array([[0, 0, 10, 10]])
        gt = np.empty((0, 4))
        result = compute_iou_matrix(pred, gt)
        assert result.shape == (1, 0)

    def test_single_box_each(self):
        """Single pred vs single gt must match compute_iou."""
        pred = np.array([[0, 0, 20, 20]])
        gt = np.array([[10, 10, 30, 30]])
        expected = compute_iou(pred[0], gt[0])
        result = compute_iou_matrix(pred, gt)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-10)

    def test_large_random_batch(self):
        """Stress test: 200 preds x 150 gts must match the scalar loop."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(0, 500, size=(200, 2))
        wh = rng.uniform(10, 100, size=(200, 2))
        pred = np.column_stack([xy, xy + wh])

        xy = rng.uniform(0, 500, size=(150, 2))
        wh = rng.uniform(10, 100, size=(150, 2))
        gt = np.column_stack([xy, xy + wh])

        expected = _iou_matrix_reference(pred, gt)
        result = compute_iou_matrix(pred, gt)
        np.testing.assert_allclose(result, expected, atol=1e-10)
