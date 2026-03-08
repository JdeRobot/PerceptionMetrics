import pytest
import numpy as np
from perceptionmetrics.utils.detection_metrics import DetectionMetricsFactory
from perceptionmetrics.utils.detection_metrics import compute_iou_matrix

@pytest.fixture
def metrics_factory():
    """Fixture to create a DetectionMetricsFactory instance (IoU=0.5)"""
    return DetectionMetricsFactory(iou_threshold=0.5)

def test_match_predictions_logic(metrics_factory):
    """Test matches are correctly assigned TP/FP according to overlap"""
    gt_boxes = np.array([[0, 0, 10, 10]])
    gt_labels = [1]
    
    pred_boxes = np.array([
        [0, 0, 9, 9],   # High overlap (Should be TP)
        [20, 20, 30, 30] # No overlap (Should be FP)
    ])
    pred_labels = [1, 1]
    pred_scores = [0.95, 0.6]

    # Call the internal matching method
    matches = metrics_factory._match_predictions(
        gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
    )
    
    results = matches[1]
    assert (0.95, 1) in results, "High overlap prediction should be a True Positive"
    assert (0.6, 0) in results, "Zero overlap prediction should be a False Positive"
    assert (None, -1) not in results, "GT was matched, so there should be no False Negative"


def test_compute_metrics(metrics_factory):
    """Test that multiple detections of the same object result in 1 TP and 1 FP"""
    gt_boxes = np.array([[10, 10, 50, 50]])
    gt_labels = [1]
    
    pred_boxes = np.array([
        [12, 12, 48, 48], # Pred A (High score)
        [11, 11, 49, 49]  # Pred B (Low score - the 'double')
    ])
    pred_labels = [1, 1]
    pred_scores = [0.90, 0.40]

    metrics_factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
    all_metrics = metrics_factory.compute_metrics()
    
    cat_metrics = all_metrics[1]


    assert cat_metrics["TP"] == 1, "Should only have 1 True Positive"
    assert cat_metrics["FP"] == 1, "The second box should be a False Positive"
    assert cat_metrics["FN"] == 0, "The object was found, so FN should be 0"
    
    # Recall = TP / (TP + FN) = 1 / (1 + 0) = 1.0
    assert cat_metrics["Recall"] == 1.0
    
    # Precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
    assert cat_metrics["Precision"] == 0.5

    # AP
    assert cat_metrics["AP"] == 1

def test_compute_iou_matrix_basic():
    """Verify that the matrix correctly maps N predictions to M ground truths."""
    pred_boxes = np.array([
        [0, 0, 10, 10],   # Pred 0
        [20, 20, 30, 30]  # Pred 1
    ])
    gt_boxes = np.array([
        [0, 0, 10, 10],   # GT 0: Exact match with Pred 0
        [0, 0, 5, 5],     # GT 1: Partial match with Pred 0
        [100, 100, 110, 110] # GT 2: No match
    ])

    matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    assert matrix.shape == (2, 3) # (num_pred, num_gt)

    assert matrix[0, 0] == 1.0, "Pred 0 vs GT 0 should be a perfect 1.0"
    assert 0 < matrix[0, 1] < 1.0, "Pred 0 vs GT 1 should be a partial overlap"
    assert matrix[1, 2] == 0.0, "Pred 1 vs GT 2 should have zero overlap"


def test_compute_coco_map_sensitivity(metrics_factory):
    """
    With IoU ≈ 0.68, the prediction is a TP for the first 4 thresholds 
    (0.50 to 0.65) and an FP for the remaining 6 thresholds (0.70 to 0.95), 
    resulting in mAP=4/10=0.4.
    """
    gt_boxes = np.array([[10, 10, 110, 110]])
    gt_labels = [1]
    
    pred_boxes = np.array([[20, 20, 120, 120]])
    pred_labels = [1]
    pred_scores = [0.99]

    metrics_factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
    coco_map = metrics_factory.compute_coco_map()

    assert np.isclose(coco_map, 0.4)
    assert coco_map < 1.0, "mAP should not be perfect for a shifted box"

def test_compute_coco_map_perfect_match(metrics_factory):
    """Test that a perfect overlap results in a 1.0 COCO mAP."""
    gt_boxes = np.array([[0, 0, 100, 100]])
    pred_boxes = np.array([[0, 0, 100, 100]])
    
    metrics_factory.update(gt_boxes, [1], pred_boxes, [1], [0.9])
    coco_map = metrics_factory.compute_coco_map()
    
    assert coco_map == 1.0, "Perfect overlap must yield perfect mAP across all thresholds"

def test_compute_coco_map_complex_multi_class(metrics_factory):
    """
    Verifies multi-class mAP by testing IoU threshold sensitivity (0.5:0.95) 
    and the correct penalization (0.0 AP) for classes present in ground truth 
    but missing from predictions.
    """
    # Class 1: 2 GTs, 2 Preds
    gt_boxes_c1 = np.array([[0, 0, 10, 8], [0, 12, 4, 22]])
    pred_boxes_c1 = np.array([[-0.5, -0.5, 10.5, 8.5], [0, 18, 3.5, 21.5]])
    pred_scores_c1 = [0.9, 0.65] # High score is TP (IoU 0.81), Low is FP (IoU 0.31)
    
    # Class 2: 3 GTs, 3 Preds
    gt_boxes_c2 = np.array([[15, 0, 20, 5], [15, 8, 19, 13], [15, 16, 21, 22]])
    pred_boxes_c2 = np.array([[14.8, 15.8, 21.2, 22.2], [14.5, 0.5, 19.5, 5.5], [16, 9, 18, 11]])
    pred_scores_c2 = [0.95, 0.80, 0.71] # 0.95=TP(0.88), 0.80=TP(0.68), 0.71=FP(0.20)

    gt_boxes = np.concatenate([gt_boxes_c1, gt_boxes_c2])
    gt_labels = [1, 1, 2, 2, 2]
    pred_boxes = np.concatenate([pred_boxes_c1, pred_boxes_c2])
    pred_labels = [1, 1, 2, 2, 2]
    pred_scores = pred_scores_c1 + pred_scores_c2

    metrics_factory.update(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)

    mAP_coco = metrics_factory.compute_coco_map()

    assert 0.38 <= mAP_coco <= 0.42, f"Expected mAP ~0.39, got {mAP_coco}"
    assert mAP_coco < 0.6, "mAP should be penalized for boxes with IoU < 0.95"
    
    metrics_factory.update(
        np.array([[100, 100, 110, 110]]), [3], # New GT for class 3
        np.empty((0, 4)), [], []               # No predictions
    )
    mAP_with_empty_class = metrics_factory.compute_coco_map()
    
    # The new mean should be (AP_c1 + AP_c2 + 0.0) / 3
    assert mAP_with_empty_class < mAP_coco
    assert np.isclose(mAP_with_empty_class, (mAP_coco * 2) / 3)

def test_coco_map_missing_class_logic(metrics_factory):
    """
    Test if mAP calculation correctly handles a class that exists in the 
    dataset (gt_counts) but received zero predictions in the raw_data.
    """
    metrics_factory.gt_counts = {1: 1}  
    
    metrics_factory.raw_data = [
        (
            np.array([[10, 10, 20, 20]]), # gt_boxes
            np.array([1]),                # gt_labels
            np.array([]),                 # pred_boxes (Empty)
            np.array([]),                 # pred_labels
            np.array([]),                 # pred_scores
        )
    ]

    mAP = metrics_factory.compute_coco_map()
    assert mAP == 0.0, f"Expected mAP 0.0 for a missed class, but got {mAP}"

def test_coco_map_empty_vs_non_empty_class(metrics_factory):
    """
    Verify the mean is calculated correctly across multiple classes 
    when one is a perfect match and one is a total miss.
    """
    # Class 1: 1 GT, 1 Perfect Match (AP = 1.0)
    # Class 2: 1 GT, 0 Matches (AP = 0.0)
    metrics_factory.gt_counts = {1: 1, 2: 1}
    
    metrics_factory.raw_data = [
        (
            np.array([[0, 0, 10, 10], [50, 50, 60, 60]]), # GTs
            np.array([1, 2]),                             # Labels
            np.array([[0, 0, 10, 10]]),                   # Only one Pred
            np.array([1]),                                # Label for Pred
            np.array([0.99]),                             # Score
        )
    ]

    mAP = metrics_factory.compute_coco_map()
    
    # (AP_Class1 + AP_Class2) / 2  => (1.0 + 0.0) / 2 = 0.5
    assert np.isclose(mAP, 0.5), f"Expected mAP of 0.5, but got {mAP}"
