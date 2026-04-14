import numpy as np
from PIL import Image


def load_mask(file):
    """
    Load a mask image and convert it into a 2D array of class IDs.

    The mask is converted to grayscale so that each pixel represents
    a class label.
    """
    img = Image.open(file).convert("L")
    return np.array(img)


def compute_error_map(gt, pred):
    """
    Create a visual error map.

    - Green = correct prediction
    - Red   = incorrect prediction

    Both inputs must have the same shape.
    """
    H, W = gt.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)

    correct = gt == pred
    incorrect = gt != pred

    error_map[correct] = [0, 255, 0]
    error_map[incorrect] = [255, 0, 0]

    return error_map


def compute_class_metrics(gt, pred):
    """
    Compute per-class accuracy.

    For each class in the ground truth:
    - Count total pixels
    - Count correctly predicted pixels
    - Compute accuracy

    Returns a list of results for display.
    """
    unique_classes = np.unique(gt)
    results = []

    for cls in unique_classes:
        total = int(np.sum(gt == cls))
        correct = int(np.sum((gt == cls) & (pred == cls)))
        acc = correct / total if total > 0 else 0.0

        results.append(
            {
                "class_id": int(cls),
                "total_pixels": total,
                "correct_pixels": correct,
                "accuracy": round(float(acc), 4),
            }
        )

    return results
