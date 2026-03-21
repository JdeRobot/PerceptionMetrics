from typing import List, Optional

import numpy as np
from PIL import Image
import supervision as sv


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image using supervision.
    Adapts to different supervision versions.

    :param image: PIL Image
    :type image: Image.Image
    :param boxes: Bounding boxes in xyxy format
    :type boxes: np.ndarray
    :param class_ids: Class indices
    :type class_ids: np.ndarray
    :param class_names: List of class names corresponding to class_ids
    :type class_names: List[str]
    :param scores: Confidence scores (optional)
    :type scores: np.ndarray
    :return: Annotated image as numpy array
    :rtype: np.ndarray
    """
    image = np.array(image)

    # Create supervision detections
    detections = sv.Detections(xyxy=boxes, class_id=class_ids, confidence=scores)

    # Construct labels
    labels = []
    for i in range(len(class_ids)):
        if i < len(class_names):
            name = class_names[i]
        else:
            name = str(class_ids[i])

        if scores is not None and i < len(scores):
            labels.append(f"{name}: {scores[i]:.2f}")
        else:
            labels.append(name)

    try:
        # Try older style (BoxAnnotator handles labels)
        try:
            palette = sv.Color.DEFAULT
        except AttributeError:
            palette = sv.ColorPalette.default()

        annotator = sv.BoxAnnotator(
            color=palette, text_scale=0.7, text_thickness=1, text_padding=2
        )
        ann_image = annotator.annotate(
            scene=image, detections=detections, labels=labels
        )
    except TypeError:
        # Fallback for newer supervision (BoxAnnotator + LabelAnnotator)
        box_annotator = sv.BoxAnnotator()
        ann_image = box_annotator.annotate(scene=image, detections=detections)

        label_annotator = sv.LabelAnnotator()
        ann_image = label_annotator.annotate(
            scene=ann_image, detections=detections, labels=labels
        )

    return ann_image
