def postprocess_detection(
    output: dict, confidence_threshold: float = 0.5, max_detections: int = 100
):
    """Post-process torchvision model output.

    :param output: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :type output: dict
    :param confidence_threshold: Confidence threshold to filter boxes.
    :type confidence_threshold: float
    :param max_detections: Maximum number of best detections to keep per image after filtering.
    :type max_detections: int
    :return: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :rtype: dict
    """
    if confidence_threshold > 0:
        keep_mask = output["scores"] >= confidence_threshold
        output = {
            "boxes": output["boxes"][keep_mask],
            "labels": output["labels"][keep_mask],
            "scores": output["scores"][keep_mask],
        }

    if max_detections > 0:
        limit = min(max_detections, output["scores"].shape[0])
        if limit > 0:
            limited_idx = output["scores"].argsort(descending=True)[:limit]
            output = {
                "boxes": output["boxes"][limited_idx],
                "labels": output["labels"][limited_idx],
                "scores": output["scores"][limited_idx],
            }

    return output
