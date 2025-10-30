def postprocess_detection(output: dict, confidence_threshold: float = 0.5):
    """Post-process torchvision model output.

    :param output: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :type output: dict
    :param confidence_threshold: Confidence threshold to filter boxes.
    :type confidence_threshold: float
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
    return output
