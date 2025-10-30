import torch
from torchvision.ops import nms


CLASS_NMS_OFFSET = 7680  # offset to apply to boxes for class-wise NMS


def postprocess_detection(
    output: torch.Tensor,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
):
    """Post-process YOLO model output.

    :param output: Tensor of shape [num_classes + 4, num_anchors] containing bounding box predictions and class logits.
    :type output: torch.Tensor
    :param confidence_threshold: Confidence threshold to filter boxes.
    :type confidence_threshold: float
    :param nms_threshold: IoU threshold for Non-Maximum Suppression (NMS).
    :type nms_threshold: float
    :return: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :rtype: dict
    """
    # Split boxes and class logits
    boxes_xywh = output[:4, :].T  # [8400, 4] (cx, cy, w, h) in pixels
    cls_logits = output[4:, :].T  # [8400, 28]

    # Get boxes above confidence threshold
    i, j = torch.where(cls_logits > confidence_threshold)
    boxes_xywh = boxes_xywh[i]
    scores = cls_logits[i, j]
    labels = j

    # Convert (cx,cy,w,h) -> (x1,y1,x2,y2)
    cx, cy, w, h = boxes_xywh.unbind(1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # Apply class-wise NMS
    offset = labels * CLASS_NMS_OFFSET
    keep_idx = nms(boxes_xyxy + offset[:, None], scores, nms_threshold)
    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    return {"boxes": boxes_xyxy, "labels": labels, "scores": scores}
