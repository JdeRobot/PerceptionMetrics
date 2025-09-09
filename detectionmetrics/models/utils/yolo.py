import torch
from torchvision.ops import nms


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
    conf, cls_id = cls_logits.max(dim=1)  # best class per candidate

    # Convert (cx,cy,w,h) -> (x1,y1,x2,y2)
    cx, cy, w, h = boxes_xywh.unbind(1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # Filter + NMS (tune thresholds as you like)
    keep = conf > confidence_threshold
    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls_id = cls_id[keep]

    # class-agnostic NMS (or do per-class if you prefer)
    keep_idx = nms(boxes_xyxy, conf, nms_threshold)
    boxes_xyxy = boxes_xyxy[keep_idx]
    conf = conf[keep_idx]
    cls_id = cls_id[keep_idx]

    return {"boxes": boxes_xyxy, "labels": cls_id, "scores": conf}
