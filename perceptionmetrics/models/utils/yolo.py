import torch
from torchvision.ops import nms


CLASS_NMS_OFFSET = 7680  # offset to apply to boxes for class-wise NMS


def postprocess_detection(
    output: torch.Tensor,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 100
):
    """Post-process YOLO model output.

    :param output: Tensor of shape [num_classes + 4, num_anchors] containing bounding box predictions and class logits.
    :type output: torch.Tensor
    :param confidence_threshold: Confidence threshold to filter boxes.
    :type confidence_threshold: float
    :param nms_threshold: IoU threshold for Non-Maximum Suppression (NMS). Some models may not perform NMS (e.g. YOLOv26).
    :type nms_threshold: float
    :return: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :rtype: dict
    """
    is_yolov26 = output.shape[1] == 6

    # Parse model output and filter by confidence threshold
    if is_yolov26:
        boxes_xyxy = output[:, :4]
        scores = output[:, 4]
        labels = output[:, 5]

        i = torch.where(scores > confidence_threshold)[0]
        boxes_xyxy = boxes_xyxy[i]
        scores = scores[i]
        labels = labels[i]
    else:
        boxes_xywh = output[:4, :].T
        cls_logits = output[4:, :].T

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

        if max_detections > 0:
            limit = min(max_detections, scores.shape[0])
            if limit > 0:
                limited_idx = scores.argsort(descending = True)[:limit]
                boxes_xyxy = boxes_xyxy[limited_idx]
                scores = scores[limited_idx]
                labels = labels[limited_idx]


    return {"boxes": boxes_xyxy, "labels": labels, "scores": scores}
