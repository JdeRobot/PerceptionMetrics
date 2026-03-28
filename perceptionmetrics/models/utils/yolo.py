import torch
from torchvision.ops import nms

CLASS_NMS_OFFSET = 7680  # offset to apply to boxes for class-wise NMS


def postprocess_detection(
    output,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 20,
):
    """Post-process YOLO model output for raw tensors or Results objects.

    :param output: Tensor of shape [num_classes + 4, num_anchors] containing bounding box
        predictions and class logits, or a YOLOv8 Results object, or a list/tuple wrapper.
    :type output: torch.Tensor
    :param confidence_threshold: Confidence threshold to filter boxes.
    :type confidence_threshold: float
    :param nms_threshold: IoU threshold for Non-Maximum Suppression (NMS). Some models may not perform NMS (e.g. YOLOv26).
    :type nms_threshold: float
    :param max_detections: Maximum number of detections to return per image.
    :type max_detections: int
    :return: Dictionary with keys 'boxes', 'labels', and 'scores'.
    :rtype: dict
    """

    # CASE 1: YOLOv8 Results object
    if hasattr(output, "boxes"):
        boxes = output.boxes.xyxy
        scores = output.boxes.conf
        labels = output.boxes.cls.long()

    # CASE 2: List / Tuple wrapper
    elif isinstance(output, (list, tuple)):
        output = output[0]
        return postprocess_detection(
            output, confidence_threshold, nms_threshold, max_detections
        )

    # CASE 3: Raw Tensor
    elif isinstance(output, torch.Tensor):
        if output.dim() == 3:
            output = output[0]

        is_yolov26 = output.shape[1] == 6

        if is_yolov26:
            boxes_xyxy = output[:, :4]
            scores = output[:, 4]
            labels = output[:, 5]

            i = torch.where(scores > confidence_threshold)[0]
            boxes_xyxy = boxes_xyxy[i]
            scores = scores[i]
            labels = labels[i]

            return {
                "boxes": boxes_xyxy,
                "labels": labels,
                "scores": scores,
            }

        else:
            # YOLOv8 format: [Features, Anchors] — transpose to [Anchors, Features]
            if output.shape[0] < output.shape[1] and output.shape[0] < 200:
                output = output.transpose(0, 1)

            num_features = output.shape[1]

            if num_features >= 5:
                boxes_xywh = output[:, :4]

                # YOLOv8 (no objectness) vs YOLOv5 (objectness at index 4)
                if num_features % 10 == 4 or num_features < 20:
                    scores, labels = output[:, 4:].max(dim=1)
                else:
                    obj_conf = output[:, 4]
                    class_conf, labels = output[:, 5:].max(dim=1)
                    scores = obj_conf * class_conf

                # Convert xywh -> xyxy
                cx, cy, w, h = boxes_xywh.unbind(1)
                boxes_xyxy = torch.stack(
                    [
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                    ],
                    dim=1,
                )
            else:
                raise ValueError(
                    f"Unexpected YOLO tensor feature count: {num_features}"
                )

        boxes = boxes_xyxy

    else:
        raise ValueError(f"Unsupported YOLO output type: {type(output)}")

    # FILTER by confidence
    keep = scores > confidence_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if boxes.numel() == 0:
        return {
            "boxes": torch.zeros((0, 4), device=boxes.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=labels.device),
            "scores": torch.zeros((0,), device=scores.device),
        }

    # CLASS-WISE NMS (original behaviour preserved)
    offset = labels.float() * CLASS_NMS_OFFSET
    keep_idx = nms(boxes + offset[:, None], scores, nms_threshold)
    boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    # LIMIT to max_detections
    if len(scores) > max_detections:
        boxes = boxes[:max_detections]
        scores = scores[:max_detections]
        labels = labels[:max_detections]

    return {"boxes": boxes, "labels": labels, "scores": scores}