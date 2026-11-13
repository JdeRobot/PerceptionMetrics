import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    Pack3DDetInputs,
)
from mmengine.registry import FUNCTIONS
import torch
from torchvision.transforms import Compose

COLLATE_FN = FUNCTIONS.get("pseudo_collate")


def detection_inference(
    model: torch.nn.Module,
    points: np.ndarray,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Run MMDetection3D inference on one loaded LiDAR point cloud.

    The returned dictionary follows PerceptionMetrics' common LiDAR detection
    contract. MMDetection3D-specific point preparation and prediction fields
    remain contained in this helper.

    :param model: Initialized MMDetection3D model
    :type model: torch.nn.Module
    :param points: Point cloud with three to five features per point
    :type points: np.ndarray
    :return: Standard detection result and MMDetection3D processed sample
    :rtype: Tuple[Dict[str, torch.Tensor], dict]
    """
    from mmdet3d.apis import inference_detector

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4, 5):
        raise ValueError(
            "points must be a two-dimensional array with 3, 4, or 5 features "
            "per point."
        )

    expected_features = points.shape[1]
    uses_multi_sweeps = False
    pipeline = model.cfg.test_dataloader.dataset.pipeline
    for transform in pipeline:
        if transform.get("type") == "LoadPointsFromFile":
            use_dim = transform.get("use_dim", expected_features)
            expected_features = (
                len(use_dim) if isinstance(use_dim, (list, tuple)) else use_dim
            )
        elif transform.get("type") == "LoadPointsFromMultiSweeps":
            uses_multi_sweeps = True

    # nuScenes multi-sweep pipelines reserve the fifth column for time lag.
    if uses_multi_sweeps:
        expected_features = max(expected_features, 5)

    if points.shape[1] < expected_features:
        padding = np.zeros(
            (points.shape[0], expected_features - points.shape[1]),
            dtype=np.float32,
        )
        points = np.concatenate((points, padding), axis=1)
    elif points.shape[1] > expected_features:
        points = points[:, :expected_features]

    with torch.no_grad():
        prediction, sample = inference_detector(model, points)

    instances = prediction.pred_instances_3d
    return {
        # MMDetection3D nuScenes models can append velocity components. PM uses
        # the shared seven-value box representation.
        "boxes_3d": instances.bboxes_3d.tensor[:, :7].detach().cpu(),
        "labels": instances.labels_3d.detach().cpu(),
        "scores": instances.scores_3d.detach().cpu(),
    }, sample


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
    has_intensity: bool = True,
    measure_processing_time: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """Get sample data for mmdetection3d models

    :param points_fname: filename of the point cloud
    :type points_fname: str
    :param model_cfg: model configuration
    :type model_cfg: dict
    :param label_fname: filename of the semantic label, defaults to None
    :type label_fname: Optional[str], optional
    :param name: sample name, defaults to None
    :type name: Optional[str], optional
    :param idx: sample numerical index, defaults to None
    :type idx: Optional[int], optional
    :param has_intensity: whether the point cloud has intensity values, defaults to True
    :type has_intensity: bool, optional
    :param measure_processing_time: whether to measure processing time, defaults to False
    :type measure_processing_time: bool, optional
    :return: sample data and optionally processing time
    :rtype: Tuple[ dict, Optional[dict] ]
    """
    sample = {
        "lidar_points": {
            "lidar_path": points_fname,
            "num_pts_feats": model_cfg.get("n_feats", 4),
        },
        "pts_semantic_mask_path": label_fname,
        "sample_id": name,
        "sample_idx": idx,
        "num_pts_feats": model_cfg.get("n_feats", 4),
        "lidar_path": points_fname,
    }

    n_feats = sample["num_pts_feats"]
    load_dim = 4 if has_intensity else 3
    transforms = [
        LoadPointsFromFile(coord_type="LIDAR", load_dim=load_dim, use_dim=n_feats)
    ]
    if sample["pts_semantic_mask_path"] is not None:
        transforms.append(
            LoadAnnotations3D(
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype="np.uint32",
                seg_offset=65536,
                dataset_type="semantickitti",
            )
        )
    transforms.append(
        Pack3DDetInputs(
            keys=["points", "pts_semantic_mask"],
            meta_keys=["sample_idx", "lidar_path", "num_pts_feats", "sample_id"],
        )
    )

    if measure_processing_time:
        start = time.perf_counter()
    transforms = Compose(transforms)
    sample = transforms(sample)
    if measure_processing_time:
        end = time.perf_counter()
        return sample, {"preprocessing": end - start}

    return sample


def inference(
    sample: dict,
    model: torch.nn.Module,
    model_cfg: dict,
    ignore_index: Optional[List[int]] = None,
    measure_processing_time: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]], Optional[dict]
]:
    """Perform inference on a sample using an mmdetection3D model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: mmdetection3D model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :param measure_processing_time: whether to measure processing time, defaults to False
    :type measure_processing_time: bool, optional
    :param ignore_index: list of class indices to ignore during inference, defaults to None
    :type ignore_index: Optional[List[int]], optional
    :return: predictions, labels (if available), sample names and optionally processing time
    :rtype: Tuple[ Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]], Optional[dict] ]
    """
    single_sample = not isinstance(sample["data_samples"], list)
    if single_sample:
        sample = COLLATE_FN([sample])

    if measure_processing_time:
        start = time.perf_counter()
    sample = model.data_preprocessor(sample, training=False)
    if measure_processing_time:
        end = time.perf_counter()
        processing_time = {"voxelization": end - start}

    inputs, data_samples = sample["inputs"], sample["data_samples"]
    has_labels = hasattr(data_samples[0].gt_pts_seg, "pts_semantic_mask")

    if measure_processing_time:
        torch.cuda.synchronize()
        start = time.perf_counter()
    outputs = model(inputs, data_samples, mode="predict")
    if measure_processing_time:
        torch.cuda.synchronize()
        end = time.perf_counter()
        processing_time["inference"] = end - start

    preds, labels, names = ([], [], []) if has_labels else ([], None, None)
    for output in outputs:
        if ignore_index is not None:
            output.pts_seg_logits.pts_seg_logits[ignore_index] = -1e9
        pred = torch.argmax(output.pts_seg_logits.pts_seg_logits, dim=0)
        preds.append(pred)
        if has_labels:
            labels.append(output.gt_pts_seg.pts_semantic_mask)
            names.append(output.metainfo["sample_id"])
    preds = torch.stack(preds, dim=0).squeeze()
    if has_labels:
        labels = torch.stack(labels, dim=0).squeeze()

    if measure_processing_time:
        return (preds, labels, names), processing_time
    else:
        return preds, labels, names
