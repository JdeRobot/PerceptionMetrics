from typing import List, Optional, Tuple

from c_gen_normal_map import gen_normal_map
import numpy as np
import torch
import utils.depth_map_utils as depth_map_utils

import detectionmetrics.utils.torch as ut


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900):
    """Project a pointcloud into a spherical projection (range image)."""
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    from_proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    from_proj_y = np.copy(proj_y)  # stope a copy in original order

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]

    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_vertex = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]
    ).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, from_proj_x, from_proj_y


def compute_normals_range(
    current_vertex, proj_H=64, proj_W=900, extrapolate=True, blur_type="gaussian"
):
    """Compute normals for each point using range image-based method."""
    proj_range, proj_vertex, from_proj_x, from_proj_y = range_projection(current_vertex)
    proj_range = depth_map_utils.fill_in_fast(
        proj_range, extrapolate=extrapolate, blur_type=blur_type
    )

    # generate normal image
    normal_data = gen_normal_map(proj_range, proj_vertex, proj_H, proj_W)
    unproj_normal_data = normal_data[from_proj_y, from_proj_x]

    return unproj_normal_data


def collate_fn(samples: List[dict]) -> dict:
    """Collate function for batching samples

    :param samples: list of sample dictionaries
    :type samples: List[dict]
    :return: collated batch dictionary
    :rtype: dict
    """
    point_num = [d["point_num"] for d in samples]
    batch_size = len(point_num)
    ref_labels = samples[0]["ref_label"]
    origin_len = samples[0]["origin_len"]
    ref_indices = [torch.from_numpy(d["ref_index"]) for d in samples]
    path = samples[0]["root"]  # [d['root'] for d in data]
    root = [d["root"] for d in samples]
    sample_id = [d["sample_id"] for d in samples]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d["point_feat"]) for d in samples]
    ref_xyz = [torch.from_numpy(d["ref_xyz"]) for d in samples]

    has_labels = samples[0]["point_label"] is not None
    if has_labels:
        labels = [torch.from_numpy(d["point_label"]) for d in samples]
    else:
        labels = [d["point_label"] for d in samples]
    normal = [torch.from_numpy(d["normal"]) for d in samples]

    return {
        "points": torch.cat(points).float(),
        "normal": torch.cat(normal).float(),
        "ref_xyz": torch.cat(ref_xyz).float(),
        "batch_idx": torch.cat(b_idx).long(),
        "batch_size": batch_size,
        "labels": torch.cat(labels).long().squeeze(1) if has_labels else labels,
        "raw_labels": torch.from_numpy(ref_labels).long() if has_labels else ref_labels,
        "origin_len": origin_len,
        "indices": torch.cat(ref_indices).long(),
        "path": path,
        "point_num": point_num,
        "root": root,
        "sample_id": sample_id,
    }


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
) -> dict:
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
    :return: Sample data dictionary
    :rtype: dict
    """
    raw_data = np.fromfile(points_fname, dtype=np.float32)
    raw_data = raw_data.reshape((-1, 4))

    labels, ref_labels = None, None
    if label_fname is not None:
        annotated_data = np.fromfile(label_fname, dtype=np.uint32)
        annotated_data = annotated_data.reshape((-1, 1))
        labels = annotated_data & 0xFFFF  # delete high 16 digits binary
        labels = labels.astype(np.uint8)
        ref_labels = labels.copy()

    xyz = raw_data[:, :3]
    feat = raw_data[:, 3:4] if model_cfg["n_feats"] > 3 else None
    origin_len = len(xyz)

    ref_pc = xyz.copy()
    ref_index = np.arange(len(ref_pc))

    mask_x = np.logical_and(
        xyz[:, 0] > model_cfg["min_volume_space"][0],
        xyz[:, 0] < model_cfg["max_volume_space"][0],
    )
    mask_y = np.logical_and(
        xyz[:, 1] > model_cfg["min_volume_space"][1],
        xyz[:, 1] < model_cfg["max_volume_space"][1],
    )
    mask_z = np.logical_and(
        xyz[:, 2] > model_cfg["min_volume_space"][2],
        xyz[:, 2] < model_cfg["max_volume_space"][2],
    )
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

    not_zero = np.logical_not(np.all(xyz[:, :3] == 0, axis=1))
    mask = np.logical_and(mask, not_zero)

    xyz = xyz[mask]
    if labels is not None:
        labels = labels[mask]
    ref_index = ref_index[mask]
    if feat is not None:
        feat = feat[mask]
    point_num = len(xyz)

    feat = np.concatenate((xyz, feat), axis=1) if feat is not None else xyz

    unproj_normal_data = compute_normals_range(feat)

    sample = {}
    sample["point_feat"] = feat
    sample["point_label"] = labels
    sample["ref_xyz"] = ref_pc
    sample["ref_label"] = ref_labels
    sample["ref_index"] = ref_index
    sample["point_num"] = point_num
    sample["origin_len"] = origin_len
    sample["normal"] = unproj_normal_data
    sample["root"] = points_fname
    sample["sample_id"] = name
    sample["idx"] = idx

    return sample


def inference(
    sample: dict, model: torch.nn.Module, model_cfg: dict
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """Perform inference on a sample using an mmdetection3D model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: mmdetection3D model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :return: predictions, labels, and sample names
    :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]
    """
    single_sample = not isinstance(sample["sample_id"], list)
    if single_sample:
        sample = collate_fn([sample])

    device = next(model.parameters()).device
    for k, v in sample.items():
        sample[k] = ut.data_to_device(v, device)

    pred = model(sample)
    pred["logits"] = torch.argmax(pred["logits"], dim=1)

    has_labels = pred["labels"][0] is not None
    preds, labels, names = ([], [], []) if has_labels else ([], None, None)

    for batch_idx in range(pred["batch_size"]):
        preds.append(pred["logits"][pred["batch_idx"] == batch_idx])
        if has_labels:
            labels.append(pred["labels"][pred["batch_idx"] == batch_idx])
            names.append(pred["sample_id"][batch_idx])

    preds = torch.stack(preds, dim=0).squeeze()
    if has_labels:
        labels = torch.stack(labels, dim=0).squeeze()

    return preds, labels, names
