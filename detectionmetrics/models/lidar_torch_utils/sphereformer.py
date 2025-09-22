import time
from typing import List, Optional, Tuple

import numpy as np
import spconv.pytorch as spconv
import torch
from util.data_util import data_prepare

import detectionmetrics.utils.torch as ut
import detectionmetrics.utils.lidar as ul


def collate_fn(samples: List[dict]) -> dict:
    """Collate function for batching samples

    :param samples: list of sample dictionaries
    :type samples: List[dict]
    :return: collated batch dictionary
    :rtype: dict
    """
    coords, xyz, feats, labels, inds_recons, fnames, sample_ids = list(zip(*samples))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    if any(label is None for label in labels):
        labels = None
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    return (
        coords,
        xyz,
        feats,
        labels,
        offset,
        inds_recons,
        list(fnames),
        list(sample_ids),
    )


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
    has_intensity: bool = True,
    measure_processing_time: bool = False
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
    :return: sample data dictionary and processing time dictionary (if measured)
    :rtype: Tuple[dict, Optional[dict]]
    """
    feats = ul.read_semantickitti_points(points_fname, has_intensity)
    feats = feats[:, : model_cfg["n_feats"]]

    labels_in = None
    if label_fname is not None:
        annotated_data = np.fromfile(label_fname, dtype=np.uint32)
        annotated_data = annotated_data.reshape((-1, 1))
        labels_in = annotated_data.astype(np.uint8).reshape(-1)

    if measure_processing_time:
        start = time.perf_counter()

    xyz = feats[:, :3]
    xyz = np.clip(xyz, model_cfg["pc_range"][0], model_cfg["pc_range"][1])

    coords, xyz, feats, labels, inds_reconstruct = data_prepare(
        xyz,
        feats,
        labels_in,
        "test",
        np.array(model_cfg["voxel_size"]),
        model_cfg["voxel_max"],
        None,
        model_cfg["xyz_norm"],
    )

    if measure_processing_time:
        end = time.perf_counter()
        processing_time = {"voxelization": end - start}

    sample = (
        coords,
        xyz,
        feats,
        labels,
        inds_reconstruct,
        points_fname,
        name,
    )

    if measure_processing_time:
        return sample, processing_time

    return sample


def inference(
    sample: dict, model: torch.nn.Module, model_cfg: dict, measure_processing_time: bool = False
) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor], List[str]], Optional[dict]]:
    """Perform inference on a sample using an mmdetection3D model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: mmdetection3D model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :param measure_processing_time: whether to measure processing time, defaults to False
    :type measure_processing_time: bool, optional
    :return: tuple of (predictions, labels, names) and processing time dictionary (if measured)
    :rtype: Tuple[Tuple[torch.Tensor, Optional[torch.Tensor], List[str]], Optional[dict]]
    """
    single_sample = not isinstance(sample[-1], list)
    if single_sample:
        sample = collate_fn([sample])

    device = next(model.parameters()).device
    sample = ut.data_to_device(sample, device)

    (
        coord,
        xyz,
        feat,
        labels,
        offset,
        inds_reconstruct,
        fnames,
        names,
    ) = sample

    if measure_processing_time:
        start = time.perf_counter()

    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]

    batch = (
        torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0)
        .long()
        .to(device)
    )

    coord = torch.cat([batch.unsqueeze(-1), coord], -1)
    spatial_shape = np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None)
    batch_size = len(fnames)

    sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_size)
    if measure_processing_time:
        end = time.perf_counter()
        processing_time = {"preprocessing": end - start}
        start = time.perf_counter()

    if measure_processing_time:
        torch.cuda.synchronize()
        start = time.perf_counter()
    preds = model(sinput, xyz, batch)
    if measure_processing_time:
        torch.cuda.synchronize()
        end = time.perf_counter()
        processing_time["inference"] = end - start

    preds = preds[inds_reconstruct, :]
    preds = torch.argmax(preds, dim=1)

    if measure_processing_time:
        return (preds, labels, names), processing_time

    return preds, labels, names
