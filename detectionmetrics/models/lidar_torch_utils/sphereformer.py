from typing import List, Optional, Tuple

import numpy as np
import spconv.pytorch as spconv
import torch
from util.data_util import data_prepare

import detectionmetrics.utils.torch as ut


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
    labels = torch.cat(labels)
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
    feats = np.fromfile(points_fname, dtype=np.float32)
    feats = feats.reshape((-1, 4))[:, : model_cfg["n_feats"]]

    annotated_data = np.fromfile(label_fname, dtype=np.uint32)
    annotated_data = annotated_data.reshape((-1, 1))
    labels = annotated_data & 0xFFFF  # delete high 16 digits binary
    labels = labels.astype(np.uint8)
    instance_label = annotated_data >> 16
    instance_label = instance_label.reshape(-1)

    xyz = feats[:, :3]

    xyz = np.clip(xyz, model_cfg["pc_range"][0], model_cfg["pc_range"][1])

    labels_in = annotated_data.astype(np.uint8).reshape(-1)

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

    sample = (
        coords,
        xyz,
        feats,
        labels,
        inds_reconstruct,
        points_fname,
        name,
    )

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

    preds = model(sinput, xyz, batch)
    preds = preds[inds_reconstruct, :]
    preds = torch.argmax(preds, dim=1)

    return preds, labels, names
