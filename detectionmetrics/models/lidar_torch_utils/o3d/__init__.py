from typing import Optional, Tuple

import numpy as np
import torch

try:
    from open3d._ml3d.datasets.utils import DataProcessing
except Exception:
    print("Open3D-ML3D not available")
from sklearn.neighbors import KDTree

from detectionmetrics.models.lidar_torch_utils.o3d import kpconv, randlanet
from detectionmetrics.utils import lidar as ul
import detectionmetrics.utils.torch as ut


def inference(
    sample: Tuple[np.ndarray, np.ndarray, ul.Sampler],
    model: torch.nn.Module,
    model_cfg: dict,
) -> torch.Tensor:
    """Perform inference on a sample using an Open3D-ML model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: Open3D-ML model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :return: predictions, labels, and sample names
    :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]
    """
    infer_complete = False
    points, projected_indices, sampler, label, name, _ = sample
    model_format = model_cfg["model_format"]
    end_th = model_cfg.get("end_th", 0.5)

    if "kpconv" in model_format:
        transform_input = kpconv.transform_input
        update_probs = kpconv.update_probs
    elif "randlanet" in model_format:
        decoder_layers = model.decoder.children()
        model_cfg["num_layers"] = sum(1 for _ in decoder_layers)
        transform_input = randlanet.transform_input
        update_probs = randlanet.update_probs
    else:
        raise ValueError(f"Unknown model type: {model_format}")

    while not infer_complete:
        # Get model input data
        input_data, selected_indices = transform_input(points, model_cfg, sampler)
        input_data = ut.data_to_device(input_data, model.device)
        if "randlanet" in model_format:
            input_data = ut.unsqueeze_data(input_data)

        # Perform inference
        with torch.no_grad():
            pred = model(*input_data)

            # TODO: check if this is consistent across different models
            if isinstance(pred, dict):
                pred = pred["out"]

        # Update probabilities if sampler is used
        if sampler is not None:
            if "kpconv" in model_format:
                sampler.test_probs = update_probs(
                    pred,
                    selected_indices,
                    sampler.test_probs,
                    lengths=input_data[-1],
                )
            else:
                sampler.test_probs = update_probs(
                    pred,
                    selected_indices,
                    sampler.test_probs,
                    model_cfg["n_classes"],
                )
            if sampler.p[sampler.p > end_th].shape[0] == sampler.p.shape[0]:
                pred = sampler.test_probs[projected_indices]
                infer_complete = True
        else:
            pred = pred.squeeze().cpu()[projected_indices].cuda()
            infer_complete = True

    if label is not None:
        label = torch.from_numpy(label.astype(np.int64)).long().cuda()

    return torch.argmax(pred.squeeze(), axis=-1), label, name


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
    has_intensity: bool = True,
) -> Tuple[np.ndarray, np.ndarray, ul.Sampler]:
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
    :return: Sample data tuple
    :rtype: Tuple[np.ndarray, np.ndarray, ul.Sampler, np.ndarray, str, int]
    """
    points = ul.read_semantickitti_points(points_fname, has_intensity)
    label = None
    if label_fname is not None:
        label, _ = ul.read_semantickitti_label(label_fname)

    # Keep only XYZ coordinates
    points = np.array(points[:, 0:3], dtype=np.float32)

    # Subsample points using a grid of given size
    grid_size = model_cfg.get("grid_size", 0.06)
    sub_points = DataProcessing.grid_subsampling(points, grid_size=grid_size)

    # Create search tree so that we can project points back to the original point cloud
    search_tree = KDTree(sub_points)
    projected_indices = np.squeeze(search_tree.query(points, return_distance=False))
    projected_indices = projected_indices.astype(np.int32)

    # Init sampler
    sampler = None
    if "sampler" in model_cfg:
        sampler = ul.Sampler(
            sub_points.shape[0],
            search_tree,
            model_cfg["sampler"],
            model_cfg["n_classes"],
        )

    return sub_points, projected_indices, sampler, label, name, idx
