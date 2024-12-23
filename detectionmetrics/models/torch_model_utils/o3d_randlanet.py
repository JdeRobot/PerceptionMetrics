from typing import List, Optional, Tuple

import numpy as np
from open3d._ml3d.datasets.utils import DataProcessing
import torch

import detectionmetrics.utils.lidar as ul


def transform_input(
    points: np.ndarray, cfg: dict, sampler: Optional[ul.Sampler] = None
) -> Tuple[
    Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ],
    np.ndarray,
]:
    """Transform point cloud data into input data for the model

    :param points: Point cloud data
    :type points: np.ndarray
    :param cfg: Dictionary containing model configuration file
    :type cfg: dict
    :param sampler: Object for sampling point cloud, defaults to None
    :type sampler: Optional[ul.Sampler], optional
    :return: Model input data and selected indices
    :rtype: Tuple[ Tuple[ torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], ], np.ndarray, ]
    """
    # Sample points if required
    selected_indices = None
    if sampler is not None:
        points, selected_indices, _ = sampler.sample(
            points, cfg.get("num_points", 45056)
        )

    # Recenter point cloud if required
    if "recenter" in cfg:
        points = ul.recenter(points, cfg["recenter"]["dims"])

    # Init model input data
    features = points.copy()
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    # Fill model input data
    num_neighbors = cfg.get("num_neighbors", 16)
    sub_sampling_ratio = cfg.get("sub_sampling_ratio", [4] * cfg["num_layers"])
    for i in range(cfg["num_layers"]):
        neighbour_idx = DataProcessing.knn_search(points, points, num_neighbors)
        sub_points = points[: points.shape[0] // sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[: points.shape[0] // sub_sampling_ratio[i], :]
        up_i = DataProcessing.knn_search(sub_points, points, 1)

        input_points.append(torch.from_numpy(points))
        input_neighbors.append(torch.from_numpy(neighbour_idx.astype(np.int64)))
        input_pools.append(torch.from_numpy(pool_i.astype(np.int64)))
        input_up_samples.append(torch.from_numpy(up_i.astype(np.int64)))

        points = sub_points

    return (
        torch.from_numpy(features),
        input_points,
        input_neighbors,
        input_pools,
        input_up_samples,
    ), selected_indices


def update_probs(
    new_probs: torch.Tensor,
    indices: torch.Tensor,
    test_probs: torch.Tensor,
    n_classes: int,
    weight: float = 0.95,
) -> torch.Tensor:
    """Update test probabilities with new model output using weighted average for a
    smooth transition between predictions

    :param new_probs: New probabilities to be added to the test probabilities
    :type new_probs: torch.Tensor
    :param indices: Corresponding indices of the new probabilities
    :type indices: torch.Tensor
    :param test_probs: Test probabilities to be updated
    :type test_probs: torch.Tensor
    :param n_classes: Number of classes
    :type n_classes: int
    :param weight: Weight used in the weighted average, defaults to 0.95
    :type weight: float, optional
    :return: Updated test probabilities
    :rtype: torch.Tensor
    """
    # Format new probabilities
    new_probs = torch.reshape(new_probs, (-1, n_classes))
    new_probs = torch.nn.functional.softmax(new_probs, dim=-1)

    # Update test probabilities using a weighted average
    if isinstance(test_probs, np.ndarray):
        test_probs = torch.tensor(test_probs, device=new_probs.device)
    test_probs[indices] = weight * test_probs[indices] + (1 - weight) * new_probs

    return test_probs
