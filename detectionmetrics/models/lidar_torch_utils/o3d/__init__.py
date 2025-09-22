import time
from typing import Optional, Tuple, Union, Dict

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
    measure_processing_time: bool = False,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str]],
    Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str], Dict[str, float]],
]:
    """Perform inference on a sample using an Open3D-ML model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: Open3D-ML model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :param measure_processing_time: whether to measure processing time, defaults to False
    :type measure_processing_time: bool, optional
    :return: predicted labels, ground truth labels, sample name and optionally processing time
    :rtype: Union[ Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str], Dict[str, float]] ]
    """
    infer_complete = False
    points, projected_indices, sampler, label, name, _ = sample
    model_format = model_cfg["model_format"]
    end_th = model_cfg.get("end_th", 0.5)

    processing_time = {"preprocessing": 0, "inference": 0}

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
        if measure_processing_time:
            start = time.perf_counter()
        input_data, selected_indices = transform_input(points, model_cfg, sampler)
        if measure_processing_time:
            end = time.perf_counter()
            processing_time["preprocessing"] += end - start

        input_data = ut.data_to_device(input_data, model.device)
        if "randlanet" in model_format:
            input_data = ut.unsqueeze_data(input_data)

        # Perform inference
        with torch.no_grad():
            if measure_processing_time:
                torch.cuda.synchronize()
                start = time.perf_counter()
            pred = model(*input_data)
            if measure_processing_time:
                torch.cuda.synchronize()
                end = time.perf_counter()
                processing_time["inference"] += end - start

            # TODO: check if this is consistent across different models
            if isinstance(pred, dict):
                pred = pred["out"]

        # Update probabilities if sampler is used
        if measure_processing_time:
            start = time.perf_counter()
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
        if measure_processing_time:
            end = time.perf_counter()
            processing_time["postprocessing"] += end - start

    if label is not None:
        label = torch.from_numpy(label.astype(np.int64)).long().cuda()

    result = torch.argmax(pred.squeeze(), axis=-1), label, name

    # Return processing time if needed
    if measure_processing_time:
        return result, processing_time

    return result


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
    has_intensity: bool = True,
    measure_processing_time: bool = False,
) -> Tuple[
    Union[
        Tuple[np.ndarray, np.ndarray, ul.Sampler, np.ndarray, str, int],
        Tuple[np.ndarray, np.ndarray, ul.Sampler, np.ndarray, str, int],
        Dict[str, float],
    ]
]:
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
    :rtype: Union[ Tuple[np.ndarray, np.ndarray, ul.Sampler, np.ndarray, str, int], Tuple[np.ndarray, np.ndarray, ul.Sampler, np.ndarray, str, int], Dict[str, float] ]
    """
    points = ul.read_semantickitti_points(points_fname, has_intensity)
    label = None
    if label_fname is not None:
        label, _ = ul.read_semantickitti_label(label_fname)

    if measure_processing_time:
        start = time.perf_counter()

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

    if measure_processing_time:
        end = time.perf_counter()

    sample = sub_points, projected_indices, sampler, label, name, idx

    # Return processing time if needed
    if measure_processing_time:
        processing_time = {"preprocessing": end - start}
        return sample, processing_time

    return sample


def reset_sampler(sampler: ul.Sampler, num_points: int, num_classes: int):
    """Reset sampler object probabilities

    :param sampler: Sampler object
    :type sampler: ul.Sampler
    :param num_points: Number of points in the point cloud
    :type num_points: int
    :param num_classes: Number of semantic classes
    :type num_classes: int
    """
    sampler.p = np.random.rand(num_points) * 1e-3
    sampler.min_p = float(np.min(sampler.p[-1]))
    sampler.test_probs = np.zeros((num_points, num_classes), dtype=np.float32)
    return sampler
