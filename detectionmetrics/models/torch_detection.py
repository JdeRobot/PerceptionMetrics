from collections import defaultdict
import os
import time
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from detectionmetrics.datasets import detection as dm_detection_dataset
from detectionmetrics.models import detection as dm_detection_model


def data_to_device(
    data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    device: torch.device
) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Move detection input or target data (dict or list of dicts) to the specified device.

    :param data: Detection data (a single dict or list of dicts with tensor values)
    :type data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    :param device: Device to move data to
    :type device: torch.device
    :return: Data with all tensors moved to the target device
    :rtype: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    """
    if isinstance(data, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

    elif isinstance(data, list):
        return [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            for item in data
        ]

    else:
        raise TypeError(f"Expected a dict or list of dicts, got {type(data)}")


def get_data_shape(data: Union[torch.Tensor, tuple]) -> tuple:
    """Get the shape of the provided data

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :return: Data shape
    :rtype: Union[tuple, list]
    """
    if isinstance(data, tuple):
        return data[0].shape
    return data.shape


def get_computational_cost(
    model: Any,
    dummy_input: Union[torch.Tensor, tuple, list],
    model_fname: Optional[str] = None,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> pd.DataFrame:
    """
    Get different metrics related to the computational cost of a model.

    :param model: TorchScript or PyTorch model (segmentation, detection, etc.)
    :type model: Any
    :param dummy_input: Dummy input data (Tensor, Tuple, or List of Dicts for detection)
    :type dummy_input: Union[torch.Tensor, tuple, list]
    :param model_fname: Optional path to model file for size estimation
    :type model_fname: Optional[str]
    :param runs: Number of timed runs
    :type runs: int
    :param warm_up_runs: Warm-up iterations before timing
    :type warm_up_runs: int
    :return: DataFrame with size, inference time, parameter count, etc.
    :rtype: pd.DataFrame
    """

    # Compute model size if applicable
    size_mb = os.path.getsize(model_fname) / 1024**2 if model_fname else None

    # Format input consistently
    if isinstance(dummy_input, (torch.Tensor, tuple)):
        dummy_tuple = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    else:
        dummy_tuple = dummy_input  # e.g., list of dicts for detection

    # Warm-up
    for _ in range(warm_up_runs):
        with torch.no_grad():
            if hasattr(model, "inference"):
                model.inference(*dummy_tuple)
            else:
                model(*dummy_tuple)

    # Measure inference time
    inference_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            if hasattr(model, "inference"):
                model.inference(*dummy_tuple)
            else:
                model(*dummy_tuple)
        torch.cuda.synchronize()
        inference_times.append(time.time() - start)

    # Get number of parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Get input shape
    input_shape = get_data_shape(dummy_input)
    input_shape_str = "x".join(map(str, input_shape))

    result = {
        "input_shape": [input_shape_str],
        "n_params": [n_params],
        "size_mb": [size_mb],
        "inference_time_s": [np.mean(inference_times)],
    }

    return pd.DataFrame.from_dict(result)



class ImageDetectionTorchDataset(Dataset):
    """
    Dataset for image detection PyTorch models.

    :param dataset: Image detection dataset
    :type dataset: ImageDetectionDataset
    :param transform: Transformation to be applied to the image and boxes (jointly)
    :type transform: callable
    :param splits: Dataset splits to use (e.g., ["train", "val", "test"])
    :type splits: List[str]
    """

    def __init__(
        self,
        dataset: dm_detection_dataset.ImageDetectionDataset,
        transform=transforms.Compose,
        splits: List[str] = ["test"]
    ):
        # Filter split and make file paths global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        self.dataset = dataset
        self.dataset.make_fname_global()

        self.transform = transform

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load image and annotations, apply transforms.

        :param idx: Sample index
        :return: Tuple of (sample_id, image_tensor, target_dict)
        """
        row = self.dataset.dataset.iloc[idx]
        image_path = row["image"]
        ann_path = row["annotation"]

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.dataset.read_annotation(ann_path)

        # Convert boxes/labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # [N, 4]
        labels = torch.as_tensor(labels, dtype=torch.int64)  # [N]

        target = {
            "boxes": boxes,     # shape [N, 4] in [x1, y1, x2, y2] format
            "labels": labels,   # shape [N]
        }

        if self.transform:
            image, target = self.transform(image, target)

        return self.dataset.dataset.index[idx], image, target

#TODO  TorchImageDetectionModel(dm_detection_model.ImageDetectionModel):