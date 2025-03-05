from collections import defaultdict
import os
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from detectionmetrics.datasets import dataset as dm_dataset
from detectionmetrics.models import model as dm_model
from detectionmetrics.models import torch_model_utils as tmu
import detectionmetrics.utils.lidar as ul
import detectionmetrics.utils.metrics as um


def data_to_device(
    data: Union[tuple, list], device: torch.device
) -> Union[tuple, list]:
    """Move provided data to given device (CPU or GPU)

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param device: Device to move data to
    :type device: torch.device
    :return: Data moved to device
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.to(device) if torch.is_tensor(d) else data_to_device(d, device)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data


def get_data_shape(data: Union[tuple, list]) -> Union[tuple, list]:
    """Get the shape of the provided data

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :return: Data shape
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            tuple(d.shape) if torch.is_tensor(d) else get_data_shape(d) for d in data
        )
    elif torch.is_tensor(data):
        return tuple(data.shape)
    else:
        return tuple(data.shape)


def unsqueeze_data(data: Union[tuple, list], dim: int = 0) -> Union[tuple, list]:
    """Unsqueeze provided data along given dimension

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param dim: Dimension that will be unsqueezed, defaults to 0
    :type dim: int, optional
    :return: Unsqueezed data
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.unsqueeze(dim) if torch.is_tensor(d) else unsqueeze_data(d, dim)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.unsqueeze(dim)
    else:
        return data


def get_computational_cost(
    model: Any,
    dummy_input: torch.Tensor,
    model_fname: Optional[str] = None,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> dict:
    """Get different metrics related to the computational cost of the model

    :param model: Either a TorchScript model or an arbitrary PyTorch module
    :type model: Any
    :param dummy_input: Dummy input data for the model
    :type dummy_input: torch.Tensor
    :param model_fname: Model filename used to measure model size, defaults to None
    :type model_fname: Optional[str], optional
    :param runs: Number of runs to measure inference time, defaults to 30
    :type runs: int, optional
    :param warm_up_runs: Number of warm-up runs, defaults to 5
    :type warm_up_runs: int, optional
    :return: Dictionary containing computational cost information
    """
    # Get model size if possible
    if model_fname is not None:
        size_mb = os.path.getsize(model_fname) / 1024**2
    else:
        size_mb = None

    # Measure inference time with GPU synchronization
    dummy_tuple = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    for _ in range(warm_up_runs):
        model(*dummy_tuple)

    inference_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start_time = time.time()
        model(*dummy_tuple)
        torch.cuda.synchronize()
        end_time = time.time()
        inference_times.append(end_time - start_time)

    return {
        "input_shape": get_data_shape(dummy_input),
        "n_params": sum(p.numel() for p in model.parameters()),
        "size_mb": size_mb,
        "inference_time_s": np.mean(inference_times),
    }


class CustomResize(torch.nn.Module):
    """Custom rescale transformation for PyTorch

    :param target_size: Target size for the image
    :type target_size: Tuple[int, int]
    :param keep_aspect: Flag to keep aspect ratio
    :type keep_aspect: bool, defaults to False
    :param interpolation: Interpolation mode for resizing (e.g. NEAREST, BILINEAR)
    :type interpolation: F.InterpolationMode, defaults to F.InterpolationMode.BILINEAR
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        keep_aspect: bool = False,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.keep_aspect = keep_aspect
        self.interpolation = interpolation

    def forward(self, image: Image.Image) -> Image.Image:
        new_size = self.target_size
        if self.keep_aspect:
            h, w = image.size
            resize_factor = max((self.target_size[0] / h, self.target_size[1] / w))
            new_size = int(h * resize_factor), int(w * resize_factor)

        if new_size != image.size:
            image = F.resize(image, new_size, self.interpolation)

        if self.keep_aspect:
            image = F.center_crop(image, self.target_size)

        return image


class ImageSegmentationTorchDataset(Dataset):
    """Dataset for image segmentation PyTorch models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param transform: Transformation to be applied to images
    :type transform: transforms.Compose
    :param target_transform: Transformation to be applied to labels
    :type target_transform: transforms.Compose
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    """

    def __init__(
        self,
        dataset: dm_dataset.ImageSegmentationDataset,
        transform: transforms.Compose,
        target_transform: transforms.Compose,
        splits: List[str] = ["test"],
    ):
        # Filter split and make filenames global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        self.dataset = dataset
        self.dataset.make_fname_global()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        """Prepare sample data: image and label

        :param idx: Sample index
        :type idx: int
        :return: Image and corresponding label tensor or PIL image
        :rtype: Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]
        """
        image = Image.open(self.dataset.dataset.iloc[idx]["image"])
        label = Image.open(self.dataset.dataset.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return self.dataset.dataset.index[idx], image, label


class LiDARSegmentationTorchDataset(Dataset):
    """Dataset for LiDAR segmentation PyTorch models

    :param dataset: LiDAR segmentation dataset
    :type dataset: LiDARSegmentationDataset
    :param model_cfg: Dictionary containing model configuration
    :type model_cfg: dict
    :param preprocess: Function for preprocessing point clouds
    :type preprocess: callable
    :param n_classes: Number of classes estimated by the model
    :type n_classes: int
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    """

    def __init__(
        self,
        dataset: dm_dataset.LiDARSegmentationDataset,
        model_cfg: dict,
        preprocess: callable,
        n_classes: int,
        splits: str = ["test"],
    ):
        # Filter split and make filenames global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        self.dataset = dataset
        self.dataset.make_fname_global()

        self.model_cfg = model_cfg
        self.preprocess = preprocess
        self.n_classes = n_classes

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Prepare sample data: point cloud and label

        :param idx: Sample index
        :type idx: int
        :return: Point cloud and corresponding label tensor or numpy arrays
        :rtype: Tuple[np.ndarray, np.ndarray,]
        """
        # Read the point cloud and its labels
        points = self.dataset.read_points(self.dataset.dataset.iloc[idx]["points"])
        semantic_label, instance_label = self.dataset.read_label(
            self.dataset.dataset.iloc[idx]["label"]
        )

        # Preprocess point cloud
        preprocessed_points, search_tree, projected_indices = self.preprocess(
            points, self.model_cfg
        )

        # Init sampler
        sampler = None
        if "sampler" in self.model_cfg:
            sampler = ul.Sampler(
                preprocessed_points.shape[0],
                search_tree,
                self.model_cfg["sampler"],
                self.n_classes,
            )

        return (
            self.dataset.dataset.index[idx],
            preprocessed_points,
            projected_indices,
            (semantic_label, instance_label),
            sampler,
        )


class TorchImageSegmentationModel(dm_model.ImageSegmentationModel):

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        model_cfg: str,
        ontology_fname: str,
    ):
        """Image segmentation model for PyTorch framework

        :param model: Either the filename of a TorchScript model or the model already loaded into an arbitrary PyTorch module.
        :type model: Union[str, torch.nn.Module]
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        """
        # Get device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If 'model' contains a string, check that it is a valid filename and load model
        if isinstance(model, str):
            assert os.path.isfile(model), "TorchScript Model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model)
                model_type = "compiled"
            except:
                print("Model is not a TorchScript model. Loading as a PyTorch module.")
                model = torch.load(model)
                model_type = "native"
        # Otherwise, check that it is a PyTorch module
        elif isinstance(model, torch.nn.Module):
            model_fname = None
            model_type = "native"
        else:
            raise ValueError("Model must be either a filename or a PyTorch module")

        # Init parent class and model
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)
        self.model = self.model.to(self.device).eval()

        # Init transformations for input images, output labels, and GT labels
        self.transform_input = []
        self.transform_label = []

        if "image_size" in self.model_cfg:
            self.transform_input += [
                CustomResize(
                    tuple(self.model_cfg["image_size"]),
                    keep_aspect=self.model_cfg.get("keep_aspect", False),
                    interpolation=F.InterpolationMode.BILINEAR,
                )
            ]
            self.transform_label += [
                CustomResize(
                    tuple(self.model_cfg["image_size"]),
                    keep_aspect=self.model_cfg.get("keep_aspect", False),
                    interpolation=F.InterpolationMode.NEAREST,
                )
            ]

        try:
            self.transform_input += [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
            self.transform_label += [
                transforms.ToImage(),
                transforms.ToDtype(torch.int64),
            ]
        except AttributeError:  # adapt for older versions of torchvision transforms v2
            self.transform_input += [
                transforms.ToImageTensor(),
                transforms.ConvertDtype(torch.float32),
            ]
            self.transform_label += [
                transforms.ToImageTensor(),
                transforms.ToDtype(torch.int64),
            ]

        if "normalization" in self.model_cfg:
            self.transform_input += [
                transforms.Normalize(
                    mean=self.model_cfg["normalization"]["mean"],
                    std=self.model_cfg["normalization"]["std"],
                )
            ]

        self.transform_input = transforms.Compose(self.transform_input)
        self.transform_label = transforms.Compose(self.transform_label)
        self.transform_output = transforms.Compose(
            [
                lambda x: torch.argmax(x.squeeze(), axis=0).squeeze().to(torch.uint8),
                transforms.ToPILImage(),
            ]
        )

    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: segmenation result as PIL image
        :rtype: Image.Image
        """
        tensor = self.transform_input(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model(tensor)

            # TODO: check if this is consistent across different models
            if isinstance(result, dict):
                result = result["out"]

        return self.transform_output(result)

    def eval(
        self,
        dataset: dm_dataset.ImageSegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: str | List[str], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Check that predictions_outdir is provided if results_per_sample is True
        if results_per_sample and predictions_outdir is None:
            raise ValueError(
                "If results_per_sample is True, predictions_outdir must be provided"
            )

        # Create predictions output directory if needed
        if predictions_outdir is not None:
            os.makedirs(predictions_outdir, exist_ok=True)

        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Retrieve ignored label indices
        ignored_label_indices = []
        for ignored_class in self.model_cfg.get("ignored_classes", []):
            ignored_label_indices.append(dataset.ontology[ignored_class]["idx"])

        # Get PyTorch dataloader
        dataset = ImageSegmentationTorchDataset(
            dataset,
            transform=self.transform_input,
            target_transform=self.transform_label,
            splits=[split] if isinstance(split, str) else split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.model_cfg.get("batch_size", 1),
            num_workers=self.model_cfg.get("num_workers", 1),
        )

        # Init metrics
        metrics_factory = um.MetricsFactory(self.n_classes)

        # Evaluation loop
        with torch.no_grad():
            pbar = tqdm(dataloader, leave=True)
            for idx, image, label in pbar:
                # Perform inference
                with torch.no_grad():
                    pred = self.model.inference(
                        image.to(self.device),
                        [
                            dict(
                                ori_shape=image.shape[2:],
                                img_shape=image.shape[2:],
                                pad_shape=image.shape[2:],
                                padding_size=[0, 0, 0, 0],
                            )
                        ]
                        * image.shape[0],
                    )
                    # pred = self.model(image.to(self.device))
                    if isinstance(pred, dict):
                        pred = pred["out"]

                # Get valid points masks depending on ignored label indices
                if ignored_label_indices:
                    valid_mask = torch.ones_like(label, dtype=torch.bool)
                    for ignored_label_idx in ignored_label_indices:
                        valid_mask *= label != ignored_label_idx
                else:
                    valid_mask = None

                # Convert labels if needed
                if lut_ontology is not None:
                    label = lut_ontology[label]

                # Prepare data and update metrics factory
                label = label.squeeze(dim=1).cpu().numpy()
                pred = torch.argmax(pred, axis=1).cpu().numpy()
                if valid_mask is not None:
                    valid_mask = valid_mask.squeeze(dim=1).cpu().numpy()

                metrics_factory.update(pred, label, valid_mask)

                # Store predictions and results per sample if required
                if predictions_outdir is not None:
                    for i, (sample_idx, sample_pred, sample_label) in enumerate(
                        zip(idx, pred, label)
                    ):
                        if results_per_sample:
                            sample_valid_mask = (
                                valid_mask[i] if valid_mask is not None else None
                            )
                            sample_mf = um.MetricsFactory(n_classes=self.n_classes)
                            sample_mf.update(
                                sample_pred, sample_label, sample_valid_mask
                            )
                            sample_df = um.get_metrics_dataframe(
                                sample_mf, self.ontology
                            )
                            sample_df.to_csv(
                                os.path.join(predictions_outdir, f"{sample_idx}.csv")
                            )
                        sample_pred = Image.fromarray(
                            np.squeeze(sample_pred).astype(np.uint8)
                        )
                        sample_pred.save(
                            os.path.join(predictions_outdir, f"{sample_idx}.png")
                        )

        return um.get_metrics_dataframe(metrics_factory, self.ontology)

    def get_computational_cost(self, runs: int = 30, warm_up_runs: int = 5) -> dict:
        """Get different metrics related to the computational cost of the model

        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
        """
        dummy_input = torch.randn(1, 3, *self.model_cfg["image_size"]).to(self.device)
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )


class TorchLiDARSegmentationModel(dm_model.LiDARSegmentationModel):

    def __init__(
        self, model: Union[str, torch.nn.Module], model_cfg: str, ontology_fname: str
    ):
        """LiDAR segmentation model for PyTorch framework

        :param model: Either the filename of a TorchScript model or the model already loaded into an arbitrary PyTorch module.
        :type model: Union[str, torch.nn.Module]
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        """
        # Get device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If 'model' contains a string, check that it is a valid filename and load model
        if isinstance(model, str):
            assert os.path.isfile(model), "TorchScript Model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model)
                model_type = "compiled"
            except Exception:
                print("Model is not a TorchScript model. Loading as a PyTorch module.")
                model = torch.load(model)
                model_type = "native"
        # Otherwise, check that it is a PyTorch module
        elif isinstance(model, torch.nn.Module):
            model_fname = None
            model_type = "native"
        else:
            raise ValueError("Model must be either a filename or a PyTorch module")

        # Init parent class and model
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)
        self.model = self.model.to(self.device).eval()

        # Init model specific functions
        if self.model_cfg["input_format"] == "o3d_randlanet":  # Open3D RandLaNet
            self.preprocess = tmu.preprocess
            self.transform_input = tmu.o3d_randlanet.transform_input
            self.update_probs = tmu.o3d_randlanet.update_probs
            self.model_cfg["num_layers"] = sum(1 for _ in self.model.decoder.children())
        if self.model_cfg["input_format"] == "o3d_kpconv":  # Open3D KPConv
            self.preprocess = tmu.preprocess
            self.transform_input = tmu.o3d_kpconv.transform_input
            self.update_probs = tmu.o3d_kpconv.update_probs
        else:
            self.preprocess = tmu.preprocess
            self.transform_input = tmu.transform_input
            self.update_probs = tmu.update_probs

        # Transformation for output labels
        self.transform_output = (
            lambda x: torch.argmax(x.squeeze(), axis=-1).squeeze().to(torch.uint8)
        )

    def inference(self, points: np.ndarray) -> np.ndarray:
        """Perform inference for a single point cloud

        :param points: Point cloud xyz array
        :type points: np.ndarray
        :return: Segmenation result as a point cloud with label indices
        :rtype: np.ndarray
        """
        # Preprocess point cloud
        points, search_tree, projected_indices = self.preprocess(points, self.model_cfg)

        # Init sampler if needed
        sampler = None
        if "sampler" in self.model_cfg:
            end_th = self.model_cfg.get("end_th", 0.5)
            sampler = ul.Sampler(
                points.shape[0],
                search_tree,
                self.model_cfg["sampler"],
                self.n_classes,
            )

        # Iterate over the sampled point cloud until all points reach the end threshold.
        # If no sampler is provided, the inference is performed in a single step.
        infer_complete = False
        while not infer_complete:
            # Get model input data
            input_data, selected_indices = self.transform_input(
                points, self.model_cfg, sampler
            )
            input_data = data_to_device(input_data, self.device)
            if self.model_cfg["input_format"] != "o3d_kpconv":
                input_data = unsqueeze_data(input_data)

            # Perform inference
            with torch.no_grad():
                result = self.model(*input_data)

                # TODO: check if this is consistent across different models
                if isinstance(result, dict):
                    result = result["out"]

            # Update probabilities if sampler is used
            if sampler is not None:
                if self.model_cfg["input_format"] == "o3d_kpconv":
                    sampler.test_probs = self.update_probs(
                        result,
                        selected_indices,
                        sampler.test_probs,
                        lengths=input_data[-1],
                    )
                else:
                    sampler.test_probs = self.update_probs(
                        result,
                        selected_indices,
                        sampler.test_probs,
                        self.n_classes,
                    )
                if sampler.p[sampler.p > end_th].shape[0] == sampler.p.shape[0]:
                    result = sampler.test_probs[projected_indices]
                    infer_complete = True
            else:
                result = result.squeeze().cpu()[projected_indices].cuda()
                infer_complete = True

        return self.transform_output(result).cpu().numpy()

    def eval(
        self,
        dataset: dm_dataset.LiDARSegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR segmentation dataset

        :param dataset: LiDAR segmentation dataset for which the evaluation will be performed
        :type dataset: LiDARSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: str | List[str], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Check that predictions_outdir is provided if results_per_sample is True
        if results_per_sample and predictions_outdir is None:
            raise ValueError(
                "If results_per_sample is True, predictions_outdir must be provided"
            )

        # Create predictions output directory if needed
        if predictions_outdir is not None:
            os.makedirs(predictions_outdir, exist_ok=True)

        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Retrieve ignored label indices
        ignored_label_indices = []
        for ignored_class in self.model_cfg.get("ignored_classes", []):
            ignored_label_indices.append(dataset.ontology[ignored_class]["idx"])

        # Get PyTorch dataset (no dataloader to avoid complexity with batching samplers)
        dataset = LiDARSegmentationTorchDataset(
            dataset,
            model_cfg=self.model_cfg,
            preprocess=self.preprocess,
            n_classes=self.n_classes,
            splits=[split] if isinstance(split, str) else split,
        )

        # Init metrics
        metrics_factory = um.MetricsFactory(self.n_classes)

        # Evaluation loop
        end_th = self.model_cfg.get("end_th", 0.5)
        with torch.no_grad():
            pbar = tqdm(dataset, total=len(dataset), leave=True)
            for idx, points, projected_indices, (label, _), sampler in pbar:
                # Iterate over the sampled point cloud until all points reach the end
                # threshold. If no sampler is provided, the inference is performed in a
                # single step.
                infer_complete = False
                while not infer_complete:
                    # Get model input data
                    input_data, selected_indices = self.transform_input(
                        points, self.model_cfg, sampler
                    )
                    input_data = data_to_device(input_data, self.device)
                    if self.model_cfg["input_format"] != "o3d_kpconv":
                        input_data = unsqueeze_data(input_data)

                    # Perform inference
                    with torch.no_grad():
                        pred = self.model(*input_data)

                        # TODO: check if this is consistent across different models
                        if isinstance(pred, dict):
                            pred = pred["out"]

                    if sampler is not None:
                        if self.model_cfg["input_format"] == "o3d_kpconv":
                            sampler.test_probs = self.update_probs(
                                pred,
                                selected_indices,
                                sampler.test_probs,
                                lengths=input_data[-1],
                            )
                        else:
                            sampler.test_probs = self.update_probs(
                                pred,
                                selected_indices,
                                sampler.test_probs,
                                self.n_classes,
                            )
                        if sampler.p[sampler.p > end_th].shape[0] == sampler.p.shape[0]:
                            pred = sampler.test_probs[projected_indices]
                            infer_complete = True
                    else:
                        pred = pred.squeeze().cpu()[projected_indices].cuda()
                        infer_complete = True

                # Get valid points masks depending on ignored label indices
                label = torch.tensor(label, device=self.device)
                if ignored_label_indices:
                    valid_mask = torch.ones_like(label, dtype=torch.bool)
                    for idx in ignored_label_indices:
                        valid_mask *= label != idx
                else:
                    valid_mask = None

                # Convert labels if needed
                if lut_ontology is not None:
                    label = lut_ontology[label]

                # Prepare data and update metrics factory
                label = label.cpu().unsqueeze(0).numpy()
                pred = self.transform_output(pred)
                pred = pred.cpu().unsqueeze(0).to(torch.int64).numpy()
                if valid_mask is not None:
                    valid_mask = valid_mask.cpu().unsqueeze(0).numpy()

                metrics_factory.update(pred, label, valid_mask)

                # Store predictions and results per sample if required
                if predictions_outdir is not None:
                    for i, (sample_idx, sample_pred, sample_label) in enumerate(
                        zip(idx, pred, label)
                    ):
                        if results_per_sample:
                            sample_valid_mask = (
                                valid_mask[i] if valid_mask is not None else None
                            )
                            sample_mf = um.MetricsFactory(n_classes=self.n_classes)
                            sample_mf.update(
                                sample_pred, sample_label, sample_valid_mask
                            )
                            sample_df = um.get_metrics_dataframe(
                                sample_mf, self.ontology
                            )
                            sample_df.to_csv(
                                os.path.join(predictions_outdir, f"{sample_idx}.csv")
                            )
                        pred.tofile(
                            os.path.join(predictions_outdir, f"{sample_idx}.bin")
                        )

        return um.get_metrics_dataframe(metrics_factory, self.ontology)

    def get_computational_cost(self, runs: int = 30, warm_up_runs: int = 5) -> dict:
        """Get different metrics related to the computational cost of the model

        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
        """
        # Build dummy input data (process is a bit complex for LiDAR models)
        dummy_points = np.random.rand(1000000, 4)
        dummy_points, search_tree, _ = self.preprocess(dummy_points, self.model_cfg)

        sampler = None
        if "sampler" in self.model_cfg:
            sampler = ul.Sampler(
                point_cloud_size=dummy_points.shape[0],
                search_tree=search_tree,
                sampler_name=self.model_cfg["sampler"],
                num_classes=self.n_classes,
            )

        dummy_input, _ = self.transform_input(dummy_points, self.model_cfg, sampler)
        dummy_input = data_to_device(dummy_input, self.device)
        if self.model_cfg["input_format"] != "o3d_kpconv":
            dummy_input = unsqueeze_data(dummy_input)

        # Get computational cost
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
