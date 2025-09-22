import importlib
import os
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.transforms import v2 as transforms
    from torchvision.transforms.v2 import functional as F
except ImportError:
    from torchvision.transforms import transforms
    from torchvision.transforms import functional as F
from tqdm import tqdm

from detectionmetrics.datasets import dataset as dm_dataset
from detectionmetrics.models import model as dm_model
import detectionmetrics.utils.metrics as um
import detectionmetrics.utils.torch as ut


AVAILABLE_MODEL_FORMATS_LIDAR = ["o3d_randlanet", "o3d_kpconv", "mmdet3d"]


def raise_unknown_model_format_lidar(model_format: str) -> None:
    """Raise an exception if the LiDAR model format is unknown

    :param input_format: Model format string
    :type input_format: str
    """
    msg = f"Unknown model format: {model_format}."
    msg += f"Available formats: {AVAILABLE_MODEL_FORMATS_LIDAR}"
    raise Exception(msg)


def get_computational_cost(
    model: Any,
    dummy_input: torch.Tensor,
    model_fname: Optional[str] = None,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> pd.DataFrame:
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
    :return: DataFrame containing computational cost information
    :rtype: pd.DataFrame
    """
    # Get model size if possible
    if model_fname is not None:
        size_mb = os.path.getsize(model_fname) / 1024**2
    else:
        size_mb = None

    # Measure inference time with GPU synchronization
    dummy_tuple = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    for _ in range(warm_up_runs):
        if hasattr(model, "inference"):  # e.g. mmsegmentation models
            model.inference(
                *dummy_tuple,
                [
                    dict(
                        ori_shape=dummy_tuple[0].shape[2:],
                        img_shape=dummy_tuple[0].shape[2:],
                        pad_shape=dummy_tuple[0].shape[2:],
                        padding_size=[0, 0, 0, 0],
                    )
                ]
                * dummy_tuple[0].shape[0],
            )
        else:
            model(*dummy_tuple)

    inference_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start_time = time.time()
        if hasattr(model, "inference"):  # e.g. mmsegmentation models
            model.inference(
                *dummy_tuple,
                [
                    dict(
                        ori_shape=dummy_tuple[0].shape[2:],
                        img_shape=dummy_tuple[0].shape[2:],
                        pad_shape=dummy_tuple[0].shape[2:],
                        padding_size=[0, 0, 0, 0],
                    )
                ]
                * dummy_tuple[0].shape[0],
            )
        else:
            model(*dummy_tuple)
        torch.cuda.synchronize()
        end_time = time.time()
        inference_times.append(end_time - start_time)

    result = {
        "input_shape": ["x".join(map(str, ut.get_data_shape(dummy_input)))],
        "n_params": [sum(p.numel() for p in model.parameters())],
        "size_mb": [size_mb],
        "inference_time_s": [np.mean(inference_times)],
    }

    return pd.DataFrame.from_dict(result)


class CustomResize(torch.nn.Module):
    """Custom rescale transformation for PyTorch. If only one dimension is provided,
    the aspect ratio is preserved.

    :param width: Target width for resizing
    :type width: Optional[int], optional
    :param height: Target height for resizing
    :type height: Optional[int], optional
    :param interpolation: Interpolation mode for resizing (e.g. NEAREST, BILINEAR)
    :type interpolation: F.InterpolationMode, defaults to F.InterpolationMode.BILINEAR
    :param closest_divisor: Closest divisor for the target size, defaults to 16. Only applies to the dimension not provided.
    :type closest_divisor: int, optional
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        closest_divisor: int = 16,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.closest_divisor = closest_divisor

    def forward(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        old_size = (h, w)

        if self.width is None:
            w = int((self.height / image.size[1]) * image.size[0])
            h = self.height
        if self.height is None:
            h = int((self.width / image.size[0]) * image.size[1])
            w = self.width

        h = round(h / self.closest_divisor) * self.closest_divisor
        w = round(w / self.closest_divisor) * self.closest_divisor
        new_size = (h, w)

        if new_size != old_size:
            image = F.resize(image, new_size, self.interpolation)

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
    """Dataset for LiDAR segmentation PyTorch - Open3D-ML models

    :param dataset: LiDAR segmentation dataset
    :type dataset: LiDARSegmentationDataset
    :param model_cfg: Dictionary containing model configuration
    :type model_cfg: dict
    :param get_sample: Function for loading sample data
    :type get_sample: callable
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    """

    def __init__(
        self,
        dataset: dm_dataset.LiDARSegmentationDataset,
        model_cfg: dict,
        get_sample: callable,
        splits: str = ["test"],
    ):
        # Filter split and make filenames global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        self.dataset = dataset
        self.dataset.make_fname_global()
        self.model_cfg = model_cfg
        self.get_sample = get_sample

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(self, idx: int):
        """Prepare sample data

        :param idx: Sample index
        :type idx: int
        :return: Sample data required by the model
        """
        return self.get_sample(
            points_fname=self.dataset.dataset.iloc[idx]["points"],
            model_cfg=self.model_cfg,
            label_fname=self.dataset.dataset.iloc[idx]["label"],
            name=self.dataset.dataset.index[idx],
            idx=idx,
            has_intensity=self.dataset.has_intensity,
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
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # If 'model' contains a string, check that it is a valid filename and load model
        if isinstance(model, str):
            assert os.path.isfile(model), "TorchScript Model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model, map_location=self.device)
                model_type = "compiled"
            except:
                print("Model is not a TorchScript model. Loading as a PyTorch module.")
                model = torch.load(model, map_location=self.device)
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

        if "resize" in self.model_cfg:
            self.transform_input += [
                CustomResize(
                    width=self.model_cfg["resize"].get("width", None),
                    height=self.model_cfg["resize"].get("height", None),
                    interpolation=F.InterpolationMode.BILINEAR,
                )
            ]
            self.transform_label += [
                CustomResize(
                    width=self.model_cfg["resize"].get("width", None),
                    height=self.model_cfg["resize"].get("height", None),
                    interpolation=F.InterpolationMode.NEAREST,
                )
            ]

        if "crop" in self.model_cfg:
            crop_size = (
                self.model_cfg["crop"]["height"],
                self.model_cfg["crop"]["width"],
            )
            self.transform_input += [transforms.CenterCrop(crop_size)]
            self.transform_label += [transforms.CenterCrop(crop_size)]

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

    def predict(self, image: Image.Image) -> Image.Image:
        """Perform prediction for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: Segmentation result as a PIL image
        :rtype: Image.Image
        """
        tensor = self.transform_input(image).unsqueeze(0).to(self.device)
        result = self.predict(tensor)
        return self.transform_output(result)

    def predict(self, tensor_in: torch.Tensor) -> torch.Tensor:
        """Perform inference for a tensor

        :param tensor_in: Input point cloud tensor
        :type tensor_in: torch.Tensor
        :return: Segmentation result as tensor
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            # Perform inference
            if hasattr(self.model, "inference"):  # e.g. mmsegmentation models
                tensor_out = self.model.inference(
                    tensor_in.to(self.device),
                    [
                        dict(
                            ori_shape=tensor_in.shape[2:],
                            img_shape=tensor_in.shape[2:],
                            pad_shape=tensor_in.shape[2:],
                            padding_size=[0, 0, 0, 0],
                        )
                    ]
                    * tensor_in.shape[0],
                )
            else:
                tensor_out = self.model(tensor_in.to(self.device))

            if isinstance(tensor_out, dict):
                tensor_out = tensor_out["out"]

        return tensor_out

    def eval(
        self,
        dataset: dm_dataset.ImageSegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
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
                pred = self.predict(image)

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

    def get_computational_cost(
        self, image_size: Tuple[int], runs: int = 30, warm_up_runs: int = 5
    ) -> dict:
        """Get different metrics related to the computational cost of the model

        :param image_size: Image size used for inference
        :type image_size: Tuple[int]
        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
        """
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
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
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # If 'model' contains a string, check that it is a valid filename and load model
        if isinstance(model, str):
            assert os.path.isfile(model), "TorchScript Model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model, map_location=self.device)
                model_type = "compiled"
            except Exception:
                print("Model is not a TorchScript model. Loading as a PyTorch module.")
                model = torch.load(model, map_location=self.device)
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

        # Init specific attributes and update model configuration
        self.model_format = self.model_cfg["model_format"]

        # Init model specific functions
        model_format = self.model_format.split("_")[0]
        model_utils_module_str = (
            f"detectionmetrics.models.lidar_torch_utils.{model_format}"
        )
        try:
            model_utils_module = importlib.import_module(model_utils_module_str)
        except ImportError:
            raise_unknown_model_format_lidar(model_format)
        self._get_sample = model_utils_module.get_sample
        self._inference = model_utils_module.inference

    def predict(
        self,
        points_fname: str,
        has_intensity: bool = True,
    ) -> np.ndarray:
        """Perform prediction for a single point cloud

        :param points_fname: Point cloud in SemanticKITTI .bin format
        :type points_fname: str
        :param has_intensity: Whether the point cloud has intensity values, defaults to True
        :type has_intensity: bool, optional
        :return: Segmenation result as a point cloud with label indices
        :rtype: np.ndarray
        """
        # Preprocess point cloud
        sample = self._get_sample(
            points_fname, self.model_cfg, has_intensity=has_intensity
        )
        pred, _, _ = self.inference(sample, self.model, self.model_cfg)

        return pred.squeeze().cpu().numpy()

    def eval(
        self,
        dataset: dm_dataset.LiDARSegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR segmentation dataset

        :param dataset: LiDAR segmentation dataset for which the evaluation will be performed
        :type dataset: LiDARSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
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
        dataset = LiDARSegmentationTorchDataset(
            dataset,
            self.model_cfg,
            self._get_sample,
            splits=[split] if isinstance(split, str) else split,
        )

        # Init metrics
        metrics_factory = um.MetricsFactory(self.n_classes)

        # Evaluation loop
        with torch.no_grad():
            pbar = tqdm(dataset, total=len(dataset), leave=True)
            for sample in pbar:
                # Perform inference
                pred, label, name = self.inference(sample, self.model, self.model_cfg)

                # Get valid points masks depending on ignored label indices
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
                label = label.cpu().numpy()
                pred = pred.cpu().numpy()
                if valid_mask is not None:
                    valid_mask = valid_mask.cpu().numpy()

                metrics_factory.update(pred, label, valid_mask)

                # Store predictions and results per sample if required
                if predictions_outdir is not None:
                    for i, (sample_name, sample_pred, sample_label) in enumerate(
                        zip(name, pred, label)
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
                                os.path.join(predictions_outdir, f"{sample_name}.csv")
                            )
                        pred.tofile(
                            os.path.join(predictions_outdir, f"{sample_name}.bin")
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
        dummy_points, _, sampler = self.preprocess(dummy_points, self.model_cfg)

        dummy_input, _ = self.transform_input(dummy_points, self.model_cfg, sampler)
        dummy_input = ut.data_to_device(dummy_input, self.device)
        if self.model_format != "o3d_kpconv":
            dummy_input = ut.unsqueeze_data(dummy_input)

        # Get computational cost
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
