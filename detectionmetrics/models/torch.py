import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
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
    model: torch.jit.ScriptModule,
    model_fname: str,
    dummy_input: torch.Tensor,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> dict:
    """Get different metrics related to the computational cost of the model

    :param model: TorchScript model
    :type model: torch.jit.ScriptModule
    :param model_fname: Path to the model file
    :type model_fname: str
    :param dummy_input: Dummy input data for the model
    :type dummy_input: torch.Tensor
    :param runs: Number of runs to measure inference time, defaults to 30
    :type runs: int, optional
    :param warm_up_runs: Number of warm-up runs, defaults to 5
    :type warm_up_runs: int, optional
    :return: Dictionary containing computational cost information
    """

    computational_cost = {}

    computational_cost["input_shape"] = get_data_shape(dummy_input)
    computational_cost["size_mb"] = os.path.getsize(model_fname) / 1024**2
    computational_cost["n_params"] = sum(p.numel() for p in model.parameters())

    # Measure inference time with GPU synchronization
    dummy_input = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    for _ in range(warm_up_runs):
        model(*dummy_input)

    inference_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start_time = time.time()
        model(*dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        inference_times.append(end_time - start_time)

    computational_cost["time_s"] = np.mean(inference_times)

    return computational_cost


class ImageSegmentationTorchDataset(Dataset):
    """Dataset for image segmentation PyTorch models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param transform: Transformation to be applied to images
    :type transform: transforms.Compose
    :param target_transform: Transformation to be applied to labels
    :type target_transform: transforms.Compose
    :param split: Split to be used from the dataset, defaults to "all"
    :type split: str, optional
    """

    def __init__(
        self,
        dataset: dm_dataset.ImageSegmentationDataset,
        transform: transforms.Compose,
        target_transform: transforms.Compose,
        split: str = "all",
    ):
        # Filter split and make filenames global
        if split != "all":
            dataset.dataset = dataset.dataset[dataset.dataset["split"] == split]
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
        return image, label


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
    :param split: Split to be used from the dataset, defaults to "all"
    :type split: str, optional
    """

    def __init__(
        self,
        dataset: dm_dataset.LiDARSegmentationDataset,
        model_cfg: dict,
        preprocess: callable,
        n_classes: int,
        split: str = "all",
    ):
        # Filter split and make filenames global
        if split != "all":
            dataset.dataset = dataset.dataset[dataset.dataset["split"] == split]
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
            preprocessed_points,
            projected_indices,
            (semantic_label, instance_label),
            sampler,
        )


class TorchImageSegmentationModel(dm_model.ImageSegmentationModel):

    def __init__(self, model_fname: str, model_cfg: str, ontology_fname: str):
        """Image segmentation model for PyTorch framework

        :param model_fname: PyTorch model saved using TorchScript
        :type model_fname: str
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        """
        super().__init__(model_fname, ontology_fname, model_cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check that provided path exist and load model
        assert os.path.isfile(model_fname), "Model file not found"
        self.model = torch.jit.load(model_fname).to(self.device)
        self.model.eval()

        # Init transformations for input images, output labels, and GT labels
        self.transform_input = []
        self.transform_label = []

        if "image_size" in self.model_cfg:
            self.transform_input += [
                transforms.Resize(tuple(self.model_cfg["image_size"]))
            ]
            self.transform_label += [
                transforms.Resize(
                    tuple(self.model_cfg["image_size"]),
                    interpolation=transforms.InterpolationMode.NEAREST_EXACT,
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
        split: str = "all",
        ontology_translation: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will
        be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split to be used from the dataset, defaults to "all"
        :type split: str, optional
        :param ontology_translation: JSON file containing translation between dataset
        and model output ontologies
        :type ontology_translation: str, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Retrieve ignored label indices
        self.ignored_label_indices = []
        for ignored_class in self.model_cfg.get("ignored_classes", []):
            self.ignored_label_indices.append(dataset.ontology[ignored_class]["idx"])

        # Get PyTorch dataloader
        dataset = ImageSegmentationTorchDataset(
            dataset,
            transform=self.transform_input,
            target_transform=self.transform_label,
            split=split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.model_cfg.get("batch_size", 1),
            num_workers=self.model_cfg.get("num_workers", 1),
        )

        # Init metrics
        results = {}
        iou = um.IoU(self.n_classes)
        cm = um.ConfusionMatrix(self.n_classes)

        # Evaluation loop
        with torch.no_grad():
            pbar = tqdm(dataloader, leave=True)
            for image, label in pbar:
                # Perform inference
                with torch.no_grad():
                    pred = self.model(image.to(self.device))
                    if isinstance(pred, dict):
                        pred = pred["out"]

                # Get valid points masks depending on ignored label indices
                if self.ignored_label_indices:
                    valid_mask = torch.ones_like(label, dtype=torch.bool)
                    for idx in self.ignored_label_indices:
                        valid_mask *= label != idx
                else:
                    valid_mask = None

                # Convert labels if needed
                if lut_ontology is not None:
                    label = lut_ontology[label]

                # Prepare data and update confusion matrix
                label = label.squeeze(dim=1).cpu()
                pred = torch.argmax(pred, axis=1).cpu()
                if valid_mask is not None:
                    valid_mask = valid_mask.squeeze(dim=1).cpu()

                cm.update(
                    pred.numpy(),
                    label.numpy(),
                    valid_mask.numpy() if valid_mask is not None else None,
                )

                # Prepare data and update IoU
                label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
                label = label.permute(0, 3, 1, 2)
                pred = torch.nn.functional.one_hot(pred, num_classes=self.n_classes)
                pred = pred.permute(0, 3, 1, 2)
                if valid_mask is not None:
                    valid_mask = valid_mask.unsqueeze(1).repeat(1, self.n_classes, 1, 1)

                iou.update(
                    pred.numpy(),
                    label.numpy(),
                    valid_mask.numpy() if valid_mask is not None else None,
                )

        # Get metrics results
        iou_per_class, iou = iou.compute()
        acc_per_class, acc = cm.get_accuracy()
        iou_per_class = [float(n) for n in iou_per_class]
        acc_per_class = [float(n) for n in acc_per_class]

        # Build results dataframe
        results = {}
        for class_name, class_data in self.ontology.items():
            results[class_name] = {
                "iou": iou_per_class[class_data["idx"]],
                "acc": acc_per_class[class_data["idx"]],
            }
        results["global"] = {"iou": iou, "acc": acc}

        return pd.DataFrame(results)

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
            self.model, self.model_fname, dummy_input, runs, warm_up_runs
        )


class TorchLiDARSegmentationModel(dm_model.LiDARSegmentationModel):

    def __init__(self, model_fname: str, model_cfg: str, ontology_fname: str):
        """LiDAR segmentation model for PyTorch framework

        :param model_fname: PyTorch model saved using TorchScript
        :type model_fname: str
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        """
        super().__init__(model_fname, ontology_fname, model_cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check that provided path exist and load model
        assert os.path.isfile(model_fname), "Model file not found"
        self.model = torch.jit.load(model_fname).to(self.device)
        self.model.eval()

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
        split: str = "all",
        ontology_translation: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR segmentation dataset

        :param dataset: LiDAR segmentation dataset for which the evaluation will
        be performed
        :type dataset: LiDARSegmentationDataset
        :param split: Split to be used from the dataset, defaults to "all"
        :type split: str, optional
        :param ontology_translation: JSON file containing translation between dataset
        and model output ontologies
        :type ontology_translation: str, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Retrieve ignored label indices
        self.ignored_label_indices = []
        for ignored_class in self.model_cfg.get("ignored_classes", []):
            self.ignored_label_indices.append(dataset.ontology[ignored_class]["idx"])

        # Get PyTorch dataset (no dataloader to avoid complexity with batching samplers)
        dataset = LiDARSegmentationTorchDataset(
            dataset,
            model_cfg=self.model_cfg,
            preprocess=self.preprocess,
            n_classes=self.n_classes,
            split=split,
        )

        # Init metrics
        iou = um.IoU(self.n_classes)
        cm = um.ConfusionMatrix(self.n_classes)

        # Evaluation loop
        results = {}
        end_th = self.model_cfg.get("end_th", 0.5)
        with torch.no_grad():
            pbar = tqdm(dataset, total=len(dataset), leave=True)
            for points, projected_indices, (label, _), sampler in pbar:
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
                if self.ignored_label_indices:
                    valid_mask = torch.ones_like(label, dtype=torch.bool)
                    for idx in self.ignored_label_indices:
                        valid_mask *= label != idx
                else:
                    valid_mask = None

                # Convert labels if needed
                if lut_ontology is not None:
                    label = lut_ontology[label]

                # Prepare data and update confusion matrix
                label = label.cpu().unsqueeze(0)
                pred = self.transform_output(pred).cpu().unsqueeze(0).to(torch.int64)
                if valid_mask is not None:
                    valid_mask = valid_mask.cpu().unsqueeze(0)

                cm.update(
                    pred.numpy(),
                    label.numpy(),
                    valid_mask.numpy() if valid_mask is not None else None,
                )

                # Prepare data and update IoU
                label = torch.nn.functional.one_hot(label, self.n_classes)
                label = label.permute(0, 2, 1)
                pred = torch.nn.functional.one_hot(pred, self.n_classes)
                pred = pred.permute(0, 2, 1)
                if valid_mask is not None:
                    valid_mask = valid_mask.unsqueeze(1).repeat(1, self.n_classes, 1)

                iou.update(
                    pred.numpy(),
                    label.numpy(),
                    valid_mask.numpy() if valid_mask is not None else None,
                )

        # Get metrics results
        iou_per_class, iou = iou.compute()
        acc_per_class, acc = cm.get_accuracy()
        iou_per_class = [float(n) for n in iou_per_class]
        acc_per_class = [float(n) for n in acc_per_class]

        # Build results dataframe
        results = {}
        for class_name, class_data in self.ontology.items():
            if class_data["idx"] not in self.model_cfg.get("ignored_indices", []):
                results[class_name] = {
                    "iou": iou_per_class[class_data["idx"]],
                    "acc": acc_per_class[class_data["idx"]],
                }
        results["global"] = {"iou": iou, "acc": acc}

        results = pd.DataFrame(results)
        results.index.name = "metric"

        return results

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
            self.model, self.model_fname, dummy_input, runs, warm_up_runs
        )
