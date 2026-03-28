from copy import copy
import os
import time
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors
from tqdm.auto import tqdm

from perceptionmetrics.datasets import detection as detection_dataset
from perceptionmetrics.models import detection as detection_model
from perceptionmetrics.utils import detection_metrics as um
from perceptionmetrics.utils import image as ui


def get_resize_args(resize_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the resize arguments for torchvision.transforms.Resize from the configuration.

    :param resize_cfg: Resize configuration dictionary
    :return: Dictionary with arguments for transforms.Resize
    """
    resize_args = {"interpolation": transforms.InterpolationMode.BILINEAR}
    fixed_h = resize_cfg.get("height")
    fixed_w = resize_cfg.get("width")
    min_side = resize_cfg.get("min_side")
    max_side = resize_cfg.get("max_side")

    if fixed_h is not None and fixed_w is not None:
        if min_side is not None:
            raise ValueError(
                "Resize config cannot satisfy both fixed dimensions (width/height) and min_side. They are mutually exclusive."
            )
        resize_args["size"] = (fixed_h, fixed_w)
    elif min_side is not None:
        resize_args["size"] = min_side
        if fixed_h is not None or fixed_w is not None:
            raise ValueError(
                "Resize config cannot satisfy both fixed dimensions (width/height) and min_side. They are mutually exclusive."
            )
    else:
        raise ValueError(
            "Resize config must contain either 'height' and 'width' or 'min_side' and 'max_side'."
        )

    if max_side is not None:
        resize_args["max_size"] = max_side

    return resize_args


def data_to_device(
    data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    device: torch.device,
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
    """Dataset for image detection PyTorch models

    :param dataset: Image detection dataset
    :type dataset: ImageDetectionDataset
    :param transform: Transformation to be applied to images
    :type transform: transforms.Compose
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    """

    def __init__(
        self,
        dataset: detection_dataset.ImageDetectionDataset,
        transform: transforms.Compose,
        splits: List[str] = ["test"],
    ):
        self.dataset = copy(dataset)

        # Filter split and make filenames global
        self.dataset.dataset = self.dataset.dataset[
            self.dataset.dataset["split"].isin(splits)
        ]

        # Use the dataset's make_fname_global method instead of manual path joining
        self.dataset.make_fname_global()

        self.transform = transform

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load image and annotations, apply transforms.

        :param idx: Sample index
        :return: Tuple of (sample_id, image_tensor, target_dict)
        """
        row = self.dataset.dataset.iloc[idx]
        image_path = row["image"]
        ann_path = row["annotation"]

        image = Image.open(image_path).convert("RGB")
        boxes, category_indices = self.dataset.read_annotation(ann_path)

        # Convert boxes/labels to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        boxes = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(image.height, image.width)
        )
        category_indices = torch.as_tensor(category_indices, dtype=torch.int64)

        target = {
            "boxes": boxes,  # [N, 4]
            "labels": category_indices,  # [N]
        }

        if self.transform:
            image, target = self.transform(image, target)

        return self.dataset.dataset.index[idx], image, target


class TorchImageDetectionModel(detection_model.ImageDetectionModel):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        model_cfg: str,
        ontology_fname: str,
        device: torch.device = None,
    ):
        """Image detection model for PyTorch framework

        :param model: Either the filename of a TorchScript model or the model already loaded into a PyTorch module.
        :type model: Union[str, torch.nn.Module]
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        :param device: torch.device to use (optional). If not provided, will auto-select cuda, mps, or cpu.
        """
        # Get device (GPU, MPS, or CPU) if not provided
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        # Load model from file or use passed instance
        if isinstance(model, str):
            assert os.path.isfile(model), "Torch model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model, map_location=self.device)
                model_type = "compiled"
            except Exception:
                print(
                    "Model is not a TorchScript model. Loading as native PyTorch model."
                )

                from ultralytics.nn.tasks import DetectionModel
                import torch.serialization

                torch.serialization.add_safe_globals([DetectionModel])

                model = torch.load(model, map_location=self.device, weights_only=False)

                if isinstance(model, dict):
                    if "model" in model:
                        model = model["model"]
                    elif "state_dict" in model:
                        model = model["state_dict"]

                model_type = "native"
        elif isinstance(model, torch.nn.Module):
            model_fname = None
            model_type = "native"
        else:
            raise ValueError("Model must be a filename or a torch.nn.Module")

        # Init parent class
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)
        self.model = self.model.to(self.device).eval()

        # FORCE CPU + FLOAT32 FIX
        if str(self.device) == "cpu":
            try:
                self.model.float()
            except:
                try:
                    self.model.model.float()
                except:
                    pass

        # Load post-processing functions for specific model formats
        self.model_format = self.model_cfg.get("model_format", "torchvision")
        if self.model_format == "yolo":
            from perceptionmetrics.models.utils.yolo import postprocess_detection
        elif self.model_format == "torchvision":
            from perceptionmetrics.models.utils.torchvision import postprocess_detection
        else:
            raise ValueError(f"Unsupported model_format: {self.model_format}")

        self.postprocess_detection = postprocess_detection

        # Load confidence and NMS thresholds from config
        self.confidence_threshold = self.model_cfg.get("confidence_threshold", 0.5)
        self.nms_threshold = self.model_cfg.get("nms_threshold", 0.3)
        self.max_detections = self.model_cfg.get("max_detections", 20)

        # Standardize post-processing arguments
        self.postprocess_args = [
            self.confidence_threshold,
            self.nms_threshold,
            self.max_detections,
        ]

        # Build idx -> class name mapping for ontology
        self.idx_to_class_name = {}
        if isinstance(self.ontology, dict) and "classes" in self.ontology:
            for item in self.ontology["classes"]:
                self.idx_to_class_name[item["id"]] = item["name"]
        elif isinstance(self.ontology, list):
            for i, name in enumerate(self.ontology):
                self.idx_to_class_name[i] = name
        else:
            for k, v in self.ontology.items():
                if isinstance(v, dict):
                    idx = v.get("idx", k)
                    self.idx_to_class_name[int(idx)] = k
                elif isinstance(v, int):
                    self.idx_to_class_name[v] = k
                elif str(k).isdigit():
                    self.idx_to_class_name[int(k)] = str(v)

        # Build input transforms
        self.transform_input = []

        resize_cfg = self.model_cfg.get("resize")
        if resize_cfg is not None:
            resize_args = get_resize_args(resize_cfg)
            self.transform_input.append(transforms.Resize(**resize_args))
        else:
            print("'resize_cfg' missing in model config. No resizing will be applied.")

        if "crop" in self.model_cfg:
            crop_size = (
                self.model_cfg["crop"]["height"],
                self.model_cfg["crop"]["width"],
            )
            self.transform_input += [transforms.CenterCrop(crop_size)]

        try:
            self.transform_input += [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        except AttributeError:
            self.transform_input += [
                transforms.ToImageTensor(),
                transforms.ConvertDtype(torch.float32),
            ]

        self.transform_input = transforms.Compose(self.transform_input)

    def predict(
        self, image: Image.Image, return_sample: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], torch.Tensor]]:
        """Perform prediction and scale boxes back to original size."""
        orig_w, orig_h = image.size
        sample = self.transform_input(image).unsqueeze(0).to(self.device)

        # Capture input dimensions after preprocessing
        _, _, target_h, target_w = sample.shape

        result = self.inference(sample)

        # SCALE BOXES BACK TO ORIGINAL IMAGE SIZE
        if "boxes" in result and result["boxes"].numel() > 0:
            boxes = result["boxes"].clone()
            # Scaling coordinates from model input space back to original image space
            boxes[:, [0, 2]] *= orig_w / target_w
            boxes[:, [1, 3]] *= orig_h / target_h
            result["boxes"] = boxes

        # MAP INDICES TO NAMES FOR DISPLAY
        if "labels" in result:
            labels = result["labels"]
            class_names = [
                self.idx_to_class_name.get(int(l), f"class_{l}") for l in labels
            ]
            result["class_names"] = class_names

        if return_sample:
            return result, sample
        else:
            return result

    def inference(self, tensor_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform inference for a tensor"""
        with torch.no_grad():
            if hasattr(self.model, "predict") and not isinstance(
                self.model, torch.nn.Module
            ):
                results = self.model.predict(tensor_in)
            else:
                results = self.model(tensor_in)

        if isinstance(results, list) and len(results) > 0:
            result = results[0]
        else:
            result = results

        result = self.postprocess_detection(result, *self.postprocess_args)
        return result

    def eval(
        self,
        dataset: detection_dataset.ImageDetectionDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
        save_visualizations: bool = False,
        progress_callback=None,
        metrics_callback=None,
    ) -> pd.DataFrame:
        """Evaluate model over a detection dataset and compute metrics"""
        if (results_per_sample or save_visualizations) and predictions_outdir is None:
            raise ValueError(
                "predictions_outdir required if results_per_sample or save_visualizations is True"
            )

        if predictions_outdir is not None:
            os.makedirs(predictions_outdir, exist_ok=True)

        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        if lut_ontology is not None:
            lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        torch_dataset = ImageDetectionTorchDataset(
            dataset,
            transform=self.transform_input,
            splits=[split] if isinstance(split, str) else split,
        )

        num_workers = (
            0 if progress_callback is not None else self.model_cfg.get("num_workers", 0)
        )

        dataloader = DataLoader(
            torch_dataset,
            batch_size=self.model_cfg.get("batch_size", 1),
            num_workers=num_workers,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

        iou_threshold = self.model_cfg.get("iou_threshold", 0.5)
        evaluation_step = self.model_cfg.get("evaluation_step", None)
        if evaluation_step == 0:
            evaluation_step = None

        metrics_factory = um.DetectionMetricsFactory(
            iou_threshold=iou_threshold, num_classes=self.n_classes
        )

        total_samples = len(dataloader.dataset)
        processed_samples = 0

        with torch.no_grad():
            iterator = (
                dataloader
                if progress_callback is not None
                else tqdm(dataloader, leave=True)
            )

            for image_ids, images, targets in iterator:
                if not images or any(image.numel() == 0 for image in images):
                    continue

                images = torch.stack(images).to(self.device)
                if hasattr(self.model, "predict") and not isinstance(
                    self.model, torch.nn.Module
                ):
                    batch_results = self.model.predict(images)
                else:
                    batch_results = self.model(images)

                for i in range(len(images)):
                    gt = targets[i]
                    r = batch_results[i]
                    pred = self.postprocess_detection(r, *self.postprocess_args)

                    image_tensor = images[i]
                    sample_id = image_ids[i]

                    if lut_ontology is not None:
                        gt["labels"] = lut_ontology[gt["labels"]]

                    metrics_factory.update(
                        gt["boxes"],
                        gt["labels"],
                        pred["boxes"],
                        pred["labels"],
                        pred["scores"],
                    )

                    if predictions_outdir is not None:
                        pred_boxes = pred["boxes"].cpu().numpy()
                        pred_labels = pred["labels"].cpu().numpy()
                        pred_scores = pred["scores"].cpu().numpy()

                        if results_per_sample:
                            out_data = []
                            for box, label, score in zip(
                                pred_boxes, pred_labels, pred_scores
                            ):
                                class_name = self.idx_to_class_name.get(
                                    int(label), f"class_{label}"
                                )
                                out_data.append(
                                    {
                                        "image_id": sample_id,
                                        "label": class_name,
                                        "score": float(score),
                                        "bbox": box.tolist(),
                                    }
                                )

                            df = pd.DataFrame(out_data)
                            df.to_json(
                                os.path.join(predictions_outdir, f"{sample_id}.json"),
                                orient="records",
                                indent=2,
                            )

                            sample_mf = um.DetectionMetricsFactory(
                                iou_threshold=iou_threshold, num_classes=self.n_classes
                            )
                            sample_mf.update(
                                gt["boxes"],
                                gt["labels"],
                                pred["boxes"],
                                pred["labels"],
                                pred["scores"],
                            )
                            sample_df = sample_mf.get_metrics_dataframe(self.ontology)
                            sample_df.to_csv(
                                os.path.join(
                                    predictions_outdir, f"{sample_id}_metrics.csv"
                                )
                            )

                        if save_visualizations:
                            pil_image = transforms.ToPILImage()(image_tensor.cpu())
                            gt_boxes = gt["boxes"].cpu().numpy()
                            gt_labels = gt["labels"].cpu().numpy()
                            gt_class_names = [
                                self.idx_to_class_name.get(int(l), str(l))
                                for l in gt_labels
                            ]
                            pred_class_names = [
                                self.idx_to_class_name.get(int(l), str(l))
                                for l in pred_labels
                            ]

                            image_gt = ui.draw_detections(
                                pil_image.copy(),
                                gt_boxes,
                                gt_labels,
                                gt_class_names,
                                scores=None,
                            )
                            image_pred = ui.draw_detections(
                                pil_image.copy(),
                                pred_boxes,
                                pred_labels,
                                pred_class_names,
                                scores=pred_scores,
                            )

                            pil_gt = Image.fromarray(image_gt)
                            pil_pred = Image.fromarray(image_pred)

                            combined_image = Image.new(
                                "RGB",
                                (
                                    pil_gt.width + pil_pred.width,
                                    max(pil_gt.height, pil_pred.height),
                                ),
                            )
                            combined_image.paste(pil_gt, (0, 0))
                            combined_image.paste(pil_pred, (pil_gt.width, 0))
                            combined_image.save(
                                os.path.join(predictions_outdir, f"{sample_id}.jpg")
                            )

                    processed_samples += 1
                    if progress_callback is not None:
                        progress_callback(processed_samples, total_samples)

                    if (
                        metrics_callback is not None
                        and evaluation_step is not None
                        and processed_samples % evaluation_step == 0
                    ):
                        metrics_callback(
                            metrics_factory.get_metrics_dataframe(self.ontology),
                            processed_samples,
                            total_samples,
                        )

        return {
            "metrics_df": metrics_factory.get_metrics_dataframe(self.ontology),
            "metrics_factory": metrics_factory,
        }

    def get_computational_cost(
        self, image_size: Tuple[int], runs: int = 30, warm_up_runs: int = 5
    ) -> dict:
        """Get computational cost metrics like inference time

        :param image_size: Size of input image (H, W)
        :type image_size: Tuple[int]
        :param runs: Number of repeated runs to average over
        :type runs: int
        :param warm_up_runs: Warm-up runs before timing
        :type warm_up_runs: int
        :return: Dictionary with computational cost details
        :rtype: dict
        """
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
