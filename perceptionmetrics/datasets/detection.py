from abc import abstractmethod
import json
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from perceptionmetrics.datasets.perception import PerceptionDataset
import perceptionmetrics.utils.conversion as uc
import perceptionmetrics.utils.detection_metrics as um


class DetectionDataset(PerceptionDataset):
    """Abstract perception detection dataset class."""

    @abstractmethod
    def read_annotation(self, fname: str):
        """Read detection annotation from a file.

        :param fname: Annotation file name
        """
        raise NotImplementedError

    def get_label_count(self, splits: Optional[List[str]] = None):
        """Count detection labels per class for given splits.

        :param splits: List of splits to consider
        :return: Numpy array of label counts per class
        :raises ValueError: If any requested split is not present in the dataset
        """
        if splits is None:
            splits = ["train", "val"]

        self._validate_splits(splits)
        df = self.dataset[self.dataset["split"].isin(splits)]
        n_classes = max(c["idx"] for c in self.ontology.values()) + 1
        label_count = np.zeros(n_classes, dtype=np.uint64)

        for annotation_file in tqdm(df["annotation"], desc="Counting labels"):
            annots = self.read_annotation(annotation_file)
            for annot in annots:
                class_idx = annot[
                    "category_id"
                ]  # Should override the key category_id if needed in specific dataset class
                label_count[class_idx] += 1

        return label_count


class ImageDetectionDataset(DetectionDataset):
    """Image detection dataset class."""

    def make_fname_global(self):
        """Convert relative filenames in 'image' and 'annotation' columns to global paths."""
        if self.dataset_dir is not None:
            self.dataset["image"] = self.dataset["image"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset["annotation"] = self.dataset["annotation"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset_dir = None

    def read_annotation(self, fname: str):
        """Read detection annotation from a file.

        Override this based on annotation format (e.g., COCO JSON, XML, TXT).

        :param fname: Annotation filename
        :return: Parsed annotations (e.g., list of dicts)
        """
        # TODO implement COCO or VOC parsing in their classes separately.
        raise NotImplementedError("Implement annotation reading logic")

    @staticmethod
    def _pred_label_to_idx(label: Union[int, str], pred_ontology: dict) -> int:
        """Convert a prediction label to its class index.

        Predictions saved by the model evaluation store the class name, while
        other prediction files may store the class index directly.

        :param label: Class name or class index read from the prediction file
        :type label: Union[int, str]
        :param pred_ontology: Ontology used by the predictions
        :type pred_ontology: dict
        :return: Class index
        :rtype: int
        :raises ValueError: If the class name is not in the prediction ontology
        """
        if isinstance(label, str):
            if label in pred_ontology:
                return pred_ontology[label]["idx"]
            # A numeric string such as "1" is a class index, as before
            if label.isdigit():
                return int(label)
            raise ValueError(
                f"Unknown class name '{label}' in predictions. Expected one of "
                f"the prediction ontology classes: {list(pred_ontology)}"
            )
        return int(label)

    def eval_preds(
        self,
        predictions_dir: str,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[dict] = None,
        translation_direction: str = "dataset_to_model",
        pred_ontology: Optional[dict] = None,
        ignored_classes: Optional[List[str]] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Evaluate pre-computed predictions stored on disk against GT annotations.

        :param predictions_dir: Root directory containing prediction annotation files.
        :type predictions_dir: str
        :param split: Split or splits to evaluate, defaults to "test"
        :type split: Union[str, List[str]], optional
        :param ontology_translation: Translation dictionary between GT and prediction ontologies. Only required when the two ontologies differ.
        :type ontology_translation: Optional[dict], optional
        :param translation_direction: Direction of the ontology translation. ``"dataset_to_model"`` maps GT labels to the prediction ontology. ``"model_to_dataset"`` maps predictions to the GT ontology. Defaults to ``"dataset_to_model"``.
        :type translation_direction: str, optional
        :param pred_ontology: Ontology used by the predictions. If ``None``, it is assumed to match the GT ontology.
        :type pred_ontology: Optional[dict], optional
        :param ignored_classes: List of class names to exclude from evaluation. These class names must exist in the GT ontology.
        :type ignored_classes: Optional[List[str]], optional
        :param results_per_sample: If ``True``, per-sample results are saved next to each prediction file inside predictions_dir.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        splits = [split] if isinstance(split, str) else split
        self._validate_splits(splits)

        df = self.dataset[self.dataset["split"].isin(splits)]

        # Determine the evaluation ontology and build a LUT if needed
        eval_ontology = self.ontology
        lut_ontology = None

        if pred_ontology is None:
            pred_ontology = self.ontology

        if pred_ontology != self.ontology:
            if ontology_translation is None:
                raise ValueError(
                    "'ontology_translation' must be provided when GT and prediction "
                    "ontologies differ."
                )
            if translation_direction == "dataset_to_model":
                eval_ontology = pred_ontology
                lut_ontology = uc.get_ontology_conversion_lut(
                    self.ontology, pred_ontology, ontology_translation
                )
            else:
                lut_ontology = uc.get_ontology_conversion_lut(
                    pred_ontology, self.ontology, ontology_translation
                )

        n_classes = len(eval_ontology)

        # Retrieve ignored label indices
        ignored_label_indices = []
        if ignored_classes:
            for cls_name in ignored_classes:
                ignored_label_indices.append(self.ontology[cls_name]["idx"])

        # Init metrics
        metrics_factory = um.DetectionMetricsFactory(num_classes=n_classes)

        pbar = tqdm(df.iterrows(), total=len(df), leave=True)
        for sample_name, row in pbar:
            pbar.set_description(f"Evaluating sample: {sample_name}")

            # Read GT annotation
            gt_ann_fname = row["annotation"]
            if self.dataset_dir is not None:
                gt_ann_fname = os.path.join(self.dataset_dir, gt_ann_fname)

            gt_boxes, gt_labels = self.read_annotation(gt_ann_fname)
            gt_boxes = (
                np.array(gt_boxes, dtype=np.float32).reshape(-1, 4)
                if len(gt_boxes) > 0
                else np.zeros((0, 4), dtype=np.float32)
            )
            gt_labels = np.array(gt_labels, dtype=np.int64)

            # Build valid mask from ignored classes
            if ignored_label_indices:
                valid_mask = np.ones(len(gt_labels), dtype=bool)
                for idx in ignored_label_indices:
                    valid_mask &= gt_labels != idx
                gt_boxes = gt_boxes[valid_mask]
                gt_labels = gt_labels[valid_mask]

            # Read predictions
            pred_file = os.path.join(predictions_dir, f"{sample_name}.json")
            if not os.path.isfile(pred_file):
                raise FileNotFoundError(f"Prediction file not found: {pred_file}")

            with open(pred_file, "r") as f:
                preds = json.load(f)

            pred_boxes = [p["bbox"] for p in preds]
            pred_labels = [
                self._pred_label_to_idx(p["label"], pred_ontology) for p in preds
            ]
            pred_scores = [p["score"] for p in preds]

            pred_boxes = (
                np.array(pred_boxes, dtype=np.float32).reshape(-1, 4)
                if pred_boxes
                else np.zeros((0, 4), dtype=np.float32)
            )
            pred_labels = np.array(pred_labels, dtype=np.int64)
            pred_scores = np.array(pred_scores, dtype=np.float32)

            # Apply ontology translation
            if lut_ontology is not None:
                if translation_direction == "dataset_to_model":
                    gt_labels = lut_ontology[gt_labels]
                else:
                    pred_labels = lut_ontology[pred_labels]

            metrics_factory.update(
                gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
            )

            # Per-sample results
            if results_per_sample:
                sample_mf = um.DetectionMetricsFactory(num_classes=n_classes)
                sample_mf.update(
                    gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
                )
                sample_df = sample_mf.get_metrics_dataframe(eval_ontology)
                sample_csv = os.path.join(predictions_dir, f"{sample_name}_metrics.csv")
                sample_df.to_csv(sample_csv)

        return metrics_factory.get_metrics_dataframe(eval_ontology)


class LiDARDetectionDataset(DetectionDataset):
    """LiDAR detection dataset class."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        dataset_dir: str,
        ontology: dict,
        is_kitti_format: bool = True,
    ):
        super().__init__(dataset, dataset_dir, ontology)
        self.is_kitti_format = is_kitti_format

    def make_fname_global(self):
        if self.dataset_dir is not None:
            self.dataset["points"] = self.dataset["points"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset["annotation"] = self.dataset["annotation"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset_dir = None

    def read_annotation(self, fname: str):
        """Read LiDAR detection annotation.

        For example, read KITTI format label files or custom format.

        :param fname: Annotation file path
        :return: Parsed annotations (e.g., list of dicts)
        """
        # TODO Implement format specific parsing
        raise NotImplementedError("Implement LiDAR detection annotation reading")

    def eval_preds(
        self,
        predictions_dir: str,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[dict] = None,
        translation_direction: str = "dataset_to_model",
        pred_ontology: Optional[dict] = None,
        ignored_classes: Optional[List[str]] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Evaluate pre-computed predictions stored on disk against GT annotations.

        :param predictions_dir: Root directory containing prediction annotation files.
        :type predictions_dir: str
        :param split: Split or splits to evaluate, defaults to "test"
        :type split: Union[str, List[str]], optional
        :param ontology_translation: Translation dictionary between GT and prediction ontologies. Only required when the two ontologies differ.
        :type ontology_translation: Optional[dict], optional
        :param translation_direction: Direction of the ontology translation. ``"dataset_to_model"`` maps GT labels to the prediction ontology. ``"model_to_dataset"`` maps predictions to the GT ontology. Defaults to ``"dataset_to_model"``.
        :type translation_direction: str, optional
        :param pred_ontology: Ontology used by the predictions. If ``None``, it is assumed to match the GT ontology.
        :type pred_ontology: Optional[dict], optional
        :param ignored_classes: List of class names to exclude from evaluation. These class names must exist in the GT ontology.
        :type ignored_classes: Optional[List[str]], optional
        :param results_per_sample: If ``True``, per-sample results are saved next to each prediction file inside predictions_dir.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError(
            "eval_preds is not yet implemented for LiDARDetectionDataset"
        )
