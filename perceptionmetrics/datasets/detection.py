from abc import abstractmethod
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from perceptionmetrics.datasets.perception import PerceptionDataset


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
            "eval_preds is not yet implemented for ImageDetectionDataset"
        )


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
