from abc import abstractmethod
import os
from typing import List, Optional

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
        """
        if splits is None:
            splits = ["train", "val"]

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
