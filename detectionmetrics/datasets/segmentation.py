from abc import ABC, abstractmethod
import os
import shutil
from typing import List, Optional, Tuple
from typing_extensions import Self

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from detectionmetrics.datasets.perecption import PerceptionDataset
import detectionmetrics.utils.io as uio
import detectionmetrics.utils.conversion as uc

class SegmentationDataset(PerceptionDataset):
    """Abstract perception dataset class."""

    @abstractmethod
    def read_label(self, fname: str) -> np.ndarray:
        """Read label from an image file

        :param fname: Image file containing labels
        :type fname: str
        :return: Numpy array containing labels
        :rtype: np.ndarray
        """
        raise NotImplementedError


class ImageSegmentationDataset(SegmentationDataset):
    """Parent image segmentation dataset class

    :param dataset: Image segmentation dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    :param is_label_rgb: Whether the labels are in RGB format or not, defaults to False
    :type is_label_rgb: bool, optional
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        dataset_dir: str,
        ontology: dict,
        is_label_rgb: bool = False,
    ):
        super().__init__(dataset, dataset_dir, ontology)
        self.is_label_rgb = is_label_rgb

    def make_fname_global(self):
        """Get all relative filenames in dataset and make global"""
        if self.dataset_dir is not None:
            self.dataset["image"] = self.dataset["image"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset["label"] = self.dataset["label"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset_dir = None  # dataset_dir=None -> filenames must be relative

    def export(
        self,
        outdir: str,
        new_ontology: Optional[dict] = None,
        ontology_translation: Optional[dict] = None,
        ignored_classes: Optional[List[str]] = None,
        resize: Optional[Tuple[int, int]] = None,
        include_label_count: bool = True,
    ):
        """Export dataset dataframe and image files in SemanticKITTI format. Optionally, modify ontology before exporting.

        :param outdir: Directory where Parquet and images files will be stored
        :type outdir: str
        :param new_ontology: Target ontology definition
        :type new_ontology: dict
        :param ontology_translation: Ontology translation dictionary, defaults to None
        :type ontology_translation: Optional[dict], optional
        :param ignored_classes: Classes to ignore from the old ontology, defaults to []
        :type ignored_classes: Optional[List[str]], optional
        :param resize: Resize images and labels to the given dimensions, defaults to None
        :type resize: Optional[Tuple[int, int]], optional
        :param include_label_count: Whether to include class weights in the dataset, defaults to True
        :type include_label_count: bool, optional
        """
        os.makedirs(outdir, exist_ok=True)

        pbar = tqdm(self.dataset.iterrows())

        # Check if ontology conversion is needed and possible
        if new_ontology is not None and ontology_translation is None:
            raise ValueError("Ontology translation must be provided")
        if ontology_translation is not None and new_ontology is None:
            raise ValueError("New ontology must be provided")

        # Create ontology conversion lookup table if needed and get number of classes
        ontology_conversion_lut = None
        if new_ontology is not None:
            ontology_conversion_lut = uc.get_ontology_conversion_lut(
                old_ontology=self.ontology,
                new_ontology=new_ontology,
                ontology_translation=ontology_translation,
                ignored_classes=ignored_classes,
            )
            n_classes = max(c["idx"] for c in new_ontology.values()) + 1
        else:
            n_classes = max(c["idx"] for c in self.ontology.values()) + 1

        # Check if label count is missing and create empty array if needed
        label_count_missing = include_label_count and (
            not self.has_label_count or new_ontology is not None
        )
        if label_count_missing:
            label_count = np.zeros(n_classes, dtype=np.uint64)

        # Export each sample
        for sample_name, row in pbar:
            pbar.set_description(f"Exporting sample: {sample_name}")

            # Create each split directory
            split = row["split"]
            split_dir = os.path.join(outdir, split)
            if not os.path.isdir(split_dir):
                os.makedirs(split_dir, exist_ok=True)

            # Init target filenames for both images and labels
            rel_image_fname = os.path.join(split, f"image-{sample_name}.png")
            rel_label_fname = os.path.join(split, f"label-{sample_name}.png")

            image_fname = row["image"]
            label_fname = row["label"]
            if self.dataset_dir is not None:
                image_fname = os.path.join(self.dataset_dir, image_fname)
                if label_fname:
                    label_fname = os.path.join(self.dataset_dir, label_fname)

            # If image mode is not appropriate: read, convert, and rewrite image
            if uio.get_image_mode(image_fname) != "RGB" or resize is not None:
                image = cv2.imread(image_fname, 1)  # convert to RGB

                # Resize image if needed
                if resize is not None:
                    image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(outdir, rel_image_fname), image)

            # If image mode is appropriate simply copy image to new location
            else:
                shutil.copy2(image_fname, os.path.join(outdir, rel_image_fname))
            self.dataset.at[sample_name, "image"] = rel_image_fname

            # Same for labels (plus ontology conversion and label count if needed)
            if label_fname:
                image_mode = uio.get_image_mode(label_fname)
                if (
                    image_mode != "L"
                    or ontology_conversion_lut is not None
                    or resize is not None
                    or label_count_missing
                ):
                    # Read and convert label from RGB to L
                    label = self.read_label(label_fname)

                    # Convert label to new ontology if needed
                    if ontology_conversion_lut is not None:
                        label = ontology_conversion_lut[label]

                    # Resize label if needed
                    if resize is not None:
                        label = cv2.resize(
                            label, resize, interpolation=cv2.INTER_NEAREST
                        )

                    # Update label count if needed
                    if label_count_missing:
                        indices, counts = np.unique(label, return_counts=True)
                        label_count[indices] += counts.astype(np.uint64)

                    cv2.imwrite(os.path.join(outdir, rel_label_fname), label)
                else:
                    shutil.copy2(label_fname, os.path.join(outdir, rel_label_fname))

                self.dataset.at[sample_name, "label"] = rel_label_fname

        # Update dataset directory and ontology if needed
        self.dataset_dir = outdir
        self.ontology = new_ontology if new_ontology is not None else self.ontology

        # Write ontology and store relative path in dataset attributes
        if label_count_missing:
            for class_data in self.ontology.values():
                class_data["label_count"] = int(label_count[class_data["idx"]])

        ontology_fname = "ontology.json"
        self.dataset.attrs = {"ontology_fname": ontology_fname}
        uio.write_json(os.path.join(outdir, ontology_fname), self.ontology)

        # Store dataset as Parquet file containing relative filenames
        self.dataset.to_parquet(os.path.join(outdir, "dataset.parquet"))

    def read_label(self, fname: str) -> np.ndarray:
        """Read label from an image file

        :param fname: Image file containing labels
        :type fname: str
        :return: Numpy array containing labels
        :rtype: np.ndarray
        """
        if self.is_label_rgb:
            label_rgb = cv2.imread(fname)[:, :, ::-1]
            label = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
            for class_data in self.ontology.values():
                idx = class_data["idx"]
                rgb = list(class_data["rgb"])
                label[(label_rgb == rgb).all(axis=2)] = idx
        else:
            label = cv2.imread(fname, 0)  # convert to L
        return label

    def get_label_count(self, splits: Optional[List[str]] = None):
        """Get label count for each class in the dataset

        :param splits: Dataset splits to consider, defaults to ["train", "val"]
        :type splits: List[str], optional
        :return: Label count for the dataset
        :rtype: np.ndarray
        """
        if splits is None:
            splits = ["train", "val"]

        df = self.dataset[self.dataset["split"].isin(splits)]
        n_classes = max(c["idx"] for c in self.ontology.values()) + 1
        label_count = np.zeros(n_classes, dtype=np.uint64)
        pbar = tqdm(df["label"], total=len(df))
        for label_fname in pbar:
            label = self.read_label(label_fname)
            indices, counts = np.unique(label, return_counts=True)
            label_count[indices] += counts.astype(np.uint64)

        return label_count


class LiDARSegmentationDataset(SegmentationDataset):
    """Parent lidar segmentation dataset class

    :param dataset: LiDAR segmentation dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    :param is_kitti_format: Whether the linked files in the dataset are stored in SemanticKITTI format or not, defaults to True
    :type is_kitti_format: bool, optional
    """

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
        """Get all relative filenames in dataset and make global"""
        if self.dataset_dir is not None:
            self.dataset["points"] = self.dataset["points"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset["label"] = self.dataset["label"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset_dir = None  # dataset_dir=None -> filenames must be relative

    def export(
        self,
        outdir: str,
        new_ontology: Optional[dict] = None,
        ontology_translation: Optional[dict] = None,
        ignored_classes: Optional[List[str]] = [],
    ):
        """Export dataset dataframe and LiDAR files in SemanticKITTI format. Optionally, modify ontology before exporting.

        :param outdir: Directory where Parquet and LiDAR files will be stored
        :type outdir: str
        :param new_ontology: Target ontology definition
        :type new_ontology: dict
        :param ontology_translation: Ontology translation dictionary, defaults to None
        :type ontology_translation: Optional[dict], optional
        :param ignored_classes: Classes to ignore from the old ontology, defaults to []
        :type ignored_classes: Optional[List[str]], optional
        """
        os.makedirs(outdir, exist_ok=True)

        if new_ontology is not None and ontology_translation is None:
            raise ValueError("Ontology translation must be provided")
        if ontology_translation is not None and new_ontology is None:
            raise ValueError("New ontology must be provided")

        ontology_conversion_lut = None
        if new_ontology is not None:
            ontology_conversion_lut = uc.get_ontology_conversion_lut(
                old_ontology=self.ontology,
                new_ontology=new_ontology,
                ontology_translation=ontology_translation,
                ignored_classes=ignored_classes,
            )

        pbar = tqdm(self.dataset.iterrows())

        for sample_name, row in pbar:
            pbar.set_description(f"Exporting sample: {sample_name}")

            # Create each split directory
            split = row["split"]
            split_dir = os.path.join(outdir, split)
            if not os.path.isdir(split_dir):
                os.makedirs(split_dir, exist_ok=True)

            # Init target filenames for both points and labels
            rel_points_fname = os.path.join(split, f"points-{sample_name}.bin")
            rel_label_fname = os.path.join(split, f"label-{sample_name}.label")

            points_fname = row["points"]
            label_fname = row["label"]
            if self.dataset_dir is not None:
                points_fname = os.path.join(self.dataset_dir, points_fname)
                if label_fname:
                    label_fname = os.path.join(self.dataset_dir, label_fname)

            # If format is not appropriate: read, convert, and rewrite sample
            if not self.is_kitti_format or ontology_conversion_lut is not None:
                points = self.read_points(points_fname)
                label, _ = self.read_label(label_fname)
                if ontology_conversion_lut is not None:
                    label = ontology_conversion_lut[label]
                points.tofile(os.path.join(outdir, rel_points_fname))
                label.tofile(os.path.join(outdir, rel_label_fname))
            else:
                shutil.copy2(points_fname, os.path.join(outdir, rel_points_fname))
                shutil.copy2(label_fname, os.path.join(outdir, rel_label_fname))

            self.dataset.at[sample_name, "points"] = rel_points_fname
            self.dataset.at[sample_name, "label"] = rel_label_fname

        self.dataset_dir = outdir

        # Write ontology and store relative path in dataset attributes
        ontology_fname = "ontology.json"
        self.dataset.attrs = {"ontology_fname": ontology_fname}
        uio.write_json(os.path.join(outdir, ontology_fname), self.ontology)

        # Store dataset as Parquet file containing relative filenames
        self.dataset.to_parquet(os.path.join(outdir, "dataset.parquet"))

    @staticmethod
    def read_points(fname: str) -> np.ndarray:
        """Read points from a binary file in SemanticKITTI format

        :param fname: Binary file containing points
        :type fname: str
        :return: Numpy array containing points
        :rtype: np.ndarray
        """
        points = np.fromfile(fname, dtype=np.float32)
        return points.reshape((-1, 4))

    @staticmethod
    def read_label(fname: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read labels from a binary file in SemanticKITTI format

        :param fname: Binary file containing labels
        :type fname: str
        :return: Numpy arrays containing semantic and instance labels
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        label = np.fromfile(fname, dtype=np.uint32)
        label = label.reshape((-1))
        semantic_label = label & 0xFFFF
        instance_label = label >> 16
        return semantic_label.astype(np.int32), instance_label.astype(np.int32)

