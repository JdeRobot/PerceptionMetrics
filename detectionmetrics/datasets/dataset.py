from abc import ABC, abstractmethod
import os
import shutil
from typing import Tuple
from typing_extensions import Self

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import detectionmetrics.utils.io as uio


class SegmentationDataset(ABC):
    """Abstract segmentation dataset class

    :param dataset: Segmentation dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    """

    def __init__(self, dataset: pd.DataFrame, dataset_dir: str, ontology: dict):
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.ontology = ontology

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def make_fname_global(self):
        """Get all relative filenames in dataset and make global"""
        raise NotImplementedError

    def append(self, new_dataset: Self):
        """Append another dataset with common ontology

        :param new_dataset: Dataset to be appended
        :type new_dataset: Self
        """
        assert self.ontology == new_dataset.ontology, "Ontologies don't match"

        # Global filenames to avoid dealing with each dataset relative location
        self.make_fname_global()
        new_dataset.make_fname_global()

        # Simply concatenate pandas dataframes
        self.dataset = pd.concat(
            [self.dataset, new_dataset.dataset], verify_integrity=True
        )


class ImageSegmentationDataset(SegmentationDataset):
    """Parent image segmentation dataset class

    :param dataset: Image segmentation dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    """

    def __init__(self, dataset: pd.DataFrame, dataset_dir: str, ontology: dict):
        super().__init__(dataset, dataset_dir, ontology)

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

    def export(self, outdir: str):
        """Export dataset dataframe and image files

        :param outdir: Directory where Parquet and image files will be stored
        :type outdir: str
        """
        os.makedirs(outdir, exist_ok=True)

        pbar = tqdm(self.dataset.iterrows())

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
            if uio.get_image_mode(image_fname) != "RGB":
                image = cv2.imread(image_fname, 1)  # convert to RGB
                cv2.imwrite(os.path.join(outdir, rel_image_fname), image)
            # if image mode is appropriate simply copy image to new location
            else:
                shutil.copy2(image_fname, os.path.join(outdir, rel_image_fname))
            self.dataset.at[sample_name, "image"] = rel_image_fname

            # Same for labels
            if label_fname:
                if uio.get_image_mode(label_fname) != "L":
                    label = cv2.imread(label_fname, 0)  # convert to L
                    cv2.imwrite(os.path.join(outdir, rel_label_fname), label)
                else:
                    shutil.copy2(label_fname, os.path.join(outdir, rel_label_fname))
                self.dataset.at[sample_name, "label"] = rel_label_fname

        self.dataset_dir = outdir

        # Write ontology and store relative path in dataset attributes
        ontology_fname = "ontology.json"
        self.dataset.attrs = {"ontology_fname": ontology_fname}
        uio.write_json(os.path.join(outdir, ontology_fname), self.ontology)

        # Store dataset as Parquet file containing relative filenames
        self.dataset.to_parquet(os.path.join(outdir, "dataset.parquet"))


class LiDARSegmentationDataset(SegmentationDataset):
    """Parent lidar segmentation dataset class

    :param dataset: LiDAR segmentation dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    :param is_kitti_format: Whether the linked files in the dataset are stored in
    SemanticKITTI format or not, defaults to True
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

    def export(self, outdir: str):
        """Export dataset dataframe and LiDAR files in SemanticKITTI format

        :param outdir: Directory where Parquet and LiDAR files will be stored
        :type outdir: str
        """
        os.makedirs(outdir, exist_ok=True)

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
            if not self.is_kitti_format:
                points = self.read_points(points_fname)
                label, _ = self.read_label(label_fname)
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
