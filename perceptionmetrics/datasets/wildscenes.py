from collections import OrderedDict
import logging
import os
from typing import Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset


def build_dataset(
    dataset_dir: str, split_fnames: dict, ontology: dict
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries from Wildscenes dataset structure

    :param dataset_dir: Directory where both RGB images and annotations have been extracted to
    :type dataset_dir: str
    :param split_fnames: Dictionary that contains the paths where train, val, and test split files (.csv) have been extracted to
    :type split_dir: str
    :param ontology: Ontology definition as found in the official repo
    :type ontology: dict
    :return: Dataset and onotology
    :rtype: Tuple[dict, dict]
    """
    # Check that provided paths exist and ensure they are absolute
    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(dataset_dir), "Dataset directory not found"
    dataset_dir, _ = os.path.split(dataset_dir)

    for split_fname in split_fnames.values():
        assert os.path.isfile(split_fname), f"{split_fname} split file not found"

    # Load and adapt ontology
    parsed_ontology = {}
    ontology_iter = zip(ontology["classes"], ontology["palette"], ontology["cidx"])
    for name, color, idx in ontology_iter:
        parsed_ontology[name] = {"idx": idx, "rgb": color}

    # Get samples filenames
    train_split = pd.read_csv(split_fnames["train"])
    train_split["split"] = "train"

    val_split = pd.read_csv(split_fnames["val"])
    val_split["split"] = "val"

    test_split = pd.read_csv(split_fnames["test"])
    test_split["split"] = "test"

    samples_data = pd.concat([train_split, val_split, test_split])

    if "hist_path" in samples_data.columns:
        samples_data = samples_data.drop(columns=["hist_path"])

    # Build dataset as ordered python dictionary
    dataset = OrderedDict()
    skipped_samples = []
    for _, sample_data in samples_data.iterrows():
        sample_name, data_fname, label_fname, split = sample_data

        sample_dir, sample_base_name = os.path.split(data_fname)
        sample_base_name, _ = os.path.splitext(sample_base_name)
        scene = os.path.split(os.path.split(sample_dir)[0])[-1]

        data_fname = os.path.join(dataset_dir, data_fname)
        label_fname = os.path.join(dataset_dir, label_fname)

        if not os.path.isfile(data_fname) or not os.path.isfile(label_fname):
            missing_file = "data" if not os.path.isfile(label_fname) else "label"
            logging.warning(f"Missing {missing_file} for {sample_name}. Skipped!")
            skipped_samples.append(sample_name)
            continue

        dataset[sample_name] = (data_fname, label_fname, scene, split)

    # Report results
    print(f"Samples retrieved: {len(dataset)} / {len(samples_data)}")
    if skipped_samples:
        print("Skipped samples:")
        for sample_name in skipped_samples:
            print(f"\n\t{sample_name}")

    return dataset, parsed_ontology


class WildscenesImageSegmentationDataset(
    segmentation_dataset.ImageSegmentationDataset
):
    """Specific class for Wildscenes-styled image segmentation datasets. All data can
    be downloaded from the official repo:
    dataset  -> https://data.csiro.au/collection/csiro:61541
    split    -> https://github.com/csiro-robotics/WildScenes/tree/main/data/splits/opt2d

    :param dataset_dir: Directory where dataset images and labels are stored (Wildscenes2D)
    :type dataset_dir: str
    :param split_dir: Directory where train, val, and test files (.csv) have been extracted to
    :type split_dir: str
    """

    def __init__(self, dataset_dir: str, split_dir: str):
        split_fnames = {
            "train": os.path.join(split_dir, "train.csv"),
            "val": os.path.join(split_dir, "val.csv"),
            "test": os.path.join(split_dir, "test.csv"),
        }

        # Ontology definition as found in the official repo (https://github.com/csiro-robotics/WildScenes/blob/main/wildscenes/tools/utils2d.py)
        METAINFO = {
            "classes": (
                "unlabelled",
                "asphalt",
                "dirt",
                "mud",
                "water",
                "gravel",
                "other-terrain",
                "tree-trunk",
                "tree-foliage",
                "bush",
                "fence",
                "structure",
                "pole",
                "vehicle",
                "rock",
                "log",
                "other-object",
                "sky",
                "grass",
            ),
            "palette": [
                (0, 0, 0),
                (255, 165, 0),
                (60, 180, 75),
                (255, 225, 25),
                (0, 130, 200),
                (145, 30, 180),
                (70, 240, 240),
                (240, 50, 230),
                (210, 245, 60),
                (230, 25, 75),
                (0, 128, 128),
                (170, 110, 40),
                (255, 250, 200),
                (128, 0, 0),
                (170, 255, 195),
                (128, 128, 0),
                (250, 190, 190),
                (0, 0, 128),
                (128, 128, 128),
            ],
            "cidx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        }
        dataset, ontology = build_dataset(dataset_dir, split_fnames, METAINFO)

        # Convert to Pandas
        cols = ["image", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        dataset.attrs = {"ontology": ontology}

        super().__init__(dataset, dataset_dir, ontology)


class WildscenesLiDARSegmentationDataset(
    segmentation_dataset.LiDARSegmentationDataset
):
    """Specific class for Wildscenes-styled LiDAR segmentation datasets. All data can
    be downloaded from the official repo:
    dataset  -> https://data.csiro.au/collection/csiro:61541
    split    -> https://github.com/csiro-robotics/WildScenes/tree/main/data/splits/opt3d

    :param dataset_dir: Directory where dataset images and labels are stored (Wildscenes3D)
    :type dataset_dir: str
    :param split_dir: Directory where train, val, and test files (.csv) have been extracted to
    :type split_dir: str
    """

    def __init__(self, dataset_dir: str, split_dir: str):
        split_fnames = {
            "train": os.path.join(split_dir, "train.csv"),
            "val": os.path.join(split_dir, "val.csv"),
            "test": os.path.join(split_dir, "test.csv"),
        }

        # Ontology definition as found in the official repo (https://github.com/csiro-robotics/WildScenes/blob/main/wildscenes/tools/utils3d.py)
        METAINFO = {
            "classes": (
                "unlabelled",
                "bush",
                "dirt",
                "fence",
                "grass",
                "gravel",
                "log",
                "mud",
                "other-object",
                "other-terrain",
                "rock",
                "sky",
                "structure",
                "tree-foliage",
                "tree-trunk",
                "water",
            ),
            "palette": [
                (0, 0, 0),
                (230, 25, 75),
                (60, 180, 75),
                (0, 128, 128),
                (128, 128, 128),
                (145, 30, 180),
                (128, 128, 0),
                (255, 225, 25),
                (250, 190, 190),
                (70, 240, 240),
                (170, 255, 195),
                (0, 0, 128),
                (170, 110, 40),
                (210, 245, 60),
                (240, 50, 230),
                (0, 130, 200),
            ],
            "cidx": [255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        }
        dataset, ontology = build_dataset(dataset_dir, split_fnames, METAINFO)

        # Convert to Pandas
        cols = ["points", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        dataset.attrs = {"ontology": ontology}

        super().__init__(dataset, dataset_dir, ontology, has_intensity=False)
