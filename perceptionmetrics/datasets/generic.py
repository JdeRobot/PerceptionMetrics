from collections import OrderedDict
from itertools import zip_longest
import os
import random
from typing import Optional, Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.io as uio


def get_random_rgb(class_index: int) -> Tuple[int, int, int]:
    """Uses the given class index as seed to generate a random RGB color

    :param class_index: Class index to be used as seed
    :type class_index: int
    :return: random RGB color
    :rtype: Tuple[int, int, int]
    """
    random.seed(class_index)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def build_dataset(
    data_suffix: str,
    label_suffix: str,
    ontology_fname: str,
    train_dataset_dir: Optional[str] = None,
    val_dataset_dir: Optional[str] = None,
    test_dataset_dir: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries

    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param data_suffix: File suffix to be used to filter data
    :type data_suffix: str
    :param label_suffix: File suffix to be used to filter labels
    :type label_suffix: str
    :param ontology_fname: JSON file containing either a list of classes or a dictionary with class names as keys and class indexes + rgb as values
    :type ontology_fname: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    :return: Dataset and onotology
    :rtype: Tuple[dict, dict]
    """
    # Define dataset directories and ensure they are absolute paths
    dataset_dirs = {
        "train": train_dataset_dir,
        "val": val_dataset_dir,
        "test": test_dataset_dir,
    }
    dataset_dirs = {
        split: os.path.abspath(d) for split, d in dataset_dirs.items() if d is not None
    }
    if not dataset_dirs:
        raise ValueError("At least one dataset directory must be provided")

    # Load and adapt ontology
    assert os.path.isfile(ontology_fname), "Ontology file not found"
    original_ontology = uio.read_json(ontology_fname)

    try:
        ontology = {}
        if isinstance(original_ontology, list):
            for idx, name in enumerate(original_ontology):
                ontology[name] = {"idx": idx, "rgb": get_random_rgb(idx)}
        elif isinstance(original_ontology, dict):
            for idx, name in enumerate(original_ontology.keys()):
                ontology[name] = {
                    "idx": original_ontology[name].get("idx", idx),
                    "rgb": original_ontology[name].get("rgb", get_random_rgb(idx)),
                }
        else:
            raise ValueError("Ontology format not supported")
    except Exception as e:
        print(f"Error loading ontology: {e}")
        raise

    # Build dataset as ordered python dictionary
    dataset = OrderedDict()

    do_wildcards_match = len(data_suffix.split("*")) == len(label_suffix.split("*"))
    assert do_wildcards_match, "Suffixes must have the same number of wildcards"

    for split, dataset_dir in dataset_dirs.items():
        data_pattern = os.path.join(dataset_dir, data_suffix)
        label_pattern = os.path.join(dataset_dir, label_suffix)

        data_pattern_splitted = data_pattern.split("*")
        label_pattern_splitted = label_pattern.split("*")

        all_wildcard_matches = uio.extract_wildcard_matches(data_pattern)
        for sample_matches in all_wildcard_matches:
            data_fname = zip_longest(data_pattern_splitted, sample_matches)
            label_fname = zip_longest(label_pattern_splitted, sample_matches)

            data_fname = "".join(a + (b or "") for a, b in data_fname)
            label_fname = "".join(a + (b or "") for a, b in label_fname)

            assert os.path.isfile(data_fname), f"Data file not found: {data_fname}"
            assert os.path.isfile(label_fname), f"Label file not found: {data_fname}"

            sample_name = "_".join(sample_matches)
            dataset[sample_name] = (data_fname, label_fname, split)

    return dataset, ontology


class GenericImageSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Generic class for image segmentation datasets.

    :param data_suffix: File suffix to be used to filter data
    :type data_suffix: str
    :param label_suffix: File suffix to be used to filter labels
    :type label_suffix: str
    :param ontology_fname: JSON file containing either a list of classes or a dictionary with class names as keys and class indexes + rgb as values
    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    """

    def __init__(
        self,
        data_suffix: str,
        label_suffix: str,
        ontology_fname: str,
        train_dataset_dir: Optional[str] = None,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
    ):
        dataset, ontology = build_dataset(
            data_suffix,
            label_suffix,
            ontology_fname,
            train_dataset_dir,
            val_dataset_dir,
            test_dataset_dir,
        )

        # Convert to Pandas
        cols = ["image", "label", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        # Select dataset directory
        all_dataset_dirs = [train_dataset_dir, val_dataset_dir, test_dataset_dir]
        dataset_dir = [d for d in all_dataset_dirs if d is not None][0]

        super().__init__(dataset, dataset_dir, ontology)


class GenericLiDARSegmentationDataset(segmentation_dataset.LiDARSegmentationDataset):
    """Generic class for LiDAR segmentation datasets.

    :param data_suffix: File suffix to be used to filter data
    :type data_suffix: str
    :param label_suffix: File suffix to be used to filter labels
    :type label_suffix: str
    :param ontology_fname: JSON file containing either a list of classes or a dictionary with class names as keys and class indexes + rgb as values
    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    """

    def __init__(
        self,
        data_suffix: str,
        label_suffix: str,
        ontology_fname: str,
        train_dataset_dir: Optional[str] = None,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
    ):
        dataset, ontology = build_dataset(
            data_suffix,
            label_suffix,
            ontology_fname,
            train_dataset_dir,
            val_dataset_dir,
            test_dataset_dir,
        )

        # Convert to Pandas
        cols = ["points", "label", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        # Select dataset directory
        all_dataset_dirs = [train_dataset_dir, val_dataset_dir, test_dataset_dir]
        dataset_dir = [d for d in all_dataset_dirs if d is not None][0]

        super().__init__(dataset, dataset_dir, ontology)
