from collections import OrderedDict
from glob import glob
import os
from typing import Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.io as uio

# Default split presented in the paper
DEFAULT_SPLIT = {
    "creek": "test",
    "park-1": "test",
    "park-2": "train",
    "park-8": "val",
    "trail": "train",
    "trail-3": "train",
    "trail-4": "train",
    "trail-5": "val",
    "trail-6": "train",
    "trail-7": "test",
    "trail-9": "train",
    "trail-10": "train",
    "trail-11": "train",
    "trail-12": "train",
    "trail-13": "test",
    "trail-14": "train",
    "trail-15": "train",
    "village": "train",
}


def build_dataset(
    data_dir: str,
    labels_dir: str,
    ontology_fname: str,
    split_sequences: dict,
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries

    :param data_dir: Directory containing data
    :type data_dir: str
    :param labels_dir: Directory containing labels (in RGB format)
    :type labels_dir: str
    :param ontology_fname: text file containing the dataset ontology (RUGD_annotation-colormap.txt)
    :type ontology_fname: str
    :param split_sequences: Dictionary containing the split sequences for train, val, and test
    :type split_sequences: dict
    :return: Dataset and onotology
    :rtype: Tuple[dict, dict]
    """
    # Check that provided paths exist and ensure they are absolute
    data_dir = os.path.abspath(data_dir)
    labels_dir = os.path.abspath(labels_dir)
    assert os.path.isdir(data_dir), "Images directory not found"
    assert os.path.isdir(labels_dir), "Labels directory not found"

    # Load and adapt ontology
    assert os.path.isfile(ontology_fname), "Ontology file not found"
    original_ontology = uio.read_txt(ontology_fname)

    try:
        ontology = {}
        for class_data in original_ontology:
            class_idx, class_name, r, g, b = class_data.split(" ")
            ontology[class_name] = {
                "idx": int(class_idx),
                "rgb": (int(r), int(g), int(b)),
            }
    except Exception as e:
        print(f"Error loading ontology: {e}")
        raise

    # Build dataset as ordered python dictionary
    dataset = OrderedDict()

    for data_fname in glob(os.path.join(data_dir, "*/*.png")):
        label_fname = data_fname.replace(data_dir, labels_dir)
        assert os.path.isfile(label_fname), f"Label file not found: {label_fname}"

        sample_name, _ = os.path.splitext(os.path.basename(data_fname))
        scene_name = sample_name.split("_")[0]
        split = split_sequences[scene_name]

        dataset[sample_name] = (data_fname, label_fname, split)

    return dataset, ontology


class RUGDImageSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Specific class for RUGD-styled image segmentation dataset.

    :param images_dir: Directory containing images
    :type images_dir: str
    :param labels_dir: Directory containing labels (in RGB format)
    :type labels_dir: str
    :param ontology_fname: text file containing the dataset ontology (RUGD_annotation-colormap.txt)
    :type ontology_fname: str
    :param split_sequences: Dictionary containing the split sequences for train, val, and test, defaults to DEFAULT_SPLIT
    :type split_sequences: dict, optional
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        ontology_fname: str,
        split_sequences: dict = DEFAULT_SPLIT,
    ):
        dataset, ontology = build_dataset(
            images_dir,
            labels_dir,
            ontology_fname,
            split_sequences,
        )

        # Convert to Pandas
        cols = ["image", "label", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        # Initialize parent class
        super().__init__(dataset, images_dir, ontology, is_label_rgb=True)
