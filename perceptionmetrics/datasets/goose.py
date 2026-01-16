from collections import OrderedDict
from glob import glob
import os
from typing import Optional, Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.conversion as uc


def build_dataset(
    data_type: str,
    data_suffix: str,
    label_suffix: str,
    train_dataset_dir: Optional[str] = None,
    val_dataset_dir: Optional[str] = None,
    test_dataset_dir: Optional[str] = None,
    is_goose_ex: bool = False,
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries from GOOSE dataset structure

    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param data_type: Data to be read (e.g. images or lidar)
    :type data_type: str
    :param data_suffix: File suffix to be used to filter data (e.g., windshield_vis.png or vls128.bin)
    :type data_suffix: str
    :param label_suffix: File suffix to be used to filter labels (e.g., vis_labelids.png or goose.label)
    :type label_suffix: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    :param is_goose_ex: Whether the dataset is GOOSE Ex or GOOSE, defaults to False
    :type is_goose_ex: bool, optional
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

    # Get ontology filename
    ontology_fname = os.path.join(
        next(iter(dataset_dirs.values())), "goose_label_mapping.csv"
    )
    if ontology_fname is None:
        raise ValueError("No valid dataset directory found")

    # Load and adapt ontology
    assert os.path.isfile(ontology_fname), "Ontology file not found"

    ontology, ontology_dataframe = ({}, pd.read_csv(ontology_fname))
    for idx, (name, _, _, color) in ontology_dataframe.iterrows():
        ontology[name] = {"idx": idx, "rgb": uc.hex_to_rgb(color)}

    # Build dataset as ordered python dictionary
    dataset = OrderedDict()
    for split, dataset_dir in dataset_dirs.items():
        train_data = os.path.join(dataset_dir, f"{data_type}/{split}/*/*_{data_suffix}")
        for data_fname in glob(train_data):
            sample_dir, sample_base_name = os.path.split(data_fname)

            # GOOSE Ex uses a different label file naming convention
            if is_goose_ex:
                sample_base_name = "sequence" + sample_base_name.split("_sequence")[-1]
            else:
                sample_base_name = sample_base_name.split("__")[-1]

            sample_base_name = sample_base_name.split("_" + data_suffix)[0]

            scene = os.path.split(sample_dir)[-1]
            sample_name = f"{scene}-{sample_base_name}"

            if is_goose_ex:
                label_base_name = f"{scene}_{sample_base_name}_{label_suffix}"
            else:
                label_base_name = f"{scene}__{sample_base_name}_{label_suffix}"

            label_fname = os.path.join(
                dataset_dir, "labels", split, scene, label_base_name
            )
            label_fname = None if not os.path.isfile(label_fname) else label_fname

            data_fname = os.path.join(dataset_dir, data_fname)
            dataset[sample_name] = (data_fname, label_fname, scene, split)

    return dataset, ontology


class GOOSEImageSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Specific class for GOOSE-styled image segmentation datasets. All data can be
    downloaded from the official webpage (https://goose-dataset.de):
    train -> https://goose-dataset.de/storage/goose_2d_train.zip
    val   -> https://goose-dataset.de/storage/goose_2d_val.zip
    test  -> https://goose-dataset.de/storage/goose_2d_test.zip

    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    """

    def __init__(
        self,
        train_dataset_dir: Optional[str] = None,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
    ):
        dataset, ontology = build_dataset(
            "images",
            "windshield_vis.png",
            "labelids.png",
            train_dataset_dir,
            val_dataset_dir,
            test_dataset_dir,
        )

        # Convert to Pandas
        cols = ["image", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        # Select dataset directory
        all_dataset_dirs = [train_dataset_dir, val_dataset_dir, test_dataset_dir]
        dataset_dir = [d for d in all_dataset_dirs if d is not None][0]

        super().__init__(dataset, dataset_dir, ontology)


class GOOSELiDARSegmentationDataset(segmentation_dataset.LiDARSegmentationDataset):
    """Specific class for GOOSE-styled LiDAR segmentation datasets. All data can be
    downloaded from the official webpage (https://goose-dataset.de):
    train -> https://goose-dataset.de/storage/gooseEx_3d_train.zip
    val   -> https://goose-dataset.de/storage/gooseEx_3d_val.zip
    test  -> https://goose-dataset.de/storage/gooseEx_3d_test.zip

    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    :param is_goose_ex: Whether the dataset is GOOSE Ex or GOOSE, defaults to False
    :type is_goose_ex: bool, optional
    """

    def __init__(
        self,
        train_dataset_dir: Optional[str] = None,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
        is_goose_ex: bool = False,
    ):
        dataset, ontology = build_dataset(
            "lidar",
            "pcl.bin" if is_goose_ex else "vls128.bin",
            "goose.label",
            train_dataset_dir,
            val_dataset_dir,
            test_dataset_dir,
            is_goose_ex=is_goose_ex,
        )

        # Convert to Pandas
        cols = ["points", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        # Select dataset directory
        all_dataset_dirs = [train_dataset_dir, val_dataset_dir, test_dataset_dir]
        dataset_dir = [d for d in all_dataset_dirs if d is not None][0]

        super().__init__(dataset, dataset_dir, ontology)
