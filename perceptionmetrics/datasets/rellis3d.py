from collections import OrderedDict
import logging
import os
from typing import Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.io as uio


def build_dataset(
    dataset_dir: str, split_fnames: dict, ontology_fname: str
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries from Rellis3D dataset structure

    :param dataset_dir: Directory where both RGB images and annotations have been extracted to
    :type dataset_dir: str
    :param split_fnames: Dictionary that contains the paths where train, val, and test split files (.lst) have been extracted to
    :type split_dir: str
    :param ontology_fname: YAML file contained in the ontology compressed directory
    :type ontology_fname: str
    :return: Dataset and onotology
    :rtype: Tuple[dict, dict]
    """
    # Check that provided paths exist and ensure they are absolute
    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(dataset_dir), "Dataset directory not found"
    for split_fname in split_fnames.values():
        assert os.path.isfile(split_fname), f"{split_fname} split file not found"
    assert os.path.isfile(ontology_fname), "Ontology file not found"

    # Load and adapt ontology
    names, colors = uio.read_yaml(ontology_fname)
    ontology = {names[idx]: {"idx": idx, "rgb": tuple(colors[idx])} for idx in names}

    # Get samples filenames
    train_split = [
        s.split(" ") + ["train"] for s in uio.read_txt(split_fnames["train"])
    ]
    val_split = [s.split(" ") + ["val"] for s in uio.read_txt(split_fnames["val"])]
    test_split = [s.split(" ") + ["test"] for s in uio.read_txt(split_fnames["test"])]

    samples_data = train_split + val_split + test_split

    # Build dataset as ordered python dictionary
    dataset = OrderedDict()
    skipped_samples = []
    for data_fname, label_fname, split in samples_data:
        sample_dir, sample_base_name = os.path.split(data_fname)
        sample_base_name, _ = os.path.splitext(sample_base_name)
        scene = os.path.split(os.path.split(sample_dir)[0])[-1]
        sample_name = f"{scene}-{sample_base_name}"

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

    return dataset, ontology


class Rellis3DImageSegmentationDataset(
    segmentation_dataset.ImageSegmentationDataset
):
    """Specific class for Rellis3D-styled image segmentation datasets. All data can
    be downloaded from the official repo (https://github.com/unmannedlab/RELLIS-3D):
    images   -> https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV
    labels   -> https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5
    split    -> https://drive.google.com/file/d/1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ
    ontology -> https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6

    :param dataset_dir: Directory where both RGB images and annotations have been extracted to
    :type dataset_dir: str
    :param split_dir: Directory where train, val, and test files (.lst) have been extracted to
    :type split_dir: str
    :param ontology_fname: YAML file contained in the ontology compressed directory
    :type ontology_fname: str
    """

    def __init__(self, dataset_dir: str, split_dir: str, ontology_fname: str):
        split_fnames = {
            "train": os.path.join(split_dir, "train.lst"),
            "val": os.path.join(split_dir, "val.lst"),
            "test": os.path.join(split_dir, "test.lst"),
        }
        dataset, ontology = build_dataset(dataset_dir, split_fnames, ontology_fname)

        # Convert to Pandas
        cols = ["image", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        dataset.attrs = {"ontology": ontology}

        super().__init__(dataset, dataset_dir, ontology)


class Rellis3DLiDARSegmentationDataset(
    segmentation_dataset.LiDARSegmentationDataset
):
    """Specific class for Rellis3D-styled LiDAR segmentation datasets. All data can
    be downloaded from the official repo (https://github.com/unmannedlab/RELLIS-3D):
    points   -> https://drive.google.com/file/d/1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH
    labels   -> https://drive.google.com/file/d/12bsblHXtob60KrjV7lGXUQTdC5PhV8Er
    split    -> https://drive.google.com/file/d/1raQJPySyqDaHpc53KPnJVl3Bln6HlcVS
    ontology -> https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6

    :param dataset_dir: Directory where both points and labels have been extracted to
    :type dataset_dir: str
    :param split_dir: Directory where train, val, and test files (.lst) have been extracted to
    :type split_dir: str
    :param ontology_fname: YAML file contained in the ontology compressed directory
    :type ontology_fname: str
    """

    def __init__(self, dataset_dir: str, split_dir: str, ontology_fname: str):
        split_fnames = {
            "train": os.path.join(split_dir, "pt_train.lst"),
            "val": os.path.join(split_dir, "pt_val.lst"),
            "test": os.path.join(split_dir, "pt_test.lst"),
        }
        dataset, ontology = build_dataset(dataset_dir, split_fnames, ontology_fname)

        # Convert to Pandas
        cols = ["points", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        dataset.attrs = {"ontology": ontology}

        super().__init__(dataset, dataset_dir, ontology)
