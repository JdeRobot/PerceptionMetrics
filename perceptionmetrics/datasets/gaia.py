import os
from typing import Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.io as uio


def build_dataset(dataset_fname: str) -> Tuple[pd.DataFrame, str, dict]:
    """Build dataset and ontology dictionaries from GAIA-like dataset structure

    :param dataset_fname: Parquet dataset filename
    :type dataset_fname: str
    :return: Dataset dataframe and directory, and onotology
    :rtype: Tuple[pd.DataFrame, str, dict]
    """
    # Check that provided path exist
    assert os.path.isfile(dataset_fname), "Dataset file not found"

    # Load dataset Parquet file
    dataset = pd.read_parquet(dataset_fname)
    dataset_dir = os.path.dirname(dataset_fname)

    # Read ontology file
    try:
        ontology_fname = dataset.attrs["ontology_fname"]
    except KeyError:
        ontology_fname =  "ontology.json"

    ontology_fname = os.path.join(dataset_dir, ontology_fname)
    assert os.path.isfile(ontology_fname), "Ontology file not found"

    ontology = uio.read_json(ontology_fname)
    for name, data in ontology.items():
        ontology[name]["rgb"] = tuple(data["rgb"])

    # Report results
    print(f"Samples retrieved: {len(dataset)}")

    return dataset, dataset_dir, ontology


class GaiaImageSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Specific class for GAIA-styled image segmentation datasets

    :param dataset_fname: Parquet dataset filename
    :type dataset_fname: str
    """

    def __init__(self, dataset_fname: str):
        dataset, dataset_dir, ontology = build_dataset(dataset_fname)
        super().__init__(dataset, dataset_dir, ontology)


class GaiaLiDARSegmentationDataset(segmentation_dataset.LiDARSegmentationDataset):
    """Specific class for GAIA-styled LiDAR segmentation datasets

    :param dataset_fname: Parquet dataset filename
    :type dataset_fname: str
    """

    def __init__(self, dataset_fname: str):
        dataset, dataset_dir, ontology = build_dataset(dataset_fname)
        super().__init__(dataset, dataset_dir, ontology)
