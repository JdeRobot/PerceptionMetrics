from abc import ABC, abstractmethod
import os
from typing import List, Optional
from typing_extensions import Self

import pandas as pd


class PerceptionDataset(ABC):
    """Abstract perception dataset class.

    :param dataset: Segmentation/Detection dataset as a pandas DataFrame
    :type dataset: pd.DataFrame
    :param dataset_dir: Dataset root directory
    :type dataset_dir: str
    :param ontology: Dataset ontology definition
    :type ontology: dict
    """

    def __init__(self, dataset: pd.DataFrame, dataset_dir: str, ontology: dict):
        self.dataset = dataset
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.ontology = ontology
        self.has_label_count = all("label_count" in v for v in self.ontology.values())

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
        if not self.has_label_count:
            assert self.ontology == new_dataset.ontology, "Ontologies don't match"
        else:
            # Check if classes match
            assert (
                self.ontology.keys() == new_dataset.ontology.keys()
            ), "Ontologies don't match"
            for class_name in self.ontology:
                # Check if indices, and RGB values match
                assert (
                    self.ontology[class_name]["idx"]
                    == new_dataset.ontology[class_name]["idx"]
                ), "Ontologies don't match"
                if (
                    "rgb" in self.ontology[class_name]
                    and "rgb" in new_dataset.ontology[class_name]
                ):
                    assert (
                        self.ontology[class_name]["rgb"]
                        == new_dataset.ontology[class_name]["rgb"]
                    ), "Ontologies don't match"

                # Accumulate label count
                self.ontology[class_name]["label_count"] += new_dataset.ontology[
                    class_name
                ]["label_count"]

        # Global filenames to avoid dealing with each dataset relative location
        self.make_fname_global()
        new_dataset.make_fname_global()

        # Simply concatenate pandas dataframes
        self.dataset = pd.concat(
            [self.dataset, new_dataset.dataset], verify_integrity=True
        )

    def get_label_count(self, splits: Optional[List[str]] = None):
        """Get label count for each class in the dataset

        :param splits: Dataset splits to consider, defaults to ["train", "val"]
        :type splits: List[str], optional
        :return: Label count for the dataset
        :rtype: np.ndarray
        """
        raise NotImplementedError
