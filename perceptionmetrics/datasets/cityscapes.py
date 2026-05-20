from collections import OrderedDict
from glob import glob
import os
from typing import Optional, Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
from cityscapesscripts.helpers.labels import labels


def build_dataset_ontology(use_train_id: bool = False) -> dict:
    """Build ontology dictionary from Cityscapes dataset labels

    :param use_train_id: Whether to use train IDs instead of Cityscapes label IDs, defaults to False
    :type use_train_id: bool, optional
    :return: Ontology dictionary mapping class names to label metadata
    :rtype: dict
    """
    ontology = {}
    for label in labels:
        if label.ignoreInEval:
            continue
        ontology[label.name] = {
            "idx": label.trainId if use_train_id else label.id,
            "train_id": label.trainId,
            "cityscapes_id": label.id,
            "category": label.category,
            "category_id": label.categoryId,
            "has_instances": label.hasInstances,
            "rgb": label.color,
        }
    return ontology


def build_dataset(
    train_dataset_dir: Optional[str] = None,
    val_dataset_dir: Optional[str] = None,
    test_dataset_dir: Optional[str] = None,
    image_dir: str = "leftImg8bit_trainvaltest/leftImg8bit",
    label_dir: str = "gtFine",
    image_suffix: str = "_leftImg8bit.png",
    label_suffix: str = "_gtFine_labelIds.png",
    use_train_id: bool = False,
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries from Cityscapes dataset structure

    :param train_dataset_dir: Directory containing training data, defaults to None
    :type train_dataset_dir: str, optional
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    :param image_dir: Subdirectory containing images within each dataset directory, defaults to "leftImg8bit_trainvaltest/leftImg8bit"
    :type image_dir: str, optional
    :param label_dir: Subdirectory containing labels within each dataset directory, defaults to "gtFine"
    :type label_dir: str, optional
    :param image_suffix: File suffix used to filter image files, defaults to "_leftImg8bit.png"
    :type image_suffix: str, optional
    :param label_suffix: File suffix used to filter label files, defaults to "_gtFine_labelIds.png"
    :type label_suffix: str, optional
    :param use_train_id: Whether to use train IDs instead of Cityscapes label IDs, defaults to False
    :type use_train_id: bool, optional
    :return: Dataset and ontology dictionaries
    :rtype: Tuple[dict, dict]
    """
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

    ontology = build_dataset_ontology(use_train_id=use_train_id)
    dataset = OrderedDict()

    for split, dataset_dir in dataset_dirs.items():

        image_pattern = os.path.join(
            dataset_dir,
            image_dir,
            split,
            "*",
            f"*{image_suffix}",
        )

        image_fnames = sorted(glob(image_pattern))

        if len(image_fnames) == 0:
            print(f"No images found for split={split}")
            print(f"Searched pattern: {image_pattern}")
            continue

        for image_fname in image_fnames:
            image_fname = os.path.abspath(image_fname)

            city = os.path.basename(os.path.dirname(image_fname))
            image_basename = os.path.basename(image_fname)

            sample_name = image_basename.replace(image_suffix, "")

            label_basename = f"{sample_name}{label_suffix}"

            label_fname = os.path.join(
                dataset_dir,
                label_dir,
                split,
                city,
                label_basename,
            )
            label_fname = os.path.abspath(label_fname)

            if not os.path.exists(label_fname):
                print(f"Missing label for image: {image_fname}")
                print(f"Expected label path: {label_fname}")
                continue

            dataset[sample_name] = (
                image_fname,
                label_fname,
                city,
                split,
            )

    return dataset, ontology


class CityscapesImageSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Specific class for Cityscapes-styled image segmentation datasets. The dataset can be
    downloaded from the official webpage (https://www.cityscapes-dataset.com):
    images -> leftImg8bit_trainvaltest.zip
    labels -> gtFine_trainvaltest.zip

    :param train_dataset_dir: Directory containing training data, defaults to None
    :type train_dataset_dir: str, optional
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    :param image_dir: Subdirectory containing images within each dataset directory, defaults to "leftImg8bit_trainvaltest/leftImg8bit"
    :type image_dir: str, optional
    :param label_dir: Subdirectory containing labels within each dataset directory, defaults to "gtFine"
    :type label_dir: str, optional
    :param image_suffix: File suffix used to filter image files, defaults to "_leftImg8bit.png"
    :type image_suffix: str, optional
    :param label_suffix: File suffix used to filter label files, defaults to "_gtFine_labelIds.png"
    :type label_suffix: str, optional
    :param use_train_id: Whether to use train IDs instead of Cityscapes label IDs, defaults to False
    :type use_train_id: bool, optional
    """

    def __init__(
        self,
        train_dataset_dir: Optional[str] = None,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
        image_dir: str = "leftImg8bit_trainvaltest/leftImg8bit",
        label_dir: str = "gtFine",
        image_suffix: str = "_leftImg8bit.png",
        label_suffix: str = "_gtFine_labelIds.png",
        use_train_id: bool = False,
    ):
        dataset, ontology = build_dataset(
            train_dataset_dir=train_dataset_dir,
            val_dataset_dir=val_dataset_dir,
            test_dataset_dir=test_dataset_dir,
            image_dir=image_dir,
            label_dir=label_dir,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            use_train_id=use_train_id,
        )

        cols = ["image", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        if len(dataset) == 0:
            raise ValueError(
                "No Cityscapes samples were found. Please check dataset paths."
            )

        print(f"Samples retrieved: {len(dataset)}")

        all_dataset_dirs = [train_dataset_dir, val_dataset_dir, test_dataset_dir]
        dataset_dir = [d for d in all_dataset_dirs if d is not None][0]

        super().__init__(dataset, dataset_dir, ontology)
