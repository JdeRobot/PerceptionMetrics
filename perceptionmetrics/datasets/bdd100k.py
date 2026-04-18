from glob import glob
import json
import logging
import os
from typing import Tuple, List

import pandas as pd

from perceptionmetrics.datasets.detection import ImageDetectionDataset

# BDD100K fixed 10-category ontology
BDD100K_ONTOLOGY = {
    "car": {"idx": 0, "rgb": [0, 0, 0]},
    "truck": {"idx": 1, "rgb": [0, 0, 0]},
    "bus": {"idx": 2, "rgb": [0, 0, 0]},
    "person": {"idx": 3, "rgb": [0, 0, 0]},
    "rider": {"idx": 4, "rgb": [0, 0, 0]},
    "bike": {"idx": 5, "rgb": [0, 0, 0]},
    "motor": {"idx": 6, "rgb": [0, 0, 0]},
    "traffic light": {"idx": 7, "rgb": [0, 0, 0]},
    "traffic sign": {"idx": 8, "rgb": [0, 0, 0]},
    "train": {"idx": 9, "rgb": [0, 0, 0]},
}


def build_bdd100k_dataset(
    images_dir: str, labels_dir: str
) -> Tuple[pd.DataFrame, dict]:
    """Build dataset DataFrame and ontology from BDD100K directory structure.

    Expected layout::

        <images_dir>/train/<filename>.jpg
        <images_dir>/val/<filename>.jpg
        <labels_dir>/train/<filename>.json
        <labels_dir>/val/<filename>.json

    :param images_dir: Root directory containing train/ and val/ image folders
    :type images_dir: str
    :param labels_dir: Root directory containing train/ and val/ label folders
    :type labels_dir: str
    :return: Tuple of (dataset DataFrame with columns [image, annotation, split],
             ontology dict)
    :rtype: Tuple[pd.DataFrame, dict]
    """
    ontology = dict(BDD100K_ONTOLOGY)

    rows = []
    for split in ["train", "val"]:
        split_images_dir = os.path.join(images_dir, split)
        if not os.path.isdir(split_images_dir):
            logging.warning(
                "Image split directory not found: %s; skipping.", split_images_dir
            )
            continue

        image_files = glob(os.path.join(split_images_dir, "*.jpg")) + glob(
            os.path.join(split_images_dir, "*.png")
        )
        for image_fname in sorted(image_files):
            image_basename = os.path.basename(image_fname)
            stem = os.path.splitext(image_basename)[0]
            label_fname = os.path.join(labels_dir, split, f"{stem}.json")

            if not os.path.isfile(label_fname):
                logging.warning(
                    "No matching label file for image '%s'; skipping.",
                    image_fname,
                )
                continue

            rows.append(
                {
                    "image": image_fname,
                    "annotation": label_fname,
                    "split": split,
                }
            )

    dataset = pd.DataFrame(rows)
    dataset.attrs = {"ontology": ontology}

    return dataset, ontology


class BDD100KDetectionDataset(ImageDetectionDataset):
    """BDD100K object detection dataset.

    :param images_dir: Root directory containing train/ and val/ image folders
    :type images_dir: str
    :param labels_dir: Root directory containing train/ and val/ label folders
    :type labels_dir: str
    """

    def __init__(self, images_dir: str, labels_dir: str):
        dataset, ontology = build_bdd100k_dataset(images_dir, labels_dir)
        # Paths in the DataFrame are already absolute; dataset_dir is used
        # only by PerceptionDataset base class (must not be None).
        super().__init__(dataset=dataset, dataset_dir=images_dir, ontology=ontology)

    def read_annotation(self, fname: str) -> Tuple[List[List[float]], List[int]]:
        """Read bounding boxes and category indices from a BDD100K per-image JSON.

        Objects without a ``box2d`` key (e.g. lane/poly2d annotations) are
        skipped.  Unknown categories not in the BDD100K ontology are skipped
        with a warning.

        :param fname: Path to the per-image JSON annotation file
        :type fname: str
        :return: Tuple of (boxes as [[x1, y1, x2, y2], ...], category_indices)
        :rtype: Tuple[List[List[float]], List[int]]
        """
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build reverse lookup: category name -> index
        cat_to_idx = {name: info["idx"] for name, info in self.ontology.items()}

        boxes = []
        category_indices = []

        objects = data.get("frames", [{}])[0].get("objects", [])
        for obj in objects:
            if "box2d" not in obj:
                continue

            category = obj.get("category", "")
            if category not in cat_to_idx:
                logging.warning(
                    "Unknown category '%s' in annotation '%s'; skipping.",
                    category,
                    fname,
                )
                continue

            box = obj["box2d"]
            boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            category_indices.append(cat_to_idx[category])

        return boxes, category_indices
