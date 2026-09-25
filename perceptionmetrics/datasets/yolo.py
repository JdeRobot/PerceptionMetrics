from typing import OrderedDict
from glob import glob
import logging
import os
from typing import Tuple, List, Optional

import pandas as pd
from PIL import Image

from perceptionmetrics.datasets.detection import ImageDetectionDataset
from perceptionmetrics.utils import io as uio


def build_dataset(
    dataset_fname: str, dataset_dir: Optional[str] = None, im_ext: str = "jpg"
) -> Tuple[pd.DataFrame, dict, str]:
    """Build dataset and ontology dictionaries from YOLO dataset structure

    :param dataset_fname: Path to the YAML dataset configuration file
    :type dataset_fname: str
    :param dataset_dir: Path to the directory containing images and annotations. If not provided, it will be inferred from the dataset file
    :type dataset_dir: Optional[str]
    :param im_ext: Image file extension (default is "jpg")
    :type im_ext: str
    :return: Dataset DataFrame and ontology dictionary
    :rtype: Tuple[pd.DataFrame, dict]
    """
    # Read dataset configuration from YAML file
    assert os.path.isfile(dataset_fname), f"Dataset file not found: {dataset_fname}"
    dataset_info = uio.read_yaml(dataset_fname)

    # Check that image directory exists
    if dataset_dir is None:
        dataset_dir = dataset_info.get("path")
    assert os.path.isdir(dataset_dir), f"Dataset directory not found: {dataset_dir}"

    # Build ontology from dataset configuration
    ontology = {}
    names = dataset_info["names"]

    # Support both list and dictionary formats for YOLO datasets
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    for idx, name in names.items():
        ontology[name] = {
            "idx": idx,
            "rgb": [0, 0, 0],  # Placeholder; YAML doesn't define RGB colors
        }

    # Build dataset DataFrame
    dataset = OrderedDict()
    for split in ["train", "val", "test"]:
        split_paths = dataset_info.get(split)
        if not split_paths:
            logging.warning(
                "Split '%s' is missing or has no path defined in '%s'; skipping.",
                split,
                dataset_fname,
            )
            continue

        if isinstance(split_paths, str):
            split_paths = [split_paths]

        def _make_path_abs(p: str) -> str:
            """Make path absolute if it is relative"""
            return os.path.join(dataset_dir, p) if not os.path.isabs(p) else p

        def _add_to_dataset(image_fname: str, label_fname: str, split: str) -> None:
            """Add a sample to the dataset DataFrame"""
            if os.path.isfile(image_fname) and os.path.isfile(label_fname):
                # splitext, not split("."): Roboflow exports name files like
                # "IMG_0001_jpg.rf.<hash>.jpg", and cutting at the first dot gives
                # every augmented copy the same key, so all but one get dropped
                sample_name = os.path.splitext(os.path.basename(image_fname))[0]
                dataset[sample_name] = (
                    os.path.relpath(image_fname, dataset_dir),
                    os.path.relpath(label_fname, dataset_dir),
                    split,
                )

        for sp in split_paths:
            sp = _make_path_abs(sp)

            # Parse as txt file containing list of image paths
            if sp.endswith(".txt"):
                if not os.path.isfile(sp):
                    continue

                with open(sp, "r") as f:
                    image_lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]

                for image_rel in image_lines:
                    image_rel = image_rel.replace("./", "")
                    image_fname = _make_path_abs(image_rel)

                    images_dir, labels_dir = (
                        f"{os.sep}images{os.sep}",
                        f"{os.sep}labels{os.sep}",
                    )
                    if images_dir in image_fname:
                        label_fname = (
                            labels_dir.join(image_fname.rsplit(images_dir, 1)).rsplit(
                                ".", 1
                            )[0]
                            + ".txt"
                        )
                    else:
                        label_fname = image_fname.rsplit(".", 1)[0] + ".txt"

                    _add_to_dataset(image_fname, label_fname, split)

            else:
                if "images" in sp:
                    labels_dir = sp.replace("images", "labels")
                else:
                    labels_dir = os.path.join(
                        dataset_dir, "labels", os.path.basename(sp)
                    )

                if not os.path.isdir(labels_dir):
                    continue

                for label_fname in glob(os.path.join(labels_dir, "*.txt")):
                    label_basename = os.path.basename(label_fname)
                    image_basename = label_basename.replace(".txt", f".{im_ext}")
                    image_fname = os.path.join(sp, image_basename)

                    _add_to_dataset(image_fname, label_fname, split)

    cols = ["image", "annotation", "split"]
    dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
    dataset.attrs = {"ontology": ontology}

    return dataset, ontology, dataset_dir


class YOLODataset(ImageDetectionDataset):
    """
    Specific class for YOLO-styled object detection datasets.

    :param dataset_fname: Path to the YAML dataset configuration file
    :type dataset_fname: str
    :param dataset_dir: Path to the directory containing images and annotations. If not provided, it will be inferred from the dataset file
    :type dataset_dir: Optional[str]
    :param im_ext: Image file extension (default is "jpg")
    :type im_ext: str
    """

    def __init__(
        self, dataset_fname: str, dataset_dir: Optional[str], im_ext: str = "jpg"
    ):
        # Build dataset using the same COCO object
        dataset, ontology, dataset_dir = build_dataset(
            dataset_fname, dataset_dir, im_ext
        )

        self.im_ext = im_ext
        super().__init__(dataset=dataset, dataset_dir=dataset_dir, ontology=ontology)

    def read_annotation(
        self, fname: str, image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[List[float]], List[int]]:
        """Return bounding boxes, and category indices for a given image ID.

        :param fname: Annotation path
        :type fname: str
        :param image_size: Corresponding image size in (w, h) format for converting relative bbox size to absolute. If not provided, we will assume image path
        :type image_size: Optional[Tuple[int, int]]
        :return: Tuple of (boxes, category_indices)
        """
        label = uio.read_txt(fname)
        image_fname = fname.replace(".txt", f".{self.im_ext}")
        image_fname = image_fname.replace("labels", "images")
        if image_size is None:
            image_size = Image.open(image_fname).size

        boxes = []
        category_indices = []

        im_w, im_h = image_size
        for row in label:
            category_idx, xc, yc, w, h = map(float, row.split())
            category_indices.append(int(category_idx))

            abs_xc = xc * im_w
            abs_yc = yc * im_h
            abs_w = w * im_w
            abs_h = h * im_h

            boxes.append(
                [
                    abs_xc - abs_w / 2,
                    abs_yc - abs_h / 2,
                    abs_xc + abs_w / 2,
                    abs_yc + abs_h / 2,
                ]
            )

        return boxes, category_indices
