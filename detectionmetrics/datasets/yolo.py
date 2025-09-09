from glob import glob
import os
from typing import Tuple, List, Optional

import pandas as pd
from PIL import Image

from detectionmetrics.datasets.detection import ImageDetectionDataset
from detectionmetrics.utils import io as uio


def build_dataset(
    dataset_fname: str, dataset_dir: Optional[str] = None, im_ext: str = "jpg"
) -> Tuple[pd.DataFrame, dict]:
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
        dataset_dir = dataset_info["path"]
    assert os.path.isdir(dataset_dir), f"Dataset directory not found: {dataset_dir}"

    # Build ontology from dataset configuration
    ontology = {}
    for idx, name in dataset_info["names"].items():
        ontology[name] = {
            "idx": idx,
            "rgb": [0, 0, 0],  # Placeholder; YAML doesn't define RGB colors
        }

    # Build dataset DataFrame
    rows = []
    for split in ["train", "val", "test"]:
        if split in dataset_info:
            images_dir = os.path.join(dataset_dir, dataset_info[split])
            labels_dir = os.path.join(
                dataset_dir, dataset_info[split].replace("images", "labels")
            )
            for label_fname in glob(os.path.join(labels_dir, "*.txt")):
                label_basename = os.path.basename(label_fname)
                image_basename = label_basename.replace(".txt", f".{im_ext}")
                image_fname = os.path.join(images_dir, image_basename)
                os.path.basename(image_fname)
                if not os.path.isfile(image_fname):
                    continue

                rows.append(
                    {
                        "image": os.path.join("images", split, image_basename),
                        "annotation": os.path.join("labels", split, label_basename),
                        "split": split,
                    }
                )

    dataset = pd.DataFrame(rows)
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
    ) -> Tuple[List[List[float]], List[int], List[int]]:
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
