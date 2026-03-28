from pycocotools.coco import COCO
import os
import re
import pandas as pd
from typing import Tuple, List, Optional

from perceptionmetrics.datasets.detection import ImageDetectionDataset


def find_img_dir_and_ann_file(dataset_path: str, split: str) -> Tuple[str, str]:
    """
    Find the image directory and annotation file for a given split in a COCO-style dataset.

    :param dataset_path: Path to the root of the dataset
    :type dataset_path: str
    :param split: Dataset split name (e.g., "train", "val", "test")
    :type split: str
    :return: Tuple of (image directory path, annotation file path)
    :rtype: Tuple[str, str]
    :raises FileNotFoundError: If the image directory or annotation file is not found
    """
    images_root = os.path.join(dataset_path, "images")
    img_dir = None
    pattern = re.compile(rf"{split}\d*")
    for folder in os.listdir(images_root):
        if pattern.fullmatch(folder):
            img_dir = os.path.join(images_root, folder)
            break

    ann_root = os.path.join(dataset_path, "annotations")
    ann_file_path = None
    pattern = re.compile(rf"instances_{split}\d*\.json")
    for fname in os.listdir(ann_root):
        if pattern.fullmatch(fname):
            ann_file_path = os.path.join(ann_root, fname)
            break

    if img_dir is None:
        raise FileNotFoundError(
            f"No matching image directory found for split '{split}' in {dataset_path}."
        )
    if ann_file_path is None:
        raise FileNotFoundError(
            f"No matching annotation file found for split '{split}' in {dataset_path}."
        )

    return img_dir, ann_file_path


def build_coco_dataset(
    annotation_file: str,
    image_dir: str,
    coco_obj: Optional[COCO] = None,
    split: str = "train",
) -> Tuple[pd.DataFrame, dict, str]:
    """Build dataset and ontology dictionaries from COCO dataset structure

    :param annotation_file: Path to the COCO-format JSON annotation file
    :type annotation_file: str
    :param image_dir: Path to the directory containing image files
    :type image_dir: str
    :param coco_obj: Optional pre-loaded COCO object to reuse
    :type coco_obj: COCO
    :param split: Dataset split name (e.g., "train", "val", "test")
    :type split: str
    :return: Dataset DataFrame and ontology dictionary
    :rtype: Tuple[pd.DataFrame, dict]
    """
    # Check that provided paths exist
    assert os.path.isfile(
        annotation_file
    ), f"Annotation file not found: {annotation_file}"
    assert os.path.isdir(image_dir), f"Image directory not found: {image_dir}"

    # Load COCO annotations (reuse if provided)
    if coco_obj is None:
        coco = COCO(annotation_file)
    else:
        coco = coco_obj

    # Build ontology from COCO categories
    ontology = {}
    for cat in coco.loadCats(coco.getCatIds()):
        ontology[cat["name"]] = {
            "idx": cat["id"],
            # "name": cat["name"],
            "rgb": [0, 0, 0],  # Placeholder; COCO doesn't define RGB colors
        }

    # Build dataset DataFrame from COCO image IDs
    rows = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        rows.append(
            {
                "image": img_info["file_name"],
                "annotation": str(img_id),
                "split": split,  # Use provided split parameter
            }
        )

    dataset = pd.DataFrame(rows)
    dataset.attrs = {"ontology": ontology}

    return dataset, ontology


class CocoDataset(ImageDetectionDataset):
    """
    Specific class for COCO-styled object detection datasets.

    :param annotation_file: Path to the COCO-format JSON annotation file
    :type annotation_file: str
    :param image_dir: Path to the directory containing image files
    :type image_dir: str
    :param split: Dataset split name (e.g., "train", "val", "test")
    :type split: str
    """

    def __init__(self, annotation_file: str, image_dir: str, split: str = "train"):
        # Load COCO object once - this loads all annotations into memory with efficient indexing
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.split = split

        # Build dataset using the same COCO object and split
        dataset, ontology = build_coco_dataset(
            annotation_file, image_dir, self.coco, split=split
        )

        super().__init__(dataset=dataset, dataset_dir=image_dir, ontology=ontology)

    def read_annotation(
        self, fname: str
    ) -> Tuple[List[List[float]], List[int], List[int]]:
        """Return bounding boxes and category indices for a given image ID.

        This method uses COCO's efficient indexing to load annotations on-demand.
        The COCO object maintains an internal index that allows for very fast
        annotation retrieval without needing a separate cache.

        :param fname: str (image_id in string form)
        :return: Tuple of (boxes, category_indices)
        """
        # Extract image ID (fname might be a path or ID string)
        try:
            image_id = int(os.path.basename(fname))
        except ValueError:
            raise ValueError(f"Invalid annotation ID: {fname}")

        # Use COCO's efficient indexing to get annotations for this image
        # getAnnIds() and loadAnns() are very fast due to COCO's internal indexing
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, category_indices = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            category_indices.append(ann["category_id"])

        return boxes, category_indices
