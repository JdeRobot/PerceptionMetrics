from pycocotools.coco import COCO
import os
import pandas as pd
from typing import Tuple, List, Optional

from detectionmetrics.datasets.detection import ImageDetectionDataset


def build_coco_dataset(annotation_file: str, image_dir: str, coco_obj: Optional[COCO] = None) -> Tuple[pd.DataFrame, dict]:
    """Build dataset and ontology dictionaries from COCO dataset structure

    :param annotation_file: Path to the COCO-format JSON annotation file
    :type annotation_file: str
    :param image_dir: Path to the directory containing image files
    :type image_dir: str
    :param coco_obj: Optional pre-loaded COCO object to reuse
    :type coco_obj: COCO
    :return: Dataset DataFrame and ontology dictionary
    :rtype: Tuple[pd.DataFrame, dict]
    """
    # Check that provided paths exist
    assert os.path.isfile(annotation_file), f"Annotation file not found: {annotation_file}"
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
            "rgb": [0, 0, 0]  # Placeholder; COCO doesn't define RGB colors
        }

    # Build dataset DataFrame from COCO image IDs
    rows = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        rows.append({
            "image": img_info["file_name"],
            "annotation": str(img_id),
            "split": "train"  # Default split - could be enhanced to read from COCO
        })
    
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
    """
    def __init__(self, annotation_file: str, image_dir: str):
        # Load COCO object once
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        
        # Build dataset using the same COCO object
        dataset, ontology = build_coco_dataset(annotation_file, image_dir, self.coco)
        
        super().__init__(dataset=dataset, dataset_dir=image_dir, ontology=ontology)

    def read_annotation(self, fname: str) -> Tuple[List[List[float]], List[int], List[int]]:
        """Return bounding boxes, labels, and category_ids for a given image ID.

        :param fname: str (image_id in string form)
        :return: Tuple of (boxes, labels, category_ids)
        """
        # Extract image ID (fname might be a path or ID string)
        try:
            image_id = int(os.path.basename(fname))  # handles both '123' and '/path/to/123'
        except ValueError:
            raise ValueError(f"Invalid annotation ID: {fname}")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        category_ids = []

        for ann in anns:
            # Convert [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            category_ids.append(ann["category_id"])

        return boxes, labels, category_ids

