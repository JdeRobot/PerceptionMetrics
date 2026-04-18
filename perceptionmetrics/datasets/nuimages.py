import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import cv2
from perceptionmetrics.datasets.detection import ImageDetectionDataset
from nuimages import NuImages
from nuimages.utils.utils import name_to_index_mapping
from perceptionmetrics.datasets import segmentation as segmentation_dataset

DROP = {}

def _get_rgb_from_idx(class_idx: int) -> List[int]:
    """Generate a deterministic RGB color for a class index."""
    rng = np.random.default_rng(seed=class_idx + 17)
    rgb = rng.integers(low=40, high=256, size=3)
    return [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def build_nuimages_detection_dataset(
    dataset_dir: str,
    version: str = "v1.0-mini",
    split: str = "train",
    nuim_object: Optional[NuImages] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a nuImages 2D detection dataset index.

    Iterates through the nuImages scenes and samples, collects image paths for a given camera, and constructs a dataset index along with a category ontology mapping class names to integer indices.

    :param dataset_dir: Path to the nuImages dataset root directory.
    :type dataset_dir: str
    :param version: nuImages dataset version to load, defaults to "v1.0-mini".
    :type version: str
    :param split: Dataset split to load ("train" or "val"), defaults to "train".
    :type split: str
    :param nuim_object: Optional pre-initialized NuImages object to reuse, defaults to None.
    :type nuim_object: Optional[NuImages]
    :return: Tuple containing:
             - A pandas DataFrame with columns ["image", "annotation", "split"] for each sample.
             - An ontology dictionary mapping category names to indices.
    :rtype: Tuple[pd.DataFrame, dict]
    """

    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(
        dataset_dir
    ), f"Dataset directory {dataset_dir} does not exist."

    nuim = (
        nuim_object
        if nuim_object
        else NuImages(version=version, dataroot=dataset_dir, verbose=False)
    )

    all_categories = [cat["name"] for cat in nuim.category]
    categories = [cat for cat in all_categories if cat not in DROP]

    ontology = {
        name: {"idx": i + 1, "rgb": _get_rgb_from_idx(i + 1)}
        for i, name in enumerate(categories)
    }
    cat_to_idx = {name: ontology[name]["idx"] for name in ontology}

    rows = []
    for sample in nuim.sample:
        key_camera_token = sample["key_camera_token"]
        sample_data = nuim.get("sample_data", key_camera_token)
        rows.append(
            {
                "image": os.path.join(dataset_dir, sample_data["filename"]),
                "annotation": key_camera_token,
                "split": split,
            }
        )

    dataset = pd.DataFrame(rows)
    dataset.attrs = {
        "ontology": ontology,
        "cat_to_idx": cat_to_idx,
    }

    print(
        f"Built nuimages detection dataset with {len(dataset)} samples and "
        f"{len(categories)} categories."
    )
    return dataset, ontology


def build_nuimages_segmentation_dataset(
    dataset_dir: str,
    version: str = "v1.0-mini",
    split: str = "train",
    labels_rel_dir: str = "generated/nuimages_segmentation_labels",
    nuim_object: Optional[NuImages] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a nuImages semantic segmentation dataset index and masks.

    :param dataset_dir: Path to the nuImages dataset root directory.
    :type dataset_dir: str
    :param version: nuImages dataset version to load, defaults to "v1.0-mini".
    :type version: str
    :param split: Dataset split to load ("train" or "val"), defaults to "train".
    :type split: str
    :param labels_rel_dir: Relative directory for segmentation labels, defaults to "generated/nuimages_segmentation_labels".
    :type labels_rel_dir: str
    :param nuim_object: Optional pre-initialized NuImages object to reuse, defaults to None.
    :type nuim_object: Optional[NuImages]
    :return: Tuple containing:
             - A pandas DataFrame with columns ["image", "label", "split"] for each sample.
             - An ontology dictionary mapping category names to indices.
    :rtype: Tuple[pd.DataFrame, dict]
    """

    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(
        dataset_dir
    ), f"Dataset directory {dataset_dir} does not exist."

    nuim = (
        nuim_object
        if nuim_object
        else NuImages(version=version, dataroot=dataset_dir, verbose=False)
    )

    ## For segmentation, we build semantic masks from surface annotations for keyframe images

    labels_root = os.path.join(dataset_dir, labels_rel_dir, version)
    os.makedirs(labels_root, exist_ok=True)

    name_to_global_idx = name_to_index_mapping(nuim.category)
    ontology = {"background": {"idx": 0, "rgb": [0, 0, 0]}}
    for category_name, category_idx in sorted(
        name_to_global_idx.items(), key=lambda item: item[1]
    ):
        ontology[category_name] = {
            "idx": int(category_idx),
            "rgb": _get_rgb_from_idx(int(category_idx)),
        }

    rows = []
    for sample in nuim.sample:
        key_camera_token = sample["key_camera_token"]
        sample_data = nuim.get("sample_data", key_camera_token)

        image_abs_path = os.path.join(dataset_dir, sample_data["filename"])
        semantic_mask, _ = nuim.get_segmentation(key_camera_token)
        label = semantic_mask.astype(np.uint8)

        label_abs_path = os.path.join(labels_root, f"{key_camera_token}.png")
        cv2.imwrite(label_abs_path, label)

        rows.append(
            {
                "image": image_abs_path,
                "label": label_abs_path,
                "split": split,
            }
        )

    dataset = pd.DataFrame(rows)
    dataset.attrs = {"ontology": ontology}

    print(
        f"Built nuImages segmentation dataset with {len(dataset)} samples and "
        f"{len(ontology)} classes."
    )

    return dataset, ontology


class NuImagesDetectionDataset(ImageDetectionDataset):
    """
    Dataset class for nuImages 2D object detection.

    Inherits from ImageDetectionDataset and parses 2D bounding boxes
    from nuImages.

    :param dataset_dir: Path to the nuImages dataset root.
    :type dataset_dir: str
    :param version: nuImages version to load, defaults to "v1.0-mini".
    :type version: str
    :param split: Dataset split ("train" or "val").
    :type split: str
    """

    def __init__(
        self,
        dataset_dir: str,
        version: str = "v1.0-mini",
        split: str = "train",
    ):
        """
        Initialize the nuImages 2D detection dataset.

        :param dataset_dir: Path to the nuImages dataset root directory.
        :param version: nuImages dataset version to load, defaults to "v1.0-mini".
        :param split: Dataset split to load ("train" or "val"), defaults to "train".
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.nuim = NuImages(dataroot=dataset_dir, version=version)

        dataset, ontology = build_nuimages_detection_dataset(
            dataset_dir=dataset_dir,
            version=version,
            split=split,
            nuim_object=self.nuim,
        )

        self.cat_to_idx = dataset.attrs["cat_to_idx"]
        self.ann_by_sample_data_token = {}
        for ann in self.nuim.object_ann:
            sample_data_token = ann["sample_data_token"]
            if sample_data_token not in self.ann_by_sample_data_token:
                self.ann_by_sample_data_token[sample_data_token] = []
            self.ann_by_sample_data_token[sample_data_token].append(ann)

        super().__init__(dataset=dataset, dataset_dir=dataset_dir, ontology=ontology)

    def read_annotation(self, fname: str) -> Tuple[List[List[float]], List[int]]:
        """
        Read annotations for a single nuImages sample and return 2D bounding boxes and class indices.

        :param fname: Sample token or filename.
        :type fname: str
        :return: Tuple containing:
                - List of bounding boxes ``[[x1, y1, x2, y2], ...]``.
                - List of corresponding class indices.
        :rtype: Tuple[List[List[float]], List[int]]
        """

        if isinstance(fname, str) and ("/" in fname or "\\" in fname):
            fname = os.path.basename(fname)

        sample_data = self.nuim.get("sample_data", fname)
        image_width = float(sample_data["width"])
        image_height = float(sample_data["height"])

        boxes_out = []
        labels_out = []

        for ann in self.ann_by_sample_data_token.get(fname, []):

            category_name = self.nuim.get("category", ann["category_token"])["name"]

            if category_name in DROP:
                continue

            if category_name not in self.cat_to_idx:
                continue

            x1_raw, y1_raw, x2_raw, y2_raw = ann["bbox"]

            x1 = max(0.0, min(float(x1_raw), image_width - 1.0))
            y1 = max(0.0, min(float(y1_raw), image_height - 1.0))
            x2 = max(0.0, min(float(x2_raw), image_width - 1.0))
            y2 = max(0.0, min(float(y2_raw), image_height - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_out.append([x1, y1, x2, y2])
            labels_out.append(self.cat_to_idx[category_name])

        return boxes_out, labels_out


nuimages_detection = NuImagesDetectionDataset


class NuImagesSegmentationDataset(segmentation_dataset.ImageSegmentationDataset):
    """Dataset class for nuImages 2D surface segmentation.
    Inherits from ImageSegmentationDataset and constructs semantic segmentation masks

    param dataset_dir: Path to the nuImages dataset root.
    :type dataset_dir: str
    :param version: nuImages version to load, defaults to "v1.0-mini".
    :type version: str
    :param split: Dataset split ("train" or "val").
    :type split: str
    :param labels_rel_dir: Relative directory within the dataset where generated segmentation label images will be stored, defaults to "generated/nuimages_segmentation_labels".
    :type labels_rel_dir: str

    """

    def __init__(
        self,
        dataset_dir: str,
        version: str = "v1.0-mini",
        split: str = "train",
        labels_rel_dir: str = "generated/nuimages_segmentation_labels",
    ):
        dataset_dir = os.path.abspath(dataset_dir)
        assert os.path.isdir(
            dataset_dir
        ), f"Dataset directory {dataset_dir} does not exist."

        self.nuim = NuImages(dataroot=dataset_dir, version=version)
        dataset, ontology = build_nuimages_segmentation_dataset(
            dataset_dir=dataset_dir,
            version=version,
            split=split,
            labels_rel_dir=labels_rel_dir,
            nuim_object=self.nuim,
        )

        super().__init__(dataset, dataset_dir, ontology)



import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_dir = "local/data/nuimages-v1.0-mini"
    version = "v1.0-mini"
    split = "train"
    dataset = NuImagesSegmentationDataset(dataset_dir, version, split)

    sample = dataset.dataset.iloc[10]
    label_path = sample["label"]
    img_path = sample["image"]

    # Load images
    img = cv2.imread(img_path)
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    bright_colors = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [255, 255, 0],  # yellow
        [255, 0, 255],  # magenta
        [0, 255, 255],  # cyan
        [255, 128, 0],  # orange
        [128, 0, 255],  # purple
        [0, 128, 255],  # sky blue
        [128, 255, 0],  # lime
        [255, 0, 128],  # pink
        [0, 255, 128],  # teal
    ]

    # Create color mask
    color_mask = np.zeros_like(img)
    for class_name, class_info in dataset.ontology.items():
        class_idx = class_info["idx"]
        rgb = bright_colors[class_idx % len(bright_colors)]
        color_mask[label_img == class_idx] = rgb

    # Blend original image with color mask
    overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)

    # Draw class names
    for class_name, class_info in dataset.ontology.items():
        class_idx = class_info["idx"]
        positions = np.argwhere(label_img == class_idx)
        if positions.shape[0] == 0:
            continue
        y, x = np.mean(positions, axis=0).astype(int)
        cv2.putText(
            overlay,
            class_name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Show overlay
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Image with Segmentation Overlay and Labels")
    plt.show()

