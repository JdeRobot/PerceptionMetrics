import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from perceptionmetrics.datasets.detection import ImageDetectionDataset

# Classes to drop, keep it empty keep all classes
DROP = {
    "animal",
    "movable_object.pushable_pullable",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "human.pedestrian.personal_mobility",
    "movable_object.debris",
    "static_object.bicycle_rack",
    "vehicle.trailer",
    "movable_object.barrier",
    "movable_object.trafficcone",
}


def build_nuscenes_detection_dataset(
    dataset_dir: str,
    version: str = "v1.0-mini",
    camera: str = "CAM_FRONT",
    split: str = "train",
    nusc_obj: Optional[NuScenes] = None,
) -> Tuple[pd.DataFrame, dict]:
    
    """
    Build a nuScenes 2D detection dataset index.
    
    Iterates through the nuScenes scenes and samples, collects image paths for a given camera, and constructs a dataset index along with a category ontology mapping class names to integer indices.
    
    :param dataset_dir: Path to the nuScenes dataset root directory.
    :type dataset_dir: str
    :param version: nuScenes dataset version to load, defaults to "v1.0-mini".
    :type version: str
    :param camera: Camera channel to use (e.g., "CAM_FRONT"), defaults to "CAM_FRONT".
    :type camera: str
    :param split: Dataset split to load ("train" or "val"), defaults to "train".
    :type split: str
    :param nusc_obj: Optional pre-initialized NuScenes object to reuse, defaults to None.
    :type nusc_obj: Optional[NuScenes]
    :return: Tuple containing:
             - A pandas DataFrame with columns ["image", "annotation", "split"] for each sample.
             - An ontology dictionary mapping category names to indices and RGB colors.
    :rtype: Tuple[pd.DataFrame, dict]
    """

    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(dataset_dir), f"Dataset directory not found: {dataset_dir}"

    nusc = (
        nusc_obj
        if nusc_obj
        else NuScenes(version=version, dataroot=dataset_dir, verbose=False)
    )

    # Get all nuScenes categories except the DROP list
    all_categories = [cat["name"] for cat in nusc.category]
    categories = [c for c in all_categories if c not in DROP]

    # Build ontology
    ontology = {
        name: {"idx": i + 1, "rgb": [0, 0, 0]} for i, name in enumerate(categories)
    }
    cat_to_idx = {name: ontology[name]["idx"] for name in ontology}

    # Train/val split
    scene_tokens = [s["token"] for s in nusc.scene]
    if version == "v1.0-mini":
        train_tokens = scene_tokens[:8]
        val_tokens = scene_tokens[8:]
    else:
        train_tokens = scene_tokens
        val_tokens = []

    rows = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        if split == "train" and scene_token not in train_tokens:
            continue
        if split == "val" and scene_token not in val_tokens:
            continue

        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            cam_token = sample["data"].get(camera)
            if cam_token is None:
                token = sample["next"]
                continue

            sd = nusc.get("sample_data", cam_token)
            rows.append(
                {
                    "image": os.path.join(dataset_dir, sd["filename"]),
                    "annotation": sample["token"],
                    "split": split,
                }
            )

            token = sample["next"]

    dataset = pd.DataFrame(rows)
    dataset.attrs = {"ontology": ontology, "cat_to_idx": cat_to_idx}

    print(
        f"Built nuScenes detection dataset with {len(dataset)} samples for split '{split}'"
    )
    return dataset, ontology


# Dataset class
class NuScenesDetectionDataset(ImageDetectionDataset):
    """
    Dataset class for nuScenes 2D object detection.

    Inherits from ImageDetectionDataset and parses 3D bounding boxes
    from nuScenes into 2D camera view, dropping unwanted classes.

    :param dataset_dir: Path to the nuScenes dataset root.
    :type dataset_dir: str
    :param camera: Camera channel to use (e.g., "CAM_FRONT").
    :type camera: str
    :param split: Dataset split ("train" or "val").
    :type split: str
    :param nusc: Initialized NuScenes object.
    :type nusc: NuScenes
    :param cat_to_idx: Mapping from category names to integer indices.
    :type cat_to_idx: dict
    """

    def __init__(
        self,
        dataset_dir: str,
        version: str = "v1.0-mini",
        camera: str = "CAM_FRONT",
        split: str = "train",
    ):
        """
        Initialize the nuScenes 2D detection dataset.

        :param dataset_dir: Path to the nuScenes dataset root directory.
        :param version: nuScenes dataset version to load, defaults to "v1.0-mini".
        :param camera: Camera channel to use (e.g., "CAM_FRONT"), defaults to "CAM_FRONT".
        :param split: Dataset split to load ("train" or "val"), defaults to "train".
        """
        self.dataset_dir = dataset_dir
        self.camera = camera
        self.split = split
        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=False)

        dataset, ontology = build_nuscenes_detection_dataset(
            dataset_dir, version=version, camera=camera, split=split, nusc_obj=self.nusc
        )

        self.cat_to_idx = dataset.attrs["cat_to_idx"]

        super().__init__(dataset=dataset, dataset_dir=dataset_dir, ontology=ontology)

    def read_annotation(self, fname: str) -> Tuple[List[List[float]], List[int]]:
        """
        Read annotations for a single nuScenes sample and project 3D boxes
        into 2D bounding boxes in the selected camera view.

        :param fname: Sample token or filename.
        :type fname: str
        :return: Tuple containing:
                - List of bounding boxes ``[[x1, y1, x2, y2], ...]``.
                - List of corresponding class indices.
        :rtype: Tuple[List[List[float]], List[int]]
        """

        # clean token
        if isinstance(fname, str) and ("/" in fname or "\\" in fname):
            fname = os.path.basename(fname)

        sample = self.nusc.get("sample", fname)
        cam_token = sample["data"][self.camera]

        image_path = self.nusc.get_sample_data_path(cam_token)
        img = Image.open(image_path)
        H, W = img.height, img.width

        _, boxes, cam_intrinsic = self.nusc.get_sample_data(cam_token, box_vis_level=0)

        boxes_out, labels_out = [], []

        for box in boxes:
            raw_name = box.name
            if raw_name in DROP:
                continue

            class_name = raw_name  # keep canonical nuScenes label
            if class_name not in self.cat_to_idx:
                continue

            corners = view_points(box.corners(), cam_intrinsic, normalize=True)

            x1 = np.clip(corners[0].min(), 0, W - 1)
            y1 = np.clip(corners[1].min(), 0, H - 1)
            x2 = np.clip(corners[0].max(), 0, W - 1)
            y2 = np.clip(corners[1].max(), 0, H - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_out.append([float(x1), float(y1), float(x2), float(y2)])
            labels_out.append(self.cat_to_idx[class_name])

        return boxes_out, labels_out


# Example usage
# if __name__ == "__main__":
#     nus = NuScenesDetectionDataset(version="v1.0-mini", dataset_dir="/home/tejass/Downloads/JDE_Robotics/v1.0-mini")
#     image_path = nus.dataset.iloc[0]['image']
#     annotation_token = nus.dataset.iloc[0]['annotation']

#     # Example debug: draw boxes
#     import cv2
#     img = cv2.imread(image_path)
#     boxes, labels = nus.read_annotation(annotation_token)
#     for box in boxes:
#         img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#     cv2.imwrite("debug_nuscenes.jpg", img)
