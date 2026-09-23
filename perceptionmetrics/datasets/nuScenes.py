import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes

from perceptionmetrics.datasets.detection import LiDARDetectionDataset


def build_nuscenes_lidar_detection_dataset(
    dataset_dir: str,
    version: str = "v1.0-mini",
    split: str = "train",
    nusc_object: Optional[NuScenes] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Build a nuScenes LiDAR detection dataset index.

    Each row stores the LiDAR point cloud path and the corresponding
    sample_data token used to retrieve 3D annotations.

    :param dataset_dir: Path to the nuScenes dataset root directory
    :type dataset_dir: str
    :param version: nuScenes dataset version, defaults to "v1.0-mini"
    :type version: str
    :param split: Dataset split to assign to the indexed samples
    :type split: str
    :param nusc_object: Optional pre-initialized NuScenes object
    :type nusc_object: Optional[NuScenes]
    :return: Dataset DataFrame and ontology dictionary
    :rtype: Tuple[pd.DataFrame, dict]
    """
    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(
        dataset_dir
    ), f"Dataset directory {dataset_dir} does not exist."

    nusc = (
        nusc_object
        if nusc_object
        else NuScenes(version=version, dataroot=dataset_dir, verbose=False)
    )

    ontology = {
        category["name"]: {"idx": idx + 1, "rgb": [0, 0, 0]}
        for idx, category in enumerate(nusc.category)
    }

    rows = []
    sample_names = []
    for sample in nusc.sample:
        lidar_token = sample["data"].get("LIDAR_TOP")
        if lidar_token is None:
            continue

        sample_data = nusc.get("sample_data", lidar_token)
        rows.append(
            {
                "points": os.path.join(dataset_dir, sample_data["filename"]),
                "annotation": lidar_token,
                "split": split,
            }
        )
        sample_names.append(sample["token"])

    dataset = pd.DataFrame(rows, index=sample_names)
    dataset.attrs = {"ontology": ontology}

    return dataset, ontology


class NuScenesLiDARDetectionDataset(LiDARDetectionDataset):
    """Dataset class for nuScenes LiDAR 3D object detection.

    The annotation boxes are returned as [x, y, z, dx, dy, dz, yaw] in the
    LiDAR sensor frame.

    :param dataset_dir: Path to the nuScenes dataset root directory
    :type dataset_dir: str
    :param version: nuScenes dataset version, defaults to "v1.0-mini"
    :type version: str
    :param split: Dataset split to assign to the indexed samples
    :type split: str
    """

    def __init__(
        self,
        dataset_dir: str,
        version: str = "v1.0-mini",
        split: str = "train",
    ):
        dataset_dir = os.path.abspath(dataset_dir)
        assert os.path.isdir(
            dataset_dir
        ), f"Dataset directory {dataset_dir} does not exist."

        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=False)
        dataset, ontology = build_nuscenes_lidar_detection_dataset(
            dataset_dir=dataset_dir,
            version=version,
            split=split,
            nusc_object=self.nusc,
        )
        self.cat_to_idx = {
            category_name: class_data["idx"]
            for category_name, class_data in ontology.items()
        }

        super().__init__(
            dataset=dataset,
            dataset_dir=dataset_dir,
            ontology=ontology,
            is_kitti_format=False,
        )

    def make_fname_global(self):
        """Make point cloud paths absolute while keeping annotation tokens intact."""
        if self.dataset_dir is not None:
            self.dataset["points"] = self.dataset["points"].apply(
                lambda x: os.path.join(self.dataset_dir, x)
                if x is not None and not os.path.isabs(x)
                else x
            )
            self.dataset_dir = None

    def read_points(self, fname: str) -> np.ndarray:
        """Read nuScenes LiDAR points.

        nuScenes LiDAR .bin files store x, y, z, intensity, and ring index.
        The ring index is dropped so the returned points match the usual
        x, y, z, intensity format.
        """
        return np.fromfile(fname, dtype=np.float32).reshape(-1, 5)[:, :4]

    def read_annotation(self, fname: str):
        """Read 3D annotations for a nuScenes LiDAR sample-data token.

        :param fname: nuScenes sample_data token
        :type fname: str
        :return: LiDAR-frame boxes in [x, y, z, dx, dy, dz, yaw] format and class IDs
        :rtype: Tuple[list, list]
        """
        _, sample_boxes, _ = self.nusc.get_sample_data(fname)

        boxes = []
        labels = []
        for sample_box in sample_boxes:
            category_name = sample_box.name
            if category_name not in self.cat_to_idx:
                continue

            box = [
                *sample_box.center,
                *sample_box.wlh,
                quaternion_yaw(sample_box.orientation),
            ]
            boxes.append(box)
            labels.append(self.cat_to_idx[category_name])

        return boxes, labels




if __name__ == "__main__":
    dataset = NuScenesLiDARDetectionDataset(
        dataset_dir="examples/local/nuscenes/v1.0-mini",
        version="v1.0-mini",
        split="train",
    )

    print(dataset.dataset.head())

    sample_token = dataset.dataset.index[0]
    print (f"points path: {dataset.dataset.loc[sample_token, 'points']}")
    points = dataset.read_points(dataset.dataset.loc[sample_token, "points"])
    boxes, labels = dataset.read_annotation(dataset.dataset.loc[sample_token, "annotation"])
    print (f"Sample token: {sample_token}")
    print(f"Points shape: {points.shape}")
    print(f"Boxes: {len(boxes)}, Labels: {len(labels)}")
    print (f"ontology: {dataset.ontology}")
