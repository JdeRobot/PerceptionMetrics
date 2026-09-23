from unittest.mock import patch

import numpy as np
import pandas as pd

from perceptionmetrics.datasets.nuimages import (
    NuImagesDetectionDataset,
    NuImagesSegmentationDataset,
    build_nuimages_detection_dataset,
    build_nuimages_segmentation_dataset,
)


class _FakeNuImages:
    category = [
        {"token": "cat_car", "name": "vehicle.car"},
        {"token": "cat_person", "name": "human.pedestrian.adult"},
    ]
    sample = [{"key_camera_token": "sample_data_1"}]
    object_ann = [
        {
            "sample_data_token": "sample_data_1",
            "category_token": "cat_car",
            "bbox": [-10, 5, 120, 200],
        },
        {
            "sample_data_token": "sample_data_1",
            "category_token": "cat_person",
            "bbox": [50, 60, 50, 100],
        },
    ]

    def __init__(self, dataroot=None, version=None, verbose=None):
        self.dataroot = dataroot
        self.version = version

    def get(self, table_name, token):
        if table_name == "sample_data":
            return {
                "filename": "samples/CAM_FRONT/sample.jpg",
                "width": 100,
                "height": 80,
            }
        if table_name == "category":
            return {
                "cat_car": {"name": "vehicle.car"},
                "cat_person": {"name": "human.pedestrian.adult"},
            }[token]
        raise KeyError(table_name)

    def get_segmentation(self, token):
        return np.array([[0, 1], [2, 1]], dtype=np.uint8), None


def test_nuimages_segmentation_dataset(tmp_path) -> None:
    """Regression tests for nuImages segmentation dataset building.

    Verifies that:
    - segmentation rows point to image and generated label paths;
    - background and category ontology IDs are created;
    - generated masks are written;
    - the wrapper initializes the base dataset with the built data.
    """
    with patch(
        "perceptionmetrics.datasets.nuimages.name_to_index_mapping",
        return_value={"vehicle.car": 1, "human.pedestrian.adult": 2},
    ):
        with patch("perceptionmetrics.datasets.nuimages.cv2.imwrite") as imwrite:
            dataset, ontology = build_nuimages_segmentation_dataset(
                dataset_dir=str(tmp_path),
                version="v1.0-mini",
                split="train",
                labels_rel_dir="generated/labels",
                nuim_object=_FakeNuImages(),
            )

    assert isinstance(dataset, pd.DataFrame)
    assert dataset.iloc[0]["image"].endswith("samples/CAM_FRONT/sample.jpg")
    assert dataset.iloc[0]["label"].endswith(
        "generated/labels/v1.0-mini/sample_data_1.png"
    )
    assert dataset.iloc[0]["split"] == "train"
    assert ontology["background"]["idx"] == 0
    assert ontology["vehicle.car"]["idx"] == 1
    imwrite.assert_called_once()
    assert np.array_equal(imwrite.call_args.args[1], np.array([[0, 1], [2, 1]]))

    with patch(
        "perceptionmetrics.datasets.nuimages.NuImages", return_value=_FakeNuImages()
    ):
        with patch(
            "perceptionmetrics.datasets.nuimages.build_nuimages_segmentation_dataset",
            return_value=(dataset, ontology),
        ):
            wrapped_dataset = NuImagesSegmentationDataset(
                dataset_dir=str(tmp_path),
                version="v1.0-mini",
                split="train",
            )

    assert wrapped_dataset.dataset.equals(dataset)
    assert wrapped_dataset.dataset_dir == str(tmp_path)
    assert wrapped_dataset.ontology == ontology


def test_nuimages_detection_dataset(tmp_path) -> None:
    """Regression tests for nuImages detection dataset building.

    Verifies that:
    - detection rows store sample-data tokens as annotations;
    - ontology and category lookup are built;
    - bounding boxes are clipped to image bounds;
    - invalid boxes are skipped.
    """
    dataset, ontology = build_nuimages_detection_dataset(
        dataset_dir=str(tmp_path),
        version="v1.0-mini",
        split="val",
        nuim_object=_FakeNuImages(),
    )

    assert isinstance(dataset, pd.DataFrame)
    assert dataset.iloc[0]["annotation"] == "sample_data_1"
    assert dataset.iloc[0]["split"] == "val"
    assert dataset.attrs["cat_to_idx"]["vehicle.car"] == ontology["vehicle.car"]["idx"]

    with patch(
        "perceptionmetrics.datasets.nuimages.NuImages", return_value=_FakeNuImages()
    ):
        wrapped_dataset = NuImagesDetectionDataset(
            dataset_dir=str(tmp_path),
            version="v1.0-mini",
            split="train",
        )

    boxes, labels = wrapped_dataset.read_annotation("sample_data_1")

    assert boxes == [[0.0, 5.0, 99.0, 79.0]]
    assert labels == [wrapped_dataset.cat_to_idx["vehicle.car"]]
