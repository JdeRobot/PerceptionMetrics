import json

from perceptionmetrics.datasets.coco import build_coco_dataset


def _write_coco(tmp_path, file_names):
    """Write a minimal COCO annotation file with one box per image.

    :param tmp_path: Directory to write into
    :type tmp_path: pathlib.Path
    :param file_names: Image file names to list in the annotation file
    :type file_names: List[str]
    :return: Path to the annotation file and to the image directory
    :rtype: Tuple[str, str]
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images, annotations = [], []
    for i, name in enumerate(file_names, start=1):
        images.append({"id": i, "file_name": name, "width": 64, "height": 48})
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
            }
        )
    ann_file = tmp_path / "instances_val.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "helmet"}],
            }
        )
    )
    return str(ann_file), str(image_dir)


def test_build_coco_dataset_keeps_samples_with_dots_in_name(tmp_path):
    """Every image in the annotation file must get its own row, even when the
    file names only differ after the first dot (e.g. Roboflow exports).

    :param tmp_path: pytest temporary directory
    :type tmp_path: pathlib.Path
    """
    names = [
        "IMG_0001_jpg.rf.3f9a1c7e2b.jpg",
        "IMG_0001_jpg.rf.8b21d04aa9.jpg",
        "IMG_0002_jpg.rf.1a2b3c4d5e.jpg",
        "plain.jpg",
    ]
    ann_file, image_dir = _write_coco(tmp_path, names)

    dataset, _ = build_coco_dataset(ann_file, image_dir, split="val")

    assert len(dataset) == 4
    assert list(dataset["annotation"]) == ["1", "2", "3", "4"]
    assert "IMG_0001_jpg.rf.8b21d04aa9" in dataset.index
    assert "plain" in dataset.index
