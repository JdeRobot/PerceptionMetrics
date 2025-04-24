from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    Pack3DDetInputs,
)
from mmengine.registry import FUNCTIONS
from torchvision.transforms import Compose

COLLATE_FN = FUNCTIONS.get("pseudo_collate")


def preprocess(sample):
    n_feats = sample["num_pts_feats"]
    transforms = [
        LoadPointsFromFile(coord_type="LIDAR", load_dim=4, use_dim=n_feats)
    ]
    if sample["pts_semantic_mask_path"] is not None:
        transforms.append(
            LoadAnnotations3D(
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype="np.uint32",
                seg_offset=65536,
                dataset_type="semantickitti",
            )
        )
    transforms.append(
        Pack3DDetInputs(
            keys=["points", "pts_semantic_mask"],
            meta_keys=["sample_idx", "lidar_path", "num_pts_feats", "sample_id"],
        )
    )
    transforms = Compose(transforms)
    return transforms(sample)


def inference(sample, model):
    single_sample = not isinstance(sample["data_samples"], list)
    if single_sample:
        sample = COLLATE_FN([sample])

    sample = model.data_preprocessor(sample, training=False)
    inputs, data_samples = sample["inputs"], sample["data_samples"]
    pred = model(inputs, data_samples, mode="predict")

    if single_sample:
        pred = pred[0]

    return pred
