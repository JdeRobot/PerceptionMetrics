from perceptionmetrics.datasets.gaia import (
    GaiaImageSegmentationDataset,
    GaiaLiDARSegmentationDataset,
)
from perceptionmetrics.datasets.generic import (
    GenericImageSegmentationDataset,
    GenericLiDARSegmentationDataset,
)
from perceptionmetrics.datasets.goose import (
    GOOSEImageSegmentationDataset,
    GOOSELiDARSegmentationDataset,
)
from perceptionmetrics.datasets.rellis3d import (
    Rellis3DImageSegmentationDataset,
    Rellis3DLiDARSegmentationDataset,
)
from perceptionmetrics.datasets.rugd import RUGDImageSegmentationDataset
from perceptionmetrics.datasets.wildscenes import WildscenesImageSegmentationDataset
from perceptionmetrics.datasets.nuscenes_detection import NuScenesDetectionDataset
from perceptionmetrics.datasets.nuimages import NuImagesDetectionDataset
from perceptionmetrics.datasets.nuimages import NuImagesSegmentationDataset

try:
    from perceptionmetrics.datasets.coco import CocoDataset
except ImportError:
    print("COCO dataset dependencies not available")
    CocoDataset = None
REGISTRY = {
    "gaia_image_segmentation": GaiaImageSegmentationDataset,
    "gaia_lidar_segmentation": GaiaLiDARSegmentationDataset,
    "generic_image_segmentation": GenericImageSegmentationDataset,
    "generic_lidar_segmentation": GenericLiDARSegmentationDataset,
    "goose_image_segmentation": GOOSEImageSegmentationDataset,
    "goose_lidar_segmentation": GOOSELiDARSegmentationDataset,
    "rellis3d_image_segmentation": Rellis3DImageSegmentationDataset,
    "rellis3d_lidar_segmentation": Rellis3DLiDARSegmentationDataset,
    "rugd_image_segmentation": RUGDImageSegmentationDataset,
    "wildscenes_image_segmentation": WildscenesImageSegmentationDataset,
    "nuscenes_detection": NuScenesDetectionDataset,
    "nuimages_detection": NuImagesDetectionDataset,
    "nuimages_segmentation": NuImagesSegmentationDataset,
}

if CocoDataset is not None:
    REGISTRY["coco_image_detection"] = CocoDataset
