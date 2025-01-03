from detectionmetrics.datasets.gaia import (
    GaiaImageSegmentationDataset,
    GaiaLiDARSegmentationDataset,
)
from detectionmetrics.datasets.generic import (
    GenericImageSegmentationDataset,
    GenericLiDARSegmentationDataset,
)
from detectionmetrics.datasets.goose import (
    GOOSEImageSegmentationDataset,
    GOOSELiDARSegmentationDataset,
)
from detectionmetrics.datasets.rellis3d import (
    Rellis3DImageSegmentationDataset,
    Rellis3DLiDARSegmentationDataset,
)


REGISTRY = {
    "gaia_image_segmentation": GaiaImageSegmentationDataset,
    "gaia_lidar_segmentation": GaiaLiDARSegmentationDataset,
    "generic_image_segmentation": GenericImageSegmentationDataset,
    "generic_lidar_segmentation": GenericLiDARSegmentationDataset,
    "goose_image_segmentation": GOOSEImageSegmentationDataset,
    "goose_lidar_segmentation": GOOSELiDARSegmentationDataset,
    "rellis_image_segmentation": Rellis3DImageSegmentationDataset,
    "rellis_lidar_segmentation": Rellis3DLiDARSegmentationDataset,
}
