from detectionmetrics.models.torch import (
    TorchImageSegmentationModel,
    TorchLiDARSegmentationModel,
)
from detectionmetrics.models.tensorflow import (
    TensorflowImageSegmentationModel,
)


REGISTRY = {
    "torch_image_segmentation": TorchImageSegmentationModel,
    "torch_lidar_segmentation": TorchLiDARSegmentationModel,
    "tensorflow_image_segmentation": TensorflowImageSegmentationModel,
}
