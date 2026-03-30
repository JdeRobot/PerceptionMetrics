REGISTRY = {}

try:
    from perceptionmetrics.models.torch_segmentation import (
        TorchImageSegmentationModel,
        TorchLiDARSegmentationModel,
    )

    REGISTRY["torch_image_segmentation"] = TorchImageSegmentationModel
    REGISTRY["torch_lidar_segmentation"] = TorchLiDARSegmentationModel
except (ImportError, OSError):
    print("Torch not available")

try:
    from perceptionmetrics.models.torch_detection import TorchImageDetectionModel

    REGISTRY["torch_image_detection"] = TorchImageDetectionModel
except (ImportError, OSError):
    print("Torch detection not available")

try:
    from perceptionmetrics.models.tf_segmentation import TensorflowImageSegmentationModel

    REGISTRY["tensorflow_image_segmentation"] = TensorflowImageSegmentationModel
except (ImportError, OSError):
    print("Tensorflow not available")

if not REGISTRY:
    raise ImportError("No valid deep learning framework found")
