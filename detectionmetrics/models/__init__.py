REGISTRY = {}

try:
    from detectionmetrics.models.torch_segmentation import (
        TorchImageSegmentationModel,
        TorchLiDARSegmentationModel,
    )

    REGISTRY["torch_image_segmentation"] = TorchImageSegmentationModel
    REGISTRY["torch_lidar_segmentation"] = TorchLiDARSegmentationModel
except ImportError:
    print("Torch not available")

try:
    from detectionmetrics.models.torch_detection import TorchImageDetectionModel

    REGISTRY["torch_image_detection"] = TorchImageDetectionModel
except ImportError:
    print("Torch detection not available")

try:
    from detectionmetrics.models.tensorflow import TensorflowImageSegmentationModel

    REGISTRY["tensorflow_image_segmentation"] = TensorflowImageSegmentationModel
except ImportError:
    print("Tensorflow not available")

if not REGISTRY:
    raise Exception("No valid deep learning framework found")
