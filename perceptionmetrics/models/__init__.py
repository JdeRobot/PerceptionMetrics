from perceptionmetrics.utils.exception import PerceptionMetricsException
from perceptionmetrics.utils.logging_config import get_logger, add_file_handler

_logger = get_logger(__name__)
# add_file_handler("logs/run.log")

REGISTRY = {}

try:
    from perceptionmetrics.models.torch_segmentation import (
        TorchImageSegmentationModel,
        TorchLiDARSegmentationModel,
    )

    REGISTRY["torch_image_segmentation"] = TorchImageSegmentationModel
    REGISTRY["torch_lidar_segmentation"] = TorchLiDARSegmentationModel
except ImportError:
    _logger.warning("Torch not available – segmentation models disabled.")

try:
    from perceptionmetrics.models.torch_detection import TorchImageDetectionModel

    REGISTRY["torch_image_detection"] = TorchImageDetectionModel
except ImportError:
    _logger.warning("Torch detection not available – detection model disabled.")

try:
    from perceptionmetrics.models.tf_segmentation import TensorflowImageSegmentationModel

    REGISTRY["tensorflow_image_segmentation"] = TensorflowImageSegmentationModel
except ImportError:
    _logger.warning("TensorFlow not available – segmentation model disabled.")

if not REGISTRY:
    msg = "No valid deep learning framework found (Torch / TensorFlow missing)"
    _logger.error(msg)
    raise PerceptionMetricsException(Exception(msg))
