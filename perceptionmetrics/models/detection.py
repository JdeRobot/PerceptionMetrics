from abc import abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from perceptionmetrics.datasets import detection as detection_dataset
from perceptionmetrics.models.perception import PerceptionModel


class DetectionModel(PerceptionModel):
    """Parent detection model class

    :param model: Detection model object
    :type model: Any
    :param model_type: Model type (e.g. scripted, compiled, etc.)
    :type model_type: str
    :param model_cfg: JSON file containing model configuration
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_fname: Model file or directory, defaults to None
    :type model_fname: Optional[str], optional
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        model_cfg: str,
        ontology_fname: str,
        model_fname: Optional[str] = None,
    ):
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)

    @abstractmethod
    def inference(self, data: Union[np.ndarray, Image.Image]) -> List[dict]:
        """Perform inference for a single input (image or point cloud)

        :param data: Input image or LiDAR point cloud
        :type data: Union[np.ndarray, Image.Image]
        :return: List of detection results (each a dict with bbox, confidence, class)
        :rtype: List[dict]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: detection_dataset.DetectionDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a detection dataset

        :param dataset: Detection dataset for which evaluation will be performed
        :type dataset: ImageDetecctionDataset
        :param split: Split(s) to use, defaults to "test"
        :type split: Union[str, List[str]]
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: Optional[str]
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str]
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool
        :return: DataFrame containing evaluation metrics
        :rtype: pd.DataFrame
        """
        raise NotImplementedError


class ImageDetectionModel(DetectionModel):
    """Parent image detection model class

    :param model: Detection model object
    :type model: Any
    :param model_type: Model type (e.g. scripted, compiled, etc.)
    :type model_type: str
    :param model_cfg: JSON file containing model configuration (e.g. image size or normalization parameters)
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_fname: Model file or directory, defaults to None
    :type model_fname: Optional[str], optional
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        model_cfg: str,
        ontology_fname: str,
        model_fname: Optional[str] = None,
    ):
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)

    @abstractmethod
    def inference(self, image: Image.Image) -> List[dict]:
        """Perform inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: List of detection results
        :rtype: List[dict]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: detection_dataset.ImageDetectionDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Evaluate the image detection model

        :param dataset: Image detection dataset for which the evaluation will be performed
        :type dataset: ImageDetectionDataset
        :param split: Split(s) to use, defaults to "test"
        :type split: Union[str, List[str]]
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: Optional[str]
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str]
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool
        :return: DataFrame containing evaluation metrics
        :rtype: pd.DataFrame
        """
        raise NotImplementedError


class LiDARDetectionModel(DetectionModel):
    """Parent LiDAR detection model class

    :param model: Detection model object
    :type model: Any
    :param model_type: Model type (e.g. scripted, compiled, etc.)
    :type model_type: str
    :param model_cfg: JSON file with model configuration
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_fname: Model file or directory, defaults to None
    :type model_fname: Optional[str], optional
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        model_cfg: str,
        ontology_fname: str,
        model_fname: Optional[str] = None,
    ):
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)

    @abstractmethod
    def inference(self, points: np.ndarray) -> List[dict]:
        """Perform inference for a single LiDAR point cloud

        :param points: N x 3 or N x 4 point cloud array
        :type points: np.ndarray
        :return: List of detection results
        :rtype: List[dict]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: detection_dataset.LiDARDetectionDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR detection dataset

        :param dataset: LiDAR detection dataset for which the evaluation will be performed
        :type dataset: LiDARDetectionDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]]
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: Optional[str]
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str]
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation metrics
        :rtype: pd.DataFrame
        """
        raise NotImplementedError
