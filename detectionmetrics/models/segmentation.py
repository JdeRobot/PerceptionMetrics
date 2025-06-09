from abc import ABC, abstractmethod
import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image


from detectionmetrics.datasets import segmentation as dm_segentation_dataset
from detectionmetrics.models.perception import PerceptionModel
import detectionmetrics.utils.conversion as uc
import detectionmetrics.utils.io as uio

class SegmentationModel(PerceptionModel):
    """Parent segmentation model class

    :param model: Segmentation model object
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
    def inference(
        self, points: Union[np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        """Perform inference for a single image or point cloud

        :param image: Either a numpy array (LiDAR point cloud) or a PIL image
        :type image: Union[np.ndarray, Image.Image]
        :return: Segmenation result as a point cloud or image with label indices
        :rtype: Union[np.ndarray, Image.Image]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: dm_segentation_dataset.SegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: str | List[str], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError


class ImageSegmentationModel(SegmentationModel):
    """Parent image segmentation model class

    :param model: Image segmentation model object
    :type model: Any
    :param model_type: Model type (e.g. scripted, compiled, etc.)
    :type model_type: str
    :param model_cfg: JSON file containing model configuration (e.g. image size or normalization parameters)
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_fname: Model file or directory
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
    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image.
        :type image: Image.Image
        :return: Segmenation result as PIL image
        :rtype: Image.Image
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: dm_segentation_dataset.ImageSegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: str | List[str], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError


class LiDARSegmentationModel(SegmentationModel):
    """Parent LiDAR segmentation model class

    :param model: LiDAR segmentation model object
    :type model: Any
    :param model_type: Model type (e.g. scripted, compiled, etc.)
    :type model_type: str
    :param model_cfg: JSON file containing model configuration (e.g. sampling method, input format, etc.)
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_fname: Model file or directory
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
    def inference(self, points: np.ndarray) -> np.ndarray:
        """Perform inference for a single image

        :param image: Point cloud xyz array
        :type image: np.ndarray
        :return: Segmenation result as a point cloud with label indices
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: dm_segentation_dataset.LiDARSegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR segmentation dataset

        :param dataset: LiDAR segmentation dataset for which the evaluation will be performed
        :type dataset: LiDARSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: str | List[str], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

