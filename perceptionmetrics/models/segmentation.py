from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from perceptionmetrics.datasets import segmentation as segmentation_dataset
from perceptionmetrics.models.perception import PerceptionModel
import perceptionmetrics.utils.conversion as uc
import perceptionmetrics.utils.io as uio


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

    def get_eval_lut_ontology(
        self,
        dataset_ontology: dict,
        ontology_translation: Optional[str] = None,
        translation_direction: str = "dataset_to_model",
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Build the ontology conversion LUT. When no translation file is given, both
        ontologies are matched by class name: shared names with differing indices are
        converted after issuing a warning, while differing names raise an error.

        :param dataset_ontology: Image or LiDAR dataset ontology
        :type dataset_ontology: dict
        :param ontology_translation: JSON file containing translation between dataset and model ontologies, defaults to None
        :type ontology_translation: Optional[str], optional
        :param translation_direction: Direction of the ontology translation, either 'dataset_to_model' or 'model_to_dataset', defaults to "dataset_to_model"
        :type translation_direction: str, optional
        :return: LUT converting indices in the requested direction and the ontology to be used for evaluation
        :rtype: Tuple[Optional[np.ndarray], dict]
        """
        if translation_direction == "dataset_to_model":
            old_ontology, new_ontology = dataset_ontology, self.ontology
        elif translation_direction == "model_to_dataset":
            old_ontology, new_ontology = self.ontology, dataset_ontology
        else:
            raise ValueError("Invalid translation direction")
        
        eval_ontology = new_ontology

        if ontology_translation is not None:
            ontology_translation = uio.read_json(ontology_translation)
        else:
            dataset_only = sorted(set(dataset_ontology) - set(self.ontology))
            model_only = sorted(set(self.ontology) - set(dataset_ontology))
            if dataset_only or model_only:
                raise ValueError(
                    "Dataset and model ontologies do not share the same class names, "
                    "so 'ontology_translation' is required to evaluate them together. "
                )

            if all(
                dataset_ontology[class_name]["idx"] == self.ontology[class_name]["idx"]
                for class_name in dataset_ontology
            ):
                return None, eval_ontology

            warnings.warn(
                "Dataset and model ontologies share class names but use different "
                "class indices. A conversion LUT will be built to match the indices."
            )

        lut_ontology = uc.get_ontology_conversion_lut(
            old_ontology, new_ontology, ontology_translation
        )
        
        return lut_ontology, eval_ontology

    @abstractmethod
    def predict(
        self, data: Union[np.ndarray, Image.Image], return_sample: bool = False
    ) -> Union[np.ndarray, Image.Image, Tuple[np.ndarray, Image.Image]]:
        """Perform prediction for a single data sample

        :param data: Input data sample
        :type data: Union[np.ndarray, Image.Image]
        :param return_sample: Whether to return the sample data along with predictions, defaults to False
        :type return_sample: bool, optional
        :return: Prediction result or a tuple with the detection result and the input sample tensor
        :rtype: Union[np.ndarray, Image.Image, Tuple[np.ndarray, Image.Image]]
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self, tensor_in):
        """Perform inference for a tensor

        :param tensor_in: Input tensor (image or point cloud)
        :type tensor_in: Either tf.Tensor or torch.Tensor
        :return: Segmenation result as a tensor
        :rtype: Either tf.Tensor or torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: segmentation_dataset.SegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        translations_direction: str = "dataset_to_model",
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param translations_direction: Direction of the ontology translation. Either "dataset_to_model" or "model_to_dataset", defaults to "dataset_to_model"
        :type translations_direction: str, optional
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
    def predict(
        self, image: Image.Image, return_sample: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Any]]:
        """Perform prediction for a single image

        :param image: PIL image
        :type image: Image.Image
        :param return_sample: Whether to return the sample data along with predictions, defaults to False
        :type return_sample: bool, optional
        :return: Segmentation result as a PIL image or a tuple with the segmentation result and the input sample tensor
        :rtype: Union[Image.Image, Tuple[Image.Image, Any]]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: segmentation_dataset.ImageSegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        translations_direction: str = "dataset_to_model",
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param translations_direction: Direction of the ontology translation. Either "dataset_to_model" or "model_to_dataset", defaults to "dataset_to_model"
        :type translations_direction: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def get_computational_cost(
        self,
        image_size: Tuple[int] = None,
        runs: int = 30,
        warm_up_runs: int = 5,
    ) -> dict:
        """Get different metrics related to the computational cost of the model

        :param image_size: Image size used for inference
        :type image_size: Tuple[int], optional
        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
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
    def predict(
        self,
        points_fname: str,
        has_intensity: bool = True,
        return_sample: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """Perform prediction for a single point cloud

        :param points_fname: Point cloud in SemanticKITTI .bin format
        :type points_fname: str
        :param has_intensity: Whether the point cloud has intensity values, defaults to True
        :type has_intensity: bool, optional
        :param return_sample: Whether to return the sample data along with predictions, defaults to False
        :type return_sample: bool, optional
        :return: Segmentation result as a numpy array or a tuple with the segmentation result and the input sample data
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Any]]
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        dataset: segmentation_dataset.LiDARSegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        translations_direction: str = "dataset_to_model",
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for a LiDAR segmentation dataset

        :param dataset: LiDAR segmentation dataset for which the evaluation will be performed
        :type dataset: LiDARSegmentationDataset
        :param split: Split or splits to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :param translations_direction: Direction of the ontology translation. Either "dataset_to_model" or "model_to_dataset", defaults to "dataset_to_model"
        :type translations_direction: str, optional
        :param predictions_outdir: Directory to save predictions per sample, defaults to None. If None, predictions are not saved.
        :type predictions_outdir: Optional[str], optional
        :param results_per_sample: Whether to store results per sample or not, defaults to False. If True, predictions_outdir must be provided.
        :type results_per_sample: bool, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def get_computational_cost(self, runs: int = 30, warm_up_runs: int = 5) -> dict:
        """Get different metrics related to the computational cost of the model

        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
        """
        raise NotImplementedError
