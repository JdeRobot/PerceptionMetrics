from abc import ABC, abstractmethod
import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

import perceptionmetrics.utils.conversion as uc
import perceptionmetrics.utils.io as uio


class PerceptionModel(ABC):
    """Base class for all vision perception models (e.g., segmentation, detection).

    :param model: Model object
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
        self.model = model
        self.model_type = model_type
        self.model_fname = model_fname

        # Check that provided paths exist
        assert os.path.isfile(ontology_fname), "Ontology file not found"
        assert os.path.isfile(model_cfg), "Model configuration not found"
        if self.model_fname is not None:
            assert os.path.exists(model_fname), "Model file or directory not found"

        # Read ontology and model configuration
        self.ontology = uio.read_json(ontology_fname)
        self.model_cfg = uio.read_json(model_cfg)
        self.n_classes = len(self.ontology)
        self.model_cfg["n_classes"] = self.n_classes

    @abstractmethod
    def inference(
        self, data: Union[np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image, dict]:
        """Perform inference for a single image or point cloud."""
        raise NotImplementedError

    @abstractmethod
    def eval(self, *args, **kwargs) -> pd.DataFrame:
        """Evaluate the model on the given dataset."""
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

    def get_lut_ontology(
        self, dataset_ontology: dict, ontology_translation: Optional[str] = None
    ):
        """Build ontology lookup table (leave empty if ontologies match)

        :param dataset_ontology: Image or LiDAR dataset ontology
        :type dataset_ontology: dict
        :param ontology_translation: JSON file containing translation between model and dataset ontologies, defaults to None
        :type ontology_translation: Optional[str], optional
        """
        lut_ontology = None
        if dataset_ontology != self.ontology:
            if ontology_translation is not None:
                ontology_translation = uio.read_json(ontology_translation)
            lut_ontology = uc.get_ontology_conversion_lut(
                dataset_ontology,
                self.ontology,
                ontology_translation,
                classes_to_remove=self.model_cfg.get("classes_to_remove", None),
            )
        return lut_ontology
