import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tqdm import tqdm

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
from detectionmetrics.models.model import ImageSegmentationModel
import detectionmetrics.utils.metrics as um

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def get_computational_cost(
    model: tf.Module,
    dummy_input: tf.Tensor,
    model_fname: Optional[str] = None,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> dict:
    """Get different metrics related to the computational cost of the model

    :param model: Loaded TensorFlow SavedModel
    :type model: tf.Module
    :param dummy_input: Dummy input data for the model
    :type dummy_input: tf.Tensor
    :param model_fname: Model filename used to measure model size, defaults to None
    :type model_fname: Optional[str], optional
    :param runs: Number of runs to measure inference time, defaults to 30
    :type runs: int, optional
    :param warm_up_runs: Number of warm-up runs, defaults to 5
    :type warm_up_runs: int, optional
    :return: Dictionary containing computational cost information
    """
    # Get model size (if possible) and number of parameters
    if model_fname is not None:
        size_mb = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(model_fname)
            for f in files
        )
        size_mb /= 1024**2
    else:
        size_mb = None

    n_params = sum(np.prod(var.shape) for var in model.variables.variables)

    # Measure inference time with GPU synchronization
    infer = model.signatures["serving_default"]
    for _ in range(warm_up_runs):
        _ = infer(dummy_input)

    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    inference_times = []

    for _ in range(runs):
        if has_gpu:
            tf.config.experimental.set_synchronous_execution(True)

        start_time = time.time()
        _ = infer(dummy_input)

        if has_gpu:
            tf.config.experimental.set_synchronous_execution(True)

        inference_times.append(time.time() - start_time)

    # Retrieve computational cost information
    return {
        "input_shape": dummy_input.shape.as_list(),
        "size_mb": size_mb,
        "n_params": int(n_params),
        "inference_time_s": np.mean(inference_times),
    }


def resize_image(
    image: tf.Tensor,
    target_size: Tuple[int, int],
    method: str,
    keep_aspect: bool = False,
) -> tf.Tensor:
    """Resize tensorflow image to target size

    :param image: Input image tensor
    :type image: tf.Tensor
    :param target_size: Target size for the image
    :type target_size: Tuple[int, int]
    :param method: Resizing method (e.g. bilinear, nearest)
    :type method: str
    :param keep_aspect: Whether to keep aspect ratio when resizing images. If true, resize to match smaller sides size and crop center. Defaults to False
    :type keep_aspect: bool, optional
    :return: Resized image tensor
    :rtype: tf.Tensor
    """
    # If keep_aspect is True, resize to match smaller side
    if keep_aspect:
        original_size = tf.cast(tf.shape(image)[:2], tf.float32)
        resize_size = tf.cast(tf.convert_to_tensor(target_size), tf.float32)
        scale_factor = tf.reduce_max(resize_size / original_size)
        resize_size = tf.cast(tf.round(original_size * scale_factor), tf.int32)
    else:
        resize_size = target_size

    image = tf_image.resize(images=image, size=resize_size, method=method)

    # If keep_aspect is True, crop center to match target size
    if keep_aspect:
        y0 = (resize_size[0] - target_size[0]) // 2
        x0 = (resize_size[1] - target_size[1]) // 2
        image = tf_image.crop_to_bounding_box(
            image, y0, x0, target_size[1], target_size[0]
        )

    return image


class ImageSegmentationTensorflowDataset:
    """Dataset for image segmentation Tensorflow models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param image_size: Image size in pixels (width, height)
    :type image_size: Tuple[int, int]
    :param batch_size: Batch size, defaults to 1
    :type batch_size: int, optional
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    :param lut_ontology: LUT to transform label classes, defaults to None
    :type lut_ontology: dict, optional
    :param normalization: Parameters for normalizing input images, defaults to None
    :type normalization: dict, optional
    :param keep_aspect: Whether to keep aspect ratio when resizing images. If true, resize to match smaller sides size and crop center. Defaults to False
    :type keep_aspect: bool, optional
    """

    def __init__(
        self,
        dataset: ImageSegmentationDataset,
        image_size: Tuple[int, int],
        batch_size: int = 1,
        splits: List[str] = ["test"],
        lut_ontology: Optional[dict] = None,
        normalization: Optional[dict] = None,
        keep_aspect: bool = False,
    ):
        self.image_size = image_size
        self.normalization = None
        if normalization is not None:
            mean = tf.constant(normalization["mean"], dtype=tf.float32)
            std = tf.constant(normalization["std"], dtype=tf.float32)
            self.normalization = {"mean": mean, "std": std}
        self.keep_aspect = keep_aspect

        # Filter split and make filenames global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        dataset.make_fname_global()

        self.lut_ontology = None
        if lut_ontology is not None:
            keys = tf.constant(range(len(lut_ontology)), dtype=tf.int32)
            values = tf.constant(list(lut_ontology), dtype=tf.int32)
            self.lut_ontology = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
                default_value=0,
            )

        # Build tensorflow dataset
        sample = (
            dataset.dataset.index,
            dataset.dataset["image"],
            dataset.dataset["label"],
        )
        self.dataset = tf_data.Dataset.from_tensor_slices(sample)
        self.dataset = self.dataset.map(
            self.load_data, num_parallel_calls=tf_data.AUTOTUNE
        )
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

    def read_image(self, fname: str, label=False) -> tf.Tensor:
        """Read a single image or label

        :param fname: Input image or label filename
        :type fname: str
        :param label: Whether the input data is a label or not, defaults to False
        :type label: bool, optional
        :return: Tensorflow tensor containing read image or label
        :rtype: tf.Tensor
        """
        # Read image file
        image = tf_io.read_file(fname)

        # Decode image
        n_channels = 1 if label else 3
        image = tf_image.decode_png(image, channels=n_channels)
        image.set_shape([None, None, n_channels])

        # Apply LUT for transforming ontology if required
        if label and self.lut_ontology is not None:
            image = tf.cast(
                self.lut_ontology.lookup(tf.cast(image, tf.int32)), tf.uint8
            )

        # Resize (use NN to avoid interpolation when dealing with labels)
        method = "nearest" if label else "bilinear"
        image = resize_image(image, self.image_size, method, self.keep_aspect)

        # If label, round values to avoid interpolation artifacts
        if label:
            image = tf.round(image)

        # If normalization parameters are provided, normalize image
        else:
            if self.normalization is not None:
                image = tf.cast(image, tf.float32) / 255.0
                image = (image - self.normalization["mean"]) / self.normalization["std"]

        return image

    def load_data(
        self, idx: str, images_fnames: List[str], labels_fnames: List[str]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Function for loading data for each dataset sample

        :param idx: Sample index
        :type idx: str
        :param images_fnames: List containing all image filenames
        :type images_fnames: List[str]
        :param labels_fnames: List containing all corresponding label filenames
        :type labels_fnames: List[str]
        :return: Image and label tensor pairs
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        image = self.read_image(images_fnames)
        label = self.read_image(labels_fnames, label=True)
        return idx, image, label


class TensorflowImageSegmentationModel(ImageSegmentationModel):
    """Image segmentation model for Tensorflow framework


    :param model: Either the filename of a Tensorflow model in SavedModel format or the model already loaded into an arbitrary Tensorflow or Keras model.
    :type model: Union[str, torch.nn.Module]
    :param model_cfg: JSON file containing model configuration
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    """

    def __init__(
        self,
        model: Union[str, tf.Module, tf.keras.Model],
        model_cfg: str,
        ontology_fname: str,
    ):
        # If 'model' contains a string, check that it is a valid filename and load model
        if isinstance(model, str):
            assert os.path.isdir(model), "SavedModel directory not found"
            model_fname = model
            model = tf.saved_model.load(model)
            model_type = "compiled"
        # Otherwise, check that it is a Tensorflow or Keras model
        elif isinstance(model, tf.Module) or isinstance(model, tf.keras.Model):
            model_fname = None
            model_type = "native"
        else:
            raise ValueError(
                "Model must be either a SavedModel directory or a TF/Keras model"
            )

        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)

        # Init transformation for input images
        def t_in(image):
            tensor = tf.convert_to_tensor(image)
            tensor = resize_image(
                tensor,
                target_size=self.model_cfg["image_size"],
                method="bilinear",
                keep_aspect=self.model_cfg.get("keep_aspect", False),
            )
            tensor = tf.expand_dims(tensor, axis=0)
            if "normalization" in self.model_cfg:
                mean = tf.constant(self.model_cfg["normalization"]["mean"])
                std = tf.constant(self.model_cfg["normalization"]["std"])
                tensor = tf.cast(tensor, tf.float32) / 255.0
                tensor = (tensor - mean) / std
            return tensor

        self.t_in = t_in

        # Init transformation for model output
        self.t_out = lambda x: Image.fromarray(
            tf.argmax(tf.squeeze(x), axis=2).numpy().astype(np.uint8)
        )

    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: segmenation result as PIL image
        :rtype: Image.Image
        """
        tensor = self.t_in(image)

        if self.model_type == "native":
            result = self.model(tensor)
        elif self.model_type == "compiled":
            result = self.model.signatures["serving_default"](tensor)
        else:
            raise ValueError("Model type not recognized")

        if isinstance(result, dict):
            result = list(result.values())[0]

        return self.t_out(result)

    def eval(
        self,
        dataset: ImageSegmentationDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split to be used from the dataset, defaults to "test"
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
        # Check that predictions_outdir is provided if results_per_sample is True
        if results_per_sample and predictions_outdir is None:
            raise ValueError(
                "If results_per_sample is True, predictions_outdir must be provided"
            )

        # Create predictions output directory if needed
        if predictions_outdir is not None:
            os.makedirs(predictions_outdir, exist_ok=True)

        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        dataset_ontology = dataset.ontology

        # Get Tensorflow dataset
        dataset = ImageSegmentationTensorflowDataset(
            dataset,
            image_size=self.model_cfg["image_size"],
            batch_size=self.model_cfg.get("batch_size", 1),
            splits=[split] if isinstance(split, str) else split,
            lut_ontology=lut_ontology,
            normalization=self.model_cfg.get("normalization", None),
            keep_aspect=self.model_cfg.get("keep_aspect", False),
        )

        # Retrieve ignored label indices
        ignored_label_indices = []
        for ignored_class in self.model_cfg.get("ignored_classes", []):
            ignored_label_indices.append(dataset_ontology[ignored_class]["idx"])

        # Init metrics
        metrics_factory = um.MetricsFactory(self.n_classes)

        # Evaluation loop
        pbar = tqdm(dataset.dataset)
        for idx, image, label in pbar:
            idx = idx.numpy()

            if self.model_type == "native":
                pred = self.model(image, training=False)
            elif self.model_type == "compiled":
                pred = self.model.signatures["serving_default"](image)
            else:
                raise ValueError("Model type not recognized")

            if isinstance(pred, dict):
                pred = list(pred.values())[0]

            # Get valid points masks depending on ignored label indices
            if ignored_label_indices:
                valid_mask = tf.ones_like(label, dtype=tf.bool)
                for ignored_label_idx in ignored_label_indices:
                    valid_mask *= label != ignored_label_idx
            else:
                valid_mask = None

            # Update metrics
            label = tf.squeeze(label, axis=3).numpy()
            pred = tf.argmax(pred, axis=3).numpy()
            if valid_mask is not None:
                valid_mask = tf.squeeze(valid_mask, axis=3).numpy()

            metrics_factory.update(pred, label, valid_mask)

            # Store predictions and results per sample if required
            if predictions_outdir is not None:
                for i, (sample_idx, sample_pred, sample_label) in enumerate(
                    zip(idx, pred, label)
                ):
                    sample_idx = sample_idx.decode("utf-8")
                    if results_per_sample:
                        sample_valid_mask = (
                            valid_mask[i] if valid_mask is not None else None
                        )
                        sample_mf = um.MetricsFactory(n_classes=self.n_classes)
                        sample_mf.update(sample_pred, sample_label, sample_valid_mask)
                        sample_df = um.get_metrics_dataframe(sample_mf, self.ontology)
                        sample_df.to_csv(
                            os.path.join(predictions_outdir, f"{sample_idx}.csv")
                        )
                    pred = Image.fromarray(np.squeeze(pred).astype(np.uint8))
                    pred.save(os.path.join(predictions_outdir, f"{sample_idx}.png"))

        return um.get_metrics_dataframe(metrics_factory, self.ontology)

    def get_computational_cost(self, runs: int = 30, warm_up_runs: int = 5) -> dict:
        """Get different metrics related to the computational cost of the model

        :param runs: Number of runs to measure inference time, defaults to 30
        :type runs: int, optional
        :param warm_up_runs: Number of warm-up runs, defaults to 5
        :type warm_up_runs: int, optional
        :return: Dictionary containing computational cost information
        """
        dummy_input = tf.random.normal([1, *self.model_cfg["image_size"], 3])
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
