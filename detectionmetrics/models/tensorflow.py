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
    :return: DataFrame containing computational cost information
    :rtype: pd.DataFrame
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
    result = {
        "input_shape": ["x".join(map(str, dummy_input.shape.as_list()))],
        "n_params": [int(n_params)],
        "size_mb": [size_mb],
        "inference_time_s": [np.mean(inference_times)],
    }
    return pd.DataFrame.from_dict(result)


def resize_image(
    image: tf.Tensor,
    method: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    closest_divisor: int = 16,
) -> tf.Tensor:
    """Resize tensorflow image to target size. If only one dimension is provided, the
    aspect ratio is preserved.

    :param image: Input image tensor
    :type image: tf.Tensor
    :param method: Resizing method (e.g. bilinear, nearest)
    :type method: str
    :param width: Target width for resizing
    :type width: Optional[int], optional
    :param height: Target height for resizing
    :type height: Optional[int], optional
    :param closest_divisor: Closest divisor for the target size, defaults to 16. Only applies to the dimension not provided.
    :type closest_divisor: int, optional
    :return: Resized image tensor
    :rtype: tf.Tensor
    """
    old_size = tf.cast(tf.shape(image)[:2], tf.float32)
    old_h = old_size[0]
    old_w = old_size[1]

    h, w = (old_h, old_w)
    if width is None:
        w = int((height / old_h) * old_w)
        h = height
    if height is None:
        h = int((width / old_w) * old_h)
        w = width

    h = (h / closest_divisor) * closest_divisor
    w = (w / closest_divisor) * closest_divisor
    new_size = [int(h), int(w)]

    image = tf_image.resize(
        images=image, size=tf.cast(new_size, tf.int32), method=method
    )

    return image


def crop_center(image: tf.Tensor, width: int, height: int) -> tf.Tensor:
    """Crop tensorflow image center to target size

    :param image: Input image tensor
    :type image: tf.Tensor
    :param width: Target width for cropping
    :type width: int
    :param height: Target width for cropping
    :type height: int
    :return: Cropped image tensor
    :rtype: tf.Tensor
    """
    old_size = tf.cast(tf.shape(image)[:2], tf.float32)
    old_h = old_size[0]
    old_w = old_size[1]

    offset_height = int((old_h - height) // 2)
    offset_width = int((old_w - width) // 2)

    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, height, width
    )

    return image


class ImageSegmentationTensorflowDataset:
    """Dataset for image segmentation Tensorflow models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param resize: Target size for resizing images, defaults to None
    :type resize: Optional[Tuple[int, int]], optional
    :param crop: Target size for center cropping images, defaults to None
    :type crop: Optional[Tuple[int, int]], optional
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
        resize: Optional[Tuple[int, int]] = None,
        crop: Optional[Tuple[int, int]] = None,
        batch_size: int = 1,
        splits: List[str] = ["test"],
        lut_ontology: Optional[dict] = None,
        normalization: Optional[dict] = None,
        keep_aspect: bool = False,
    ):
        self.resize = resize
        self.crop = crop
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
        if self.resize is not None:
            method = "nearest" if label else "bilinear"
            image = resize_image(
                image,
                method=method,
                width=self.resize.get("width", None),
                height=self.resize.get("height", None),
            )
        if self.crop is not None:
            image = crop_center(
                image,
                width=self.crop.get("width", None),
                height=self.crop.get("height", None),
            )

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

            if "resize" in self.model_cfg:
                tensor = resize_image(
                    method="bilinear",
                    width=self.model_cfg["resize"].get("width", None),
                    height=self.model_cfg["resize"].get("height", None),
                )

            if "crop" in self.model_cfg:
                tensor = crop_center(
                    tensor,
                    width=self.model_cfg["crop"].get("width", None),
                    height=self.model_cfg["crop"].get("height", None),
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

    def predict(self, image: Image.Image) -> Image.Image:
        """Perform prediction for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: Segmentation result as PIL image
        :rtype: Image.Image
        """
        tensor = self.t_in(image)
        result = self.predict(tensor)
        return self.t_out(result)

    def predict(self, tensor_in: tf.Tensor) -> tf.Tensor:
        """Perform inference for a tensor

        :param tensor_in: Input point cloud tensor
        :type tensor_in: tf.Tensor
        :return: Segmentation result as tensor
        :rtype: tf.Tensor
        """
        if self.model_type == "native":
            tensor_out = self.model(tensor_in, training=False)
        elif self.model_type == "compiled":
            tensor_out = self.model.signatures["serving_default"](tensor_in)
        else:
            raise ValueError("Model type not recognized")

        if isinstance(tensor_out, dict):
            tensor_out = list(tensor_out.values())[0]

        return tensor_out

    def eval(
        self,
        dataset: ImageSegmentationDataset,
        split: Union[str, List[str]] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split to be used from the dataset, defaults to "test"
        :type split: Union[str, List[str]], optional
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
            resize=self.model_cfg.get("resize", None),
            crop=self.model_cfg.get("crop", None),
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

            pred = self.predict(image)

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
        dummy_input = tf.random.normal([1, *image_size, 3])
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
