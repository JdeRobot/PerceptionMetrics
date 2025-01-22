import os
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


class ImageSegmentationTensorflowDataset:
    """Dataset for image segmentation Tensorflow models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param image_size: Image size in pixels (width, height)
    :type image_size: Tuple[int, int]
    :param batch_size: Batch size, defaults to 1
    :type batch_size: int, optional
    :param split: Split to be used from the dataset, defaults to "all"
    :type split: str, optional
    :param lut_ontology: LUT to transform label classes, defaults to None
    :type lut_ontology: dict, optional
    """

    def __init__(
        self,
        dataset: ImageSegmentationDataset,
        image_size: Tuple[int, int],
        batch_size: int = 1,
        split: str = "all",
        lut_ontology: Optional[dict] = None,
    ):
        self.image_size = image_size

        # Filter split and make filenames global
        if split != "all":
            dataset.dataset = dataset.dataset[dataset.dataset["split"] == split]
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
        fnames = (dataset.dataset["image"], dataset.dataset["label"])
        self.dataset = tf_data.Dataset.from_tensor_slices(fnames)
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
        image = tf_image.resize(images=image, size=self.image_size, method=method)
        if label:
            image = tf.round(image)

        return image

    def load_data(
        self, images_fnames: List[str], labels_fnames: List[str]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Function for loading data for each dataset sample

        :param images_fnames: List containing all image filenames
        :type images_fnames: List[str]
        :param labels_fnames: List containing all corresponding label filenames
        :type labels_fnames: List[str]
        :return: Image and label tensor pairs
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        image = self.read_image(images_fnames)
        label = self.read_image(labels_fnames, label=True)
        return image, label


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
            model = tf.saved_model.load(model)
            model_type = "compiled"
        # Otherwise, check that it is a Tensorflow or Keras model
        elif isinstance(model, tf.Module) or isinstance(model, tf.keras.Model):
            model_type = "native"
        else:
            raise ValueError(
                "Model must be either a SavedModel directory or a TF/Keras model"
            )

        super().__init__(model, model_type, model_cfg, ontology_fname)

        # Init transformation for input images
        def t_in(image):
            tensor = tf.convert_to_tensor(image)
            tensor = tf_image.resize(images=tensor, size=self.model_cfg["image_size"])
            tensor = tf.expand_dims(tensor, axis=0)
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
        split: str = "all",
        ontology_translation: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset: Image segmentation dataset for which the evaluation will be performed
        :type dataset: ImageSegmentationDataset
        :param split: Split to be used from the dataset, defaults to "all"
        :type split: str, optional
        :param ontology_translation: JSON file containing translation between dataset and model output ontologies
        :type ontology_translation: str, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Build a LUT for transforming ontology if needed
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)

        # Get Tensorflow dataset
        dataset = ImageSegmentationTensorflowDataset(
            dataset,
            image_size=self.model_cfg["image_size"],
            batch_size=self.model_cfg.get("batch_size", 1),
            split=split,
            lut_ontology=lut_ontology,
        )

        # Init metrics
        results = {}
        iou = um.IoU(self.n_classes)
        cm = um.ConfusionMatrix(self.n_classes)

        # Evaluation loop
        pbar = tqdm(dataset.dataset)
        for image, label in pbar:
            if self.model_type == "native":
                pred = self.model(image)
            elif self.model_type == "compiled":
                pred = self.model.signatures["serving_default"](image)
            else:
                raise ValueError("Model type not recognized")

            if isinstance(pred, dict):
                pred = list(pred.values())[0]

            label = tf.squeeze(label, axis=3)
            pred = tf.argmax(pred, axis=3)
            cm.update(pred.numpy(), label.numpy())

            pred = tf.one_hot(pred, self.n_classes)
            pred = tf.transpose(pred, perm=[0, 3, 1, 2])

            label = tf.one_hot(label, self.n_classes)
            label = tf.transpose(label, perm=[0, 3, 1, 2])

            iou.update(pred.numpy(), label.numpy())

        # Get metrics results
        iou_per_class, iou = iou.compute()
        acc_per_class, acc = cm.get_accuracy()
        iou_per_class = [float(n) for n in iou_per_class]
        acc_per_class = [float(n) for n in acc_per_class]

        # Build results dataframe
        results = {}
        for class_name, class_data in self.ontology.items():
            results[class_name] = {
                "iou": iou_per_class[class_data["idx"]],
                "acc": acc_per_class[class_data["idx"]],
            }
        results["global"] = {"iou": iou, "acc": acc}

        results = pd.DataFrame(results)
        results.index.name = "metric"

        return results
