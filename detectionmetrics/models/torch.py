import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
from detectionmetrics.models.model import ImageSegmentationModel
import detectionmetrics.utils.conversion as uc
import detectionmetrics.utils.io as uio
import detectionmetrics.utils.metrics as um


class ImageSegmentationTorchDataset(Dataset):
    """Dataset for image segmentation PyTorch models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param transform: Transformation to be applied to images
    :type transform: v2.Compose
    :param target_transform: Transformation to be applied to labels
    :type target_transform: v2.Compose
    :param split: Split to be used from the dataset, defaults to "all"
    :type split: str, optional
    """

    def __init__(
        self,
        dataset: ImageSegmentationDataset,
        transform: v2.Compose,
        target_transform: v2.Compose,
        split: str = "all",
    ):
        # Filter split and make filenames global
        if split != "all":
            dataset.dataset = dataset.dataset[dataset.dataset["split"] == split]
        self.dataset = dataset
        self.dataset.make_fname_global()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]:
        """Prepare sample data: image and label

        :param idx: Sample index
        :type idx: int
        :return: Image and corresponding label tensor or PIL image
        :rtype: Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]
        """
        image = Image.open(self.dataset.dataset.iloc[idx]["image"])
        label = Image.open(self.dataset.dataset.iloc[idx]["label"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class TorchImageSegmentationModel(ImageSegmentationModel):

    def __init__(self, model_fname: str, model_cfg: str, ontology_fname: str):
        """Image segmentation model for PyTorch framework

        :param model_fname: PyTorch model saved using TorchScript
        :type model_fname: str
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        """
        super().__init__(ontology_fname, model_cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check that provided path exist and load model
        assert os.path.isfile(model_fname), "Model file not found"
        self.model = torch.jit.load(model_fname).to(self.device)
        self.model.eval()

        # Init transformations for input images, output labels, and GT labels
        self.t_in = []
        self.t_label = []

        if "image_size" in self.model_cfg:
            self.t_in += [v2.Resize(tuple(self.model_cfg["image_size"]))]
            self.t_label += [
                v2.Resize(
                    tuple(self.model_cfg["image_size"]),
                    interpolation=v2.InterpolationMode.NEAREST_EXACT,
                )
            ]

        self.t_in += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        self.t_label += [v2.ToImage(), v2.ToDtype(torch.int64)]

        if "normalization" in self.model_cfg:
            self.t_in += [
                v2.Normalize(
                    mean=self.model_cfg["normalization"]["mean"],
                    std=self.model_cfg["normalization"]["std"],
                )
            ]

        self.t_in = v2.Compose(self.t_in)
        self.t_label = v2.Compose(self.t_label)
        self.t_out = v2.Compose(
            [
                lambda x: torch.argmax(x.squeeze(), axis=0).squeeze().to(torch.uint8),
                v2.ToPILImage(),
            ]
        )

    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: segmenation result as PIL image
        :rtype: Image.Image
        """
        tensor = self.t_in(image).unsqueeze(0).to(self.device)
        result = self.model(tensor)

        # TODO: check if this is consistent across different models
        if isinstance(result, dict):
            result = result["out"]

        return self.t_out(result)

    def eval(
        self,
        dataset: ImageSegmentationDataset,
        batch_size: int = 1,
        split: str = "all",
        ontology_translation: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset_test: Image segmentation dataset for which the evaluation will
        be performed
        :type dataset_test: ImageSegmentationDataset
        :param batch_size: Batch size, defaults to 1
        :type batch_size: int, optional
        :param split: Split to be used from the dataset, defaults to "all"
        :type split: str, optional
        :param ontology_translation: JSON file containing translation between dataset
        and model output ontologies
        :type ontology_translation: str, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Build a LUT for transforming ontology if needed
        lut_ontology = None
        if dataset.ontology != self.ontology:
            if ontology_translation is not None:
                ontology_translation = uio.read_json(ontology_translation)
            lut_ontology = uc.get_ontology_conversion_lut(
                dataset.ontology, self.ontology, ontology_translation
            )
            lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Get PyTorch dataloader
        dataset = ImageSegmentationTorchDataset(
            dataset,
            transform=self.t_in,
            target_transform=self.t_label,
            split=split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=16,
        )

        # Init metrics
        results = {}
        iou = um.IoU(self.n_classes)
        acc = um.Accuracy(self.n_classes)

        # Evaluation loop
        with torch.no_grad():
            pbar = tqdm(dataloader)
            for image, label in pbar:
                pred = self.model(image.to(self.device))
                if isinstance(pred, dict):
                    pred = pred["out"]

                if lut_ontology is not None:
                    label = lut_ontology[label]

                label = label.squeeze(dim=1).cpu()
                pred = torch.argmax(pred, axis=1).cpu()
                acc.update(pred.numpy(), label.numpy())

                label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
                label = label.permute(0, 3, 1, 2)

                pred = torch.nn.functional.one_hot(pred, num_classes=self.n_classes)
                pred = pred.permute(0, 3, 1, 2)

                iou.update(pred.numpy(), label.numpy())

        # Get metrics results
        iou_per_class, iou = iou.compute()
        acc_per_class, acc = acc.compute()
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

        return pd.DataFrame(results)
