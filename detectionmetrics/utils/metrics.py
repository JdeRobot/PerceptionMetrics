from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Metric(ABC):
    """Abstract class for metrics

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    @abstractmethod
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Accumulate results for a new batch

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> np.ndarray:
        """Get final values

        :return: Array containing final values
        :rtype: np.ndarray
        """
        raise NotImplementedError


class IoU(Metric):
    """Compute Intersection over Union (IoU). IoU per sample and class is accumulated
    and then the average per class is computed.

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.iou = np.zeros((0, n_classes), dtype=np.float64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Accumulate IoU values for a new set of samples

        :param pred: one-hot encoded prediction array (batch, class, width, height)
        :type pred: np.ndarray
        :param gt: one-hot encoded ground truth array (batch, class, width, height)
        :type gt: np.ndarray
        """
        assert pred.shape == gt.shape, "Pred. and GT shapes don't match"
        assert pred.shape[1] == self.n_classes, "Number of classes mismatch"

        # Flatten spatial dimensions
        batch_size = pred.shape[0]
        pred = pred.reshape(batch_size, self.n_classes, -1)
        gt = gt.reshape(batch_size, self.n_classes, -1)

        # Compute intersection and union for each sample
        intersection = np.sum(pred * gt, axis=-1)
        union = np.sum(pred + gt, axis=-1) - intersection
        iou = np.where(union > 0, intersection / union, np.nan)

        # Accumulate
        self.iou = np.append(self.iou, iou, axis=0)

    def compute(self) -> np.ndarray:
        """Get IoU (global and per class)

        :return: per class IoU, and global IoU
        :rtype: Tuple[float, np.ndarray]
        """
        return np.nanmean(self.iou, axis=0), np.nanmean(self.iou)


class Accuracy(Metric):
    r"""Compute accuracy as:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Accuracy per sample and class is accumulated and then the average per class is
    computed.

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.correct = []
        self.total = []
        self.accuracy = []
        self.correct_per_class = np.zeros((0, n_classes))
        self.total_per_class = np.zeros((0, n_classes))
        self.accuracy_per_class = np.zeros((0, n_classes))

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Accumulate accuracy values for a new set of samples

        :param pred: label encoded prediction array (batch, width, height)
        :type pred: np.ndarray
        :param gt: label encoded ground truth array (batch, width, height)
        :type gt: np.ndarray
        """
        assert pred.shape == gt.shape, "Pred. and GT shapes don't match"

        # Flatten spatial dimensions
        batch_size = pred.shape[0]
        pred = pred.reshape(batch_size, -1)
        gt = gt.reshape(batch_size, -1)

        # Accumulate total number of pixels and correct pixels
        correct = np.sum(pred == gt)
        total = np.size(gt)

        self.correct.append(correct)
        self.total.append(total)
        self.accuracy.append(correct / total)

        # Accumulate number of pixels and correct pixels per class
        correct_per_class = np.zeros((batch_size, self.n_classes))
        total_per_class = np.zeros((batch_size, self.n_classes))
        for class_idx in range(self.n_classes):
            mask = gt == class_idx
            class_total = np.sum(mask, axis=1)
            correct_per_class[:, class_idx] = np.sum((pred == gt) & mask, axis=1)
            total_per_class[:, class_idx] = class_total

            correct_per_class[class_total == 0, class_idx] = np.nan
            total_per_class[class_total == 0, class_idx] = np.nan

        self.correct_per_class = np.append(
            self.correct_per_class, correct_per_class, axis=0
        )
        self.total_per_class = np.append(self.total_per_class, total_per_class, axis=0)
        self.accuracy_per_class = np.append(
            self.accuracy_per_class, correct_per_class / total_per_class, axis=0
        )

    def compute(self) -> Tuple[float, np.ndarray]:
        """Get accuracy (global and per class)

        :return: per class accuracy, and global accuracy
        :rtype: Tuple[float, np.ndarray]
        """
        acc_per_class = np.nansum(self.correct_per_class, axis=0) / np.nansum(
            self.total_per_class, axis=0
        )
        acc = np.nansum(self.correct) / np.nansum(self.total)
        return acc_per_class, acc
