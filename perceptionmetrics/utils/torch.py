from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


def data_to_device(
    data: Union[tuple, list], device: torch.device
) -> Union[tuple, list]:
    """Move provided data to given device (CPU or GPU)

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param device: Device to move data to
    :type device: torch.device
    :return: Data moved to device
    :rtype: Union[tuple, list]
    :raises TypeError: If data is not a tensor, list, or tuple
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.to(device) if torch.is_tensor(d) else data_to_device(d, device)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        raise TypeError(
            f"data_to_device expected torch.Tensor, list, or tuple, but got {type(data)}"
        )


def get_data_shape(data: Union[tuple, list]) -> Union[tuple, list]:
    """Get the shape of the provided data

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :return: Data shape
    :rtype: Union[tuple, list]
    :raises TypeError: If data is not a tensor, list, or tuple
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            tuple(d.shape) if torch.is_tensor(d) else get_data_shape(d) for d in data
        )
    elif torch.is_tensor(data):
        return tuple(data.shape)
    else:
        raise TypeError(
            f"get_data_shape expected torch.Tensor, list, or tuple, but got {type(data)}"
        )


def unsqueeze_data(data: Union[tuple, list], dim: int = 0) -> Union[tuple, list]:
    """Unsqueeze provided data along given dimension

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param dim: Dimension that will be unsqueezed, defaults to 0
    :type dim: int, optional
    :return: Unsqueezed data
    :rtype: Union[tuple, list]
    :raises TypeError: If data is not a tensor, list, or tuple
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.unsqueeze(dim) if torch.is_tensor(d) else unsqueeze_data(d, dim)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.unsqueeze(dim)
    else:
        raise TypeError(
            f"unsqueeze_data expected torch.Tensor, list, or tuple, but got {type(data)}"
        )


def get_device_info():
    """Get the best available device(CPU or GPU) and list of available devices.

    :return: Tuple of (best_device, available_devices) where best_device is a string and avaible_devices is a list of strings.
    :rtype: tuple(str, list[str])
    """

    available_devices = []

    if torch.cuda.is_available():
        available_devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available_devices.append("mps")
    available_devices.append("cpu")

    best_device = available_devices[0]

    return best_device, available_devices


def get_resize_args(resize_cfg: dict) -> dict:
    """
    Get the resize arguments for torchvision.transforms.Resize from the configuration.

    :param resize_cfg: Resize configuration dictionary
    :type resize_cfg: dict
    :return: Dictionary with arguments for transforms.Resize
    :rtype: dict
    """
    resize_args = {"interpolation": transforms.InterpolationMode.BILINEAR}
    fixed_h = resize_cfg.get("height")
    fixed_w = resize_cfg.get("width")
    min_side = resize_cfg.get("min_side")
    max_side = resize_cfg.get("max_side")

    if fixed_h is not None and fixed_w is not None:
        if min_side is not None or max_side is not None:
            raise ValueError(
                "Resize config cannot satisfy both fixed dimensions (width/height) and "
                "min_side/max_side. They are mutually exclusive."
            )
        resize_args["size"] = (fixed_h, fixed_w)
    elif min_side is not None or max_side is not None:
        resize_args["size"] = min_side
        resize_args["max_size"] = max_side
        if fixed_h is not None or fixed_w is not None:
            raise ValueError(
                "Resize config cannot satisfy both fixed dimensions (width/height) and "
                "min_side/max_side. They are mutually exclusive."
            )
    else:
        raise ValueError(
            "Resize config must contain either 'height' and 'width' or 'min_side' "
            "and/or 'max_side'."
        )

    return resize_args


def pad_to_closest_divisor(
    tensor: torch.Tensor, closest_divisor: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad a tensor with zeros right and bottom so that its height and width are a multiple
    of closest_divisor.

    :param tensor: Input tensor of shape (..., H, W)
    :type tensor: torch.Tensor
    :param closest_divisor: The divisor to pad to
    :type closest_divisor: int
    :return: Padded tensor and original (H, W)
    :rtype: Tuple[torch.Tensor, Tuple[int, int]]
    """
    h, w = tensor.shape[-2:]
    pad_h = (closest_divisor - (h % closest_divisor)) % closest_divisor
    pad_w = (closest_divisor - (w % closest_divisor)) % closest_divisor

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    return tensor, (h, w)


def inverse_transform_boxes(
    boxes: torch.Tensor,
    orig_size: Tuple[int, int],
    input_transforms: transforms.Compose,
) -> torch.Tensor:
    """Map bounding boxes from the transformed image space back to the original image space.

    :param boxes: Bounding boxes in [N, 4] format
    :type boxes: torch.Tensor
    :param orig_size: Original image size (width, height)
    :type orig_size: tuple(int, int)
    :param input_transforms: Transforms to be applied to images
    :type input_transforms: transforms.Compose
    :return: Bounding boxes mapped back to the original image space
    :rtype: torch.Tensor
    """
    w_curr, h_curr = orig_size
    spatial_ops = []

    for t in input_transforms:
        if isinstance(t, transforms.Resize):
            size = t.size
            max_size = getattr(t, "max_size", None)
            if isinstance(size, int) or (
                isinstance(size, (list, tuple)) and len(size) == 1
            ):
                s = size if isinstance(size, int) else size[0]
                if (w_curr <= h_curr and w_curr == s) or (
                    h_curr <= w_curr and h_curr == s
                ):
                    new_h, new_w = h_curr, w_curr
                elif w_curr < h_curr:
                    new_w = s
                    new_h = int(s * h_curr / w_curr)
                else:
                    new_h = s
                    new_w = int(s * w_curr / h_curr)
                if max_size is not None:
                    if max(new_w, new_h) > max_size:
                        if new_w > new_h:
                            new_w = max_size
                            new_h = int(max_size * h_curr / w_curr)
                        else:
                            new_h = max_size
                            new_w = int(max_size * w_curr / h_curr)
            else:
                new_h, new_w = size

            spatial_ops.append((w_curr / new_w, h_curr / new_h, 0, 0))
            w_curr, h_curr = new_w, new_h

        elif isinstance(t, transforms.CenterCrop):
            if isinstance(t.size, int):
                new_h, new_w = t.size, t.size
            else:
                new_h, new_w = t.size
            top = int(round((h_curr - new_h) / 2.0))
            left = int(round((w_curr - new_w) / 2.0))
            spatial_ops.append((1.0, 1.0, left, top))
            w_curr, h_curr = new_w, new_h

    inv_boxes = boxes.clone()
    for scale_w, scale_h, left, top in reversed(spatial_ops):
        inv_boxes[:, [0, 2]] += left
        inv_boxes[:, [1, 3]] += top
        inv_boxes[:, [0, 2]] *= scale_w
        inv_boxes[:, [1, 3]] *= scale_h

    return inv_boxes
