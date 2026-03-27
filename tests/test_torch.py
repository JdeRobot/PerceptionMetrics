import torch
import pytest
from perceptionmetrics.utils.torch import data_to_device, get_data_shape, unsqueeze_data


def test_data_to_device():
    # Setup
    device = torch.device("cpu")
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 3)
    data = [t1, (t2, "not_a_tensor")]

    # Execute
    moved_data = data_to_device(data, device)

    # Verify
    assert torch.equal(moved_data[0], t1.to(device))
    assert torch.equal(moved_data[1][0], t2.to(device))
    assert moved_data[1][1] == "not_a_tensor"
    assert isinstance(moved_data, list)
    assert isinstance(moved_data[1], tuple)


def test_get_data_shape():
    # Setup
    t1 = torch.randn(2, 3)
    t2 = torch.randn(4, 5, 6)
    data = (t1, [t2, "label"])

    # Execute
    shapes = get_data_shape(data)

    # Verify
    assert shapes == ((2, 3), [(4, 5, 6), "label"])
    assert isinstance(shapes, tuple)
    assert isinstance(shapes[1], list)


def test_unsqueeze_data():
    # Setup
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 3)
    data = [t1, (t2, "text")]

    # Execute
    unsqueezed = unsqueeze_data(data, dim=0)

    # Verify
    assert unsqueezed[0].shape == (1, 2, 2)
    assert unsqueezed[1][0].shape == (1, 3, 3)
    assert unsqueezed[1][1] == "text"
    assert isinstance(unsqueezed, list)
    assert isinstance(unsqueezed[1], tuple)


def test_torch_non_tensor_passthrough():
    # Ensure non-tensors are passed through correctly
    data = "string"
    assert data_to_device(data, torch.device("cpu")) == "string"
    assert get_data_shape(data) == "string"
    assert unsqueeze_data(data) == "string"
