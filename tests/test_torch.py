import torch
import pytest
from perceptionmetrics.utils.torch import data_to_device, get_data_shape, unsqueeze_data


def test_data_to_device():
    # Setup
    device = torch.device("cpu")
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 3)
    data = [t1, (t2,)]

    # Execute
    moved_data = data_to_device(data, device)

    # Verify
    assert torch.equal(moved_data[0], t1.to(device))
    assert torch.equal(moved_data[1][0], t2.to(device))
    assert isinstance(moved_data, list)
    assert isinstance(moved_data[1], tuple)


def test_get_data_shape():
    # Setup
    t1 = torch.randn(2, 3)
    t2 = torch.randn(4, 5, 6)
    data = (t1, [t2])

    # Execute
    shapes = get_data_shape(data)

    # Verify
    assert shapes == ((2, 3), [(4, 5, 6)])
    assert isinstance(shapes, tuple)
    assert isinstance(shapes[1], list)


def test_unsqueeze_data():
    # Setup
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 3)
    data = [t1, (t2,)]

    # Execute
    unsqueezed = unsqueeze_data(data, dim=0)

    # Verify
    assert unsqueezed[0].shape == (1, 2, 2)
    assert unsqueezed[1][0].shape == (1, 3, 3)
    assert isinstance(unsqueezed, list)
    assert isinstance(unsqueezed[1], tuple)


def test_torch_raises_type_error():
    # Ensure non-tensors raise TypeError
    data = "string"
    device = torch.device("cpu")
    
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        data_to_device(data, device)
        
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        get_data_shape(data)
        
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        unsqueeze_data(data)

def test_torch_raises_type_error_nested():
    # Test nested invalid types
    data = [torch.randn(2), "invalid"]
    
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        data_to_device(data, torch.device("cpu"))
        
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        get_data_shape(data)
