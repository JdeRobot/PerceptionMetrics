"""Shared pytest fixtures and setup for the test suite."""

import sys
from unittest.mock import MagicMock


def _stub_missing_torch():
    """Stub out torch/torchvision if they are not installed."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        pass
    else:
        return

    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = False
    torch_stub.is_tensor.return_value = False

    nn_stub = MagicMock(name="torch.nn")
    functional_stub = MagicMock(name="torch.nn.functional")
    nn_stub.functional = functional_stub
    torch_stub.nn = nn_stub

    transforms_v2_stub = MagicMock(name="torchvision.transforms.v2")
    transforms_stub = MagicMock(name="torchvision.transforms")
    transforms_stub.v2 = transforms_v2_stub
    torchvision_stub = MagicMock(name="torchvision")
    torchvision_stub.transforms = transforms_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub
    sys.modules["torchvision.transforms.v2"] = transforms_v2_stub


_stub_missing_torch()
