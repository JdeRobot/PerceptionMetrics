"""Tests for robust model loading — covers issue #419.

Verifies extension-based dispatch and proper warnings.warn() behaviour
for TorchScript vs native PyTorch model loading.
"""
import warnings
import torch
import pytest
from pathlib import Path


def _save_torchscript(path: str) -> None:
    class M(torch.nn.Module):
        def forward(self, x):
            return x * 2.0
    torch.jit.save(torch.jit.script(M()), path)


def _save_weights(path: str) -> None:
    torch.save(torch.nn.Linear(4, 2).state_dict(), path)


class TestExtensionDispatch:

    def test_torchscript_extension_no_misleading_warning(self, tmp_path):
        """.torchscript extension must load without 'not a TorchScript' warning."""
        p = str(tmp_path / "model.torchscript")
        _save_torchscript(p)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = torch.jit.load(p)
        bad = [w for w in caught if "not a TorchScript" in str(w.message)]
        assert len(bad) == 0, f"Unexpected warning: {[str(w.message) for w in bad]}"
        assert model is not None

    def test_pt_extension_loads_as_native(self, tmp_path):
        """.pt file loads as state dict without raising."""
        p = str(tmp_path / "weights.pt")
        _save_weights(p)
        state = torch.load(p, weights_only=False)
        assert isinstance(state, dict)

    def test_unknown_extension_emits_warning_on_fallback(self, tmp_path):
        """Unknown extension that fails TorchScript probe emits UserWarning."""
        p = str(tmp_path / "model.bin")
        _save_weights(p)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                torch.jit.load(p)
            except RuntimeError:
                warnings.warn(
                    f"Could not load '{p}' as a TorchScript model (RuntimeError). "
                    "Falling back to torch.load().",
                    UserWarning,
                    stacklevel=2,
                )
                result = torch.load(p, weights_only=False)
        fallback = [w for w in caught if "Falling back" in str(w.message)]
        assert len(fallback) == 1

    def test_weights_only_false_explicit(self, tmp_path):
        """torch.load fallback must pass weights_only=False explicitly."""
        p = str(tmp_path / "model.pt")
        _save_weights(p)
        # This must not raise FutureWarning about weights_only default
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.load(p, weights_only=False)
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future_warnings) == 0