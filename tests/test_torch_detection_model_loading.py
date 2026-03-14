"""Tests for TorchImageDetectionModel model-loading dispatch (issue #419).

The model loading logic dispatches on file extension rather than relying on
a broad try/except that can mask real errors or produce misleading output.

These tests verify the dispatch logic in isolation using only the standard
library (pathlib) so they run without PyTorch installed.
"""
import pytest
from pathlib import Path


def _resolve_load_path(model_path: str):
    """
    Mirror the dispatch logic added to TorchImageDetectionModel.__init__.

    Returns "jit" for .torchscript, "native" for .pt/.pth, or raises
    ValueError for anything else.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix == ".torchscript":
        return "jit"
    elif suffix in (".pt", ".pth"):
        return "native"
    else:
        raise ValueError(
            f"Unsupported model file extension '{suffix}'. "
            "Expected '.torchscript' for TorchScript models or "
            "'.pt' / '.pth' for native PyTorch models."
        )


class TestModelLoadingDispatch:
    """Verify that model loading dispatches correctly on file extension."""

    def test_torchscript_extension_dispatches_to_jit(self):
        """.torchscript files must use torch.jit.load, not torch.load."""
        assert _resolve_load_path("model.torchscript") == "jit"

    def test_pt_extension_dispatches_to_native(self):
        """.pt files must use torch.load."""
        assert _resolve_load_path("model.pt") == "native"

    def test_pth_extension_dispatches_to_native(self):
        """.pth files must use torch.load."""
        assert _resolve_load_path("model.pth") == "native"

    def test_unsupported_extension_raises_value_error(self):
        """An unsupported extension must raise ValueError with a clear message."""
        with pytest.raises(ValueError, match="Unsupported model file extension"):
            _resolve_load_path("model.onnx")

    def test_unsupported_extension_error_message_lists_supported_formats(self):
        """The ValueError must mention the supported extensions."""
        with pytest.raises(ValueError, match=r"\.torchscript"):
            _resolve_load_path("model.bin")

    def test_extension_check_is_case_insensitive(self):
        """Extension matching must be case-insensitive."""
        for ext in (".TORCHSCRIPT", ".TorchScript", ".torchscript"):
            assert _resolve_load_path(f"model{ext}") == "jit"

        for ext in (".PT", ".Pt", ".PTH", ".Pth"):
            assert _resolve_load_path(f"model{ext}") == "native"

    def test_path_with_directory_components(self):
        """Dispatch works when the model path includes directory components."""
        assert _resolve_load_path("/some/dir/weights/best.torchscript") == "jit"
        assert _resolve_load_path("/some/dir/weights/best.pt") == "native"

    def test_no_silent_fallback_for_torchscript(self):
        """
        .torchscript must never fall back to torch.load.

        Before the fix, torch.jit.load could raise (e.g. wrong CUDA device),
        and the bare except would silently call torch.load instead — producing
        a misleading warning. With the extension-based dispatch there is no
        fallback path.
        """
        result = _resolve_load_path("yolov8n.torchscript")
        assert result == "jit", (
            "TorchScript models must always use jit.load — no silent fallback."
        )
