"""Unit tests for model loading robustness"""
import os
import tempfile
import pytest
import torch
from PIL import Image

from perceptionmetrics.models.torch_detection import TorchImageDetectionModel
from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel


class TestModelLoadingExceptions:
    """Test that model loading raises appropriate exceptions"""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model file (corrupted)
            bad_model = os.path.join(tmpdir, "bad_model.pt")
            with open(bad_model, "w") as f:
                f.write("this is not a pytorch model")
            
            # Create dummy ontology
            ontology = os.path.join(tmpdir, "ontology.json")
            import json
            with open(ontology, "w") as f:
                json.dump({
                    "car": {"idx": 0, "rgb": [0, 0, 0]},
                    "person": {"idx": 1, "rgb": [255, 0, 0]},
                }, f)
            
            # Create dummy config
            config = os.path.join(tmpdir, "config.json")
            with open(config, "w") as f:
                json.dump({
                    "resize": {"width": 512, "height": 512},
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    },
                    "batch_size": 1,
                    "model_format": "torchvision"
                }, f)
            
            yield {
                "bad_model": bad_model,
                "ontology": ontology,
                "config": config,
                "tmpdir": tmpdir
            }
    
    def test_detection_model_bad_file_raises_specific_error(self, temp_files):
        """Test that loading corrupted model raises RuntimeError, not generic Exception"""
        with pytest.raises(RuntimeError) as exc_info:
            TorchImageDetectionModel(
                model=temp_files["bad_model"],
                model_cfg=temp_files["config"],
                ontology_fname=temp_files["ontology"]
            )
        
        # Check error message is informative
        error_msg = str(exc_info.value)
        assert "Failed to load model" in error_msg
        assert "TorchScript error" in error_msg or "PyTorch error" in error_msg
    
    def test_segmentation_model_bad_file_raises_specific_error(self, temp_files):
        """Test that loading corrupted segmentation model raises RuntimeError"""
        with pytest.raises(RuntimeError) as exc_info:
            TorchImageSegmentationModel(
                model=temp_files["bad_model"],
                model_cfg=temp_files["config"],
                ontology_fname=temp_files["ontology"]
            )
        
        error_msg = str(exc_info.value)
        assert "Failed to load model" in error_msg
    
    def test_detection_model_missing_file_raises_file_not_found(self, temp_files):
        """Test that missing model file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            TorchImageDetectionModel(
                model="/nonexistent/path/model.pt",
                model_cfg=temp_files["config"],
                ontology_fname=temp_files["ontology"]
            )
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_segmentation_model_missing_file_raises_file_not_found(self, temp_files):
        """Test that missing segmentation model file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            TorchImageSegmentationModel(
                model="/nonexistent/path/model.pt",
                model_cfg=temp_files["config"],
                ontology_fname=temp_files["ontology"]
            )
        
        assert "Model file not found" in str(exc_info.value)
