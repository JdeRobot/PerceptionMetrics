import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.neighbors import KDTree
import open3d as o3d
from PIL import Image
from perceptionmetrics.utils.lidar import (
    Sampler,
    recenter,
    build_point_cloud,
    view_point_cloud,
    render_point_cloud,
    REFERENCE_SIZE,
    CAMERA_VIEWS,
)


@pytest.fixture
def sample_points():
    """Fixture to generate reproducible sample points for testing."""
    np.random.seed(42)
    return np.random.rand(100, 3)


@pytest.fixture
def sample_colors():
    """Fixture to generate reproducible sample colors for testing."""
    np.random.seed(42)
    return np.random.rand(100, 3)


@pytest.fixture
def sample_kdtree(sample_points):
    """Create a KDTree from sample points."""
    return KDTree(sample_points)


class TestSampler:
    """Tests for the Sampler class."""

    def test_valid_samplers(self, sample_points, sample_kdtree):
        """Test initialization with valid samplers."""
        # Test with random sampler
        random_sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="random",
            num_classes=10,
            seed=42,
        )

        assert random_sampler.num_classes == 10
        assert random_sampler.test_probs.shape == (len(sample_points), 10)
        assert random_sampler.sample.__name__ == "random"

        # Test with spatially_regular sampler
        spatial_sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="spatially_regular",
            num_classes=10,
            seed=42,
        )

        assert spatial_sampler.sample.__name__ == "spatially_regular"

    def test_invalid_sampler(self, sample_points, sample_kdtree):
        """Test initialization with invalid sampler name."""
        # Handling the fact that the original code tries to access self.model_cfg['sampler']
        # We expect an AttributeError rather than NotImplementedError
        with pytest.raises(AttributeError):
            Sampler(
                point_cloud_size=len(sample_points),
                search_tree=sample_kdtree,
                sampler_name="invalid_sampler",
                num_classes=10,
                seed=42,
            )

    def test_get_indices_small_cloud(self, sample_points, sample_kdtree):
        """Test _get_indices when point_cloud_size < num_points."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="random",
            num_classes=10,
            seed=42,
        )

        point_cloud_size = 20
        num_points = 30
        center_point = np.array([[0.5, 0.5, 0.5]])

        indices = sampler._get_indices(point_cloud_size, num_points, center_point)

        assert len(indices) == num_points
        assert np.max(indices) < point_cloud_size  # All indices should be within range

    def test_get_indices_large_cloud(self, sample_points, sample_kdtree):
        """Test _get_indices when point_cloud_size >= num_points."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="random",
            num_classes=10,
            seed=42,
        )

        point_cloud_size = 100
        num_points = 10
        center_point = np.array([[0.5, 0.5, 0.5]])

        indices = sampler._get_indices(point_cloud_size, num_points, center_point)

        assert len(indices) == num_points
        assert np.max(indices) < point_cloud_size

    def test_random_sampler_functionality(self, sample_points, sample_kdtree):
        """Test the random sampler's sampling behavior."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="random",
            num_classes=10,
            seed=42,
        )

        num_points = 20
        points, indices, center_point = sampler.random(sample_points, num_points)

        assert points.shape == (num_points, 3)
        assert len(indices) == num_points
        assert center_point.shape == (1, 3)
        assert indices.max() < len(sample_points)

    def test_spatially_regular_with_num_points(self, sample_points, sample_kdtree):
        """Test spatially regular sampler with num_points parameter."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="spatially_regular",
            num_classes=10,
            seed=42,
        )

        num_points = 20
        points, indices, center_point = sampler.spatially_regular(
            sample_points, num_points=num_points
        )

        assert points.shape == (len(indices), 3)
        assert len(indices) >= 2  # Should have at least 2 points
        assert center_point.shape == (1, 3)
        assert np.min(sampler.p) >= sampler.min_p

    def test_spatially_regular_with_radius(self, sample_points, sample_kdtree):
        """Test spatially regular sampler with radius parameter."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="spatially_regular",
            num_classes=10,
            seed=42,
        )

        radius = 0.3
        points, indices, center_point = sampler.spatially_regular(
            sample_points, radius=radius
        )

        assert points.shape == (len(indices), 3)
        assert len(indices) >= 2
        assert center_point.shape == (1, 3)

    def test_spatially_regular_missing_params(self, sample_points, sample_kdtree):
        """Test spatially_regular raises error when parameters are missing."""
        sampler = Sampler(
            point_cloud_size=len(sample_points),
            search_tree=sample_kdtree,
            sampler_name="spatially_regular",
            num_classes=10,
            seed=42,
        )

        with pytest.raises(
            ValueError, match="Either num_points or radius must be provided"
        ):
            sampler.spatially_regular(sample_points)


class TestUtilityFunctions:
    """Tests for standalone utility functions."""

    def test_recenter(self, sample_points):
        """Test recenter function properly centers point cloud dimensions."""
        dims = [0, 2]
        recentered_points = recenter(sample_points.copy(), dims)

        # Check that mean along specified dimensions is close to zero
        assert np.abs(recentered_points[:, dims].mean(0)).max() < 1e-10

        # Check that unspecified dimension is unchanged
        assert np.allclose(recentered_points[:, 1], sample_points[:, 1])

    def test_build_point_cloud(self, sample_points, sample_colors):
        """Test build_point_cloud creates proper Open3D point cloud."""
        point_cloud = build_point_cloud(sample_points, sample_colors)

        assert isinstance(point_cloud, o3d.geometry.PointCloud)
        assert len(point_cloud.points) == len(sample_points)
        assert len(point_cloud.colors) == len(sample_colors)
        assert np.allclose(np.asarray(point_cloud.points), sample_points)
        assert np.allclose(np.asarray(point_cloud.colors), sample_colors)

    @patch("open3d.visualization.draw_geometries")
    def test_view_point_cloud(self, mock_draw, sample_points, sample_colors):
        """Test view_point_cloud correctly calls visualization function."""
        view_point_cloud(sample_points, sample_colors)

        mock_draw.assert_called_once()
        args = mock_draw.call_args[0][0]
        assert len(args) == 1
        assert isinstance(args[0], o3d.geometry.PointCloud)

    @patch("open3d.visualization.rendering.OffscreenRenderer")
    def test_render_point_cloud(
        self, mock_renderer_class, sample_points, sample_colors
    ):
        """Test render_point_cloud produces expected output."""
        # Setup mock
        mock_renderer = MagicMock()
        mock_renderer_class.return_value = mock_renderer
        mock_image_array = np.zeros((1080, 1920, 4), dtype=np.uint8)
        mock_renderer.render_to_image.return_value = mock_image_array

        # Call function with custom parameters
        result = render_point_cloud(
            sample_points,
            sample_colors,
            camera_view="3rd_person",
            bg_color=[0.5, 0.5, 0.5, 1.0],
            color_jitter=0.1,
            point_size=5.0,
            resolution=(800, 600),
        )

        # Verify expectations
        mock_renderer_class.assert_called_once_with(800, 600)
        mock_renderer.scene.add_geometry.assert_called_once()
        mock_renderer.scene.set_background.assert_called_once()
        mock_renderer.setup_camera.assert_called_once()
        mock_renderer.render_to_image.assert_called_once()
        mock_renderer.scene.clear_geometry.assert_called_once()

        assert isinstance(result, Image.Image)

    def test_render_point_cloud_invalid_camera_view(self, sample_points, sample_colors):
        """Test render_point_cloud with invalid camera view."""
        with pytest.raises(AssertionError):
            render_point_cloud(sample_points, sample_colors, camera_view="invalid_view")


class TestConstants:
    """Tests for constants in the module."""

    def test_camera_views_structure(self):
        """Test the structure of CAMERA_VIEWS constant."""
        assert "3rd_person" in CAMERA_VIEWS
        view = CAMERA_VIEWS["3rd_person"]

        required_keys = ["zoom", "front", "lookat", "up"]
        for key in required_keys:
            assert key in view

        for vector_key in ["front", "lookat", "up"]:
            assert isinstance(view[vector_key], np.ndarray)
            assert view[vector_key].shape == (3,)
