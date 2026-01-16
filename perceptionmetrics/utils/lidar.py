import numpy as np
import random
from typing import List, Optional, Tuple, Union

import open3d as o3d
from PIL import Image
from sklearn.neighbors import KDTree


REFERENCE_SIZE = 100
CAMERA_VIEWS = {
    "3rd_person": {
        "zoom": 0.12,
        "front": np.array([1, 0, 0.5], dtype=np.float32),  # Camera front vector
        "lookat": np.array([1, 0.0, 0.0], dtype=np.float32),  # Point camera looks at
        "up": np.array([-0.5, 0, 1], dtype=np.float32),  # Camera up direction
    },
    "3rd_person_opposite": {
        "zoom": 0.12,
        "front": np.array([-1, 0, 0.5], dtype=np.float32),  # Camera front vector
        "lookat": np.array([1, 0.0, 0.0], dtype=np.float32),  # Point camera looks at
        "up": np.array([-0.5, 0, 1], dtype=np.float32),  # Camera up direction
    },
    "top": {
        "zoom": 0.025,
        "front": np.array([0, 0, -1], dtype=np.float32),  # Looking straight down
        "lookat": np.array([1, 0.0, 0.0], dtype=np.float32),  # Same target point
        "up": np.array([0, 1, 0], dtype=np.float32),  # Y axis is "up" in image
    },
    "side": {
        "zoom": 0.012,
        "front": np.array(
            [0, -1, 0], dtype=np.float32
        ),  # Looking from positive Y toward origin
        "lookat": np.array([1, 0.0, 0.0], dtype=np.float32),  # Same target point
        "up": np.array([0, 0, 1], dtype=np.float32),  # Z axis is up
    },
}


class Sampler:
    """Init point cloud sampler

    :param point_cloud_size: Total number of points in the point cloud
    :type point_cloud_size: int
    :param search_tree: Search tree for the point cloud data
    :type search_tree: KDTree
    :param sampler_name: Sampler name (e.g. random, spatially_regular)
    :type sampler_name: str
    :param num_classes: Number of classes in the dataset
    :type num_classes: int
    :param seed: Random seed, defaults to 42
    :type seed: int, optional
    :raises NotImplementedError: _description_
    """

    def __init__(
        self,
        point_cloud_size: int,
        search_tree: KDTree,
        sampler_name: str,
        num_classes: int,
        seed: int = 42,
    ):
        np.random.seed(seed)
        random.seed(seed)

        self.search_tree = search_tree
        self.num_classes = num_classes
        self.p = np.random.rand(point_cloud_size) * 1e-3
        self.min_p = float(np.min(self.p[-1]))

        self.test_probs = np.zeros(
            (point_cloud_size, self.num_classes), dtype=np.float32
        )

        if sampler_name == "random":
            self.sample = self.random
        elif sampler_name == "spatially_regular":
            self.sample = self.spatially_regular
        else:
            raise NotImplementedError(
                f"Sampler {self.model_cfg['sampler']} not implemented"
            )

    def _get_indices(
        self, point_cloud_size: int, num_points: int, center_point: np.ndarray
    ) -> np.ndarray:
        """Get indices to sample given a center point

        :param point_cloud_size: Current point cloud size
        :type point_cloud_size: int
        :param num_points: Number of points to sample
        :type num_points: int
        :param center_point: Center point for sampling
        :type center_point: np.ndarray
        :return: Indices of points to sample
        :rtype: np.ndarray
        """
        # Sample only if the number of points is less than the required number of points
        if point_cloud_size < num_points:
            diff = num_points - point_cloud_size
            indices = np.array(range(point_cloud_size))
            indices = list(indices) + list(random.choices(indices, k=diff))
            indices = np.asarray(indices)
        else:
            indices = self.search_tree.query(center_point, k=num_points)[1][0]

        return indices

    def random(
        self, points: np.ndarray, num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random sampling

        :param points: Point cloud data
        :type points: np.ndarray
        :param num_points: Number of points to sample
        :type num_points: int
        :return: Sampled points, and their respective indices and center point
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Get center point randomly
        center_idx = np.random.choice(len(points), 1)
        center_point = points[center_idx, :].reshape(1, -1)

        # Get indices to sample and shuffle them
        indices = self._get_indices(points.shape[0], num_points, center_point)
        random.shuffle(indices)

        # Get sampled points
        points = points[indices]

        return points, indices, center_point

    def spatially_regular(
        self,
        points: np.ndarray,
        num_points: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spatially regular sampling

        :param points: Point cloud data
        :type points: np.ndarray
        :param num_points: Number of points to sample
        :type num_points: Optional[int]
        :param radius: Radius for spatially regular sampling
        :type radius: Optional[float]
        :return: Sampled points, and their respective indices and center point
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        n = 0
        while n < 2:
            # Keep as center point the one with the lowest probability
            center_idx = np.argmin(self.p)
            center_point = points[center_idx, :].reshape(1, -1)

            # Get indices to sample
            if radius is not None:
                indices = self.search_tree.query_radius(center_point, r=radius)[0]
            elif num_points is not None:
                indices = self._get_indices(points.shape[0], num_points, center_point)
            else:
                raise ValueError("Either num_points or radius must be provided")
            n = len(indices)

            # Special case, less than 2 points in the cloud
            if n < 2:
                self.p[center_idx] += 0.001

        # Shuffle indices and sample
        random.shuffle(indices)
        points = points[indices]

        # Use inverse distance to center point as an increment in probability to be
        # sampled in the future
        dists = np.sum(np.square((points - center_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.p[indices] += delta
        new_min = float(np.min(self.p))
        self.min_p = new_min

        return points, indices, center_point


def recenter(points: np.ndarray, dims: List[int]) -> np.ndarray:
    """Recenter a point cloud along the specified dimensions

    :param points: Point cloud data
    :type points: np.ndarray
    :param dims: Dimensions to recenter
    :type dims: List[int]
    :return: Recentred point cloud data
    :rtype: np.ndarray
    """
    points[:, dims] = points[:, dims] - points.mean(0)[dims]
    return points


def build_point_cloud(
    points: np.ndarray, colors: np.ndarray
) -> o3d.geometry.PointCloud:
    """Build a point cloud

    :param points: Point cloud data
    :type points: np.ndarray
    :param colors: Colors for the point cloud data
    :type colors: np.ndarray
    :return: Point cloud
    :rtype: o3d.geometry.PointCloud
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def view_point_cloud(points: np.ndarray, colors: np.ndarray):
    """Visualize a single point cloud

    :param points: Point cloud data
    :type points: np.ndarray
    :param colors: Colors for the point cloud data
    :type colors: np.ndarray
    """
    point_cloud = build_point_cloud(points, colors)
    o3d.visualization.draw_geometries([point_cloud])


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    camera_view: Union[str, dict] = "3rd_person",
    bg_color: Optional[List[float]] = [0.0, 0.0, 0.0, 1.0],
    color_jitter: float = 0.05,
    point_size: float = 3.0,
    resolution: Tuple[int, int] = (1920, 1080),
    render_origin: bool = False,
    origin_size: float = 0.5,
) -> Image:
    """Render a given point cloud from a specific camera view and return the image

    :param points: Point cloud data
    :type points: np.ndarray
    :param colors: Colors for the point cloud data
    :type colors: np.ndarray
    :param camera_view: Camera view (either ID or dictionary containing camera definition), defaults to "3rd_person"
    :type camera_view: Union[str, dict], optional
    :param bg_color: Background color, defaults to black -> [0., 0., 0., 1.]
    :type bg_color: Optional[List[float]], optional
    :param color_jitter: Jitters the colors by a random value between [-color_jitter, color_jitter], defaults to 0.05
    :type color_jitter: float, optional
    :param point_size: Point size, defaults to 3.0
    :type point_size: float, optional
    :param resolution: Render resolution, defaults to (1920, 1080)
    :type resolution: Tuple[int, int], optional
    :param render_origin: Whether to render the origin axes, defaults to False
    :type render_origin: bool, optional
    :param origin_size: Size of the origin axes, defaults to 0.5
    :type origin_size: float, optional
    :return: Rendered point cloud
    :rtype: Image
    """
    if isinstance(camera_view, dict):
        # If camera_view is a dictionary, use it directly
        view_settings = camera_view
    elif isinstance(camera_view, str):
        # If camera_view is a string, look it up in predefined views
        assert camera_view in CAMERA_VIEWS, f"Camera view {camera_view} not implemented"
        view_settings = CAMERA_VIEWS[camera_view]

    # Add color jitter if needed
    if color_jitter > 0:
        jitter = np.random.uniform(-color_jitter, color_jitter, (points.shape[0], 1))
        colors += jitter

    # Build point cloud object
    point_cloud = build_point_cloud(points, colors)

    # Set up the offscreen renderer
    width, height = resolution
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Create material and add the point cloud to the scene
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"  # Use unlit shader for better visibility of colors
    material.sRGB_color = True
    material.point_size = point_size
    renderer.scene.add_geometry("point_cloud", point_cloud, material)

    # Add origin axes for reference
    if render_origin:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=origin_size, origin=[0, 0, 0]
        )
        coord_material = o3d.visualization.rendering.MaterialRecord()
        coord_material.shader = "defaultUnlit"  # Also unlit for visibility
        renderer.scene.add_geometry("coordinate_frame", coord_frame, coord_material)

    # Set the background color
    renderer.scene.set_background(bg_color)

    # Set up the camera
    camera_distance = 1 / view_settings["zoom"]
    camera_position = view_settings["lookat"] + view_settings["front"] * camera_distance

    renderer.setup_camera(
        vertical_field_of_view=60.0,  # Field of view in degrees
        center=view_settings["lookat"],
        eye=camera_position,
        up=view_settings["up"],
    )

    # Render the scene to an image
    image = np.asarray(renderer.render_to_image())
    image = Image.fromarray(image)

    # Cleanup
    renderer.scene.clear_geometry()

    return image


def read_semantickitti_points(fname: str, has_intensity: bool = True) -> np.ndarray:
    """Read points from a binary file in SemanticKITTI format

    :param fname: Binary file containing points
    :type fname: str
    :param has_intensity: Whether the points have intensity values, defaults to True
    :type has_intensity: bool
    :return: Numpy array containing points
    :rtype: np.ndarray
    """
    points = np.fromfile(fname, dtype=np.float32)
    points = points.reshape((-1, 4 if has_intensity else 3))
    if not has_intensity:
        empty_intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, empty_intensity], axis=1)
    return points


def read_semantickitti_label(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read labels from a binary file in SemanticKITTI format

    :param fname: Binary file containing labels
    :type fname: str
    :return: Numpy arrays containing semantic and instance labels
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    label = np.fromfile(fname, dtype=np.uint32)
    label = label.reshape((-1))
    semantic_label = label & 0xFFFF
    instance_label = label >> 16
    return semantic_label, instance_label
