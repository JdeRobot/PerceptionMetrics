import numpy as np
import random
from typing import List, Optional, Tuple

import open3d as o3d
from sklearn.neighbors import KDTree


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


def render_point_cloud(points: np.ndarray, colors: np.ndarray):
    """Render a single point cloud

    :param points: Point cloud data
    :type points: np.ndarray
    :param colors: Colors for the point cloud data
    :type colors: np.ndarray
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])
