from typing import Optional, Tuple

import numpy as np
try:
    from open3d._ml3d.datasets.utils import DataProcessing
except Exception:
    print("Open3D-ML3D not available")
from sklearn.neighbors import KDTree

from detectionmetrics.models.torch_model_utils import o3d_randlanet, o3d_kpconv


# Default functions
def preprocess(
    points: np.ndarray, cfg: Optional[dict] = {}
) -> Tuple[np.ndarray, KDTree, np.ndarray]:
    """Preprocess point cloud data

    :param points: Point cloud data
    :type points: np.ndarray
    :param cfg: Dictionary containing model configuration, defaults to {}
    :type cfg: Optional[dict], optional
    :return: Subsampled points, search tree, and projected indices
    :rtype: Tuple[np.ndarray, KDTree, np.ndarray]
    """
    # Keep only XYZ coordinates
    points = np.array(points[:, 0:3], dtype=np.float32)

    # Subsample points using a grid of given size
    grid_size = cfg.get("grid_size", 0.06)
    sub_points = DataProcessing.grid_subsampling(points, grid_size=grid_size)

    # Create search tree so that we can project points back to the original point cloud
    search_tree = KDTree(sub_points)
    projected_indices = np.squeeze(search_tree.query(points, return_distance=False))
    projected_indices = projected_indices.astype(np.int32)

    return sub_points, search_tree, projected_indices


transform_input = o3d_randlanet.transform_input
update_probs = o3d_randlanet.update_probs
