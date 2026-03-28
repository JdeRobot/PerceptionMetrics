import numpy as np
from pathlib import Path


class SemanticKITTIDataset:
    """
    Basic loader for SemanticKITTI LiDAR dataset.

    This class loads:
    - Point clouds (.bin)
    - Labels (.label)

    Each sample returns:
    (points, labels)

    points: (N, 4) -> x, y, z, intensity
    labels: (N,)   -> semantic labels
    """

    def __init__(self, dataset_dir, sequence="00"):
        self.dataset_dir = Path(dataset_dir)
        self.sequence = sequence

        # Paths
        self.velodyne_dir = self.dataset_dir / "sequences" / sequence / "velodyne"
        self.label_dir = self.dataset_dir / "sequences" / sequence / "labels"

        # Check paths
        if not self.velodyne_dir.exists():
            raise FileNotFoundError(f"Velodyne directory not found: {self.velodyne_dir}")

        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # Load files
        self.files = sorted(self.velodyne_dir.glob("*.bin"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bin_file = self.files[idx]
        label_file = self.label_dir / (bin_file.stem + ".label")

        points = self.load_point_cloud(bin_file)
        labels = self.load_labels(label_file)

        return points, labels

    @staticmethod
    def load_point_cloud(bin_path):
        """
        Load LiDAR point cloud from .bin file
        """
        return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def load_labels(label_path):
        """
        Load labels and extract semantic part
        """
        labels = np.fromfile(label_path, dtype=np.uint32)
        return labels & 0xFFFF  # remove instance info