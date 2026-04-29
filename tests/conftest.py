# tests/conftest.py
# Pre-import open3d before any test module is collected.
# test_yolo.py stubs open3d only if it's NOT already in sys.modules.
# Loading it here first ensures test_lidar.py gets the real open3d.
import open3d  # noqa: F401
