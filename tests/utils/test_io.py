import json
from PIL import Image
from unittest.mock import mock_open, patch
from perceptionmetrics.utils.io import (
    read_txt,
    read_yaml,
    read_json,
    write_json,
    get_image_mode,
    extract_wildcard_matches,
)


# Test read_txt
def test_read_txt():
    mock_content = "line1\nline2\nline3\n"
    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = read_txt("dummy.txt")
        assert result == ["line1", "line2", "line3"]


# Test read_yaml
def test_read_yaml():
    mock_yaml = "key1: value1\nkey2: value2"
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        result = read_yaml("dummy.yaml")
        assert result == {"key1": "value1", "key2": "value2"}


# Test read_json
def test_read_json():
    mock_json = json.dumps({"key": "value"})
    with patch("builtins.open", mock_open(read_data=mock_json)):
        result = read_json("dummy.json")
        assert result == {"key": "value"}


# Test write_json
def test_write_json():
    data = {"name": "pytest"}
    mock_file = mock_open()

    with patch("builtins.open", mock_file):
        write_json("dummy.json", data)

    # Retrieve all calls to write()
    written_data = "".join(call.args[0] for call in mock_file().write.call_args_list)

    assert json.loads(written_data) == data


# Test get_image_mode
def test_get_image_mode(tmp_path):
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    assert get_image_mode(str(img_path)) == "RGB"


# Test extract_wildcard_matches
def test_extract_wildcard_matches(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    with patch(
        "perceptionmetrics.utils.io.glob",
        return_value=[str(tmp_path / "file1.txt"), str(tmp_path / "file2.txt")],
    ):
        matches = extract_wildcard_matches(str(tmp_path / "*.txt"))
        assert len(matches) == 2
