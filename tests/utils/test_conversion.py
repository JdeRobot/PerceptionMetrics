import pytest
import numpy as np
from PIL import Image

from perceptionmetrics.utils import conversion as uc


# To verify hex to rgb conversion
def test_hex_to_rgb():
    assert uc.hex_to_rgb("#FF5733") == (255, 87, 51)
    assert uc.hex_to_rgb("FF5733") == (255, 87, 51)
    assert uc.hex_to_rgb("#ff5733") == (255, 87, 51)

    with pytest.raises(ValueError):
        uc.hex_to_rgb("#GGGGGG")  # Invalid HEX characters

    with pytest.raises(ValueError):
        uc.hex_to_rgb("#12345")  # Invalid HEX length


# To verify ontology to rgb lut
def test_ontology_to_rgb_lut():
    ontology = {
        "car": {"idx": 1, "rgb": (255, 0, 0)},
        "tree": {"idx": 2, "rgb": (0, 255, 0)},
    }
    lut = uc.ontology_to_rgb_lut(ontology)
    expected = np.zeros((3, 3), dtype=np.uint8)  # Index 0 should be (0,0,0)
    expected[1] = (255, 0, 0)
    expected[2] = (0, 255, 0)

    assert np.array_equal(lut, expected)

    # Test non-sequential indices
    ontology = {
        "car": {"idx": 5, "rgb": (255, 0, 0)},
        "tree": {"idx": 10, "rgb": (0, 255, 0)},
    }
    lut = uc.ontology_to_rgb_lut(ontology)
    assert lut[5].tolist() == [255, 0, 0]
    assert lut[10].tolist() == [0, 255, 0]


# To verify conversion of label array to rgb array
def test_label_to_rgb():
    ontology = {
        "car": {"idx": 1, "rgb": (255, 0, 0)},
        "tree": {"idx": 2, "rgb": (0, 255, 0)},
    }

    label_image = Image.fromarray(np.array([[1, 2], [2, 1]], dtype=np.uint8))
    rgb_image = uc.label_to_rgb(label_image, ontology)

    expected_output = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 255, 0], [255, 0, 0]]], dtype=np.uint8
    )

    # Comparing label to rgb conversion with expected output
    assert np.array_equal(np.array(rgb_image), expected_output)


# To verify mapping of new and old antologies
def test_get_ontology_conversion_lut():

    # Test without translation without ignored class
    old_ontology = {
        "car": {"idx": 1},
        "tree": {"idx": 2},
    }
    compatible_new_ontology = {
        "car": {"idx": 10},
        "tree": {"idx": 20},
    }
    lut = uc.get_ontology_conversion_lut(old_ontology, compatible_new_ontology)
    expected_lut = [0, 10, 20]
    assert np.array_equal(lut, expected_lut)

    # Test with translation without ignored class
    old_ontology = {
        "car": {"idx": 1},
        "tree": {"idx": 2},
    }
    incompatible_new_ontology = {
        "vehicle": {"idx": 10},
        "plant": {"idx": 20},
    }
    translation = {
        "car": "vehicle",
        "tree": "plant",
    }
    lut = uc.get_ontology_conversion_lut(
        old_ontology, incompatible_new_ontology, translation
    )
    expected_lut = [0, 10, 20]
    assert np.array_equal(lut, expected_lut)

    # Test with translation with ignored class
    old_ontology = {
        "car": {"idx": 1},
        "tree": {"idx": 2},
    }
    incompatible_new_ontology = {
        "vehicle": {"idx": 10},
        "plant": {"idx": 20},
    }
    translation = {
        "car": "vehicle",
        "tree": "plant",
    }
    lut = uc.get_ontology_conversion_lut(
        old_ontology, incompatible_new_ontology, translation, ["tree"]
    )
    expected_lut = [0, 10, 0]
    assert np.array_equal(lut, expected_lut)

    # Test without translation with ignored class
    old_ontology = {"car": {"idx": 1}, "tree": {"idx": 2}, "road": {"idx": 3}}
    compatible_new_ontology = {
        "car": {"idx": 10},
        "tree": {"idx": 20},
    }
    lut = uc.get_ontology_conversion_lut(
        old_ontology, compatible_new_ontology, None, ["road"]
    )
    expected_lut = [0, 10, 20, 0]
    assert np.array_equal(lut, expected_lut)

    # Test with non compatible antologies without translation
    old_ontology = {
        "car": {"idx": 1},
        "tree": {"idx": 2},
    }
    incompatible_new_ontology = {
        "vehicle": {"idx": 10},
        "plant": {"idx": 20},
    }
    with pytest.raises(AssertionError):
        uc.get_ontology_conversion_lut(old_ontology, incompatible_new_ontology)
