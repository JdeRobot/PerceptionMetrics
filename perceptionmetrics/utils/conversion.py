from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def hex_to_rgb(hex: str) -> Tuple[int, ...]:
    """Convert HEX color code to sRGB

    :param hex: HEX color code
    :type hex: str
    :return: sRGB color value
    :rtype: Tuple[int, ...]
    """
    hex = hex.strip("#")
    if len(hex) != 6:
        raise ValueError("Invalid hex code: Must be exactly 6 characters long")

    if not tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4)):
        raise ValueError("Invalid hex code: Contains non-hexadecimal characters")
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def ontology_to_rgb_lut(ontology: dict) -> np.ndarray:
    """Given an ontology definition, build a LUT that links indices and RGB values

    :param ontology: Ontology definition
    :type ontology: dict
    :return: numpy array containing RGB values per index
    :rtype: np.ndarray
    """
    max_idx = max(class_data["idx"] for class_data in ontology.values())
    lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for class_data in ontology.values():
        lut[class_data["idx"]] = class_data["rgb"]
    return lut


def label_to_rgb(label: Image.Image, ontology: dict) -> Image.Image:
    """Convert an image with raw label indices to RGB mask

    :param label: Raw label indices as PIL image
    :type label: Image.Image
    :param ontology: Ontology definition
    :type ontology: dict
    :return: RGB mask
    :rtype: Image.Image
    """
    label = np.array(label)
    lut = ontology_to_rgb_lut(ontology)
    image = lut[label]
    return Image.fromarray(image, mode="RGB")


def get_ontology_conversion_lut(
    old_ontology: dict,
    new_ontology: dict,
    ontology_translation: Optional[dict] = None,
    classes_to_remove: Optional[List[str]] = None,
    lut_dtype: Optional[np.dtype] = np.uint8,
) -> np.ndarray:
    """Build a LUT that links old ontology and new ontology indices. If class names
    don't match between the provided ontologies, user must provide an ontology
    translation dictionary with old and new class names as keys and values, respectively

    :param old_ontology: Origin ontology definition
    :type old_ontology: dict
    :param new_ontology: Target ontology definition
    :type new_ontology: dict
    :param ontology_translation: Ontology translation dictionary, defaults to None
    :type ontology_translation: Optional[dict], optional
    :param classes_to_remove: Classes to be removed from the old ontology, defaults to None
    :type classes_to_remove: Optional[List[str]], optional
    :param lut_dtype: Type for the ontology conversion LUT, defaults to np.uint8
    :type lut_dtype: Optional[np.dtype], optional
    :return: numpy array associating old and new ontology indices
    :rtype: np.ndarray
    """
    classes_to_remove = [] if classes_to_remove is None else classes_to_remove

    max_idx = max(class_data["idx"] for class_data in old_ontology.values())
    lut = np.zeros((max_idx + 1), dtype=lut_dtype)
    if ontology_translation is not None:
        # Deleting requested classes from ontology translation
        for class_name in classes_to_remove:
            if class_name in ontology_translation:
                del ontology_translation[class_name]

        # Mapping old and new class names through ontology_translation
        for old_class_name, new_class_name in ontology_translation.items():
            old_class_idx = old_ontology[old_class_name]["idx"]
            new_class_idx = new_ontology[new_class_name]["idx"]
            lut[old_class_idx] = new_class_idx
    else:
        old_ontology = old_ontology.copy()
        # Deleting classes requested from old ontology
        for class_name in classes_to_remove:
            del old_ontology[class_name]
        assert set(old_ontology.keys()) == set(  # Checking ontology compatibility
            new_ontology.keys()
        ), "Ontologies classes are not compatible"
        for class_name, class_data in old_ontology.items():
            lut[class_data["idx"]] = new_ontology[class_name]["idx"]
    return lut
