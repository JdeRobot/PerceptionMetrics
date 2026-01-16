from glob import glob
import json
import re
from typing import List

from PIL import Image
import yaml


def read_txt(fname: str) -> List[str]:
    """Read a .txt file line by line

    :param fname: .txt filename
    :type fname: str
    :return: List of lines found in the .txt file
    :rtype: List[str]
    """
    with open(fname, "r") as f:
        data = f.read().split("\n")
    return [line for line in data if line]


def read_yaml(fname: str) -> dict:
    """Read a YAML file

    :param fname: YAML filename
    :type fname: str
    :return: Dictionary containing YAML file data
    :rtype: dict
    """
    with open(fname, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def read_json(fname: str) -> dict:
    """Read a JSON file

    :param fname: JSON filename
    :type fname: str
    :return: Dictionary containing JSON file data
    :rtype: dict
    """
    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(fname: str, data: dict):
    """Write a JSON file properly indented

    :param fname: Target JSON filename
    :type fname: str
    :param data: Dictionary containing data to be dumped as a JSON file
    :type data: dict
    """
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_image_mode(fname: str) -> str:
    """Given an image retrieve its color mode using PIL

    :param fname: Input image
    :type fname: str
    :return: PIL color image mode
    :rtype: str
    """
    with Image.open(fname) as img:
        return img.mode


def extract_wildcard_matches(pattern: str) -> list:
    """Given a pattern with wildcards, extract the matches

    :param pattern: 'Globable' pattern with wildcards
    :type pattern: str
    :return: Matches found in the pattern
    :rtype: list
    """
    regex_pattern = re.escape(pattern).replace(r"\*", "(.*?)")
    regex = re.compile(regex_pattern)
    files = glob(pattern)
    matches = [regex.match(file).groups() for file in files if regex.match(file)]
    return matches
