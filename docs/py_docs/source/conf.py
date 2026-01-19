import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PerceptionMetrics'
copyright = '2026, JdeRobot'
author = 'JdeRobot'
release = '3.0.0'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"

autodoc_mock_imports = [
    "c_gen_normal_map",
    "mmdet3d",
    "mmengine",
    "open3d",
    "spconv",
    "tensorflow",
    "torch",
    "torchvision",
    "util",  # sphereformer 3rd party util folder
    "utils",  # lsk3dnet 3rd party utils folder
]

