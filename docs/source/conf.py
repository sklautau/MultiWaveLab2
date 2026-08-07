# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
project = 'MultiWaveLab'
copyright = '2026, Sofia Klautau'
author = 'Sofia Klautau'


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [] # disable extensions to avoid errors when building docs
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"  # new theme
html_static_path = ['_static']

autoapi_type = "python"

autoapi_dirs = [
    str(ROOT / "src")
]
