import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "moirestrain"
author = "Kohei"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_sitemap",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["extra"]
html_meta = {
    "description": "moirestrain is a NumPy-first package for sampling moire analysis of grating images.",
    "keywords": "sampling moire, moire, displacement, strain, phase analysis, optical metrology, Python",
    "author": author,
    "robots": "index, follow",
}

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    html_baseurl = "https://moirestrain.readthedocs.io/en/latest/"
else:
    html_baseurl = "https://kevin-tofu.github.io/moirestrain/"
