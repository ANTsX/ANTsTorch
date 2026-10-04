"""Documentation builds do not import ANTsTorch or download model weights."""
import os

project = "ANTsTorch"
author = "ANTsX contributors"
extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "README.md"]
html_theme = "sphinx_rtd_theme"
html_title = "ANTsTorch documentation"
myst_heading_anchors = 3
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
