"""
Configuration file for the Sphinx documentation builder.
Simplified version for reliable ReadtheDocs builds.
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "Hierarchical Data Simulator"
copyright = f"{datetime.now().year}, Moses Kabungo"
author = "Moses Kabungo"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------

# Add only core Sphinx extensions that are available by default
extensions = [
    # Core Sphinx extensions (these should always be available)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
]

# Optional extensions - only add if available
try:
    import sphinx_copybutton

    extensions.append("sphinx_copybutton")
except ImportError:
    pass

try:
    import sphinx_rtd_theme

    extensions.append("sphinx_rtd_theme")
except ImportError:
    pass

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": None,
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2563eb",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files (only if they exist)
html_css_files = []
if os.path.exists(os.path.join(os.path.dirname(__file__), "_static", "custom.css")):
    html_css_files.append("custom.css")

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Don't show full module names in API docs
add_module_names = False

# -- Options for autosummary extension ---------------------------------------

autosummary_generate = True
autosummary_imported_members = False

# -- Options for napoleon extension ------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# -- Options for copybutton extension ----------------------------------------
# Only configure if the extension is available
if "sphinx_copybutton" in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
    copybutton_prompt_is_regexp = True

# -- Options for autosectionlabel extension ---------------------------------

autosectionlabel_prefix_document = True

# -- LaTeX output options ----------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files.
latex_documents = [
    (
        master_doc,
        "hierarchical-simulator.tex",
        "Hierarchical Data Simulator Documentation",
        author,
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (
        master_doc,
        "hierarchical-simulator",
        "Hierarchical Data Simulator Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "hierarchical-simulator",
        "Hierarchical Data Simulator Documentation",
        author,
        "hierarchical-simulator",
        "A comprehensive hierarchical data simulator for multilevel modeling research.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

# -- Version handling --------------------------------------------------------

try:
    import hierarchical_simulator

    release = hierarchical_simulator.__version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    # Fallback if package is not installed
    pass
