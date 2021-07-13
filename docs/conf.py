# MOVE DOCS1 TO DOCS BOTTOM OF THIS FILE


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import inspect


def abspath(rel):
    """
    Take paths relative to the current file and
    convert them to absolute paths.

    Parameters
    ------------
    rel : str
      Relative path, IE '../stuff'

    Returns
    -------------
    abspath : str
      Absolute path, IE '/home/user/stuff'
    """

    # current working directory
    cwd = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    return os.path.abspath(os.path.join(cwd, rel))


extensions = ['sphinx.ext.napoleon', 'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'XMCD-projection'
copyright = '2021, Luka Skoric'
author = 'Luka Skoric'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# grab from library without installing
exec(open(abspath('../xmcd_projection/version.py')).read())
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output --------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# options for rtd-theme
html_theme_options = {
    'analytics_id': 'UA-161434837-1',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/custom.css',  # override non-wrapping tables in RTD theme
    ],
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": "Skoricius",
    "github_repo": "xmcd-projection",
    "github_version": "master",
    "conf_py_path": "/docs/",
}

# Output file base name for HTML help builder.
htmlhelp_basename = 'xmcdprojdoc'

# -- Extensions configuration ----------------------------------------

autodoc_default_options = {
    'autosummary': True,
}
