# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'StructInf'
copyright = '2023, StructInf Developers'
author = 'StructInf Developers'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# Build PDF & ePub

# -- Options for HTML output

# Disable link to GitHub


html_theme_options = {
    "prev_next_buttons_location": "both",
    "logo_only": True,
    
}
html_logo = "images/logo_project_resized.png"
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']


# -- Options for EPUB output
epub_show_urls = 'footnote'
