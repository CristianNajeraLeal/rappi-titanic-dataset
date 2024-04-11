"""
Initialization module for the src package.

Modules:
    utils: A submodule containing utility functions or classes.
    fit_model: A submodule dedicated to fitting the Titanic classification model.
"""

from . import utils, fit_model

__all__ = [
    'utils',
    'fit_model'
]
