"""Main entry into the PyBERT GUI."""
from .pybert import PyBERT
from .pybert_view import traits_view

PyBERT().configure_traits(view=traits_view)
