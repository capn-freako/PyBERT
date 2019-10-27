"""Main entry into the PyBERT GUI."""
from pybert.pybert      import PyBERT
from pybert.pybert_view import traits_view

thePyBERT = PyBERT()
thePyBERT.configure_traits(view=traits_view)
