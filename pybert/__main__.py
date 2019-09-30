"""Main entry into the PyBERT GUI."""
from pybert.pybert import PyBERT
from pybert.pybert_view import MyView

thePyBERT = PyBERT()
theView = MyView(thePyBERT.trait("solver")).traits_view
thePyBERT.configure_traits(view=theView)
