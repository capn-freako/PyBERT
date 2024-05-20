"""Main entry into the PyBERT GUI when called with python -m.

This is now largely for debug or if users want to use the python
-m option since calling `pybert` will instead point to cli.py.
"""
from pybert.gui.view import traits_view
from pybert.pybert import PyBERT


def main():
    "Run the PyBERT GUI."
    thePyBERT = PyBERT()
    thePyBERT.configure_traits(view=traits_view)


if __name__ == "__main__":
    main()
