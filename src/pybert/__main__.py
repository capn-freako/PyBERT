"""Main entry into the PyBERT GUI."""
from pybert.gui.view import traits_view
from pybert.pybert import PyBERT


def main():
    thePyBERT = PyBERT()
    thePyBERT.configure_traits(view=traits_view)


if __name__ == "__main__":
    main()
