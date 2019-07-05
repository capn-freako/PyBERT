# PyBERT

PyBERT is a serial communication link bit error rate tester simulator with a graphical user interface (GUI).

It uses the Traits/UI package of the Enthought Python Distribution (EPD) <http://www.enthought.com/products/epd.php>,
as well as the NumPy and SciPy packages.

Notice: Before using this package for any purpose, you MUST read and understand the terms put forward in the accompanying "LICENSE" file.

## Installation
Using Conda:

* `conda install -c dbanas pybert`
* `pybert`

Using Pip:
* `pip install numpy Cython`  Required by enable/chaco which use numpy's distutils.
* `pip install PyBERT`
* `pybert`

## Testing
Tox is used for the test runner and documentation builder.  By default, it runs the following
environments: py36, py37, pylint, flake8 and docs.  It will skip any missing python versions.
* `pip install tox`
* `tox`

To run a single environment such as "docs" run: `tox -e docs`

## Documentation
PyBERT documentation exists in 2 separate forms:

- For developers: 
  - pybert/doc/build/html/index.html  (See testing on how to build the documentation)

- For users:

  - Quick installation instructions at <https://github.com/capn-freako/PyBERT/wiki/instant_gratification>
  - The 'Help' tab of the PyBERT GUI
  - The PyBERT FAQ at <https://github.com/capn-freako/PyBERT/wiki/pybert_faq>
  - Sending e-mail to David Banas at <capn.freako@gmail.com>

## Acknowledgements

I would like to thank the following individuals, for their contributions to PyBERT:  

- Mark Marlett  
- Low Kian Seong  
- Amanda Bukur <amandabukur11@gmail.com>  

