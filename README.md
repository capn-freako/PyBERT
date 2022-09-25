# PyBERT

PyBERT is a serial communication link bit error rate tester simulator with a graphical user interface (GUI).

It uses the Traits/UI package of the Enthought Python Distribution (EPD) <http://www.enthought.com/products/epd.php>,
as well as the NumPy and SciPy packages.

Notice: Before using this package for any purpose, you MUST read and understand the terms put forward in the accompanying "LICENSE" file.

## Wiki

- https://github.com/capn-freako/PyBERT/wiki

## FAQ

- https://github.com/capn-freako/PyBERT/wiki/pybert_faq

## Email List

- <pybert@freelists.org>

## User Installation

- <https://github.com/capn-freako/PyBERT/wiki/instant_gratification>

## Developer Installation

- <https://github.com/capn-freako/PyBERT/wiki/dev_install>

## Testing

Tox is used for the test runner and documentation builder.  By default, it runs the following
environments: _py36_, _py37_, _pylint_, _flake8_ and _docs_.  It will skip any missing python versions.
* `pip install tox`
* `tox`

To run a single environment such as "docs" run: `tox -e docs`

## Documentation

PyBERT documentation exists in 2 separate forms:

- For developers: 
  
  - pybert/doc/build/html/index.html  (See testing on how to build the documentation)
  - https://github.com/capn-freako/PyBERT/wiki/dev_install
  
- For users:

  - Quick installation instructions at <https://github.com/capn-freako/PyBERT/wiki/instant_gratification>
  - The 'Help' tab of the PyBERT GUI
  - The PyBERT FAQ at <https://github.com/capn-freako/PyBERT/wiki/pybert_faq>
  - Sending e-mail to David Banas at <capn.freako@gmail.com>

## Acknowledgements

I would like to thank the following individuals, for their contributions to PyBERT:  

- Mark Marlett  
- Low Kian Seong  
- Amanda Bukur
- David Patterson
- Dennis Han
- Yuri Shlepnev
- Jason Ellison
- Tod Westerhoff
- Michael Gielda
- Peter Pupalaikis
