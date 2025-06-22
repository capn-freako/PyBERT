PyBERT
======

PyBERT is a serial communication link bit error rate tester simulator with a graphical user interface (GUI).

It uses the Traits/UI package of the Enthought Python Distribution (EPD) <http://www.enthought.com/products/epd.php>,
as well as the NumPy and SciPy packages.

**Notice:** Before using this package for any purpose, you MUST read and understand the terms put forward in the accompanying "LICENSE" file.

User Installation
-----------------

- <https://github.com/capn-freako/PyBERT/wiki/instant_gratification>

Developer Installation
----------------------

- <https://github.com/capn-freako/PyBERT/wiki/Developer_Instructions>

Wiki
----

- https://github.com/capn-freako/PyBERT/wiki

FAQ
---

- https://github.com/capn-freako/PyBERT/wiki/pybert_faq

Email List
----------

- <pybert@freelists.org>

Testing
-------

Tox is used for the test runner and documentation builder.  By default, it will try to unit test
for any installed/supported of versions and it will skip any missing versions.

- `pip install tox`
- `tox -p all`

To run a single environment such as "docs" run: `tox run -e docs`

Documentation
-------------

PyBERT documentation exists in 2 separate forms:

- For developers:

  - pybert/doc/build/html/index.html  (See testing on how to build the documentation)
  - https://github.com/capn-freako/PyBERT/wiki/dev_install

- For users:

  - Quick installation instructions at <https://github.com/capn-freako/PyBERT/wiki/instant_gratification>
  - The 'Help' tab of the PyBERT GUI
  - The PyBERT FAQ at <https://github.com/capn-freako/PyBERT/wiki/pybert_faq>
  - Sending e-mail to David Banas at <capn.freako@gmail.com>

Acknowledgments
---------------

I would like to thank the following individuals for their contributions to the *PyBERT* project:

**David Patterson** for being my main co-author and for his countless hours
driving the PyBERT project across the Python2<=>Python3 divide, as well as,
more recently, completely updating its build infrastructure to be more in sync.
w/ modern Python package building/testing/distribution philosophy.

**Peter Pupalaikis** for sharing his expertise w/ both Fourier transform and
S-parameter subtleties. The PyBERT source code wouldn't have nearly the
mathematical/theoretical fidelity that it does had Peter not contributed.

**Yuri Shlepnev** for his rock solid understanding of RF fundamentals, as
well as his infinite patience in helping me understand them, too. ;-)

**Dennis Han** for thoroughly beating the snot out of PyBERT w/ nothing but
love in his heart and determination in his mind to drive PyBERT further towards
a professional level of quality. Dennis has made perhaps the most significant
contributions towards making PyBERT a serious tool for the working professional
serial communications link designer.

**Todd Westerhoff** for helping me better understand what tool features really
matter to working professional link designers and which are just in the way.
Also, for some very helpful feedback, re: improving the real World PyBERT
experience for the user.

**Mark Marlett** for first introducing me to Python/NumPy/SciPy, as an
alternative to MATLAB for numerical computing and signal processing, as
well as his countless hours of tutelage, regarding the finer points of
serial communication link simulation technique. Mark is also the one,
who insisted that I take a break from development and finally write
some documentation, so that others could understand what I intended and,
hopefully, contribute. Thanks, Mark!

**Low Kian Seong** for straightening out my understanding of the real
purpose of the *description* field in the *setup.py* script.

**Jason Ellison** for many cathartic chats on the topic of quality open
source software creation, maintenance, and distribution.

**Michael Gielda** & **Denz Choe** for their contributions to the PyBERT
code base.

**The entire SciKit-RF team** for creating and supporting an absolutely
wonderful Python package for working with RF models and simulations.
