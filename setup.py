''' setup.py - Distutils setup file for PyBERT package

    David Banas
    October 22, 2014
'''

from distutils.core import setup

setup(
    name='PyBERT',
    version='0.3',
    packages=['pybert',],
    license='BSD',
    long_description=open('README.txt').read(),
    url='https://github.com/capn-freako/PyBERT/wiki',
    author='David Banas',
    author_email='capn.freako@gmail.com',
    install_requires = [
        'numpy',
        'scipy',
        'traits',
        'traitsui',
        'enable',
        'chaco',
        ],
)
