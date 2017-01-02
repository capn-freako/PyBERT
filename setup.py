''' setup.py - Distutils setup file for PyBERT package

    David Banas
    October 22, 2014
'''

from setuptools import setup

setup(
    name='PyBERT',
    version='2.0.1',
    packages=['pybert',],
    license='BSD',
    description='Serial communication link bit error rate tester simulator, written in Python.',
    long_description=open('README.txt').read(),
    url='https://github.com/capn-freako/PyBERT/wiki',
    author='David Banas',
    author_email='capn.freako@gmail.com',
    install_requires = [
        'traits == 4.4.0',
        'traitsui == 4.4.0',
        'enable == 4.4.1',
        'chaco == 4.5.0',
        'Sphinx == 1.2.3',
        'Jinja2 == 2.7.3',
        'docutils == 0.12',
        'Pygments == 2.1.3',
        'PyIBIS-AMI >= 2.0.1',
        ],
    keywords = ['bert', 'communication', 'simulator'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Emulators",
        "Topic :: System :: Networking",
        "Topic :: Utilities"
    ]
)
