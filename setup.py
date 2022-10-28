""" setup.py - Distutils setup file for PyBERT package

    David Banas
    October 22, 2014
"""

from setuptools import setup, find_packages

setup(
    name="PyBERT",
    version="3.5.7",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    license="BSD",
    description="Serial communication link bit error rate tester simulator, written in Python.",
    long_description=open("README.md").read(),
    url="https://github.com/capn-freako/PyBERT/wiki",
    author="David Banas",
    author_email="capn.freako@gmail.com",
    install_requires=[
        "chaco==5.0.0",
        "click==8.1.3",
        "enable==5.3.1",
        "numpy==1.23.3",
        "scikit-rf==0.23.1",
        "scipy==1.9.2",
        "traits==6.4.1",
        "traitsui==7.4.1",
        "PyIBIS-AMI>=3.5.0",
        "pyyaml==6.0",
        "pyside2==5.15.2.1",
    ],
    entry_points={
        "console_scripts": [
            "pybert = pybert.__main__:main",
        ]
    },
    keywords=["bert", "communication", "simulator"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Emulators",
        "Topic :: System :: Networking",
        "Topic :: Utilities"
    ],
)
