# Makefile for PyBERT project.
#
# Original author: David Banas <capn.freako@gmail.com>  
# Original date:   February 10, 2015
#
# Copyright (c) 2015 David Banas; all rights reserved World wide.

.PHONY: dflt help check tox format lint flake8 type-check docs build install upload test clean etags conda-build conda-skeleton chaco enable pyibis-ami pyibis-ami-dev pybert pybert-dev etags

PROJ_NAME := PipBERT
PROJ_FILE := pyproject.toml
PROJ_INFO := src/PipBERT.egg-info/PKG-INFO
VER_FILE := .proj_ver
VER_GETTER := get_proj_ver
PYTHON_EXEC := python
TOX_EXEC := tox
TOX_SKIP_ENV := format
PYVERS := 39 310 311 312
PLATFORMS := lin mac win

# Put it first so that "make" without arguments is like "make help".
dflt: help

check:
	${TOX_EXEC} exec -e lint -- validate-pyproject ${PROJ_FILE}

${VER_FILE}: ${PROJ_INFO}
	${PYTHON_EXEC} -m ${VER_GETTER} ${PROJ_NAME} $@

tox:
	TOX_SKIP_ENV="${TOX_SKIP_ENV}" ${TOX_EXEC}

format:
	${TOX_EXEC} run -e format

lint:
	${TOX_EXEC} run -e lint

flake8:
	${TOX_EXEC} run -e flake8

type-check:
	${TOX_EXEC} run -e type-check

docs: ${VER_FILE}
	source $< && ${TOX_EXEC} run -e docs

build:
	${TOX_EXEC} run -e build

install: ${VER_FILE}
	source $< && ${TOX_EXEC} run -e install

upload: ${VER_FILE}
	source $< && ${TOX_EXEC} run -e upload

test:
	@for V in ${PYVERS}; do \
		for P in ${PLATFORMS}; do \
			${TOX_EXEC} run -e "py$$V-$$P"; \
		done; \
	done

clean:
	rm -rf .tox docs/build/ .mypy_cache .pytest_cache .venv

conda-skeleton:
	rm -rf conda.recipe/pybert/ conda.recipe/pyibis-ami/
	conda skeleton pypi --noarch-python --output-dir=conda.recipe pybert pyibis-ami

chaco:
	conda build --numpy=1.16 conda.recipe/chaco
	conda install --use-local chaco

enable:
	conda build --numpy=1.16 conda.recipe/enable
	conda install --use-local enable

pyibis-ami:
	conda build --numpy=1.16 conda.recipe/pyibis-ami
	conda install --use-local pyibis-ami

pyibis-ami-dev:
	conda install -n pybert64 --use-local --only-deps PyAMI/
	conda develop -n pybert64 PyAMI/

pybert: pybert_bld pybert_inst
pybert_bld:
	conda build --numpy=1.16 conda.recipe/pybert
pybert_inst:
	conda install --use-local pybert

pybert-dev: pybert_bld
	conda install -n pybert64 --use-local --only-deps pybert
	conda develop -n pybert64 .

etags:
	etags -o TAGS pybert/*.py

help:
	@echo "Available targets:"
	@echo ""
	@echo "\tPip Targets"
	@echo "\t==========="
	@echo "\ttox: Run all Tox environments."
	@echo "\tcheck: Validate the 'pyproject.toml' file."
	@echo "\tformat: Run Tox 'format' environment."
	@echo "\t\tThis will run EXTREME reformatting on the code. Use with caution!"
	@echo "\tlint: Run Tox 'lint' environment. (Runs 'pylint' on the source code.)"
	@echo "\tflake8: Run Tox 'flake8' environment. (Runs 'flake8' on the source code.)"
	@echo "\ttype-check: Run Tox 'type-check' environment. (Runs 'mypy' on the source code.)"
	@echo "\tdocs: Run Tox 'docs' environment. (Runs 'sphinx' on the source code.)"
	@echo "\t\tTo view the resultant API documentation, open 'docs/build/index.html' in a browser."
	@echo "\tbuild: Run Tox 'build' environment."
	@echo "\t\tBuilds source tarball and wheel, for installing or uploading to PyPi."
	@echo "\tinstall: Run Tox 'install' environment."
	@echo "\t\tTests installation before uploading to PyPi."
	@echo "\tupload: Run Tox 'upload' environment."
	@echo "\t\tUploads source tarball and wheel to PyPi."
	@echo "\t\t(Only David Banas can do this.)"
	@echo "\ttest: Run Tox testing for all supported Python versions."
	@echo "\tclean: Remove all previous build results, virtual environments, and cache contents."
	@echo ""
	@echo "\tConda Targets"
	@echo "\t============="
	@echo "\tconda-skeleton: Rebuild Conda build recipes for PyBERT and PyIBIS-AMI."
	@echo "\tchaco: Rebuild Conda 'Chaco' package and install into activated environment."
	@echo "\tenable: Rebuild Conda 'Enable' package and install into activated environment."
	@echo "\tpyibis-ami: Rebuild Conda 'PyIBIS-AMI' package and install into activated environment."
	@echo "\tpyibis-ami-dev: Install Conda 'PyIBIS-AMI' package in 'development' mode."
	@echo "\t\tUse this to test changes to PyIBIS-AMI code in real time."
	@echo "\tpybert: Rebuild Conda 'PyBERT' package and install into activated environment."
	@echo "\tpybert-dev: Install Conda 'PyBERT' package in 'development' mode."
	@echo "\t\tUse this to test changes to PyBERT code in real time."
	@echo ""
	@echo "\tMisc. Targets"
	@echo "\t============="
	@echo "\tetags: Rebuild Emacs tags for source code."
	@echo "\tcheck: Test the project TOML file integrity."
