# Makefile for PyBERT project.
#
# Original author: David Banas <capn.freako@gmail.com>  
# Original date:   February 10, 2015
#
# Copyright (c) 2015 David Banas; all rights reserved World wide.

.PHONY: dflt help check tox format lint flake8 type-check docs upload test clean distclean

SRC_DIR := src/pybert
DOCS_DIR := docs
UV_EXEC := uv
UVX_EXEC := uvx
PYVERS := 3.10 3.11 3.12
PROJ_VER := $(shell ${UV_EXEC} version | cut -f 2 -d ' ') 
TEST_EXP ?= tests

# Put it first so that "make" without arguments is like "make help".
dflt: help

# Prevent implicit rule searching for makefiles.
$(MAKEFILE_LIST): ;

check:
	${UVX_EXEC} -w packaging>=24.2 validate-pyproject pyproject.toml

format:
	${UVX_EXEC} isort src/ tests/
	${UVX_EXEC} black src/ tests/

lint:
	${UVX_EXEC} ruff check ${SRC_DIR}
	${UVX_EXEC} flake8 --ignore=E501,E272,E241,E222,E221 ${SRC_DIR}

type-check:
	${UV_EXEC} run mypy --install-types --follow-untyped-imports ${SRC_DIR}

docs:
	pushd ${DOCS_DIR}; PROJ_VER=${PROJ_VER} ${UV_EXEC} run sphinx-build -j auto -b html source/ build/; popd

build:
	${UV_EXEC} build --clear --no-create-gitignore

upload: build
	${UVX_EXEC} uv-publish --repo pypi dist/*

upload_test: build
	${UVX_EXEC} uv-publish --repo testpypi dist/*

test:
	@for VERSION in $(PYVERS); do \
        $(UV_EXEC) run --python $$VERSION pytest -vv \
            --cov=pyibisami --cov-report=html \
            --cov-report=term-missing $(TEST_EXP); \
    done

clean:
	rm -rf .tox build/ docs/build/ .mypy_cache .pytest_cache .venv src/*.egg-info

distclean: clean
	rm -rf dist/

help:
	@echo "Available targets:"
	@echo "=================="
	@echo "\tcheck: Validate the 'pyproject.toml' file."
	@echo "\tformat: Reformats all Python source code. USE CAUTION!"
	@echo "\tlint: Run 'ruff' and 'flake8' over the source code."
	@echo "\ttype-check: Run type checking, via 'mypy', on the source code."
	@echo "\tdocs: Run 'sphinx' on the source code, to generate documentation."
	@echo "\t\tTo view the resultant API documentation, open 'docs/build/index.html' in a browser."
	@echo "\tbuild: Build both the source tarball and wheel."
	@echo "\tupload: Upload both the source tarball and wheel to PyPi."
	@echo "\tupload_test: Upload both the source tarball and wheel to TestPyPi."
	@echo "\ttest: Run tests, using all supported Python versions."
	@echo "\tclean: Remove all previous build results, virtual environments, and cache contents."
	@echo "\tdistclean: Runs a 'make clean' and removes 'dist/'."
