.PHONY: tox clean test lint etags conda-build conda-skeleton chaco enable pyibis-ami pybert

tox:
	tox

lint:
	tox -e pylint

test:
	tox -e py37

docs:
	# Docs doesn't rely on docker but does require tox to be installed via pip.
	tox -e docs

clean:
	rm -rf .tox docs/_build/ .pytest_cache .venv

conda-build:
	conda build conda.recipe/pybert

conda-skeleton:
	rm -rf conda.recipe/pybert/ conda.recipe/pyibis-ami/ \
	conda skeleton pypi --noarch-python --output-dir=conda.recipe pybert pyibis-ami

etags:
	etags -o TAGS pybert/*.py

chaco:
	conda build -c conda-forge conda.recipe/chaco

enable:
	conda build -c conda-forge conda.recipe/enable

pyibis-ami:
	conda build -c conda-forge conda.recipe/pyibis-ami

pybert:
	conda build -c conda-forge conda.recipe/pybert
