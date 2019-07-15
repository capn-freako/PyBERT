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
	conda build --numpy=1.16 conda.recipe/chaco
	conda install --use-local chaco

enable:
	conda build --numpy=1.16 conda.recipe/enable
	conda install --use-local enable

pyibis-ami:
	conda build -c conda-forge conda.recipe/pyibis-ami

pyibis-ami_dev:
	conda install -n pybert64 --use-local --only-deps PyAMI/
	conda develop -n pybert64 PyAMI/

pybert: pybert_bld pybert_inst
pybert_bld:
	conda build --numpy=1.16 conda.recipe/pybert
pybert_inst:
	conda install --use-local pybert

pybert_dev: pybert_bld
	conda install -n pybert64 --use-local --only-deps pybert
	conda develop -n pybert64 .
