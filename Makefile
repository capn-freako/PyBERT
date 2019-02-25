DOCKER=docker run --env="DISPLAY" --net=host -v ~/git/PyBERT:/data/PyBERT:rw \
	-v ~/git/PyAMI:/data/PyAMI:rw -it pybert
.PHONY: tox clean test lint docker-build docker-shell etags conda-build conda-skeleton

tox:
	$(DOCKER) tox

lint:
	$(DOCKER) tox -e pylint

test:
	$(DOCKER) tox -e py37

docs:
	# Docs doesn't rely on docker but does require tox to be installed via pip.
	tox -e docs

clean:
	rm -rf .tox docs/_build/ .pytest_cache .venv

docker-build:
	docker build -t pybert .

docker-shell:
	$(DOCKER) /bin/bash

conda-build:
	conda build conda.recipe/pybert

conda-skeleton:
	rm -rf conda.recipe/pybert/ conda.recipe/pyibis-ami/ \
	conda skeleton pypi --noarch-python --output-dir=conda.recipe pybert pyibis-ami

etags:
	etags -o TAGS pybert/*.py
