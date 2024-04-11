SHELL = /bin/bash

.PHONY: typehint
typehint:
	mypy --ignore-missing-imports src/ train.py run.py

.PHONY: test
test:
	pytest --cov=src --cov-report html tests/

.PHONY: lint
lint:
	pylint src/ train.py run.py

.PHONY: checklist
checklist: typehint test lint

.PHONY: black
black:
	black *.py

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr
	find . -type d -name __pycache__ | xargs rm -fr
	find . -type d -name mlruns | xargs rm -fr
	find . -type d -name .mypy_cache | xargs rm -fr
	find . -type d -name .pytest_cache | xargs rm -fr
	find . -type f -name '*.coverage*' | xargs rm -fr

.PHONY: train
train:
	python train.py

.PHONY: run
run:
	python run.py

.PHONY: all
all: train run

.PHONY: help
help:
	@echo "Commands:"
	@echo "TBD"