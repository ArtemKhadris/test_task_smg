# Makefile - basic automation targets
.PHONY: install prepare-data train evaluate serve run-experiments clean lint test

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

prepare-data:
	python -m src.cli prepare-data --config config/default.yaml

train:
	python -m src.cli train --config config/default.yaml

evaluate:
	python -m src.cli evaluate --config config/default.yaml

serve:
	python -m src.cli serve --config config/default.yaml

run-experiments:
	python -m src.cli run-experiments --configs config/experiments/exp1.yaml config/experiments/exp2.yaml config/experiments/exp3.yaml

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf mlruns
	rm -rf models/*.joblib
	rm -rf reports/*

lint:
	flake8 src

test:
	pytest -q

# Notes for Windows PowerShell users (run these commands instead of make):
#   Install:   python -m pip install -r requirements.txt
#   Prepare:   python -m src.cli prepare-data --config config/default.yaml
#   Train:     python -m src.cli train --config config/default.yaml
#   Evaluate:  python -m src.cli evaluate --config config/default.yaml
#   Serve:     python -m src.cli serve --config config/default.yaml
