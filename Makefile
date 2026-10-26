# Re-derivation of the numbers, tables and figures of the associated study from the stored records.
# Activate the conda environment first (conda activate dftgnn), or pass PYTHON=/path/to/python.
# Runtimes were measured on a laptop with an 8-core Apple-silicon CPU; no target needs a GPU.

PYTHON ?= python
ENV_NAME ?= dftgnn
ENV_FILE ?= env/environment-local.yml
CONDA ?= $(shell command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
export MPLBACKEND := Agg

.DEFAULT_GOAL := help
.PHONY: help env test specs-check numbers tables figures all clean

help:  ## list the targets
	@echo "Targets (run 'conda activate $(ENV_NAME)' first; none needs a GPU):"
	@echo "  make env          create or update the conda env '$(ENV_NAME)' from $(ENV_FILE), install the package (about 10 min, network; no GPU)"
	@echo "  make test         unit tests of the package and the analysis code (about 3-5 min on a laptop, no GPU)"
	@echo "  make specs-check  job specs against splits, configs, run ids and run records (about 25 s, no GPU)"
	@echo "  make numbers      re-derive the headline quantities from the per-site predictions (about 1 min, no GPU)"
	@echo "  make tables       rebuild the 18 CSV tables from the stored records and compare (about 1 s, no GPU)"
	@echo "  make figures      regenerate the figures as PDF into figures/main and figures/si (about 10 s, no GPU)"
	@echo "  make all          env test specs-check numbers tables figures (about 6 min after env, no GPU)"
	@echo "  make clean        remove caches and the generated figure PDFs"

env:  ## conda environment from env/environment-local.yml, then the package in editable mode
	$(CONDA) env create -y -n $(ENV_NAME) -f $(ENV_FILE) || $(CONDA) env update -y -n $(ENV_NAME) -f $(ENV_FILE)
	$(CONDA) run -n $(ENV_NAME) python -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest -q -rs tests analysis/tests

specs-check:
	$(PYTHON) reproduce/check_specs.py

numbers:
	$(PYTHON) reproduce/numbers.py

tables:
	$(PYTHON) reproduce/tables.py

figures:
	$(PYTHON) reproduce/figures.py

all: env test specs-check numbers tables figures

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f figures/main/*.pdf figures/si/*.pdf
