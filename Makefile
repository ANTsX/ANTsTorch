# Makefile for antstorch
# Portable Python discovery: VIRTUAL_ENV -> CONDA_PREFIX -> python3 on PATH
ifdef VIRTUAL_ENV
    DEFAULT_PYTHON := $(VIRTUAL_ENV)/bin/python
else ifdef CONDA_PREFIX
    DEFAULT_PYTHON := $(CONDA_PREFIX)/bin/python
else
    DEFAULT_PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
endif
PYTHON ?= $(DEFAULT_PYTHON)
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff
PKG    := antstorch

.PHONY: help install install-dev install-all prefetch-data test test-fast test-slow test-all test-strict test-cov lint format docs compile audit check clean build release pypi

help:
	@echo "antstorch Development & Maintenance Tasks"
	@echo "========================================="
	@echo "  make install        Install antstorch in editable mode (-e .)"
	@echo "  make install-dev    Install antstorch in editable mode with dev/test tooling"
	@echo "  make install-all    Alias for install-dev"
	@echo "  make prefetch-data  Download and cache pretrained networks and test data"
	@echo "  make test           Run unit test suite (excluding slow tests)"
	@echo "  make test-fast      Run fast CPU-only offline unit tests"
	@echo "  make test-slow      Run slow integration and benchmark tests"
	@echo "  make test-all       Run full test suite including slow tests"
	@echo "  make test-strict    Alias for test-all with --strict-markers"
	@echo "  make test-cov       Run test suite with source coverage report"
	@echo "  make lint           Run ruff check and antsxqc.lint"
	@echo "  make format         Format code with ruff"
	@echo "  make docs           Build Sphinx HTML documentation (warnings as errors)"
	@echo "  make compile        Verify bytecode compilation without SyntaxWarnings"
	@echo "  make audit          Run compile + lint + test"
	@echo "  make check          Alias for audit"
	@echo "  make build          Build distribution packages (sdist and wheel)"
	@echo "  make release        Build and upload packages to PyPI via twine"
	@echo "  make pypi           Upload packages to PyPI and push version git tag"
	@echo "  make clean          Clean build artifacts and cache files"

install:
	@echo "Installing antstorch in editable mode using: $(PYTHON)"
	$(PIP) install -e .

install-dev:
	@echo "Installing antstorch with development and test tooling using: $(PYTHON)"
	$(PIP) install -e . pytest pytest-cov coverage ruff build twine

install-all: install-dev

prefetch-data:
	@echo "Downloading ANTsTorch data and pretrained networks using: $(PYTHON)"
	$(PYTHON) tools/data/download_antstorch_data.py --strict

lint:
	@echo "Checking code style with ruff and antsxqc.lint using: $(PYTHON)"
	$(RUFF) check $(PKG) tests
	$(PYTHON) -m antsxqc.lint $(PKG)

format:
	@echo "Formatting code with ruff using: $(PYTHON)"
	$(RUFF) format $(PKG) tests
	$(RUFF) check --fix $(PKG) tests

compile:
	$(PYTHON) -W error::SyntaxWarning -m compileall $(PKG) tests

test:
	@echo "Running test suite (not slow) using: $(PYTHON)"
	$(PYTEST) -v --strict-markers -m "not slow" tests

test-fast:
	@echo "Running fast CPU offline test suite using: $(PYTHON)"
	$(PYTEST) -v --strict-markers --test-device=cpu -m "not slow and not requires_data and not cuda and not mps" tests

test-slow:
	@echo "Running slow tests using: $(PYTHON)"
	$(PYTEST) -v --strict-markers -m "slow" tests

test-all: test-strict

test-strict:
	@echo "Running full test suite including slow tests using: $(PYTHON)"
	$(PYTEST) -v --strict-markers tests

test-cov:
	@echo "Running test suite with source coverage using: $(PYTHON)"
	$(PYTEST) --strict-markers -m "not slow" --cov=$(PKG) --cov-report=term-missing tests

docs:
	@echo "Building Sphinx HTML documentation using: $(PYTHON)"
	$(PYTHON) -m sphinx -b html -n -W --keep-going docs docs/_build/html

audit: compile lint test

check: audit

clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/ dist/ src/*.egg-info/ *.egg-info/ .eggs/ docs/_build/
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type f -name ".coverage*" -delete
	find . -type d -name "htmlcov" -prune -exec rm -rf {} +

build: clean
	@echo "Building distribution packages (sdist and wheel) using: $(PYTHON)"
	$(PYTHON) -m build

release: clean build
	@echo "Uploading packages to PyPI via twine..."
	$(PYTHON) -m twine upload dist/*

pypi: clean build
	@echo "Uploading packages to PyPI via twine..."
	$(PYTHON) -m twine upload dist/*
	@VERSION=$$($(PYTHON) -c "import antstorch; print(antstorch.__version__)"); \
	echo "Creating Git tag v$$VERSION..."; \
	git tag v$$VERSION && \
	echo "Pushing tag v$$VERSION to origin..." && \
	git push origin v$$VERSION
