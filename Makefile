# Convenience wrappers. The only target you actually need is `test`, and it
# requires nothing installed.
.PHONY: help test test-pytest lint format typecheck examples coverage build clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test:  ## Run the test suite (stdlib only, no install, no network)
	python -m unittest discover -s tests -t .

test-pytest:  ## Run the test suite under pytest
	pytest

coverage:  ## Run under pytest with a coverage report (same floor CI enforces)
	pytest --cov=agentguard --cov-report=term-missing --cov-fail-under=90

lint:  ## Check style with ruff
	ruff check .
	ruff format --check --diff .

format:  ## Auto-fix style with ruff
	ruff check --fix .
	ruff format .

typecheck:  ## Type-check the package with mypy --strict
	mypy

examples:  ## Run every example end to end
	PYTHONPATH=src python examples/basic.py
	PYTHONPATH=src python examples/loop_detection.py
	PYTHONPATH=src python examples/wrapped_client.py
	PYTHONPATH=src python examples/report_demo.py
	PYTHONPATH=src python examples/streaming.py
	PYTHONPATH=src python examples/langgraph_demo.py
	PYTHONPATH=src python examples/custom_models.py
	PYTHONPATH=src python examples/checkpointing.py

build:  ## Build sdist and wheel into dist/
	python -m build
	python -m twine check dist/*

clean:  ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: lint typecheck test examples  ## Everything CI runs
