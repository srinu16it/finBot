.PHONY: help install test format lint clean pre-commit validate

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make format       Format code with black"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Remove cache and build files"
	@echo "  make pre-commit   Install pre-commit hooks"
	@echo "  make validate     Check for modifications in protected folders"

install:
	pip install -r requirements.txt
	pip install pre-commit

test:
	pytest

format:
	black enhancements/ tests/
	@echo "Note: myfirstfinbot/ folder is protected and will not be formatted"

lint:
	ruff check enhancements/ tests/
	@echo "Note: myfirstfinbot/ folder is protected and will not be linted"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf enhancements/cache/
	rm -f .coverage

pre-commit:
	pre-commit install
	pre-commit run --all-files

validate:
	python enhancements/tools/validate_protected_folders.py 