# Recommendation System Makefile

.PHONY: help train serve test clean install build-train build-serve

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  train       - Run training pipeline"
	@echo "  serve       - Start the recommendation server"
	@echo "  test        - Run all tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  build-train - Build training module into binary"
	@echo "  build-serve - Build serve module into binary"

# Install dependencies
install:
	pip install -r requirements.txt

# Run training pipeline
train:
	python -m train.main

# Start the recommendation server
serve:
	python -m serve.main

# Run all tests
test:
	python3 -m unittest discover tests

run-train:
	python -m train.main

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Build training module into binary
build-train:
	# Implementation for building train module to binary will go here
	@echo "Building train module to binary..."

# Build serve module into binary
build-serve:
	# Implementation for building serve module to binary will go here
	@echo "Building serve module to binary..."
