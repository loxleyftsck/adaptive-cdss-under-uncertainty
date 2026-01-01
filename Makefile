.PHONY: help install install-dev test lint format train evaluate mlflow-ui docker-build docker-up clean

# Default target
help:
	@echo "RL-CDSS Makefile Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests with coverage"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code (black, isort)"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train          - Train baseline models"
	@echo "  make evaluate       - Run robustness evaluation"
	@echo "  make mlflow-ui      - Start MLflow tracking UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker containers"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean cache and temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black

train:
	python scripts/experiments/run_baseline_comparison.py

evaluate:
	python scripts/experiments/run_robustness_test.py

mlflow-ui:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
	@echo "✅ Cleaned cache and temporary files"
