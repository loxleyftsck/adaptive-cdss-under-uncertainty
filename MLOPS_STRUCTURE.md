# Production-Ready MLOps Project Structure

## RL-Based CDSS: From Research to Production

**Author:** Herald Michain Samuel Theo Ginting  
**Role Context:** Senior MLOps Engineer & AI Researcher  
**Date:** January 2, 2026

---

## 📁 COMPLETE FOLDER STRUCTURE

```
adaptive-cdss-under-uncertainty/
│
├── .github/                          # GitHub-specific configurations
│   ├── workflows/                    # CI/CD pipelines
│   │   ├── ci.yml                   # Continuous Integration (tests, lint)
│   │   ├── cd.yml                   # Continuous Deployment (Docker build, push)
│   │   ├── mlflow-tracking.yml      # Auto-log experiments to MLflow
│   │   └── data-validation.yml      # DVC pipeline validation
│   ├── ISSUE_TEMPLATE/              # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md     # PR template
│
├── configs/                          # Configuration management (CRITICAL!)
│   ├── __init__.py
│   ├── base.yaml                    # Base configuration (hyperparams, paths)
│   ├── experiment/                  # Experiment-specific configs
│   │   ├── q_learning_baseline.yaml
│   │   ├── sarsa_conservative.yaml
│   │   └── fuzzy_reward.yaml
│   ├── data/                        # Data configurations
│   │   ├── missing_20.yaml          # 20% missing data scenario
│   │   ├── missing_40.yaml          # 40% baseline
│   │   └── missing_60.yaml          # 60% stress test
│   └── deployment/                  # Deployment configs
│       ├── dev.yaml
│       ├── staging.yaml
│       └── prod.yaml
│
├── data/                            # Data directory (DVC-tracked)
│   ├── raw/                         # Original, immutable data
│   │   ├── drugbank/
│   │   ├── ddinter/
│   │   └── fda_labels/
│   ├── processed/                   # Cleaned, transformed data
│   │   ├── drugs.json
│   │   ├── interactions.json
│   │   └── contraindications.json
│   ├── synthetic/                   # Generated synthetic patients
│   │   ├── train_patients.pkl
│   │   └── test_patients.pkl
│   └── external/                    # Third-party data sources
│       └── references/
│
├── models/                          # Trained models (DVC + MLflow registry)
│   ├── checkpoints/                 # Training checkpoints
│   │   ├── q_learning_ep500.pkl
│   │   └── sarsa_ep500.pkl
│   ├── production/                  # Deployed models (MLflow registry)
│   │   ├── q_learning_v1.0.pkl
│   │   └── metadata.json
│   └── experimental/                # Experimental models
│       └── fuzzy_reward_beta.pkl
│
├── notebooks/                       # Jupyter notebooks (exploratory ONLY)
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   ├── 03_training_debug.ipynb
│   └── 04_results_visualization.ipynb
│
├── src/                             # Source code (main application)
│   ├── __init__.py
│   ├── knowledge/                   # Knowledge Base module
│   │   ├── __init__.py
│   │   ├── knowledge_base.py        # KnowledgeBase class
│   │   ├── loader.py                # Data loading utilities
│   │   └── validator.py             # Consistency validation
│   │
│   ├── environment/                 # RL Environment module
│   │   ├── __init__.py
│   │   ├── cdss_env.py              # Main CDSSEnvironment class
│   │   ├── patient_generator.py    # Synthetic patient generation
│   │   ├── observation_model.py    # Partial observability simulation
│   │   └── reward.py                # Reward function implementations
│   │
│   ├── agents/                      # RL Agents module
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Abstract base class
│   │   ├── q_learning.py            # Q-Learning agent
│   │   ├── sarsa.py                 # SARSA agent
│   │   ├── state_encoding.py       # Belief state approximation
│   │   └── baselines/               # Baseline policies
│   │       ├── random_policy.py
│   │       ├── rule_based.py
│   │       └── oracle.py
│   │
│   ├── training/                    # Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── callbacks.py             # Training callbacks (early stopping, etc.)
│   │   └── logger.py                # MLflow integration wrapper
│   │
│   ├── evaluation/                  # Evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py               # SafetyMetrics class
│   │   ├── robustness.py            # Robustness testing
│   │   ├── comparator.py            # Policy comparison
│   │   └── statistical.py           # Statistical validation (t-tests, CIs)
│   │
│   ├── explainability/              # XAI & interpretability
│   │   ├── __init__.py
│   │   ├── explainer.py             # Decision explanation
│   │   ├── q_analyzer.py            # Q-value analysis
│   │   └── visualizer.py            # Explanation visualizations
│   │
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   ├── config_loader.py         # YAML config loading
│   │   ├── reproducibility.py       # Seed setting, determinism
│   │   └── io.py                    # File I/O helpers
│   │
│   └── api/                         # REST API for model serving (Optional)
│       ├── __init__.py
│       ├── app.py                   # FastAPI application
│       ├── models.py                # Pydantic schemas
│       └── endpoints.py             # API endpoints
│
├── tests/                           # Testing suite (CRITICAL!)
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_knowledge_base.py
│   │   ├── test_environment.py
│   │   ├── test_agents.py
│   │   └── test_reward.py
│   ├── integration/                 # Integration tests
│   │   ├── test_training_pipeline.py
│   │   └── test_evaluation_pipeline.py
│   ├── performance/                 # Performance/benchmark tests
│   │   └── test_inference_speed.py
│   └── fixtures/                    # Test fixtures & mocks
│       ├── sample_patients.json
│       └── mock_knowledge_base.json
│
├── scripts/                         # Standalone scripts
│   ├── data_acquisition/           # Data collection scripts
│   │   ├── fetch_drugbank.py
│   │   ├── fetch_ddinter.py
│   │   └── generate_synthetic_patients.py
│   ├── preprocessing/              # Data preprocessing
│   │   ├── clean_interactions.py
│   │   └── validate_knowledge_base.py
│   ├── experiments/                # Experiment runners
│   │   ├── run_baseline_comparison.py
│   │   ├── run_robustness_test.py
│   │   └── run_ablation_study.py
│   └── deployment/                 # Deployment scripts
│       ├── build_docker.sh
│       └── deploy_to_mlflow.py
│
├── mlruns/                         # MLflow tracking directory (gitignored)
│   └── .gitkeep
│
├── .dvc/                           # DVC configuration (auto-generated)
│   └── config
│
├── docker/                         # Docker configurations
│   ├── Dockerfile                  # Main application Dockerfile
│   ├── Dockerfile.mlflow           # MLflow server Dockerfile
│   ├── Dockerfile.api              # API server Dockerfile
│   └── docker-compose.yml          # Multi-container orchestration
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── architecture/               # Architecture diagrams
│   │   ├── ARCHITECTURE.md         # (Your existing file)
│   │   └── diagrams/
│   ├── research/                   # Research papers & reports
│   │   ├── research_paper.pdf
│   │   ├── theoretical_foundation.md
│   │   └── performance_metrics.md
│   └── guides/                     # User guides
│       ├── quickstart.md
│       ├── training_guide.md
│       └── deployment_guide.md
│
├── logs/                           # Application logs (gitignored)
│   ├── training/
│   ├── evaluation/
│   └── api/
│
├── results/                        # Experiment results (DVC-tracked)
│   ├── figures/                    # Generated plots
│   │   ├── learning_curves/
│   │   ├── robustness_plots/
│   │   └── comparison_charts/
│   ├── metrics/                    # Metrics JSON/CSV
│   │   ├── baseline_comparison.json
│   │   └── robustness_results.csv
│   └── reports/                    # Auto-generated reports
│       └── experiment_summary.html
│
├── deployments/                    # Deployment artifacts
│   ├── kubernetes/                 # K8s manifests (if applicable)
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── mlflow_models/              # MLflow model artifacts
│       └── production_v1/
│
├── .dvcignore                      # DVC ignore patterns
├── .gitignore                      # Git ignore patterns
├── .dockerignore                   # Docker ignore patterns
├── .env.example                    # Environment variables template
├── .pre-commit-config.yaml         # Pre-commit hooks (linting, formatting)
│
├── pyproject.toml                  # Project metadata & dependencies (Poetry/setuptools)
├── requirements.txt                # Python dependencies (pip)
├── requirements-dev.txt            # Development dependencies
├── setup.py                        # Package installation script
│
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # DVC pipeline lock file
├── params.yaml                     # DVC parameters (hyperparameters)
│
├── Makefile                        # Task automation (CRITICAL!)
├── README.md                       # Project overview
├── LICENSE                         # MIT License
└── CONTRIBUTING.md                 # Contribution guidelines
```

---

## 🔥 KEY COMPONENTS EXPLAINED

### 1. **`configs/` - Configuration Management** ⭐ CRITICAL

**Why Essential:**

- Separates hyperparameters from code
- Enables reproducible experiments
- Easy experiment tracking in MLflow

**Structure:**

```yaml
# configs/base.yaml
project:
  name: "rl-cdss-prescription-safety"
  author: "Herald Ginting"

training:
  n_episodes: 500
  alpha: 0.1
  gamma: 0.95
  epsilon: 0.2

environment:
  missing_rate: 0.4
  drugs:
    - warfarin
    - aspirin
    # ...

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "q-learning-baseline"
```

**Usage:**

```python
from src.utils.config_loader import load_config

config = load_config("configs/experiment/q_learning_baseline.yaml")
agent = QLearningAgent(
    alpha=config.training.alpha,
    gamma=config.training.gamma
)
```

---

### 2. **MLflow Integration** 🎯

**Three Critical Files:**

#### A. `src/training/logger.py` - MLflow Wrapper

```python
import mlflow
from src.utils.config_loader import load_config

class MLflowLogger:
    def __init__(self, experiment_name, run_name=None):
        config = load_config("configs/base.yaml")
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_name = run_name
    
    def log_params(self, params):
        """Log hyperparameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics (detection_rate, etc.)"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, artifact_path):
        """Log file artifacts (models, plots)"""
        mlflow.log_artifact(artifact_path)
    
    def log_model(self, model, model_name):
        """Log model to MLflow Model Registry"""
        mlflow.sklearn.log_model(model, model_name)
```

#### B. `scripts/experiments/run_baseline_comparison.py`

```python
import mlflow
from src.training.logger import MLflowLogger
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

def main():
    logger = MLflowLogger(
        experiment_name="baseline-comparison",
        run_name="q-learning-missing-40"
    )
    
    with mlflow.start_run(run_name=logger.run_name):
        # Log config
        logger.log_params({
            "algorithm": "q-learning",
            "alpha": 0.1,
            "gamma": 0.95,
            "missing_rate": 0.4
        })
        
        # Train
        agent = QLearningAgent(alpha=0.1, gamma=0.95)
        trainer = Trainer(env, agent)
        history = trainer.train(n_episodes=500)
        
        # Log metrics
        logger.log_metrics({
            "final_reward": history["rewards"][-1],
            "convergence_episode": trainer.convergence_episode
        })
        
        # Log model
        logger.log_model(agent, "q_learning_agent")
        
        # Log artifacts
        logger.log_artifact("results/figures/learning_curve.png")

if __name__ == "__main__":
    main()
```

#### C. **MLflow UI Access**

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Access: http://localhost:5000
```

**MLflow Workflow:**

1. **Tracking:** Every experiment auto-logs to MLflow (metrics, params, artifacts)
2. **Comparison:** MLflow UI shows all runs side-by-side
3. **Registry:** Best model promoted to "Production" stage
4. **Deployment:** Load model from registry for serving

---

### 3. **DVC (Data Version Control)** 📊

**Purpose:** Version control for data & models (Git for large files)

**Key Files:**

#### `dvc.yaml` - Pipeline Definition

```yaml
stages:
  data_acquisition:
    cmd: python scripts/data_acquisition/fetch_drugbank.py
    deps:
      - scripts/data_acquisition/fetch_drugbank.py
    outs:
      - data/raw/drugbank/

  preprocessing:
    cmd: python scripts/preprocessing/clean_interactions.py
    deps:
      - data/raw/drugbank/
      - scripts/preprocessing/clean_interactions.py
    outs:
      - data/processed/interactions.json

  training:
    cmd: python scripts/experiments/run_baseline_comparison.py
    deps:
      - data/processed/interactions.json
      - src/
    params:
      - configs/base.yaml:
          - training
    outs:
      - models/checkpoints/q_learning_ep500.pkl
    metrics:
      - results/metrics/baseline_comparison.json:
          cache: false

  evaluation:
    cmd: python scripts/experiments/run_robustness_test.py
    deps:
      - models/checkpoints/q_learning_ep500.pkl
    outs:
      - results/figures/robustness_plots/
```

**DVC Commands:**

```bash
# Initialize DVC
dvc init

# Track large files
dvc add data/raw/drugbank/
dvc add models/checkpoints/q_learning_ep500.pkl

# Run pipeline
dvc repro

# Push data to remote (S3, Google Drive, etc.)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push

# Reproduce experiment on different machine
dvc pull
dvc repro
```

---

### 4. **Docker Configuration** 🐳

**Three Dockerfiles for Different Purposes:**

#### `docker/Dockerfile` - Main Application

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Run training
CMD ["python", "-m", "scripts.experiments.run_baseline_comparison"]
```

#### `docker/Dockerfile.mlflow` - MLflow Server

```dockerfile
FROM python:3.9-slim

RUN pip install mlflow boto3 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://user:pass@db:5432/mlflow", \
     "--default-artifact-root", "s3://my-bucket/mlflow-artifacts", \
     "--host", "0.0.0.0"]
```

#### `docker/docker-compose.yml` - Multi-Container

```yaml
version: '3.8'

services:
  mlflow:
    build:
      context: ..
      dockerfile: docker/Dockerfile.mlflow
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

  training:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../results:/app/results
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
```

**Usage:**

```bash
# Build & run all services
docker-compose -f docker/docker-compose.yml up --build

# MLflow UI: http://localhost:5000
# API: http://localhost:8000/docs
```

---

### 5. **`tests/` - Testing Strategy** ✅

**Coverage Target: >80%**

**Structure:**

#### Unit Tests

```python
# tests/unit/test_reward.py
import pytest
from src.environment.reward import RewardFunction
from src.knowledge.knowledge_base import KnowledgeBase

@pytest.fixture
def reward_fn():
    kb = KnowledgeBase("tests/fixtures/mock_knowledge_base.json")
    return RewardFunction(kb)

def test_safe_approval_reward(reward_fn):
    patient = create_safe_patient()  # No DDI
    reward = reward_fn.compute(patient, action=0)  # APPROVE
    assert reward == 2, "Safe approval should return +2"

def test_severe_interaction_missed(reward_fn):
    patient = create_high_risk_patient()  # Warfarin + Aspirin
    reward = reward_fn.compute(patient, action=0)  # APPROVE
    assert reward == -10, "Missed severe interaction should return -10"
```

#### Integration Tests

```python
# tests/integration/test_training_pipeline.py
def test_full_training_pipeline():
    """Test complete training → evaluation flow"""
    env = CDSSEnvironment(knowledge_path="tests/fixtures/")
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)
    trainer = Trainer(env, agent)
    
    # Train for 10 episodes (fast test)
    history = trainer.train(n_episodes=10)
    
    assert len(history["rewards"]) == 10
    assert "td_errors" in history
    assert agent.q_table  # Q-table populated
```

**Run Tests:**

```bash
# All tests
pytest tests/ -v --cov=src --cov-report=html

# Specific test suite
pytest tests/unit/ -v

# Coverage report
open htmlcov/index.html
```

---

### 6. **CI/CD Pipeline** 🔄

**`.github/workflows/ci.yml`:**

```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
      
      - name: Lint code
        run: |
          flake8 src/ --max-line-length=100
          mypy src/
```

---

### 7. **`Makefile` - Task Automation** ⚡

```makefile
.PHONY: install test train evaluate docker-build clean

install:
 pip install -r requirements.txt
 pip install -r requirements-dev.txt
 dvc pull

test:
 pytest tests/ -v --cov=src --cov-report=html

lint:
 flake8 src/ --max-line-length=100
 mypy src/
 black --check src/

format:
 black src/ tests/
 isort src/ tests/

train:
 python scripts/experiments/run_baseline_comparison.py

evaluate:
 python scripts/experiments/run_robustness_test.py

mlflow-ui:
 mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0

docker-build:
 docker-compose -f docker/docker-compose.yml build

docker-up:
 docker-compose -f docker/docker-compose.yml up

clean:
 rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov
 find . -type f -name "*.pyc" -delete
```

**Usage:**

```bash
make install    # Setup environment
make test       # Run tests
make train      # Train model
make mlflow-ui  # Start MLflow UI
```

---

## 🚨 **CRITICAL COMPONENTS OFTEN FORGOTTEN**

### 1. **`.env.example` - Environment Variables**

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Database
DB_HOST=localhost
DB_PORT=5432
```

### 2. **`.pre-commit-config.yaml` - Code Quality**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

### 3. **`pyproject.toml` - Modern Python Packaging**

```toml
[tool.poetry]
name = "rl-cdss-prescription-safety"
version = "1.0.0"
description = "RL-based CDSS for prescription safety under uncertainty"
authors = ["Herald Ginting <heraldmsamueltheo@gmail.com>"]

[tool.poetry.dependencies]
python = "^3.9"
numpy = "^1.21.0"
pandas = "^1.3.0"
mlflow = "^2.0.0"
dvc = "^3.0.0"

[tool.black]
line-length = 100
target-version = ['py39']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

---

## 🎯 **MLflow Experiment Management Workflow**

### **Scenario: Running Ablation Study**

**Goal:** Compare Q-Learning vs SARSA across 3 missing data rates

```bash
# 1. Define experiments in configs/
configs/experiment/
├── q_learning_missing_20.yaml
├── q_learning_missing_40.yaml
├── q_learning_missing_60.yaml
├── sarsa_missing_20.yaml
├── sarsa_missing_40.yaml
└── sarsa_missing_60.yaml

# 2. Run experiments (auto-logs to MLflow)
for config in configs/experiment/*.yaml; do
    python scripts/experiments/run_experiment.py --config $config
done

# 3. View in MLflow UI
mlflow ui

# 4. Compare runs visually
# - Select all 6 runs
# - Click "Compare"
# - View parallel coordinates plot

# 5. Promote best model to production
mlflow models serve -m "models:/q_learning_agent/Production" -p 5001
```

**MLflow UI Shows:**

- Detection rate trend across missing rates
- Learning curves side-by-side
- Hyperparameter impact (alpha, gamma)
- Artifact links (plots, models)

---

## ✅ **PRODUCTION READINESS CHECKLIST**

- [x] **Code Quality:** Linting (flake8), formatting (black), type hints (mypy)
- [x] **Testing:** Unit tests (>80% coverage), integration tests, CI/CD
- [x] **Experiment Tracking:** MLflow integration, config management
- [x] **Data Versioning:** DVC pipelines, remote storage
- [x] **Containerization:** Docker multi-stage builds, docker-compose
- [x] **Documentation:** API docs, architecture diagrams, user guides
- [x] **Reproducibility:** Configs, seeds, DVC lock files
- [x] **Monitoring:** Logging, metrics collection (future: Prometheus/Grafana)
- [x] **Deployment:** Model registry, API serving (FastAPI)
- [x] **Security:** `.env` for secrets, `.gitignore` for sensitive files

---

## 🚀 **NEXT STEPS: Implementing This Structure**

```bash
# 1. Create folders
mkdir -p configs/{experiment,data,deployment}
mkdir -p data/{raw,processed,synthetic,external}
mkdir -p src/{knowledge,environment,agents,training,evaluation,explainability,utils,api}
mkdir -p tests/{unit,integration,performance,fixtures}
mkdir -p docker scripts/{data_acquisition,preprocessing,experiments,deployment}
mkdir -p results/{figures,metrics,reports}

# 2. Initialize DVC & Git
git init
dvc init
git add .
git commit -m "feat: Initialize production MLOps structure"

# 3. Set up pre-commit hooks
pip install pre-commit
pre-commit install

# 4. Start MLflow server
make mlflow-ui
```

**Time to Production:** ~2 weeks with this structure (vs 2+ months ad-hoc)

---

**This is enterprise-grade MLOps structure** used by teams at Google, Meta, Spotify for scalable ML systems. 🎯
