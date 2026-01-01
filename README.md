# RL-Based CDSS: Prescription Safety Under Uncertainty

[![CI](https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty/actions/workflows/ci.yml/badge.svg)](https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Reinforcement Learning for Prescription Safety Under Incomplete Patient Information**  
> A research project exploring adaptive clinical decision support using POMDP-based RL agents

**Author:** Herald Michain Samuel Theo Ginting  
**Repository:** [github.com/loxleyftsck/adaptive-cdss-under-uncertainty](https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty)

---

## 🎯 Project Overview

This project implements reinforcement learning agents (Q-Learning, SARSA) for making safe prescription decisions despite 40-60% missing patient data. Using a Partially Observable Markov Decision Process (POMDP) framework, we demonstrate that adaptive agents can achieve ≥85% severe interaction detection while outperforming static rule-based systems under uncertainty.

### Key Features

- ✅ **POMDP Framework:** Handles partial observability via stochastic data masking
- ✅ **Safety-Centered Rewards:** Balances interaction detection vs alert fatigue
- ✅ **Synthetic Environment:** Controlled experimentation with ground truth validation  
- ✅ **Comprehensive Evaluation:** Robustness testing across missing data rates (20%-80%)
- ✅ **MLOps Integration:** MLflow tracking, DVC pipelines, Docker deployment
- ✅ **Production-Ready:** CI/CD, testing (>80% coverage target), type hints, documentation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker & Docker Compose

### Installation

```bash
# Clone repository
git clone https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty.git
cd adaptive-cdss-under-uncertainty

# Install dependencies
make install

# Install development dependencies (for testing, linting)
make install-dev
```

### Running Experiments

```bash
# Start MLflow tracking UI
make mlflow-ui  # Access: http://localhost:5000

# Train baseline Q-Learning agent
make train

# Run robustness evaluation
make evaluate
```

### Docker Deployment

```bash
# Build containers
make docker-build

# Start full stack (MLflow + training)
make docker-up
```

---

## 📊 Results Summary

| Method | Detection @ 40% Missing | False Alarm Rate | Robustness (60% Missing) |
|--------|-------------------------|------------------|--------------------------|
| **RL (Q-Learning)** | **87%** | 12% | **72%** |
| **RL (SARSA)** | **85%** | 14% | **70%** |
| **Rule-Based CDSS** | 67% | 10% | 45% |
| **Random Policy** | 25% | 50% | 25% |

**Key Finding:** RL agents maintain acceptable safety performance (≥85%) under 40% missing data and degrade gracefully, outperforming rule-based systems by +25 percentage points at 60% missing.

---

## 📁 Project Structure

```
├── src/                  # Source code
│   ├── knowledge/       # Knowledge Base (DDI, contraindications)
│   ├── environment/     # RL Environment (POMDP simulation)
│   ├── agents/          # RL Agents (Q-Learning, SARSA, baselines)
│   ├── training/        # Training loop & MLflow integration
│   ├── evaluation/      # Metrics, robustness testing
│   └── utils/           # Configuration, logging, I/O
│
├── configs/             # YAML configurations (hyperparameters, experiments)
├── data/                # Knowledge base data (DVC-tracked)
├── models/              # Trained models (MLflow registry)
├── results/             # Experiment results (plots, metrics)
├── tests/               # Testing suite (unit, integration)
├── docker/              # Docker configurations
├── scripts/             # Standalone scripts (data acquisition, experiments)
└── docs/                # Documentation (research paper, architecture)
```

---

## 🧪 Development

### Running Tests

```bash
# All tests with coverage
make test

# Unit tests only
pytest tests/unit/ -v

# View coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format
```

### Configuration Management

Experiments are configured via YAML files in `configs/`:

```yaml
# configs/experiment/q_learning_baseline.yaml
training:
  algorithm: "q_learning"
  alpha: 0.1
  gamma: 0.95
  n_episodes: 500

environment:
  missing_rate: 0.4  # 40% missing data
```

Load configs in code:

```python
from src.utils.config_loader import load_config

config = load_config("configs/experiment/q_learning_baseline.yaml")
agent = QLearningAgent(alpha=config.training.alpha)
```

---

## 📚 Documentation

- **[Research Paper](docs/research/RL_Prescription_Safety_Under_Uncertainty_Ginting2026.pdf)** - Full academic paper (12 pages)
- **[Architecture](docs/architecture/ARCHITECTURE.md)** - System design & components
- **[Theoretical Foundation](docs/research/theoretical_foundation.md)** - POMDP formulation, RL algorithms
- **[Performance Metrics](docs/research/performance_metrics.md)** - Evaluation framework
- **[MLOps Structure](MLOPS_STRUCTURE.md)** - Production infrastructure guide

---

## 🔬 Methodology

### POMDP Formulation

```
State (S):        True patient condition (hidden)
Observation (Ω):  Partial data (40% missing via stochastic masking)
Actions (A):      {APPROVE, WARN, SUGGEST_ALT, REQUEST_DATA}
Reward (R):       Safety-centered (-10 for severe miss, +3 for correct warning)
```

### RL Algorithms

- **Q-Learning (Off-Policy):** Fast convergence, optimal policy learning
- **SARSA (On-Policy):** Conservative exploration, safer for medical domains

### Evaluation Protocol

1. **Training:** 500 episodes with ε-greedy exploration (ε=0.2)
2. **Evaluation:** 100 episodes, greedy policy (ε=0.0)
3. **Robustness:** Test across 20%, 40%, 60%, 80% missing data rates
4. **Statistical Validation:** 5 independent runs, paired t-tests, 95% CI

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 📬 Contact

**Herald Michain Samuel Theo Ginting**  

- Email: <heraldmsamueltheo@gmail.com>  
- GitHub: [@loxleyftsck](https://github.com/loxleyftsck)  
- Repository: [adaptive-cdss-under-uncertainty](https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty)

---

## 🙏 Acknowledgments

- Knowledge sources: DrugBank, DDInter, FDA Drug Interaction Database
- Inspiration: POMDP research in medical decision-making
- MLOps tools: MLflow, DVC, Docker

---

## ⚠️ Disclaimer

This is a research prototype using synthetic data. **NOT intended for clinical use.** All experiments conducted in controlled environments without patient risk. For clinical deployment, rigorous validation with real EHR data and regulatory approval required.

---

**Citation:**

If you use this work, please cite:

```bibtex
@misc{ginting2026rlcdss,
  title={Reinforcement Learning Approach to Prescription Safety Under Incomplete Patient Information},
  author={Ginting, Herald Michain Samuel Theo},
  year={2026},
  howpublished={\url{https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty}}
}
```
