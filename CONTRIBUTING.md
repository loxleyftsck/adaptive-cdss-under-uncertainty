# Contributing to RL-CDSS

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🔄 Development Workflow

### 1. Fork & Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/adaptive-cdss-under-uncertainty.git
cd adaptive-cdss-under-uncertainty
```

### 2. Set Up Development Environment

```bash
# Install dependencies
make install-dev

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Write code following project conventions
- Add tests for new functionality
- Update documentation as needed

### 5. Run Quality Checks

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test
```

### 6. Commit & Push

```bash
git add .
git commit -m "feat: add amazing feature"
git push origin feature/your-feature-name
```

### 7. Submit Pull Request

- Go to GitHub and create Pull Request
- Fill in PR template
- Wait for review

---

## 📝 Code Standards

### Python Style

- **Formatter:** black (line length: 100)
- **Import sorter:** isort (black profile)
- **Linter:** flake8, pylint
- **Type hints:** mypy (preferred but not required)

### Naming Conventions

- **Classes:** PascalCase (`QLearningAgent`)
- **Functions/variables:** snake_case (`compute_reward`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_EPISODES`)
- **Private methods:** `_method_name`

### Docstrings

Use Google-style docstrings:

```python
def compute_reward(patient: Patient, action: int) -> float:
    """Calculate reward for action given true patient state.
    
    Args:
        patient: Patient object with complete clinical data
        action: Action index (0-3)
    
    Returns:
        Scalar reward value
    
    Raises:
        ValueError: If action index invalid
    """
    pass
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Multi-component tests
├── performance/    # Speed/memory tests
└── fixtures/       # Test data
```

### Writing Tests

```python
import pytest
from src.environment.reward import RewardFunction

def test_safe_approval_reward():
    """Test reward for safe prescription approval."""
    reward_fn = RewardFunction(mock_knowledge_base)
    patient = create_safe_patient()
    
    reward = reward_fn.compute(patient, action=0)
    
    assert reward == 2, "Safe approval should return +2"
```

### Coverage Target

- **Minimum:** 80% code coverage
- **Goal:** 90% for core modules

---

## 📋 Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(agents): add DQN implementation for scalability

fix(reward): correct penalty for missed severe interactions

docs(readme): update installation instructions

test(environment): add robustness test cases
```

---

## 🔍 Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass (`make test`)
- [ ] Documentation updated
- [ ] Linters pass (`make lint`)
- [ ] Code formatted (`make format`)
- [ ] Commit messages follow convention

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Screenshots (if applicable)
```

---

## 🐛 Bug Reports

Use GitHub Issues with "bug" label.

**Include:**

- Python version
- OS (Windows/Linux/Mac)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Minimal code example

---

## 💡 Feature Requests

Use GitHub Issues with "enhancement" label.

**Include:**

- Use case description
- Proposed solution
- Alternatives considered
- Implementation ideas (optional)

---

## 📚 Documentation

### Update When Changing

- **README.md:** Major features, installation, quick start
- **Architecture docs:** System design changes
- **API docs:** New/modified functions
- **Comments:** Complex logic

---

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone.

### Our Standards

**Positive behavior:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior:**

- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## ❓ Questions?

- Open a GitHub Discussion
- Email: <heraldmsamueltheo@gmail.com>

Thank you for contributing! 🚀
