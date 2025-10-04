# Contributing to RNAdegron

Thank you for your interest in contributing to RNAdegron! We welcome contributions from the community.

**Note:** RNAdegron is an RNA degradation prediction system that uses embeddings from the original [RiNALMo](https://github.com/lbcb-sci/RiNALMo) language model.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your contribution
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites
- Python >= 3.8
- CUDA >= 11.8 (for GPU support)
- Git

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/photodoc1960/RNAdegron.git
cd RNAdegron

# Add upstream remote
git remote add upstream https://github.com/photodoc1960/RNAdegron.git

# Create conda environment
conda env create -f environment.yml
conda activate rnadegron

# Install RiNALMo for embeddings
pip install git+https://github.com/lbcb-sci/RiNALMo.git
pip install flash-attn==2.3.2

# Install in development mode
pip install -e .

# Install additional development dependencies
pip install pytest black flake8 mypy
```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Fixes** - Fix issues in the codebase
2. **New Features** - Add new functionality
3. **Documentation** - Improve or add documentation
4. **Performance Improvements** - Optimize existing code
5. **Tests** - Add or improve test coverage
6. **Examples** - Add usage examples or tutorials

### Before You Start

- Check existing issues and pull requests to avoid duplication
- For major changes, open an issue first to discuss your proposed changes
- Ensure your contribution aligns with the project's goals

## Code Style Guidelines

### Python Style

Follow PEP 8 guidelines:

```bash
# Format code with black
black your_file.py

# Check style with flake8
flake8 your_file.py

# Type checking with mypy
mypy your_file.py
```

### Key Style Points

- **Indentation:** 4 spaces (no tabs)
- **Line Length:** Maximum 100 characters
- **Imports:** Organize imports in standard library, third-party, local order
- **Naming:**
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings:

```python
def predict_degradation(sequence: str, model_path: str) -> np.ndarray:
    """
    Predict RNA degradation for a given sequence.

    Args:
        sequence: RNA sequence string (A, C, G, U)
        model_path: Path to trained model checkpoint

    Returns:
        Degradation predictions array of shape (seq_len, 5)

    Raises:
        ValueError: When sequence contains invalid nucleotides
        FileNotFoundError: When model_path doesn't exist
    """
    pass
```

### Type Hints

- Use type hints for function signatures
- Import types from `typing` module when needed

```python
from typing import List, Dict, Optional, Tuple

def process_sequences(sequences: List[str], max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Process RNA sequences."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=rinalmo tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

```python
def test_degradation_prediction_shape():
    """Test that degradation predictions have correct dimensions."""
    model = load_model("best_weights/fold0top1.ckpt")
    seq = "ACGUACGU"
    predictions = predict_degradation(model, seq)
    assert predictions.shape == (8, 5)  # 8 nucleotides, 5 degradation targets
```

## Pull Request Process

### Before Submitting

1. **Update your fork:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests and checks:**
   ```bash
   pytest
   black .
   flake8 .
   mypy .
   ```

3. **Update documentation:**
   - Update README if adding features
   - Add docstrings to new functions
   - Update CHANGELOG.md

### Submitting the PR

1. **Create a clear PR title:**
   - `Fix: Brief description` for bug fixes
   - `Feature: Brief description` for new features
   - `Docs: Brief description` for documentation
   - `Refactor: Brief description` for code refactoring

2. **Write a comprehensive PR description:**
   - What changes were made
   - Why these changes were necessary
   - How to test the changes
   - Any breaking changes or migration notes

3. **Link related issues:**
   - Use "Fixes #123" to automatically close issues

### Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Reporting Bugs

### Before Reporting

- Check if the bug has already been reported
- Verify the bug exists in the latest version
- Collect relevant information

### Bug Report Template

```markdown
**Description:**
Brief description of the bug

**To Reproduce:**
Steps to reproduce the behavior:
1. Run command '...'
2. With parameters '...'
3. See error

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- CUDA version: [e.g., 11.8]
- RiNALMo version: [e.g., 1.0.0]

**Error Messages:**
```
Paste error messages here
```

**Additional Context:**
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Template

```markdown
**Feature Description:**
Clear description of the proposed feature

**Use Case:**
Why is this feature needed? What problem does it solve?

**Proposed Solution:**
How should this feature work?

**Alternatives Considered:**
What other approaches did you consider?

**Additional Context:**
Any other relevant information, mockups, or examples
```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Example Workflow

```bash
# Create feature branch
git checkout -b feature/add-new-task

# Make changes and commit
git add .
git commit -m "feat(features): add secondary structure integration

Implement secondary structure features for degradation prediction.
Includes BPP matrix computation and graph distance calculations.

Fixes #42"

# Push and create PR
git push origin feature/add-structure-features
```

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues for similar questions
3. Open a new issue with the "question" label
4. Join our community discussions (if available)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project acknowledgments

Thank you for contributing to RNAdegron! 🧬

---

## Acknowledgments

RNAdegron builds upon [RiNALMo](https://github.com/lbcb-sci/RiNALMo). Please respect their license terms when contributing.
