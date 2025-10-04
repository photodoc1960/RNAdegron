# Changelog

All notable changes to RNAdegron will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite
- `.gitignore` for clean repository
- `CHANGELOG.md` for version tracking
- `CONTRIBUTING.md` with developer guidelines
- `QUICK_START.md` for rapid onboarding
- Proper attribution to original RiNALMo project

### Fixed
- **Critical:** Hardcoded sequence length assumptions in `predict_v7.py`
- **Critical:** File overwriting race conditions in `Dataset_v7.py`
- **Critical:** NaN handling in rollback mechanism (`train_pl_v7.py`)
- **Critical:** Memory leaks from improper tensor handling
- **Critical:** Uninitialized `is_best` variable in `train_v7.py`

### Changed
- Renamed project from "RiNALMo" to "RNAdegron" to reflect its specific purpose
- Updated all documentation to clarify this is an RNA degradation prediction system
- Enhanced README with clear attribution to original RiNALMo

## [1.0.0] - 2024-XX-XX

### Added - RNA Degradation Prediction System

#### Core Features
- **v7 Pipeline** - Advanced RNA degradation prediction
  - RiNALMo embedding integration (640-dim)
  - Structural feature incorporation (BPP, graph distances, nearest paired/unpaired)
  - ΔG energy values from thermodynamic calculations

#### Model Architecture
- Multi-head attention with BPP integration (16 heads)
- Convolutional transformer encoder (5 layers, 2560 hidden dim)
- Position-aware validation system
- Cluster-based sample weighting

#### Training Strategy
- 5-fold cross-validation pipeline
- Pseudo-labeling for semi-supervised learning
- Alternating supervised/pseudo-labeled training (5 epochs + 2 epochs)
- Automatic rollback mechanism (threshold: 0.002)
- Position-level uncertainty filtering

#### Infrastructure
- Multi-process safe data loading with file locking
- Memory-efficient tensor operations
- Comprehensive error handling with NaN detection
- CSV logging for training metrics
- Automated checkpoint management

### Pipeline Components

#### Data Processing
- `Dataset_v7.py` - Data loading with structural features
- `serialize_embeddings_v7.py` - RiNALMo embedding extraction
- `Functions_v7.py` - Training utilities and metrics

#### Model Training
- `train_v7.py` - Main supervised training
- `train_pl_v7.py` - Pseudo-label fine-tuning
- `pretrain_v7.py` - Optional unsupervised pretraining

#### Prediction
- `predict_v7.py` - Ensemble prediction generation
- `pseudo_predict_v7.py` - Pseudo-label generation
- `get_best_weights_v7.py` - Model selection

#### Utilities
- `Logger.py` - CSV logging system
- `Metrics.py` - MCRMSE and evaluation metrics
- `LrScheduler.py` - Learning rate scheduling
- `cluster_weighting.py` - Cluster-based sample weighting
- `position_aware_validation.py` - Position-aware early stopping
- `visualization_v7.py` - Result visualization

### Shell Scripts
- `full_pipeline_script.sh` - Complete pipeline automation
- `run_v7.sh` - Training execution
- `run_pl_v7.sh` - Pseudo-label training
- `predict_v7.sh` - Prediction generation
- `pseudo_predict_v7.sh` - Pseudo-label generation

### Dependencies
- PyTorch >= 2.0
- RiNALMo (for embeddings)
- ViennaRNA (for structure prediction)
- Flash Attention 2.3.2
- Python >= 3.8
- CUDA >= 11.8

### Documentation
- Comprehensive README.md
- Installation instructions (conda & pip)
- Usage examples and tutorials
- Model architecture documentation
- API reference for v7 pipeline

### Attribution
This project builds upon **RiNALMo** (RiboNucleic Acid Language Model):
- Uses RiNALMo embeddings for sequence representation
- Leverages pre-trained giga-v1 model (650M parameters)
- Extends capabilities for RNA degradation prediction

**RiNALMo Citation:**
Penić, R. J., Vlašić, T., Huber, R. G., Wan, Y., & Šikić, M. (2024).
RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks.
arXiv preprint arXiv:2403.00043.

### License
- **RNAdegron Code:** MIT License
- **RiNALMo Integration:** Complies with Apache 2.0 (code) and CC BY 4.0 (model parameters)

---

## Version History Notes

### Version Naming
- **v7.x** - Current production version with full structural features
- **v6.x** - Previous stable version
- **v5.x** - Legacy version (archived)

### Migration Guide
If upgrading from v6 to v7:
1. Re-extract features using `serialize_embeddings_v7.py`
2. Update configuration for new structural features
3. Retrain models with v7 architecture
4. Use new prediction scripts for inference

---

[Unreleased]: https://github.com/photodoc1960/RNAdegron/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/photodoc1960/RNAdegron/releases/tag/v1.0.0
