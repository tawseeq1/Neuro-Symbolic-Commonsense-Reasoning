# Neuro-Symbolic Commonsense Reasoning Pipeline

A staged, modular pipeline for neuro-symbolic commonsense reasoning that combines neural language models with symbolic logic rules.

## Quick Start (<5 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py

# 3. Train neural baseline
python scripts/train.py --dataset commonsenseqa --config configs/models/roberta-large.yaml

# 4. Train neuro-symbolic model
python scripts/train.py --dataset commonsenseqa --config configs/models/roberta-large.yaml --symbolic psl --fusion pipeline --lambda_logic 0.1

# 5. Evaluate and analyze
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset commonsenseqa --output reports/analysis.json
```

## Alternative: Run Example Experiment

For a quick demonstration with smaller models and datasets:

```bash
# Run the example experiment script
python examples/run_experiment.py
```

This will run three experiments comparing neural-only, pipeline fusion, and joint fusion approaches.

## Architecture

- **Neural Layer**: Transformer-based models (RoBERTa, DeBERTa) for sequence classification and multiple choice
- **Symbolic Layer**: Probabilistic Soft Logic (PSL) with weighted rules and exceptions
- **Knowledge Layer**: ConceptNet/ATOMIC-style knowledge graph retrieval
- **Fusion**: Pipeline and joint training modes with differentiable logic regularizers

## Key Features

- **Modular Design**: Clean separation of data, models, rules, and reasoning
- **Multiple Baselines**: Entailment classifier, multiple-choice head, and bi-encoder retrieval
- **Standardized Schema**: Unified data format across CommonsenseQA, Winogrande, PIQA, SocialIQA
- **Interpretable Reasoning**: Error analysis with active rules and facts
- **Calibration**: Temperature scaling for well-behaved probabilities
- **Comprehensive Evaluation**: Accuracy, F1, ECE, rule satisfaction rates

## Repository Structure

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Neural model implementations
├── rules/          # Symbolic logic rules
├── reasoning/      # Neuro-symbolic fusion
├── training/       # Training loops and utilities
└── evaluation/     # Metrics and analysis

configs/            # YAML configurations
data/               # Downloaded benchmarks
rules/              # FOL/PSL rule files
scripts/            # Training and evaluation scripts
reports/            # Results and error analyses
```

## Supported Datasets

- **CommonsenseQA**: Question with 5 options
- **Winogrande**: Winograd-style pronoun resolution
- **PIQA**: Physical commonsense with two solutions
- **SocialIQA**: Social commonsense with 3 options

## Model Types

- **Type A**: Entailment classifier (AutoModelForSequenceClassification)
- **Type B**: Multiple-choice head (AutoModelForMultipleChoice)
- **Type C**: Bi-encoder for knowledge retrieval

## Symbolic Integration

- **PSL Rules**: Weighted logical implications with exceptions
- **Fusion Modes**: Pipeline (neural → symbolic) and joint training
- **T-norms**: Łukasiewicz and Gödel for rule satisfaction
- **Calibration**: Temperature scaling for probability calibration

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- spaCy 3.6+
- WandB for logging

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd neuro-symbolic-commonsense

# Install dependencies
pip install -r requirements.txt

# Install spaCy model for entity extraction
python -m spacy download en_core_web_sm

# Optional: Install in development mode
pip install -e .
```

## Usage Examples

### Basic Training

```bash
# Train neural baseline on CommonsenseQA
python scripts/train.py \
    --dataset commonsenseqa \
    --config configs/models/roberta-large.yaml

# Train with symbolic reasoning (pipeline fusion)
python scripts/train.py \
    --dataset commonsenseqa \
    --config configs/models/roberta-large.yaml \
    --symbolic psl \
    --fusion pipeline \
    --alpha 0.5

# Train with joint fusion
python scripts/train.py \
    --dataset commonsenseqa \
    --config configs/models/roberta-large.yaml \
    --symbolic psl \
    --fusion joint \
    --lambda_logic 0.1 \
    --alpha 0.5
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset commonsenseqa \
    --output reports/evaluation.json
```

### Custom Rules

You can define custom symbolic rules in YAML format:

```yaml
# rules/custom_rules.yaml
rules:
  - name: "my_rule"
    antecedent: "Condition(x)"
    consequent: "Result(x)"
    weight: 0.8
    rule_type: "default"
```

## Key Features

- **Modular Design**: Clean separation of data, models, rules, and reasoning
- **Multiple Baselines**: Entailment classifier, multiple-choice head, and bi-encoder retrieval
- **Standardized Schema**: Unified data format across CommonsenseQA, Winogrande, PIQA, SocialIQA
- **Interpretable Reasoning**: Error analysis with active rules and facts
- **Calibration**: Temperature scaling for well-behaved probabilities
- **Comprehensive Evaluation**: Accuracy, F1, ECE, rule satisfaction rates 