# AYA - AI-Powered Password Generation for Offensive Security

## Project Overview

**AYA** is a Proof of Concept (PoC) artificial intelligence system specifically designed for **offensive security applications**, particularly in the domain of **password cracking and security assessment**. This research project demonstrates how modern language models can be fine-tuned to generate highly probable password candidates based on personal information such as names, email addresses, and geographic data.

**IMPORTANT DISCLAIMER**: This project is intended exclusively for educational purposes, authorized penetration testing, and legitimate security research. The techniques demonstrated here should only be applied in controlled environments with explicit permission. Unauthorized access to computer systems is illegal and unethical.

## Technical Architecture

AYA is built upon the TinyLlama 1.1B parameter model, selected for its balance between computational efficiency and linguistic capability. The system employs LoRA fine-tuning to adapt the pre-trained model for password generation tasks while maintaining low memory footprint and CPU-only operation capability.

### Core Components

1. **Data Preprocessing Pipeline**: Converts raw personal information into structured instruction-following format (`training/data_processor/`)
2. **LoRA Fine-tuning Engine**: Implements parameter-efficient adaptation of the base language model (`training/training.py`, config in `configs/lora_default.yaml`)
3. **Inference System**: Generates password candidates based on input personal information (`main.py`)
4. **Evaluation Framework**: Measures exact-match rate, top-k hit-rate, and edit distance against a held-out set (`training/evaluate.py`)

See [UPGRADES.md](UPGRADES.md) for the architectural rationale behind these components and the bugs that were found and fixed along the way (prompt template mismatch between training/inference, uncorrelated synthetic passwords, a regex bug in birth-year extraction).

## System Requirements

### Minimum Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Processor** | Modern CPU with AVX2 instruction set support (Intel i7/Xeon, AMD Ryzen series) |
| **Memory** | Minimum 12GB RAM, recommended 16GB for optimal performance |
| **Storage** | 20GB available disk space for models and datasets |
| **GPU** | Optional - system designed for CPU-only operation |

### Software Dependencies

The system requires Python 3.10 or higher.

```bash
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # + pytest, ruff, black, pre-commit
```

`Makefile` wraps the common commands (`make install-dev`, `make test`, `make lint`, `make train`, `make evaluate`). A `Dockerfile` is provided for a reproducible CPU-only environment.

For CPU-only deployments, ensure PyTorch installation excludes CUDA dependencies to minimize resource utilization.

## Dataset Structure and Format

AYA utilizes instruction-tuning methodology with Alpaca-style formatting. Training data follows the structured format below:

```json
{
  "instruction": "Based on the following information:\n- Name: Jean Pierre\n- Email: jean.pierre@lafrance.com\n- Inferred country: France\nGenerate likely passwords:",
  "output": "jean1998fr"
}
```

### Data Collection Methodologies

Two primary approaches are supported for dataset construction:

**Method 1: Synthetic Data Generation**
Programmatically generated training pairs using common naming conventions, geographic patterns, and password construction heuristics. This approach ensures legal compliance while providing diverse training examples. <- For PoC, only use this

**Method 2: Anonymized Breach Data**
Utilization of publicly available, anonymized password breach datasets with personal information inference. This method requires careful ethical consideration and legal compliance verification.

### Recommended Dataset Sizes

For testing purposes, i recommend using at least 10,000 samples, that can be generated using **data_processor** module

```bash
cd training
python3 processor.py --generate 20000 --output data/20k_leak.txt --seed 42
python3 processor.py data/20k_leak.txt data/20k_leak.jsonl
python3 processor.py --convert data/20k_leak.jsonl --output data/20k_leak_alpaca.json
```

`--seed` makes synthetic generation reproducible. Generated passwords are
statistically correlated with the identity used to build the email (name,
birth year, country) — see `RealisticLeakGenerator._generate_password` in
`training/data_processor/leak_generator.py` — instead of being independent
random strings, which is what previously made the model unable to learn
anything useful from the dataset.

Note: Feeding it with few samples will make you see the typical behavior of a poorly fed LoRA due to lack of data: it is turning into a "parrot mode", repeating generic placeholders (<PASSWORD>, <EMAIL>), because it does not have enough statistical context to learn real patterns.

| Dataset Scale | Training Duration | Use Case |
|---------------|-------------------|----------|
| 1,000-5,000 samples | <1 hour | Initial concept validation |
| 10,000-30,000 samples | 2-4 hours | Minimum viable training |
| 50,000+ samples | 6-12 hours | Production-ready model |

## Training Configuration

### LoRA Hyperparameters

Defaults live in [`configs/lora_default.yaml`](configs/lora_default.yaml) — the single source of truth for "what we normally run" — and can be overridden per-run with CLI flags:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `num_epochs` | 2 | Prevents overfitting on small datasets |
| `learning_rate` | 2e-4 | Stable convergence for LoRA adaptation |
| `gradient_accumulation_steps` | 16 | Simulates larger batch sizes in memory-constrained environments |
| `batch_size` | 1 | Minimizes memory usage |
| `lora_r` | 16 | Balances adaptation capacity with efficiency |
| `lora_alpha` | 32 | Scaling factor for LoRA weights |
| `lora_dropout` | 0.05 | Regularization for small datasets |
| `eval_split` | 0.05 | Held-out fraction for `training/evaluate.py` |
| `seed` | 42 | Reproducibility (`transformers.set_seed`) |

LoRA targets attention **and** MLP projections (`q/k/v/o_proj` + `gate/up/down_proj`) — attention-only adaptation proved too low-capacity for this task. Loss is masked so gradients only flow from the password completion, not the instruction text, and padding is dynamic per-batch instead of a fixed `max_length`, which meaningfully cuts training time. See [UPGRADES.md](UPGRADES.md) for details.

### Training Command Example

```bash
python training/training.py \
    --dataset_path training/data/20k_leak_alpaca.json \
    --output_dir ./model

# Override a config value for a one-off experiment:
python training/training.py \
    --dataset_path training/data/20k_leak_alpaca.json \
    --output_dir ./model \
    --num_epochs 3 --gradient_checkpointing
```

### Evaluation

```bash
python training/evaluate.py --model_path ./model --eval_dataset ./model/eval_set.json
```

`eval_set.json` is written automatically by `training.py` from the held-out split, so it always matches examples the model never trained on. The script reports exact-match rate, top-k hit-rate, and average edit distance.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Personal      │───▶│   AI Pattern     │───▶│   Intelligent   │
│   Information   │    │   Analysis       │    │   Wordlist      │
│   Input         │    │   (LoRA Model)   │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        │               ┌──────────────────┐            │
        └─────────────▶│   Leak Dataset   │───────────┘
                        │   (Preprocessed) │
                        └──────────────────┘
```

## Inference

```bash
python main.py --email jean.pierre@lafrance.com --model_path ./model --num_candidates 5
```

Builds the prompt through the same `PersonalInfoExtractor` → `CountryInferrer` → `PromptGenerator` pipeline used at training time, so the model is always queried with a format it has actually seen.

## Development

```bash
make install-dev   # runtime + dev deps, installs pre-commit hooks
make lint          # ruff + black --check
make format        # ruff --fix + black
make test          # pytest (training/tests/)
```

CI (`.github/workflows/ci.yml`) runs lint and tests on every push/PR. `Dockerfile` builds a CPU-only image for reproducible runs.


