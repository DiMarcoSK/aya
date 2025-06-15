# AYA - AI-Powered Password Generation for Offensive Security

## Project Overview

**AYA** is a Proof of Concept (PoC) artificial intelligence system specifically designed for **offensive security applications**, particularly in the domain of **password cracking and security assessment**. This research project demonstrates how modern language models can be fine-tuned to generate highly probable password candidates based on personal information such as names, email addresses, and geographic data.

**IMPORTANT DISCLAIMER**: This project is intended exclusively for educational purposes, authorized penetration testing, and legitimate security research. The techniques demonstrated here should only be applied in controlled environments with explicit permission. Unauthorized access to computer systems is illegal and unethical.

## Technical Architecture

AYA is built upon the TinyLlama 1.1B parameter model, selected for its balance between computational efficiency and linguistic capability. The system employs LoRA fine-tuning to adapt the pre-trained model for password generation tasks while maintaining low memory footprint and CPU-only operation capability.

### Core Components

1. **Data Preprocessing Pipeline**: Converts raw personal information into structured instruction-following format
2. **LoRA Fine-tuning Engine**: Implements parameter-efficient adaptation of the base language model
3. **Inference System**: Generates password candidates based on input personal information
4. **Evaluation Framework**: Assesses generated password quality and relevance

## System Requirements

### Minimum Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Processor** | Modern CPU with AVX2 instruction set support (Intel i7/Xeon, AMD Ryzen series) |
| **Memory** | Minimum 12GB RAM, recommended 16GB for optimal performance |
| **Storage** | 20GB available disk space for models and datasets |
| **GPU** | Optional - system designed for CPU-only operation |

### Software Dependencies

The system requires Python 3.8 or higher with the following packages:

```bash
pip install torch transformers datasets peft accelerate
```

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

usage@example:~$ cd training
usage@example:~$ python3 processor.py --generate 20000 --output data/20k_leak.txt
Generated 20000 synthetic leaks to generated_leaks_20.txt
```

Note: Feeding it with few samples will make you see the typical behavior of a poorly fed LoRA due to lack of data: it is turning into a "parrot mode", repeating generic placeholders (<PASSWORD>, <EMAIL>), because it does not have enough statistical context to learn real patterns.

| Dataset Scale | Training Duration | Use Case |
|---------------|-------------------|----------|
| 1,000-5,000 samples | <1 hour | Initial concept validation |
| 10,000-30,000 samples | 2-4 hours | Minimum viable training |
| 50,000+ samples | 6-12 hours | Production-ready model |

## Training Configuration

### LoRA Hyperparameters

The following configuration has been optimized for CPU-only training environments:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_train_epochs` | 2-3 | Prevents overfitting on small datasets |
| `learning_rate` | 2e-4 to 5e-4 | Stable convergence for LoRA adaptation |
| `gradient_accumulation_steps` | 16+ | Simulates larger batch sizes in memory-constrained environments |
| `per_device_train_batch_size` | 1 | Minimizes memory usage |
| `lora_r` | 8-16 | Balances adaptation capacity with efficiency |
| `lora_alpha` | 32 | Scaling factor for LoRA weights |

### Training Command Example

```bash
python training.py \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-tokenizer \
    --dataset_path dataset_alpaca.json \
    --output_dir ./model/
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Personal      │───▶│   AI Pattern     │───▶│   Intelligent   │
│   Information   │    │   Analysis       │    │   Wordlist      │
│   Input         │    │   (LoRA Model)   │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        │               ┌──────────────────┐           │
        └──────────────▶│   Leak Dataset   │───────────┘
                        │   (Preprocessed) │
                        └──────────────────┘
```



