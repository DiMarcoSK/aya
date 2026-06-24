"""LoRA fine-tuning entrypoint for AYA.

Key design choices that differ from a naive SFT script, and why:

- Loss is masked on the prompt tokens (see `tokenize_example`): only the
  password completion contributes to the gradient. Training on the full
  prompt text wastes most of the gradient signal on re-deriving boilerplate
  instruction text the model already knows, slowing convergence.
- Padding is dynamic per-batch (`DynamicPaddingCollator`), not a fixed
  `max_length` for every example. Most prompts are far shorter than the
  configured `max_length`; fixed padding multiplies compute for no benefit.
- The prompt format is built exclusively through `PromptGenerator`, the
  same class used by the data pipeline and by `main.py` at inference time,
  so the model is never evaluated on a template it didn't train on.
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_CONFIG_PATH, load_yaml_config
from data_processor.prompt_generator import PromptGenerator

logger = logging.getLogger("aya.training")


def parse_arguments():
    # A first pass just to find --config before building the real parser,
    # so YAML values can populate argparse defaults (CLI flags still win
    # when explicitly passed).
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, remaining_argv = config_parser.parse_known_args()
    defaults = load_yaml_config(config_args.config)

    parser = argparse.ArgumentParser(description="Fine-tune language model with LoRA", parents=[config_parser])
    parser.add_argument("--model_name", type=str, default=defaults.get("model_name", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
                       help="Model name or path from HuggingFace Hub")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to training dataset (JSON format)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the trained model")
    parser.add_argument("--max_length", type=int, default=defaults.get("max_length", 256),
                       help="Maximum sequence length (prompt + completion)")
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 1),
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults.get("gradient_accumulation_steps", 16),
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=defaults.get("learning_rate", 2e-4),
                       help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=defaults.get("num_epochs", 2),
                       help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=defaults.get("lora_r", 16),
                       help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=defaults.get("lora_alpha", 32),
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=defaults.get("lora_dropout", 0.05),
                       help="LoRA dropout rate")
    parser.add_argument("--eval_split", type=float, default=defaults.get("eval_split", 0.05),
                       help="Fraction of data held out for evaluation (0 to disable)")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                       default=defaults.get("gradient_checkpointing", False),
                       help="Trade compute for memory; lets you raise batch size on CPU/low-VRAM GPUs")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42),
                       help="Random seed for reproducibility")
    return parser.parse_args(remaining_argv)


def resolve_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model_and_tokenizer(model_name: str):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device, dtype = resolve_device_dtype()
    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    return model, tokenizer, device


def setup_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Attention projections alone proved too low-capacity for this task;
        # MLP projections are included since most of a Llama-style block's
        # representational capacity sits there.
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def tokenize_example(example: dict, tokenizer, prompt_generator: PromptGenerator, max_length: int) -> dict[str, list[int]]:
    """Tokenize a single example with the prompt masked out of the loss.

    `labels` mirrors `input_ids` except prompt-token positions are set to
    -100 (the value `CrossEntropyLoss` ignores), so gradients only flow
    from predicting the password completion.
    """
    built = prompt_generator.build_training_example(
        example["instruction"], example["output"], eos_token=tokenizer.eos_token
    )

    prompt_ids = tokenizer(built["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(built["completion"], add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


@dataclass
class DynamicPaddingCollator:
    """Pads each batch to the longest sequence in that batch instead of a
    fixed length, avoiding wasted compute on near-empty padded sequences."""
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def prepare_dataset(dataset_path: str, tokenizer, max_length: int, eval_split: float, output_dir: str, seed: int):
    logger.info("Loading dataset from: %s", dataset_path)
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    prompt_generator = PromptGenerator()

    raw_train, raw_eval = dataset, None
    if eval_split and eval_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        raw_train, raw_eval = split["train"], split["test"]

        # Persisted so `training/evaluate.py` measures the same held-out
        # examples the model never trained on, instead of needing a
        # separately curated eval file.
        eval_path = os.path.join(output_dir, "eval_set.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(list(raw_eval), f, ensure_ascii=False, indent=2)
        logger.info("Held-out eval set (%d examples) saved to %s", len(raw_eval), eval_path)

    train_dataset = raw_train.map(
        lambda example: tokenize_example(example, tokenizer, prompt_generator, max_length),
        remove_columns=raw_train.column_names,
    )
    eval_dataset = None
    if raw_eval is not None:
        eval_dataset = raw_eval.map(
            lambda example: tokenize_example(example, tokenizer, prompt_generator, max_length),
            remove_columns=raw_eval.column_names,
        )

    return train_dataset, eval_dataset


def create_training_arguments(args, device: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=device == "cuda",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps" if args.eval_split > 0 else "no",
        eval_steps=100,
        report_to="none",
        remove_unused_columns=False,
        # Padding is dynamic per-batch, so length-grouping keeps batches
        # internally homogeneous and minimizes padding waste further.
        group_by_length=True,
        warmup_steps=50,
        weight_decay=0.01,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_arguments()
    set_seed(args.seed)
    logger.info("Seed set to %d for reproducibility", args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(args.model_name)

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    lora_config = setup_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = prepare_dataset(
        args.dataset_path, tokenizer, args.max_length, args.eval_split, args.output_dir, args.seed
    )

    training_args = create_training_arguments(args, device)
    data_collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed, model saved to %s", args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
