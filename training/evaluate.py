"""Evaluation framework for AYA.

The README has long described an "Evaluation Framework" as a core
component, but no code measured anything — there was no way to tell
whether a training run actually improved password prediction. This script
closes that gap: it loads a held-out set (alpaca-style instruction/output
pairs, e.g. the `test` split saved alongside a training run) and reports:

- exact_match_rate: candidate == ground-truth password, verbatim
- top_k_hit_rate: ground truth appears in the top-k sampled candidates
- avg_edit_distance: mean Levenshtein distance between best candidate and
  ground truth (lower is better; a cheap proxy for "how close" misses are)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processor.prompt_generator import PromptGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained AYA LoRA adapter on a held-out set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--eval_dataset", type=str, required=True,
                       help="JSON file with a list of {instruction, output} examples")
    parser.add_argument("--top_k", type=int, default=5, help="Number of sampled candidates per example")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=200, help="Cap evaluation set size for speed")
    return parser.parse_args()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def load_eval_examples(path: str, max_examples: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:max_examples]


def generate_candidates(model, tokenizer, prompt: str, top_k: int, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=top_k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for output in outputs:
        completion = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        candidates.append(completion.split("\n")[0].strip())
    return candidates


def main():
    args = parse_arguments()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    prompt_generator = PromptGenerator()
    examples = load_eval_examples(args.eval_dataset, args.max_examples)
    print(f"Evaluating on {len(examples)} held-out examples...")

    exact_matches = 0
    top_k_hits = 0
    edit_distances = []

    for i, example in enumerate(examples, start=1):
        prompt = prompt_generator.build_inference_prompt(example["instruction"])
        ground_truth = example["output"].strip()

        candidates = generate_candidates(model, tokenizer, prompt, args.top_k, args.max_new_tokens)

        if candidates and candidates[0] == ground_truth:
            exact_matches += 1
        if ground_truth in candidates:
            top_k_hits += 1

        best_distance = min(levenshtein(c, ground_truth) for c in candidates) if candidates else len(ground_truth)
        edit_distances.append(best_distance)

        if i % 20 == 0:
            print(f"  ...{i}/{len(examples)} processed")

    n = len(examples) or 1
    print("\n=== Evaluation Results ===")
    print(f"Examples evaluated:    {len(examples)}")
    print(f"Exact match rate:      {exact_matches / n:.2%}")
    print(f"Top-{args.top_k} hit rate:        {top_k_hits / n:.2%}")
    print(f"Avg best edit distance: {sum(edit_distances) / n:.2f}")


if __name__ == "__main__":
    main()
