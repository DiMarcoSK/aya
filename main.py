"""AYA inference CLI.

Builds the inference prompt through the exact same `PromptGenerator` /
`PersonalInfoExtractor` / `CountryInferrer` pipeline used to build the
training dataset (see `training/data_processor/`). Previously this script
hand-wrote a differently-worded prompt, so the model was queried with a
template it never saw during training — that mismatch alone was enough to
make generations look random.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))
from data_processor.country_inferrer import CountryInferrer
from data_processor.personal_info_extractor import PersonalInfoExtractor
from data_processor.prompt_generator import PromptGenerator

logger = logging.getLogger("aya.inference")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate probable password candidates from an email address")
    parser.add_argument("--email", type=str, required=True, help="Target email address")
    parser.add_argument("--model_path", type=str, default="./model", help="Path to the LoRA adapter directory")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                       help="Base model the adapter was trained on")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of password candidates to generate")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max tokens to generate per candidate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    return parser.parse_args()


def build_prompt(email: str) -> str:
    extractor = PersonalInfoExtractor()
    country_inferrer = CountryInferrer()
    prompt_generator = PromptGenerator()

    personal_info = extractor.extract_personal_info(email)
    country = country_inferrer.infer_country(email, personal_info["cleaned_name"])
    instruction = prompt_generator.generate_detailed_prompt(email, personal_info, country)
    return prompt_generator.build_inference_prompt(instruction)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_arguments()

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("Missing dependency: %s. Run: pip install -r requirements.txt", exc)
        sys.exit(1)

    if not os.path.isdir(args.model_path):
        logger.error("model_path '%s' does not exist. Train a model first (see training/training.py).", args.model_path)
        sys.exit(1)

    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    prompt = build_prompt(args.email)
    inputs = tokenizer(prompt, return_tensors="pt")

    logger.info("Generating %d password candidates for %s...", args.num_candidates, args.email)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_candidates,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for output in outputs:
        completion = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
        candidate = completion.split("\n")[0].strip()
        if candidate:
            candidates.append(candidate)

    print("Candidates:")
    for candidate in dict.fromkeys(candidates):  # dedupe, preserve order
        print(f"  - {candidate}")


if __name__ == "__main__":
    main()
