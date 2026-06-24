from data_processor.prompt_generator import PromptGenerator


def _sample_personal_info():
    return {
        "cleaned_name": "jean pierre",
        "first_name": "jean",
        "last_name": "pierre",
        "numbers": ["1998"],
        "potential_birth_years": ["1998"],
    }


def test_inference_prompt_matches_training_prompt_prefix():
    """Regression test for the original bug: training built one template,
    inference built another, so the model never saw at inference time the
    format it learned during training."""
    generator = PromptGenerator()
    instruction = generator.generate_detailed_prompt(
        "jean.pierre@lafrance.com", _sample_personal_info(), "France"
    )

    inference_prompt = generator.build_inference_prompt(instruction)
    training_example = generator.build_training_example(instruction, "jean1998fr", eos_token="</s>")

    assert inference_prompt == training_example["prompt"]


def test_build_training_example_appends_eos_to_completion_only():
    generator = PromptGenerator()
    example = generator.build_training_example("do X", "password123", eos_token="</s>")
    assert example["completion"] == "password123</s>"
    assert "</s>" not in example["prompt"]


def test_generate_detailed_prompt_handles_missing_fields():
    generator = PromptGenerator()
    instruction = generator.generate_detailed_prompt(
        "user@example.com", {"cleaned_name": "user"}, "Unknown"
    )
    assert "Not available" in instruction
    assert "None" in instruction
