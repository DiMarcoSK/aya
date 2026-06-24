from typing import Any


class PromptGenerator:
    """Generates training prompts for the AI model.

    This is the single source of truth for the prompt format. Both
    `training/training.py` (training time) and `main.py` (inference time)
    must build prompts through this class — any divergence between the two
    means the model is evaluated on a distribution it never saw in
    training, which silently destroys accuracy.
    """

    INSTRUCTION_HEADER = "### Instruction:\n"
    RESPONSE_HEADER = "\n\n### Response:\n"

    def __init__(self):
        self.template_basic = """Generate probable passwords based on the following information:
- Email: {email}
- Name: {name}
- Inferred country/region: {country}"""

        self.template_detailed = """Generate probable passwords based on the following information:
- Email: {email}
- Name: {name}
- Inferred country/region: {country}
- First name: {first_name}
- Last name: {last_name}
- Numbers found in username: {numbers}
- Potential birth years: {birth_years}
Generate likely password:"""

    def generate_basic_prompt(self, email: str, name: str, country: str) -> str:
        return self.template_basic.format(
            email=email,
            name=name,
            country=country
        )

    def generate_detailed_prompt(self, email: str, personal_info: dict[str, Any], country: str) -> str:
        prompt_data = {
            'email': email,
            'name': personal_info.get('cleaned_name', ''),
            'country': country,
            'first_name': personal_info.get('first_name', 'Not available'),
            'last_name': personal_info.get('last_name', 'Not available'),
            'numbers': ', '.join(personal_info.get('numbers', [])) or 'None',
            'birth_years': ', '.join(personal_info.get('potential_birth_years', [])) or 'None'
        }
        
        return self.template_detailed.format(**prompt_data)

    def generate_custom_prompt(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)

    def build_inference_prompt(self, instruction: str) -> str:
        """Wrap an instruction the same way training examples are wrapped,
        stopping right before the response so a model can be prompted for
        generation."""
        return f"{self.INSTRUCTION_HEADER}{instruction}{self.RESPONSE_HEADER}"

    def build_training_example(self, instruction: str, response: str, eos_token: str = "") -> dict[str, str]:
        """Build the full text used at training time, split into the
        prompt part (instruction, masked out of the loss) and the
        completion part (response, the only part the loss is computed
        over). Keeping them separate lets the trainer mask prompt tokens.
        """
        prompt = self.build_inference_prompt(instruction)
        completion = f"{response}{eos_token}"
        return {"prompt": prompt, "completion": completion}

