from typing import Dict, Any


class PromptGenerator:
    """Generates training prompts for the AI model"""
    
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

    def generate_detailed_prompt(self, email: str, personal_info: Dict[str, Any], country: str) -> str:
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

