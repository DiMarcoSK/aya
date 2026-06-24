import re


class PersonalInfoExtractor:
    """Extracts personal information from email addresses and usernames"""
    
    def __init__(self):
        self.year_pattern = re.compile(r'(?:19|20)\d{2}')
        self.number_pattern = re.compile(r'\d+')
        self.separator_pattern = re.compile(r'[._-]')

    def clean_name(self, name_part: str) -> str:
        """Clean name by removing numbers and special characters"""
        name_clean = re.sub(r'[^a-zA-Z]', ' ', name_part)
        name_clean = re.sub(r'\s+', ' ', name_clean).strip()
        return name_clean

    def extract_name_parts(self, name: str) -> dict[str, str]:
        """Extract first name, last name, and middle names"""
        info = {}
        name_parts = name.split()
        
        if len(name_parts) >= 2:
            info['first_name'] = name_parts[0]
            info['last_name'] = name_parts[-1]
            if len(name_parts) > 2:
                info['middle_names'] = ' '.join(name_parts[1:-1])
        elif len(name_parts) == 1:
            info['first_name'] = name_parts[0]
        
        return info

    def extract_numbers(self, text: str) -> list[str]:
        """Extract all numbers from text"""
        return self.number_pattern.findall(text)

    def extract_years(self, text: str) -> list[str]:
        """Extract potential birth years (19XX, 20XX)"""
        return self.year_pattern.findall(text)

    def analyze_username_patterns(self, username: str) -> dict[str, any]:
        """Analyze username for common patterns"""
        patterns = {
            'has_numbers': bool(self.number_pattern.search(username)),
            'has_separators': bool(self.separator_pattern.search(username)),
            'length': len(username),
            'starts_with_number': username[0].isdigit() if username else False,
            'ends_with_number': username[-1].isdigit() if username else False,
        }
        return patterns

    def extract_personal_info(self, email: str) -> dict[str, any]:
        """Extract comprehensive personal information from email"""
        username = email.split('@')[0] if '@' in email else email
        name_clean = self.clean_name(username)
        
        info = {
            'username': username,
            'cleaned_name': name_clean,
            'numbers': self.extract_numbers(username),
            'potential_birth_years': self.extract_years(username),
            'username_patterns': self.analyze_username_patterns(username)
        }
        
        info.update(self.extract_name_parts(name_clean))
        
        return info

