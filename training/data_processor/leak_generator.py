import random
import string
from pathlib import Path


class RealisticLeakGenerator:
    def __init__(self):
        self.name_patterns = {
            'prefixes': ['al', 'an', 'ar', 'br', 'ca', 'ch', 'da', 'de', 'el', 'fr', 'ga', 'ja', 'jo', 'ka', 'la', 'ma', 'mi', 'pa', 'ro', 'sa', 'th'],
            'middle': ['a', 'e', 'i', 'o', 'u', 'r', 'n', 'l', 's', 't'],
            'suffixes': ['son', 'sen', 'ez', 'ov', 'ski', 'icz', 'sson', 'dez', 'nez', 'ber', 'ler', 'ner', 'der', 'ter']
        }
        
        self.domain_tlds = {
            'com': 0.45, 'net': 0.08, 'org': 0.05, 'edu': 0.03, 'gov': 0.01,
            'com.br': 0.06, 'com.mx': 0.03, 'es': 0.04, 'fr': 0.03, 'de': 0.03,
            'ru': 0.04, 'cn': 0.02, 'jp': 0.02, 'uk': 0.03, 'ca': 0.02,
            'au': 0.02, 'in': 0.03, 'it': 0.02, 'pl': 0.02, 'nl': 0.01
        }
        
        self.email_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'live', 'icloud', 'mail', 'proton', 'yandex']
        self.password_components = {
            'words': ['admin', 'user', 'pass', 'login', 'access', 'secret', 'private', 'secure', 'home', 'work', 'love', 'life', 'family', 'money', 'power'],
            'keyboard': ['qwerty', 'asdf', 'zxcv', '1qaz', '2wsx', '3edc', 'qazwsx'],
            'numbers': lambda: str(random.randint(1, 9999)).zfill(random.choice([2, 3, 4]))
        }

    def _generate_name(self, length_range=(4, 12)):
        length = random.randint(*length_range)
        name = random.choice(self.name_patterns['prefixes'])
        
        while len(name) < length - 3:
            name += random.choice(self.name_patterns['middle'])
            if random.random() < 0.3:
                name += random.choice(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'])
        
        if random.random() < 0.4:
            name += random.choice(self.name_patterns['suffixes'])
        
        return name[:length]

    def _generate_domain(self):
        if random.random() < 0.7:
            provider = random.choice(self.email_providers)
            tld = random.choices(list(self.domain_tlds.keys()), weights=list(self.domain_tlds.values()))[0]
            return f"{provider}.{tld}"
        else:
            company = self._generate_name((4, 10))
            tld = random.choices(list(self.domain_tlds.keys()), weights=list(self.domain_tlds.values()))[0]
            return f"{company}.{tld}"

    def _generate_username(self):
        first = self._generate_name((3, 10))
        last = self._generate_name((3, 12))
        
        patterns = [
            lambda: f"{first}.{last}",
            lambda: f"{first}_{last}",
            lambda: f"{first}{last}",
            lambda: f"{first}.{last}{random.randint(1, 999)}",
            lambda: f"{first}{last}{random.randint(1, 999)}",
            lambda: f"{first}{random.randint(1980, 2024)}",
            lambda: f"{first[0]}.{last}",
            lambda: f"{first}.{last[0]}",
            lambda: f"{last}.{first}",
            lambda: f"{first}_{random.randint(1, 999)}",
            lambda: f"{last}{random.randint(1, 99)}",
            lambda: f"{first}{last[0]}{random.randint(1, 99)}"
        ]
        
        return random.choice(patterns)()

    def _generate_password(self):
        password_type = random.choices(
            ['simple', 'name_year', 'word_number', 'keyboard', 'complex'],
            weights=[0.35, 0.25, 0.20, 0.15, 0.05]
        )[0]
        
        if password_type == 'simple':
            return ''.join([str(random.randint(0, 9)) for _ in range(random.randint(6, 12))])
        
        elif password_type == 'name_year':
            name = self._generate_name((4, 8))
            year = random.randint(1970, 2024)
            return f"{name}{year}"
        
        elif password_type == 'word_number':
            word = random.choice(self.password_components['words'])
            number = self.password_components['numbers']()
            return f"{word}{number}"
        
        elif password_type == 'keyboard':
            base = random.choice(self.password_components['keyboard'])
            number = self.password_components['numbers']()
            return f"{base}{number}"
        
        else:
            length = random.randint(8, 16)
            chars = string.ascii_letters + string.digits
            if random.random() < 0.3:
                chars += "!@#$%&*"
            return ''.join(random.choices(chars, k=length))

    def generate_single_entry(self):
        username = self._generate_username()
        domain = self._generate_domain()
        password = self._generate_password()
        
        return f"{username}@{domain}", password

    def generate_leak_file(self, output_file, num_entries):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for i in range(num_entries):
                email, password = self.generate_single_entry()
                f.write(f"{email}:{password}\n")
                
                if (i + 1) % 10000 == 0:
                    print(f"Generated {i + 1:,} entries...")

    def preview_sample(self, num_samples=10):
        print(f"Sample of {num_samples} entries:")
        for i in range(num_samples):
            email, password = self.generate_single_entry()
            print(f"{email:35} : {password}")