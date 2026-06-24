import random
import string
from pathlib import Path


class RealisticLeakGenerator:
    """Generates synthetic email:password leak data where the password is
    statistically correlated with the identity used to build the email,
    mimicking how real users build passwords from their own name/birth
    year/locale. Without this correlation there is no learnable signal
    between the prompt features and the target password."""

    def __init__(self):
        self.name_patterns = {
            'prefixes': ['al', 'an', 'ar', 'br', 'ca', 'ch', 'da', 'de', 'el', 'fr', 'ga', 'ja', 'jo', 'ka', 'la', 'ma', 'mi', 'pa', 'ro', 'sa', 'th'],
            'middle': ['a', 'e', 'i', 'o', 'u', 'r', 'n', 'l', 's', 't'],
            'suffixes': ['son', 'sen', 'ez', 'ov', 'ski', 'icz', 'sson', 'dez', 'nez', 'ber', 'ler', 'ner', 'der', 'ter']
        }

        # TLD -> country-style suffix people sometimes append to passwords
        self.domain_tlds = {
            'com': 0.45, 'net': 0.08, 'org': 0.05, 'edu': 0.03, 'gov': 0.01,
            'com.br': 0.06, 'com.mx': 0.03, 'es': 0.04, 'fr': 0.03, 'de': 0.03,
            'ru': 0.04, 'cn': 0.02, 'jp': 0.02, 'uk': 0.03, 'ca': 0.02,
            'au': 0.02, 'in': 0.03, 'it': 0.02, 'pl': 0.02, 'nl': 0.01
        }
        self.tld_country_code = {
            'com.br': 'br', 'es': 'es', 'fr': 'fr', 'de': 'de', 'ru': 'ru',
            'cn': 'cn', 'jp': 'jp', 'uk': 'uk', 'ca': 'ca', 'au': 'au',
            'in': 'in', 'it': 'it', 'pl': 'pl', 'nl': 'nl', 'com.mx': 'mx',
        }

        self.email_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'live', 'icloud', 'mail', 'proton', 'yandex']
        self.password_components = {
            'words': ['admin', 'user', 'pass', 'login', 'access', 'secret', 'private', 'secure', 'home', 'work', 'love', 'life', 'family', 'money', 'power'],
            'keyboard': ['qwerty', 'asdf', 'zxcv', '1qaz', '2wsx', '3edc', 'qazwsx'],
            'numbers': lambda: str(random.randint(1, 9999)).zfill(random.choice([2, 3, 4]))
        }
        self.leet_map = {'a': '@', 'o': '0', 'i': '1', 'e': '3', 's': '$'}

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
        tld = random.choices(list(self.domain_tlds.keys()), weights=list(self.domain_tlds.values()))[0]
        if random.random() < 0.7:
            provider = random.choice(self.email_providers)
            return f"{provider}.{tld}", tld
        else:
            company = self._generate_name((4, 10))
            return f"{company}.{tld}", tld

    def _generate_identity(self):
        """Generate a coherent identity: first/last name, optional birth
        year, and a country code derived from the email domain — the
        password generator below conditions on this same identity."""
        first = self._generate_name((3, 10))
        last = self._generate_name((3, 12))
        has_birth_year = random.random() < 0.6
        birth_year = random.randint(1965, 2006) if has_birth_year else None
        domain, tld = self._generate_domain()
        country_code = self.tld_country_code.get(tld)
        return {
            'first': first,
            'last': last,
            'birth_year': birth_year,
            'domain': domain,
            'country_code': country_code,
        }

    def _generate_username(self, identity):
        first, last = identity['first'], identity['last']

        patterns = [
            lambda: f"{first}.{last}",
            lambda: f"{first}_{last}",
            lambda: f"{first}{last}",
            lambda: f"{first}.{last}{random.randint(1, 999)}",
            lambda: f"{first}{last}{random.randint(1, 999)}",
            lambda: f"{first}{identity['birth_year'] or random.randint(1980, 2024)}",
            lambda: f"{first[0]}.{last}",
            lambda: f"{first}.{last[0]}",
            lambda: f"{last}.{first}",
            lambda: f"{first}_{random.randint(1, 999)}",
            lambda: f"{last}{random.randint(1, 99)}",
            lambda: f"{first}{last[0]}{random.randint(1, 99)}"
        ]

        return random.choice(patterns)()

    def _apply_leet(self, text):
        return ''.join(self.leet_map.get(c, c) if random.random() < 0.5 else c for c in text)

    def _generate_password(self, identity):
        """Generate a password. The majority of patterns are derived from
        the identity (name, birth year, country code) so the dataset
        carries a learnable statistical relationship between the prompt
        features and the target output. A minority remain uncorrelated,
        matching the share of real users whose passwords are unrelated
        to their identity."""
        first, last = identity['first'], identity['last']
        year = identity['birth_year'] or random.randint(1970, 2024)
        country_code = identity['country_code']

        pattern_weights = {
            'name_year': 0.30,
            'name_year_country': 0.12 if country_code else 0.0,
            'lastname_year': 0.12,
            'name_capitalized_year': 0.10,
            'name_leet_year': 0.08,
            'word_number': 0.12,
            'keyboard': 0.08,
            'random_complex': 0.08,
        }
        patterns = list(pattern_weights.keys())
        weights = list(pattern_weights.values())
        password_type = random.choices(patterns, weights=weights)[0]

        if password_type == 'name_year':
            return f"{first}{year}"

        elif password_type == 'name_year_country':
            return f"{first}{year}{country_code}"

        elif password_type == 'lastname_year':
            return f"{last}{year}"

        elif password_type == 'name_capitalized_year':
            return f"{first.capitalize()}{year}"

        elif password_type == 'name_leet_year':
            return f"{self._apply_leet(first)}{year}"

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
        identity = self._generate_identity()
        username = self._generate_username(identity)
        password = self._generate_password(identity)

        return f"{username}@{identity['domain']}", password

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
