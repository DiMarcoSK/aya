import tldextract
from typing import Tuple, Optional, Dict


class DomainAnalyzer:
    """Analyzes email domains and TLDs for country inference"""
    
    def __init__(self):
        self.tld_to_country = {
            'com': 'United States/Global',
            'org': 'Global',
            'net': 'Global',
            'edu': 'United States',
            'gov': 'United States',
            'mil': 'United States',
            'uk': 'United Kingdom',
            'co.uk': 'United Kingdom',
            'de': 'Germany',
            'fr': 'France',
            'it': 'Italy',
            'es': 'Spain',
            'ru': 'Russia',
            'cn': 'China',
            'jp': 'Japan',
            'br': 'Brazil',
            'ca': 'Canada',
            'au': 'Australia',
            'in': 'India',
            'mx': 'Mexico',
            'nl': 'Netherlands',
            'se': 'Sweden',
            'no': 'Norway',
            'dk': 'Denmark',
            'fi': 'Finland',
            'pl': 'Poland',
            'cz': 'Czech Republic',
            'hu': 'Hungary',
            'ro': 'Romania',
            'tr': 'Turkey',
            'kr': 'South Korea',
            'th': 'Thailand',
            'vn': 'Vietnam',
            'sg': 'Singapore',
            'my': 'Malaysia',
            'id': 'Indonesia'
        }

    def extract_domain_parts(self, domain: str) -> Tuple[str, str, str]:
        """Extract subdomain, domain, and TLD from domain string"""
        extracted = tldextract.extract(domain.lower())
        return extracted.subdomain, extracted.domain, extracted.suffix

    def get_country_by_tld(self, tld: str) -> Optional[str]:
        """Get country by TLD"""
        return self.tld_to_country.get(tld.lower())

    def analyze_domain(self, domain: str) -> Dict[str, str]:
        """Analyze domain and return comprehensive information"""
        subdomain, domain_base, tld = self.extract_domain_parts(domain)
        
        return {
            'subdomain': subdomain,
            'domain': domain_base,
            'tld': tld,
            'full_domain': domain.lower(),
            'country_by_tld': self.get_country_by_tld(tld)
        }

