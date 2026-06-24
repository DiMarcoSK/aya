from collections import Counter

from .domain_analyzer import DomainAnalyzer
from .language_detector import LanguageDetector


class CountryInferrer:
    # Combines multiple sources to infer user's country
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.domain_analyzer = DomainAnalyzer()

    def infer_country(self, email: str, name: str) -> str:
        # Infer country based on email domain and name language
        if '@' not in email:
            return 'Unknown'
        
        domain = email.split('@')[1]
        domain_info = self.domain_analyzer.analyze_domain(domain)

        # TLD is a far more reliable signal than language detection on short,
        # noisy strings (e.g. langdetect can misread "lafrance" as Finnish),
        # so a specific country TLD takes precedence over language guesses.
        tld_country = domain_info['country_by_tld']
        if tld_country and tld_country not in ('United States/Global', 'Global'):
            return tld_country

        texts_to_analyze = [domain_info['domain'], name]
        languages = self.language_detector.get_languages_from_text(texts_to_analyze)

        country_indicators = []
        for lang in languages:
            country = self.language_detector.get_country_by_language(lang)
            if country:
                country_indicators.append(country)

        if country_indicators:
            counts = Counter(country_indicators)
            return counts.most_common(1)[0][0]

        if tld_country:
            return tld_country

        return 'Unknown'

    def get_confidence_score(self, email: str, name: str) -> float:
        # Get confidence score for country inference (0.0 to 1.0)
        domain = email.split('@')[1] if '@' in email else ''
        domain_info = self.domain_analyzer.analyze_domain(domain)
        
        score = 0.0
        
        if domain_info['country_by_tld'] and domain_info['country_by_tld'] != 'Global':
            score += 0.4
        texts_to_analyze = [domain_info['domain'], name]
        languages = self.language_detector.get_languages_from_text(texts_to_analyze)
        
        if languages:
            score += 0.6
        
        return min(score, 1.0)
