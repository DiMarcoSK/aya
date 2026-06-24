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
        
        country_indicators = []

        if domain_info['country_by_tld']:
            country_indicators.append(domain_info['country_by_tld'])
        
        texts_to_analyze = [domain_info['domain'], name]
        languages = self.language_detector.get_languages_from_text(texts_to_analyze)
        
        for lang in languages:
            country = self.language_detector.get_country_by_language(lang)
            if country:
                country_indicators.append(country)
        
        # Return most common country or unknown
        if country_indicators:
            return max(set(country_indicators), key=country_indicators.count)
        
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
