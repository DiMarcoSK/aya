from langdetect import LangDetectException, detect


class LanguageDetector:
    """Handles language detection and country inference"""
    
    def __init__(self):
        self.lang_to_country = {
            'pt': 'Brazil/Portuguese-speaking countries',
            'fr': 'France/French-speaking countries', 
            'de': 'Germany/German-speaking countries',
            'es': 'Spain/Spanish-speaking countries',
            'it': 'Italy',
            'en': 'English-speaking countries (US/UK/AUS/CA)',
            'ru': 'Russia',
            'ja': 'Japan',
            'zh-cn': 'China',
            'zh-tw': 'Taiwan',
            'nl': 'Netherlands',
            'sv': 'Sweden',
            'da': 'Denmark',
            'no': 'Norway',
            'fi': 'Finland',
            'pl': 'Poland',
            'cs': 'Czech Republic',
            'hu': 'Hungary',
            'ro': 'Romania',
            'tr': 'Turkey',
            'ar': 'Arabic-speaking countries',
            'ko': 'South Korea',
            'th': 'Thailand',
            'vi': 'Vietnam',
            'id': 'Indonesia',
            'ms': 'Malaysia',
            'hi': 'India'
        }

    def detect_language(self, text: str) -> str | None:
        """Detect language of given text"""
        try:
            if len(text.strip()) < 3:
                return None
            return detect(text)
        except (LangDetectException, Exception):
            return None

    def get_country_by_language(self, language: str) -> str | None:
        """Get country/region by language code"""
        return self.lang_to_country.get(language)

    def get_languages_from_text(self, texts: list[str]) -> list[str]:
        """Get languages from multiple text sources"""
        languages = []
        for text in texts:
            if text:
                lang = self.detect_language(text)
                if lang:
                    languages.append(lang)
        return languages