from data_processor.domain_analyzer import DomainAnalyzer


def test_get_country_by_tld_known_suffix():
    analyzer = DomainAnalyzer()
    assert analyzer.get_country_by_tld("fr") == "France"


def test_get_country_by_tld_unknown_suffix_returns_none():
    analyzer = DomainAnalyzer()
    assert analyzer.get_country_by_tld("xyz123") is None


def test_analyze_domain_extracts_parts_and_country():
    analyzer = DomainAnalyzer()
    result = analyzer.analyze_domain("mail.example.fr")
    assert result["domain"] == "example"
    assert result["tld"] == "fr"
    assert result["country_by_tld"] == "France"
