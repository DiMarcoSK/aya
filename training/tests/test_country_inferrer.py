from data_processor.country_inferrer import CountryInferrer


def test_infer_country_uses_tld_when_available():
    inferrer = CountryInferrer()
    country = inferrer.infer_country("jean.pierre@lafrance.fr", "jean pierre")
    assert "France" in country


def test_infer_country_returns_unknown_without_at_symbol():
    inferrer = CountryInferrer()
    assert inferrer.infer_country("not-an-email", "name") == "Unknown"


def test_confidence_score_is_within_bounds():
    inferrer = CountryInferrer()
    score = inferrer.get_confidence_score("jean.pierre@lafrance.fr", "jean pierre")
    assert 0.0 <= score <= 1.0
