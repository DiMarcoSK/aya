from data_processor.personal_info_extractor import PersonalInfoExtractor


def test_extract_personal_info_splits_first_and_last_name():
    extractor = PersonalInfoExtractor()
    info = extractor.extract_personal_info("jean.pierre@example.com")
    assert info["first_name"] == "jean"
    assert info["last_name"] == "pierre"


def test_extract_personal_info_finds_birth_year_in_username():
    extractor = PersonalInfoExtractor()
    info = extractor.extract_personal_info("thomas1995@example.com")
    assert "1995" in info["potential_birth_years"]


def test_extract_personal_info_handles_single_word_username():
    extractor = PersonalInfoExtractor()
    info = extractor.extract_personal_info("admin@example.com")
    assert info["first_name"] == "admin"
    assert "last_name" not in info


def test_analyze_username_patterns_detects_numbers_and_separators():
    extractor = PersonalInfoExtractor()
    patterns = extractor.analyze_username_patterns("john.doe99")
    assert patterns["has_numbers"] is True
    assert patterns["has_separators"] is True
    assert patterns["ends_with_number"] is True
    assert patterns["starts_with_number"] is False
