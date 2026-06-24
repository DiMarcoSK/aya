from data_processor.data_validator import DataValidator


def test_validate_email_accepts_well_formed_address():
    validator = DataValidator()
    assert validator.validate_email("jean.pierre@lafrance.com") is True


def test_validate_email_rejects_malformed_address():
    validator = DataValidator()
    assert validator.validate_email("not-an-email") is False
    assert validator.validate_email("") is False


def test_validate_password_rejects_control_characters():
    validator = DataValidator()
    assert validator.validate_password("pass\nword") is False
    assert validator.validate_password("password123") is True


def test_validate_password_enforces_length_bounds():
    validator = DataValidator()
    assert validator.validate_password("") is False
    assert validator.validate_password("a" * 129) is False
    assert validator.validate_password("a" * 128) is True


def test_clean_entry_splits_on_first_colon_only():
    validator = DataValidator()
    email, password = validator.clean_entry("User@Example.com:pa:ss:word\n")
    assert email == "user@example.com"
    assert password == "pa:ss:word"


def test_clean_entry_rejects_lines_without_separator():
    validator = DataValidator()
    email, password = validator.clean_entry("not-a-valid-line")
    assert email is None
    assert password is None


def test_validate_entry_reports_reason_for_invalid_email():
    validator = DataValidator()
    is_valid, reason = validator.validate_entry("bad-email", "password123")
    assert is_valid is False
    assert "Invalid email" in reason
