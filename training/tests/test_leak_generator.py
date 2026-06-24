import random

from data_processor.leak_generator import RealisticLeakGenerator


def test_generate_single_entry_returns_email_and_password():
    random.seed(0)
    generator = RealisticLeakGenerator()
    email, password = generator.generate_single_entry()
    assert "@" in email
    assert len(password) > 0


def test_password_correlates_with_identity_above_chance():
    """Regression test for the original bug: usernames and passwords were
    generated independently, so the dataset carried no learnable
    relationship between identity features and the password. At least a
    meaningful share of generated passwords must now contain the
    identity's first or last name or birth year."""
    random.seed(1)
    generator = RealisticLeakGenerator()

    correlated = 0
    n = 500
    for _ in range(n):
        identity = generator._generate_identity()
        password = generator._generate_password(identity).lower()
        if identity["first"].lower() in password or identity["last"].lower() in password:
            correlated += 1
        elif identity["birth_year"] and str(identity["birth_year"]) in password:
            correlated += 1

    assert correlated / n > 0.4


def test_generate_leak_file_writes_requested_number_of_lines(tmp_path):
    random.seed(2)
    generator = RealisticLeakGenerator()
    output_file = tmp_path / "leaks.txt"

    generator.generate_leak_file(str(output_file), 25)

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25
    assert all(":" in line for line in lines)
