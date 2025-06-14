import re
from typing import Tuple, Optional


class DataValidator:
    """Validates email and password entries"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.min_password_length = 1
        self.max_password_length = 128

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        if not email or len(email) > 254:
            return False
        return bool(self.email_pattern.match(email))

    def validate_password(self, password: str) -> bool:
        """Validate password constraints"""
        if not password:
            return False
        
        if len(password) < self.min_password_length or len(password) > self.max_password_length:
            return False
        
        # Check for problematic characters
        if any(char in password for char in ['\n', '\r', '\t']):
            return False
        
        return True

    def validate_entry(self, email: str, password: str) -> Tuple[bool, Optional[str]]:
        """Validate complete entry and return status with error message"""
        if not self.validate_email(email):
            return False, f"Invalid email format: {email}"
        
        if not self.validate_password(password):
            return False, f"Invalid password for {email}"
        
        return True, None

    def clean_entry(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Clean and extract email:password from line"""
        line = line.strip()
        if not line or ':' not in line:
            return None, None
        
        parts = line.split(':', 1)
        if len(parts) != 2:
            return None, None
        
        email = parts[0].strip().lower()
        password = parts[1].strip()
        
        return email, password


