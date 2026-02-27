import re

def is_valid_email(email):
    """
    Check if the given string is a valid email address.
    
    A valid email address has the following characteristics:
    - It contains exactly one '@' symbol
    - The part before '@' (local part) is not empty and contains only alphanumeric characters, dots, underscores, or hyphens
    - The part after '@' (domain part) is not empty and contains at least one dot
    - The domain part contains only alphanumeric characters, dots, or hyphens
    - The domain part does not start or end with a dot or hyphen
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Check if email is a string
    if not isinstance(email, str):
        return False
    
    # Check for exactly one @ symbol
    if email.count('@') != 1:
        return False
    
    # Split into local and domain parts
    local_part, domain_part = email.split('@')
    
    # Check if both parts are non-empty
    if not local_part or not domain_part:
        return False
    
    # Check local part: alphanumeric, dots, underscores, or hyphens
    if not re.match(r'^[a-zA-Z0-9._-]+$', local_part):
        return False
    
    # Check domain part: not empty, contains at least one dot, 
    # only alphanumeric, dots, or hyphens, doesn't start/end with dot/hyphen
    if not re.match(r'^[a-zA-Z0-9.-]+$', domain_part):
        return False
    
    if '.' not in domain_part:
        return False
    
    if domain_part.startswith('.') or domain_part.startswith('-') or domain_part.endswith('.') or domain_part.endswith('-'):
        return False
    
    return True