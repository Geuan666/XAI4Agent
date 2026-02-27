def change_base(x: int, base: int):
    """Change numerical base of input number x to base.
    return string representation after the conversion.
    base numbers are less than 10.
    >>> change_base(8, 3)
    '22'
    >>> change_base(8, 2)
    '1000'
    >>> change_base(7, 2)
    '111'
    """
    if x == 0:
        return '0'
    
    result = ''
    while x > 0:
        result = str(x % base) + result
        x //= base
    
    return result

# Test the function
print("Testing change_base function:")
print(f"change_base(8, 3) = {change_base(8, 3)}")  # Should be '22'
print(f"change_base(8, 2) = {change_base(8, 2)}")  # Should be '1000'
print(f"change_base(7, 2) = {change_base(7, 2)}")  # Should be '111'
print(f"change_base(0, 2) = {change_base(0, 2)}")  # Should be '0'