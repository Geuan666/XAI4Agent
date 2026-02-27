def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
    while b:
        a, b = b, a % b
    return a


# Test the function
if __name__ == "__main__":
    # Test cases from docstring
    print(greatest_common_divisor(3, 5))   # Should output 1
    print(greatest_common_divisor(25, 15)) # Should output 5
    
    # Additional test cases
    print(greatest_common_divisor(12, 8))  # Should output 4
    print(greatest_common_divisor(17, 13)) # Should output 1 (both prime)