def compare_one(a, b):
    """
    Create a function that takes integers, floats, or strings representing
    real numbers, and returns the larger variable in its given variable type.
    Return None if the values are equal.
    Note: If a real number is represented as a string, the floating point might be . or ,

    compare_one(1, 2.5) ➞ 2.5
    compare_one(1, "2,3") ➞ "2,3"
    compare_one("5,1", "6") ➞ "6"
    compare_one("1", 1) ➞ None
    """
    def parse_value(x):
        """Convert value to float for comparison"""
        if isinstance(x, str):
            # Replace comma with dot for decimal point
            x = x.replace(',', '.')
        return float(x)

    # Parse both values for comparison
    parsed_a = parse_value(a)
    parsed_b = parse_value(b)

    # Compare and return appropriate value
    if parsed_a > parsed_b:
        return a
    elif parsed_b > parsed_a:
        return b
    else:
        return None
