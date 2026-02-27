def multiply(a, b):
    """Complete the function that takes two integers and returns 
    the product of their unit digits.
    Assume the input is always valid.
    Examples:
    multiply(148, 412) should return 16.
    multiply(19, 28) should return 72.
    multiply(2020, 1851) should return 0.
    multiply(14,-15) should return 20.
    """
    unit_a = abs(a) % 10
    unit_b = abs(b) % 10
    return unit_a * unit_b

# Test the examples from the docstring
print("Testing multiply function:")
print(f"multiply(148, 412) = {multiply(148, 412)} (expected: 16)")
print(f"multiply(19, 28) = {multiply(19, 28)} (expected: 72)")
print(f"multiply(2020, 1851) = {multiply(2020, 1851)} (expected: 0)")
print(f"multiply(14, -15) = {multiply(14, -15)} (expected: 20)")