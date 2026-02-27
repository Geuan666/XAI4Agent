from main import parse_nested_parens

# Test with the provided example
result = parse_nested_parens('(()()) ((())) () ((())()())')
print(result)
expected = [2, 3, 1, 3]
print(f"Expected: {expected}")
print(f"Got: {result}")
print(f"Match: {result == expected}")