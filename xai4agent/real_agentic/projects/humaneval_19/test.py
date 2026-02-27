from main import sort_numbers

# Test with the example from the docstring
result = sort_numbers('three one five')
print(f"Input: 'three one five'")
print(f"Output: '{result}'")
print(f"Expected: 'one three five'")
print(f"Test passed: {result == 'one three five'}")