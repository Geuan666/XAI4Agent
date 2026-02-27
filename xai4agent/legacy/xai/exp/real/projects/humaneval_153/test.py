from main import Strongest_Extension

# Test case 1: From the docstring example
result1 = Strongest_Extension('Slices', ['SErviNGSliCes', 'Cheese', 'StuFfed'])
print(f"Test 1: {result1}")
# Expected: Slices.SErviNGSliCes

# Test case 2: From the docstring example
result2 = Strongest_Extension('my_class', ['AA', 'Be', 'CC'])
print(f"Test 2: {result2}")
# Expected: my_class.AA

# Additional test cases
result3 = Strongest_Extension('TestClass', ['ABC', 'def', 'GHI'])
print(f"Test 3: {result3}")

result4 = Strongest_Extension('Example', ['AbCd', 'EfGh', 'IjKl'])
print(f"Test 4: {result4}")