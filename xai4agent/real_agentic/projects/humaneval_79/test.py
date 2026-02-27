from main import decimal_to_binary

# Test the examples from the docstring
print("Testing decimal_to_binary(15):")
result1 = decimal_to_binary(15)
print(f"Result: {result1}")
print(f"Expected: db1111db")
print(f"Match: {result1 == 'db1111db'}")

print("\nTesting decimal_to_binary(32):")
result2 = decimal_to_binary(32)
print(f"Result: {result2}")
print(f"Expected: db100000db")
print(f"Match: {result2 == 'db100000db'}")

# Additional tests
print("\nTesting additional cases:")
print(f"decimal_to_binary(0): {decimal_to_binary(0)}")
print(f"decimal_to_binary(1): {decimal_to_binary(1)}")
print(f"decimal_to_binary(8): {decimal_to_binary(8)}")