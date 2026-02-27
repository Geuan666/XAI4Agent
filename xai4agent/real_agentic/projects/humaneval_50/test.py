from main import encode_shift, decode_shift

# Test the functions
test_string = "hello"
encoded = encode_shift(test_string)
decoded = decode_shift(encoded)

print(f"Original: {test_string}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Match: {test_string == decoded}")