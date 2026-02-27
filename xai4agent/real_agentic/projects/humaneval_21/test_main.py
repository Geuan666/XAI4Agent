from main import rescale_to_unit

# Test with the provided example
result = rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
print("Result:", result)
print("Expected: [0.0, 0.25, 0.5, 0.75, 1.0]")
print("Match:", result == [0.0, 0.25, 0.5, 0.75, 1.0])

# Test edge cases
print("\nEdge case - all same numbers:")
result2 = rescale_to_unit([5.0, 5.0, 5.0])
print("Result:", result2)
print("Expected: [0.0, 0.0, 0.0]")
print("Match:", result2 == [0.0, 0.0, 0.0])

print("\nEdge case - two numbers:")
result3 = rescale_to_unit([10.0, 20.0])
print("Result:", result3)
print("Expected: [0.0, 1.0]")
print("Match:", result3 == [0.0, 1.0])