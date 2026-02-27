from main import total_match

# Test cases from the docstring
print("Test 1:", total_match([], []))
print("Test 2:", total_match(['hi', 'admin'], ['hI', 'Hi']))
print("Test 3:", total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']))
print("Test 4:", total_match(['hi', 'admin'], ['hI', 'hi', 'hi']))
print("Test 5:", total_match(['4'], ['1', '2', '3', '4', '5']))

# Additional verification
print("Test 6:", total_match(['a', 'b', 'c'], ['d', 'e']))  # Should return ['d', 'e'] since 3 < 4
print("Test 7:", total_match(['x', 'y'], ['z', 'w']))      # Should return ['x', 'y'] since 4 == 4 (first list)