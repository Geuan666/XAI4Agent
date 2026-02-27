from main import filter_by_substring

# Test case 1: Empty list
result1 = filter_by_substring([], 'a')
print(f"Test 1 - Empty list: {result1}")
assert result1 == []

# Test case 2: List with strings containing the substring
result2 = filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
print(f"Test 2 - Strings with 'a': {result2}")
assert result2 == ['abc', 'bacd', 'array']

# Test case 3: No matches
result3 = filter_by_substring(['hello', 'world'], 'z')
print(f"Test 3 - No matches: {result3}")
assert result3 == []

# Test case 4: All match
result4 = filter_by_substring(['cat', 'bat', 'rat'], 'a')
print(f"Test 4 - All match: {result4}")
assert result4 == ['cat', 'bat', 'rat']

print("All tests passed!")