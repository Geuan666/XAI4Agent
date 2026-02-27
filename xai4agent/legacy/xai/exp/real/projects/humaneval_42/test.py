from main import incr_list

# Test case 1
result1 = incr_list([1, 2, 3])
print(f"incr_list([1, 2, 3]) = {result1}")
expected1 = [2, 3, 4]
print(f"Expected: {expected1}")
print(f"Test 1 passed: {result1 == expected1}")

# Test case 2
result2 = incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
print(f"\nincr_list([5, 3, 5, 2, 3, 3, 9, 0, 123]) = {result2}")
expected2 = [6, 4, 6, 3, 4, 4, 10, 1, 124]
print(f"Expected: {expected2}")
print(f"Test 2 passed: {result2 == expected2}")