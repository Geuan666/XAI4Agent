from main import largest_smallest_integers

# Test cases from the docstring
print("Test 1:", largest_smallest_integers([2, 4, 1, 3, 5, 7]))  # Expected: (None, 1)
print("Test 2:", largest_smallest_integers([]))                 # Expected: (None, None)
print("Test 3:", largest_smallest_integers([0]))                # Expected: (None, None)

# Additional test cases
print("Test 4:", largest_smallest_integers([-1, -2, -3]))       # Expected: (-1, None)
print("Test 5:", largest_smallest_integers([1, 2, 3]))          # Expected: (None, 1)
print("Test 6:", largest_smallest_integers([-5, -2, 3, 7]))     # Expected: (-2, 3)