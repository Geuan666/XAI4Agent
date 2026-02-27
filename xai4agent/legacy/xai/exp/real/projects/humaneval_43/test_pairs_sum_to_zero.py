def pairs_sum_to_zero(l):
    """
    pairs_sum_to_zero takes a list of integers as an input.
    it returns True if there are two distinct elements in the list that
    sum to zero, and False otherwise.
    >>> pairs_sum_to_zero([1, 3, 5, 0])
    False
    >>> pairs_sum_to_zero([1, 3, -2, 1])
    False
    >>> pairs_sum_to_zero([1, 2, 3, 7])
    False
    >>> pairs_sum_to_zero([2, 4, -5, 3, 5, 7])
    True
    >>> pairs_sum_to_zero([1])
    False
    """
    # Check if list has less than 2 elements
    if len(l) < 2:
        return False
    
    # Use a set to store numbers we've seen
    seen = set()
    
    for num in l:
        # Check if the complement (negative) of current number exists in seen
        if -num in seen:
            return True
        # Add current number to seen set
        seen.add(num)
    
    return False

# Test cases
print("Testing pairs_sum_to_zero function:")
print(f"pairs_sum_to_zero([1, 3, 5, 0]) = {pairs_sum_to_zero([1, 3, 5, 0])}")  # Expected: False
print(f"pairs_sum_to_zero([1, 3, -2, 1]) = {pairs_sum_to_zero([1, 3, -2, 1])}")  # Expected: False
print(f"pairs_sum_to_zero([1, 2, 3, 7]) = {pairs_sum_to_zero([1, 2, 3, 7])}")  # Expected: False
print(f"pairs_sum_to_zero([2, 4, -5, 3, 5, 7]) = {pairs_sum_to_zero([2, 4, -5, 3, 5, 7])}")  # Expected: True
print(f"pairs_sum_to_zero([1]) = {pairs_sum_to_zero([1])}")  # Expected: False

# Additional test cases
print("\nAdditional test cases:")
print(f"pairs_sum_to_zero([1, -1]) = {pairs_sum_to_zero([1, -1])}")  # Expected: True
print(f"pairs_sum_to_zero([0, 0]) = {pairs_sum_to_zero([0, 0])}")  # Expected: True
print(f"pairs_sum_to_zero([]) = {pairs_sum_to_zero([])}")  # Expected: False