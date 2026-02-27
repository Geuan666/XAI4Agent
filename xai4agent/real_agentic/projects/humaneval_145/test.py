def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    in ascending order according to the sum of their digits.
    Note: if there are several items with similar sum of their digits,
    order them based on their index in original list.

    For example:
    >>> order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]
    >>> order_by_points([]) == []
    """
    def digit_sum(n):
        # Calculate sum of digits, handling negative numbers
        return sum(int(digit) for digit in str(abs(n)))
    
    # Create pairs of (digit_sum, original_index, value) for sorting
    indexed_nums = [(digit_sum(num), i, num) for i, num in enumerate(nums)]
    
    # Sort by digit sum first, then by original index for stability
    indexed_nums.sort(key=lambda x: (x[0], x[1]))
    
    # Extract just the values in the new order
    return [num for _, _, num in indexed_nums]

# Test
result = order_by_points([1, 11, -1, -11, -12])
print("Result:", result)
print("Expected: [-1, -11, 1, -12, 11]")

# Let's also test the empty list case
result2 = order_by_points([])
print("Empty result:", result2)