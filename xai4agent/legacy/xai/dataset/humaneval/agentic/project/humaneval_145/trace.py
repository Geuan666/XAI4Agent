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
        """Calculate the sum of digits of an integer."""
        return sum(int(digit) for digit in str(abs(n)))
    
    # Create list of tuples (digit_sum, original_index, value) 
    # and sort by digit_sum first, then by original_index for stability
    indexed_nums = [(digit_sum(num), i, num) for i, num in enumerate(nums)]
    indexed_nums.sort(key=lambda x: (x[0], x[1]))
    
    # Return just the values
    return [num for _, _, num in indexed_nums]

# Test the example case
result = order_by_points([1, 11, -1, -11, -12])
expected = [-1, -11, 1, -12, 11]
print("Result:", result)
print("Expected:", expected)
print("Match:", result == expected)

# Let's trace through what should happen:
print("\nTracing:")
nums = [1, 11, -1, -11, -12]
for i, num in enumerate(nums):
    print(f"Index {i}: {num} -> digit_sum = {sum(int(d) for d in str(abs(num)))}")