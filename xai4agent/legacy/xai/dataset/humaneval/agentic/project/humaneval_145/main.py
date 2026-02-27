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

    # Create list of (digit_sum, original_index, value) tuples and sort
    indexed_items = [(digit_sum(num), i, num) for i, num in enumerate(nums)]
    indexed_items.sort(key=lambda x: (x[0], x[1]))

    # Return just the values
    return [item[2] for item in indexed_items]
