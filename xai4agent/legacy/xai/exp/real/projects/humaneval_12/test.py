def sum_even_numbers(numbers):
    """
    Given a list of integers, return the sum of all even numbers.
    
    Args:
        numbers: A list of integers
        
    Returns:
        int: The sum of all even numbers in the list
    """
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total

# Test cases
print(sum_even_numbers([1, 2, 3, 4, 5, 6]))  # Expected: 12 (2 + 4 + 6)
print(sum_even_numbers([1, 3, 5, 7]))        # Expected: 0 (no even numbers)
print(sum_even_numbers([2, 4, 6, 8]))        # Expected: 20 (2 + 4 + 6 + 8)
print(sum_even_numbers([]))                  # Expected: 0 (empty list)
print(sum_even_numbers([-2, -1, 0, 1, 2]))   # Expected: 0 (-2 + 0 + 2)