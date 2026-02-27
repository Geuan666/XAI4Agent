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