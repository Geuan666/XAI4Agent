def unique_digits(x):
    """Given a list of positive integers x. return a sorted list of all 
    elements that hasn't any even digit.

    Note: Returned list should be sorted in increasing order.
    
    For example:
    >>> unique_digits([15, 33, 1422, 1])
    [1, 15, 33]
    >>> unique_digits([152, 323, 1422, 10])
    []
    """
    result = []
    
    for num in x:
        # Convert number to string to check each digit
        num_str = str(num)
        has_even_digit = False
        
        # Check each digit
        for digit_char in num_str:
            digit = int(digit_char)
            if digit % 2 == 0:  # Even digit found
                has_even_digit = True
                break
                
        # If no even digit found, add to result
        if not has_even_digit:
            result.append(num)
    
    # Return sorted result
    return sorted(result)