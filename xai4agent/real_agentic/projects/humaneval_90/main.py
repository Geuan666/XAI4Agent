def next_smallest(lst):
    """
    You are given a list of integers.
    Write a function next_smallest() that returns the 2nd smallest element of the list.
    Return None if there is no such element.
    
    next_smallest([1, 2, 3, 4, 5]) == 2
    next_smallest([5, 1, 4, 3, 2]) == 2
    next_smallest([]) == None
    next_smallest([1, 1]) == None
    """
    if len(lst) < 2:
        return None
    
    # Get unique elements and sort them
    unique_elements = sorted(set(lst))
    
    # If we don't have at least 2 unique elements, return None
    if len(unique_elements) < 2:
        return None
    
    # Return the second smallest element
    return unique_elements[1]