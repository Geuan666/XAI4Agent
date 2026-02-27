def add(lst):
    """Given a non-empty list of integers lst. add the even elements that are at odd indices..


    Examples:
        add([4, 2, 6, 7]) ==> 2 
    """
    total = 0
    for i in range(1, len(lst), 2):  # Start at index 1 (first odd index), step by 2
        if lst[i] % 2 == 0:  # Check if element is even
            total += lst[i]
    return total