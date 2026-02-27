def is_simple_power(x, n):
    """Your task is to write a function that returns true if a number x is a simple
    power of n and false in other cases.
    x is a simple power of n if n**int=x
    For example:
    is_simple_power(1, 4) => true
    is_simple_power(2, 2) => true
    is_simple_power(8, 2) => true
    is_simple_power(3, 2) => false
    is_simple_power(3, 1) => false
    is_simple_power(5, 3) => false
    """
    # Special case: if x is 1, then n^0 = 1 for any n != 0
    if x == 1:
        return True

    # Special case: if n is 1, then 1^k = 1 for any k, so only x=1 works
    if n == 1:
        return False

    # Special case: if n is 0, then 0^k = 0 for k > 0, and 0^0 is undefined
    if n == 0:
        return x == 0

    # For other cases, we need to check if x is a power of n
    # Start with n^1 and keep multiplying by n until we reach or exceed x
    power = n
    while power < x:
        power *= n

    return power == x
