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
    # Special case: if n is 0, then x must be 0 (only valid when n^k = 0, which happens when n=0 and k>0)
    if n == 0:
        return x == 0
    
    # Special case: if n is 1, then x must be 1 (since 1^k = 1 for all k)
    if n == 1:
        return x == 1
    
    # Special case: if n is -1, then x must be 1 or -1 (since (-1)^k = 1 when k is even, (-1)^k = -1 when k is odd)
    if n == -1:
        return x == 1 or x == -1
    
    # Special case: if x is 1, then any non-zero n works (since n^0 = 1 for any n != 0)
    if x == 1:
        return True
    
    # Special case: if x is 0, then n must also be 0 (but 0^k = 0 only when k > 0, and 0^0 is undefined)
    if x == 0:
        return False
    
    # For positive n and positive x
    if n > 0 and x > 0:
        current = 1
        while current < x:
            current *= n
        return current == x
    
    # For negative n and positive x
    if n < 0 and x > 0:
        # Check if x could be a power of |n| with even exponent
        current = 1
        power = 0
        while current < x:
            power += 1
            current *= abs(n)
        return current == x and power % 2 == 0
    
    # For negative n and negative x  
    if n < 0 and x < 0:
        # Check if |x| could be a power of |n| with odd exponent
        current = 1
        power = 0
        while current < abs(x):
            power += 1
            current *= abs(n)
        return current == abs(x) and power % 2 == 1
    
    # For positive n and negative x
    if n > 0 and x < 0:
        return False
    
    return False