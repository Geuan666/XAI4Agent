def closest_integer(value):
    '''
    Create a function that takes a value (string) representing a number
    and returns the closest integer to it. If the number is equidistant
    from two integers, round it away from zero.

    Examples
    >>> closest_integer("10")
    10
    >>> closest_integer("15.3")
    15

    Note:
    Rounding away from zero means that if the given number is equidistant
    from two integers, the one you should return is the one that is the
    farthest from zero. For example closest_integer("14.5") should
    return 15 and closest_integer("-14.5") should return -15.
    '''
    num = float(value)
    if num >= 0:
        # For positive numbers, we round normally, but if exactly .5, we round up
        if num - int(num) == 0.5:
            return int(num) + 1
        else:
            return round(num)
    else:
        # For negative numbers, we also round normally, but if exactly .5, we round down (away from zero)
        if num - int(num) == -0.5:
            return int(num) - 1
        else:
            return round(num)

# Test cases
print(closest_integer("10"))   # Should output: 10
print(closest_integer("15.3")) # Should output: 15
print(closest_integer("14.5")) # Should output: 15 (rounding away from zero)
print(closest_integer("-14.5")) # Should output: -15 (rounding away from zero)