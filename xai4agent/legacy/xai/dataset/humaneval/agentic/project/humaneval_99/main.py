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
    # For numbers that are exactly halfway between two integers,
    # we round away from zero
    if num >= 0:
        # Positive numbers: round up (away from zero) when exactly halfway
        return int(num + 0.5)
    else:
        # Negative numbers: round down (away from zero) when exactly halfway
        return int(num - 0.5)
