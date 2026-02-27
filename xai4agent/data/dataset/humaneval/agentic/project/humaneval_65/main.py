def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    """
    x_str = str(x)
    num_digits = len(x_str)

    # 如果shift大于数字位数，返回反转的数字
    if shift > num_digits:
        return x_str[::-1]

    # 处理shift为0或负数的情况
    shift = shift % num_digits

    # 循环右移
    if shift == 0:
        return x_str
    else:
        return x_str[-shift:] + x_str[:-shift]
