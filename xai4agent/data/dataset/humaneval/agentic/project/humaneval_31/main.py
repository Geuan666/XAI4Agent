def is_prime(n):
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(13441)
    True
    >>> is_prime(61)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    """
    # 处理小于2的情况
    if n < 2:
        return False

    # 2是素数
    if n == 2:
        return True

    # 偶数不是素数（除了2）
    if n % 2 == 0:
        return False

    # 只需检查到sqrt(n)，且只检查奇数
    import math
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True
