def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    # 从最小的质因数2开始
    largest_factor = 1
    divisor = 2

    while divisor * divisor <= n:
        # 当n能被divisor整除时
        while n % divisor == 0:
            largest_factor = divisor
            n //= divisor
        divisor += 1

    # 如果n大于1，那么它本身就是一个质因数（且是最大的）
    if n > 1:
        largest_factor = n

    return largest_factor
