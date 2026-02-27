def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    # 从 n//2 开始向下遍历，找到的第一个因数就是最大的真因数
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
