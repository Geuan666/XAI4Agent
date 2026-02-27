def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    It must be implemented like this:
    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]
    """
    def count_ones(n):
        return bin(n).count('1')

    # 对于负数，我们按照其绝对值的二进制中1的个数排序，如果相同则按数值排序
    # 对于正数和0，我们按照二进制中1的个数排序，如果相同则按数值排序
    return sorted(arr, key=lambda x: (count_ones(abs(x)), x))
