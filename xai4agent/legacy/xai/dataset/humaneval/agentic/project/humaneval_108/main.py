def count_nums(arr):
    """
    Write a function count_nums which takes an array of integers and returns
    the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative:
    e.g. -123 has signed digits -1, 2, and 3.
    >>> count_nums([]) == 0
    >>> count_nums([-1, 11, -11]) == 1
    >>> count_nums([1, 1, 2]) == 3
    """
    def sum_of_digits(n):
        # Convert to string to handle digits
        s = str(abs(n))
        digits = [int(d) for d in s]
        # If the number is negative, make the first digit negative
        if n < 0:
            digits[0] = -digits[0]
        return sum(digits)

    count = 0
    for num in arr:
        if sum_of_digits(num) > 0:
            count += 1
    return count
