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
    count = 0
    for num in arr:
        if num == 0:
            # Sum of digits of 0 is 0, so it doesn't count
            continue
        elif num > 0:
            # For positive numbers, sum all digits
            digit_sum = sum(int(digit) for digit in str(num))
            if digit_sum > 0:
                count += 1
        else:
            # For negative numbers, make first digit negative
            num_str = str(abs(num))
            digit_sum = -int(num_str[0]) + sum(int(digit) for digit in num_str[1:])
            if digit_sum > 0:
                count += 1
    return count