def specialFilter(nums):
    """Write a function that takes an array of numbers as input and returns
    the number of elements in the array that are greater than 10 and both
    first and last digits of a number are odd (1, 3, 5, 7, 9).
    For example:
    specialFilter([15, -73, 14, -15]) => 1
    specialFilter([33, -2, -3, 45, 21, 109]) => 2
    """
    # 奇数数字集合
    odd_digits = {'1', '3', '5', '7', '9'}

    count = 0
    for num in nums:
        # 检查数字是否大于10
        if num > 10:
            # 转换为字符串以获取首位和末位数字
            num_str = str(num)
            first_digit = num_str[0]
            last_digit = num_str[-1]

            # 检查首位和末位数字是否都是奇数
            if first_digit in odd_digits and last_digit in odd_digits:
                count += 1

    return count
