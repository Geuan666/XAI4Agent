def simplify(x, n):
    """Your task is to implement a function that will simplify the expression
    x * n. The function returns True if x * n evaluates to a whole number and False
    otherwise. Both x and n, are string representation of a fraction, and have the following format,
    <numerator>/<denominator> where both numerator and denominator are positive whole numbers.

    You can assume that x, and n are valid fractions, and do not have zero as denominator.

    simplify("1/5", "5/1") = True
    simplify("1/6", "2/1") = False
    simplify("7/10", "10/2") = False
    """
    # 解析第一个分数
    num1, den1 = map(int, x.split('/'))
    # 解析第二个分数
    num2, den2 = map(int, n.split('/'))

    # 计算乘积：(num1/den1) * (num2/den2) = (num1*num2)/(den1*den2)
    result_num = num1 * num2
    result_den = den1 * den2

    # 判断结果是否为整数：如果分子能被分母整除，则为整数
    return result_num % result_den == 0
