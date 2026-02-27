def do_algebra(operator, operand):
    """
    Given two lists operator, and operand. The first list has basic algebra operations, and
    the second list is a list of integers. Use the two given lists to build the algebric
    expression and return the evaluation of this expression.

    The basic algebra operations:
    Addition ( + )
    Subtraction ( - )
    Multiplication ( * )
    Floor division ( // )
    Exponentiation ( ** )

    Example:
    operator['+', '*', '-']
    array = [2, 3, 4, 5]
    result = 2 + 3 * 4 - 5
    => result = 9

    Note:
        The length of operator list is equal to the length of operand list minus one.
        Operand is a list of of non-negative integers.
        Operator list has at least one operator, and operand list has at least two operands.

    """
    # 创建操作数和操作符的副本以避免修改原始列表
    ops = operator[:]
    nums = operand[:]

    # 首先处理幂运算 (**)
    i = 0
    while i < len(ops):
        if ops[i] == '**':
            nums[i] = nums[i] ** nums[i + 1]
            nums.pop(i + 1)
            ops.pop(i)
        else:
            i += 1

    # 然后处理乘法 (*) 和除法 (//)
    i = 0
    while i < len(ops):
        if ops[i] == '*':
            nums[i] = nums[i] * nums[i + 1]
            nums.pop(i + 1)
            ops.pop(i)
        elif ops[i] == '//':
            nums[i] = nums[i] // nums[i + 1]
            nums.pop(i + 1)
            ops.pop(i)
        else:
            i += 1

    # 最后处理加法 (+) 和减法 (-)
    result = nums[0]
    for i in range(len(ops)):
        if ops[i] == '+':
            result += nums[i + 1]
        elif ops[i] == '-':
            result -= nums[i + 1]

    return result
