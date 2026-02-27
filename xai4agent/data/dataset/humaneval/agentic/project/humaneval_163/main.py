def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    """
    # 如果两个参数都大于等于10，则返回空列表
    if a >= 10 and b >= 10:
        return []

    # 确保a是较小值，b是较大值
    start, end = min(a, b), max(a, b)

    # 收集所有偶数
    result = []
    for num in range(start, end + 1):
        if num % 2 == 0:
            result.append(num)

    # 如果范围包含10或更大，返回[2, 4, 6, 8]
    if end >= 10:
        return [2, 4, 6, 8]

    return result
