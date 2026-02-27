def add(lst):
    """Given a non-empty list of integers lst. add the even elements that are at odd indices..


    Examples:
        add([4, 2, 6, 7]) ==> 2
    """
    total = 0
    for i in range(1, len(lst), 2):  # 从索引1开始，步长为2，只遍历奇数索引
        if lst[i] % 2 == 0:  # 检查该元素是否为偶数
            total += lst[i]
    return total
