def iscube(a):
    '''
    Write a function that takes an integer a and returns True
    if this ingeger is a cube of some integer number.
    Note: you may assume the input is always valid.
    Examples:
    iscube(1) ==> True
    iscube(2) ==> False
    iscube(-1) ==> True
    iscube(64) ==> True
    iscube(0) ==> True
    iscube(180) ==> False
    '''
    # 处理特殊情况：0的立方是0
    if a == 0:
        return True

    # 对于正数，计算立方根并检查是否为整数
    if a > 0:
        cube_root = round(a ** (1/3))
        return cube_root ** 3 == a

    # 对于负数，取绝对值计算立方根，然后检查其立方是否等于原数的绝对值
    else:
        abs_a = abs(a)
        cube_root = round(abs_a ** (1/3))
        return cube_root ** 3 == abs_a
