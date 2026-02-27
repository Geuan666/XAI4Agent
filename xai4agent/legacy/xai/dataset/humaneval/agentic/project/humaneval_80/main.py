def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 consecutive letters are distinct
    For example:
    is_happy(a) => False
    is_happy(aa) => False
    is_happy(abcd) => True
    is_happy(aabb) => False
    is_happy(adb) => True
    is_happy(xyy) => False
    """
    # 长度至少为3
    if len(s) < 3:
        return False

    # 检查每组3个连续字符是否都不相同
    for i in range(len(s) - 2):
        # 获取三个连续字符
        three_chars = s[i:i+3]
        # 如果三个字符中有重复，则不是happy字符串
        if len(set(three_chars)) != 3:
            return False

    return True
