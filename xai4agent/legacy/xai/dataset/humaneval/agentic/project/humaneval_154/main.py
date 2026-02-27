def cycpattern_check(a , b):
    """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word
    cycpattern_check("abcd","abd") => False
    cycpattern_check("hello","ell") => True
    cycpattern_check("whassup","psus") => False
    cycpattern_check("abab","baa") => True
    cycpattern_check("efef","eeff") => False
    cycpattern_check("himenss","simen") => True

    """
    # 如果b为空，则返回True（空字符串是任何字符串的子串）
    if not b:
        return True

    # 如果b比a长，则不可能是a的子串
    if len(b) > len(a):
        return False

    # 生成b的所有旋转形式并检查是否为a的子串
    # 一个字符串的所有旋转可以通过将字符串连接到自身然后取子串得到
    double_b = b + b
    for i in range(len(b)):
        rotation = double_b[i:i+len(b)]
        if rotation in a:
            return True

    return False
