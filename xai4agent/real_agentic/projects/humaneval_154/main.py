def cycpattern_check(a , b):
    """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word
    cycpattern_check("abcd","abd") => False
    cycpattern_check("hello","ell") => True
    cycpattern_check("whassup","psus") => False
    cycpattern_check("abab","baa") => True
    cycpattern_check("efef","eeff") => False
    cycpattern_check("himenss","simen") => True

    """
    # If b is empty, it should be considered as a substring of a
    if not b:
        return True
    
    # If b is longer than a, it cannot be a substring
    if len(b) > len(a):
        return False
    
    # Generate all possible rotations of b
    # A rotation of string b can be obtained by slicing: b[i:] + b[:i]
    for i in range(len(b)):
        rotation = b[i:] + b[:i]
        if rotation in a:
            return True
    
    # If no rotation of b is found in a, return False
    return False